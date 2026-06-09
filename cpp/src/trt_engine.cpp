#include "trt_engine.h"
#include "cuda_preprocess.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        fprintf(stderr, "[TRT] %s\n", msg);
    }
}

bool ExecutionContext::allocate(const InferConfig& config,
                                nvinfer1::ICudaEngine* engine,
                                int input_index,
                                const std::vector<int>& out_indices,
                                const std::vector<size_t>& out_sizes,
                                size_t in_size) {
    if (input_index >= 0) {
        if (cudaMalloc(&input_device, in_size) != cudaSuccess) {
            fprintf(stderr, "[TRT] ctx=%d: Failed to allocate input device buffer\n", slot_id);
            return false;
        }
        input_host = static_cast<float*>(malloc(in_size));
        if (!input_host) return false;
    }

    for (size_t i = 0; i < out_indices.size(); ++i) {
        void* dev_ptr = nullptr;
        if (cudaMalloc(&dev_ptr, out_sizes[i]) != cudaSuccess) {
            fprintf(stderr, "[TRT] ctx=%d: Failed to allocate output device buffer %zu\n", slot_id, i);
            return false;
        }
        output_device_ptrs.push_back(dev_ptr);

        float* host_ptr = static_cast<float*>(malloc(out_sizes[i]));
        if (!host_ptr) return false;
        output_hosts.push_back(host_ptr);
    }

    if (cudaStreamCreateWithFlags(&own_stream, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "[TRT] ctx=%d: Failed to create CUDA stream\n", slot_id);
        return false;
    }

    fprintf(stderr, "[TRT] ctx=%d: Allocated exclusive buffers (input=%zu bytes, outputs=%zu)\n",
            slot_id, in_size, out_sizes.size());
    return true;
}

void ExecutionContext::free() {
    if (input_device) { cudaFree(input_device); input_device = nullptr; }
    if (input_host) { ::free(input_host); input_host = nullptr; }

    for (auto* p : output_device_ptrs) { if (p) cudaFree(p); }
    output_device_ptrs.clear();

    for (auto* p : output_hosts) { if (p) ::free(p); }
    output_hosts.clear();

    if (own_stream) { cudaStreamDestroy(own_stream); own_stream = nullptr; }
    if (trt_ctx) { trt_ctx->destroy(); trt_ctx = nullptr; }
}

TRTEngine::TRTEngine() = default;

TRTEngine::~TRTEngine() {
    unload();
}

bool TRTEngine::load(const InferConfig& config) {
    config_ = config;

    cudaSetDevice(config_.device_id);

    std::ifstream engine_file(config_.engine_path, std::ios::binary);
    if (!engine_file.is_open()) {
        fprintf(stderr, "[TRT] Failed to open engine: %s\n", config_.engine_path.c_str());
        return false;
    }

    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) return false;

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        fprintf(stderr, "[TRT] Failed to deserialize engine\n");
        return false;
    }

    const int nb = engine_->getNbBindings();
    for (int i = 0; i < nb; ++i) {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            vol *= static_cast<size_t>(dims.d[d] > 0 ? dims.d[d] : 1);
        }

        if (engine_->bindingIsInput(i)) {
            input_index_ = i;
            input_size_ = vol * sizeof(float);
        } else {
            output_indices_.push_back(i);
            output_sizes_.push_back(vol * sizeof(float));
        }
    }

    if (!create_context_pool()) {
        fprintf(stderr, "[TRT] Failed to create context pool\n");
        return false;
    }

    fprintf(stderr, "[TRT] Engine loaded with %d exclusive execution contexts\n",
            static_cast<int>(contexts_.size()));
    return true;
}

bool TRTEngine::create_context_pool() {
    int pool_sz = config_.context_pool_size;
    if (pool_sz <= 0) pool_sz = 1;

    for (int i = 0; i < pool_sz; ++i) {
        auto ctx = std::make_unique<ExecutionContext>();
        ctx->slot_id = i;

        ctx->trt_ctx = engine_->createExecutionContext();
        if (!ctx->trt_ctx) {
            fprintf(stderr, "[TRT] Failed to create execution context %d\n", i);
            return false;
        }

        if (!ctx->allocate(config_, engine_, input_index_,
                           output_indices_, output_sizes_, input_size_)) {
            fprintf(stderr, "[TRT] Failed to allocate buffers for context %d\n", i);
            ctx->free();
            return false;
        }

        free_contexts_.push_back(ctx.get());
        contexts_.push_back(std::move(ctx));
    }

    return true;
}

void TRTEngine::unload() {
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        free_contexts_.clear();
    }
    pool_cv_.notify_all();

    for (auto& ctx : contexts_) {
        ctx->free();
    }
    contexts_.clear();

    if (engine_) { engine_->destroy(); engine_ = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
}

ExecutionContext* TRTEngine::acquire_context(int timeout_ms) {
    std::unique_lock<std::mutex> lock(pool_mutex_);

    if (free_contexts_.empty()) {
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
        if (!pool_cv_.wait_until(lock, deadline, [this] { return !free_contexts_.empty(); })) {
            fprintf(stderr, "[TRT] WARNING: context pool exhausted, timeout after %d ms\n", timeout_ms);
            return nullptr;
        }
    }

    ExecutionContext* ctx = free_contexts_.front();
    free_contexts_.pop_front();
    return ctx;
}

void TRTEngine::release_context(ExecutionContext* ctx) {
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        free_contexts_.push_back(ctx);
    }
    pool_cv_.notify_one();
}

std::vector<Detection> TRTEngine::infer(const unsigned char* input_data,
                                        int width, int height, int channels,
                                        cudaStream_t stream) {
    if (!is_loaded() || input_index_ < 0) return {};

    ExecutionContext* ctx = acquire_context(200);
    if (!ctx) {
        fprintf(stderr, "[TRT] FATAL: No available execution context — dropping frame\n");
        return {};
    }

    auto result = infer_with_context(ctx, input_data, width, height, channels, stream);
    release_context(ctx);
    return result;
}

std::vector<Detection> TRTEngine::infer_with_context(ExecutionContext* ctx,
                                                     const unsigned char* input_data,
                                                     int width, int height, int channels,
                                                     cudaStream_t external_stream) {
    cudaStream_t stream = external_stream ? external_stream : ctx->own_stream;

    cuda_preprocess_resize_normalize(
        input_data, width, height, channels,
        static_cast<float*>(ctx->input_device),
        config_.input_w, config_.input_h,
        nullptr, nullptr,
        true, true, stream);

    std::vector<void*> bindings(engine_->getNbBindings(), nullptr);
    bindings[input_index_] = ctx->input_device;
    for (size_t i = 0; i < output_indices_.size(); ++i) {
        bindings[output_indices_[i]] = ctx->output_device_ptrs[i];
    }

    ctx->trt_ctx->enqueueV2(bindings.data(), stream, nullptr);

    int out_count = 0;
    if (!output_indices_.empty()) {
        cudaMemcpyAsync(ctx->output_hosts[0], ctx->output_device_ptrs[0],
                        output_sizes_[0], cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        out_count = static_cast<int>(output_sizes_[0] / sizeof(float)) / 6;
    }

    return postprocess(ctx->output_hosts[0], 0, out_count);
}

std::vector<std::vector<Detection>> TRTEngine::infer_batch(
    const std::vector<const unsigned char*>& inputs,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    const std::vector<int>& channels_list,
    cudaStream_t stream) {
    std::vector<std::vector<Detection>> results;
    for (size_t b = 0; b < inputs.size(); ++b) {
        results.push_back(infer(inputs[b], widths[b], heights[b], channels_list[b], stream));
    }
    return results;
}

std::vector<Detection> TRTEngine::postprocess(float* output_host, int batch_idx, int count) {
    std::vector<Detection> detections;
    if (!output_host || count <= 0) return detections;

    for (int i = 0; i < count; ++i) {
        float* row = output_host + i * 6;
        float conf = row[4] * row[5];
        if (conf < config_.score_threshold) continue;

        Detection det;
        det.x1 = row[0];
        det.y1 = row[1];
        det.x2 = row[2];
        det.y2 = row[3];
        det.confidence = conf;
        det.class_id = static_cast<int>(row[5]);
        detections.push_back(det);
    }

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            float ix1 = std::max(detections[i].x1, detections[j].x1);
            float iy1 = std::max(detections[i].y1, detections[j].y1);
            float ix2 = std::min(detections[i].x2, detections[j].x2);
            float iy2 = std::min(detections[i].y2, detections[j].y2);

            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;

            float area_i = (detections[i].x2 - detections[i].x1) *
                           (detections[i].y2 - detections[i].y1);
            float area_j = (detections[j].x2 - detections[j].x1) *
                           (detections[j].y2 - detections[j].y1);
            float union_area = area_i + area_j - inter;

            if (union_area > 0 && inter / union_area > config_.nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    std::vector<Detection> filtered;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!suppressed[i]) filtered.push_back(detections[i]);
    }

    return filtered;
}
