#include "trt_engine.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        fprintf(stderr, "[TRT] %s\n", msg);
    }
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

    context_ = engine_->createExecutionContext();
    if (!context_) {
        fprintf(stderr, "[TRT] Failed to create execution context\n");
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

    return allocate_buffers();
}

void TRTEngine::unload() {
    free_buffers();

    if (context_) { context_->destroy(); context_ = nullptr; }
    if (engine_) { engine_->destroy(); engine_ = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
}

bool TRTEngine::allocate_buffers() {
    if (input_index_ >= 0) {
        if (cudaMalloc(&input_device_, input_size_) != cudaSuccess) {
            fprintf(stderr, "[TRT] Failed to allocate input buffer\n");
            return false;
        }
        input_host_ = static_cast<float*>(malloc(input_size_));
        if (!input_host_) return false;
    }

    for (size_t i = 0; i < output_indices_.size(); ++i) {
        void* dev_ptr = nullptr;
        if (cudaMalloc(&dev_ptr, output_sizes_[i]) != cudaSuccess) {
            fprintf(stderr, "[TRT] Failed to allocate output buffer %zu\n", i);
            return false;
        }
        output_device_ptrs_.push_back(dev_ptr);

        float* host_ptr = static_cast<float*>(malloc(output_sizes_[i]));
        if (!host_ptr) return false;
        output_hosts_.push_back(host_ptr);
    }

    return true;
}

void TRTEngine::free_buffers() {
    if (input_device_) { cudaFree(input_device_); input_device_ = nullptr; }
    if (input_host_) { free(input_host_); input_host_ = nullptr; }

    for (auto* p : output_device_ptrs_) { if (p) cudaFree(p); }
    output_device_ptrs_.clear();

    for (auto* p : output_hosts_) { if (p) free(p); }
    output_hosts_.clear();
}

std::vector<Detection> TRTEngine::infer(const unsigned char* input_data,
                                        int width, int height, int channels,
                                        cudaStream_t stream) {
    if (!is_loaded() || input_index_ < 0) return {};

    size_t input_vol = static_cast<size_t>(config_.input_w) * config_.input_h * 3;
    size_t input_bytes = input_vol * sizeof(float);

    cuda_preprocess_resize_normalize(
        input_data, width, height, channels,
        static_cast<float*>(input_device_),
        config_.input_w, config_.input_h,
        config_.score_threshold > 0 ? nullptr : nullptr,
        config_.score_threshold > 0 ? nullptr : nullptr,
        true, true, stream);

    std::vector<void*> bindings(engine_->getNbBindings(), nullptr);
    bindings[input_index_] = input_device_;
    for (size_t i = 0; i < output_indices_.size(); ++i) {
        bindings[output_indices_[i]] = output_device_ptrs_[i];
    }

    context_->enqueueV2(bindings.data(), stream, nullptr);

    int out_count = 0;
    if (!output_indices_.empty()) {
        cudaMemcpyAsync(output_hosts_[0], output_device_ptrs_[0],
                        output_sizes_[0], cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        out_count = static_cast<int>(output_sizes_[0] / sizeof(float)) / 6;
    }

    return postprocess(output_hosts_[0], 0, out_count);
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
