#ifndef TRT_ENGINE_H
#define TRT_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <NvInfer.h>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

struct InferConfig {
    std::string engine_path;
    int max_batch_size = 1;
    int input_h = 640;
    int input_w = 640;
    float score_threshold = 0.5f;
    float nms_threshold = 0.45f;
    bool fp16 = true;
    int device_id = 0;
    int context_pool_size = 4;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

struct ExecutionContext {
    nvinfer1::IExecutionContext* trt_ctx = nullptr;
    void* input_device = nullptr;
    std::vector<void*> output_device_ptrs;
    float* input_host = nullptr;
    std::vector<float*> output_hosts;
    cudaStream_t own_stream = nullptr;
    int slot_id = -1;

    bool allocate(const InferConfig& config,
                  nvinfer1::ICudaEngine* engine,
                  int input_index,
                  const std::vector<int>& output_indices,
                  const std::vector<size_t>& output_sizes,
                  size_t input_size);
    void free();
};

class TRTEngine {
public:
    TRTEngine();
    ~TRTEngine();

    bool load(const InferConfig& config);
    void unload();

    std::vector<Detection> infer(const unsigned char* input_data,
                                 int width, int height, int channels,
                                 cudaStream_t stream = nullptr);

    std::vector<std::vector<Detection>> infer_batch(
        const std::vector<const unsigned char*>& inputs,
        const std::vector<int>& widths,
        const std::vector<int>& heights,
        const std::vector<int>& channels_list,
        cudaStream_t stream = nullptr);

    int get_input_h() const { return config_.input_h; }
    int get_input_w() const { return config_.input_w; }
    bool is_loaded() const { return engine_ != nullptr; }

    ExecutionContext* acquire_context(int timeout_ms = 100);
    void release_context(ExecutionContext* ctx);

    int pool_size() const { return static_cast<int>(contexts_.size()); }

private:
    bool create_context_pool();
    std::vector<Detection> infer_with_context(ExecutionContext* ctx,
                                              const unsigned char* input_data,
                                              int width, int height, int channels,
                                              cudaStream_t stream);
    std::vector<Detection> postprocess(float* output_host, int batch_idx, int count);

    InferConfig config_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;

    std::vector<std::unique_ptr<ExecutionContext>> contexts_;
    std::deque<ExecutionContext*> free_contexts_;
    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;

    int input_index_ = -1;
    std::vector<int> output_indices_;
    size_t input_size_ = 0;
    std::vector<size_t> output_sizes_;

    Logger logger_;
};

#endif
