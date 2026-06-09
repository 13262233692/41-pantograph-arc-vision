#ifndef TRT_ENGINE_H
#define TRT_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
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
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
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

private:
    bool allocate_buffers();
    void free_buffers();
    std::vector<Detection> postprocess(float* output_host, int batch_idx, int count);

    InferConfig config_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    void* input_device_ = nullptr;
    std::vector<void*> output_device_ptrs_;
    float* input_host_ = nullptr;
    std::vector<float*> output_hosts_;

    int input_index_ = -1;
    std::vector<int> output_indices_;
    size_t input_size_ = 0;
    std::vector<size_t> output_sizes_;

    Logger logger_;
};

#endif
