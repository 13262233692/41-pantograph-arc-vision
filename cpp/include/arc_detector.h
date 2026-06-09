#ifndef ARC_DETECTOR_H
#define ARC_DETECTOR_H

#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include <mutex>
#include <unordered_map>
#include "trt_engine.h"
#include "cuda_preprocess.h"

struct ArcFlashEvent {
    float x1, y1, x2, y2;
    float confidence;
    int64_t pts_ns;
    int32_t stream_index;
    double intensity;
};

struct PantographROI {
    float x1, y1, x2, y2;
    float confidence;
};

struct ArcDetectorConfig {
    std::string visible_engine_path;
    std::string uv_engine_path;
    int tensorrt_device = 0;
    float visible_score_thresh = 0.5f;
    float uv_score_thresh = 0.4f;
    float nms_thresh = 0.45f;
    float uv_gain = 3.0f;
    float uv_blend_weight = 0.6f;
    int roi_padding = 20;
    int model_input_h = 640;
    int model_input_w = 640;
    int context_pool_size = 4;
};

struct StreamInferContext {
    std::unique_ptr<TRTEngine> visible_engine;
    std::unique_ptr<TRTEngine> uv_engine;
    int stream_index;
    bool initialized = false;
};

class ArcDetector {
public:
    explicit ArcDetector(const ArcDetectorConfig& config);
    ~ArcDetector();

    bool init();
    void shutdown();

    bool init_stream(int stream_index);
    void shutdown_stream(int stream_index);

    std::vector<ArcFlashEvent> detect(
        const unsigned char* visible_gpu,
        const unsigned char* uv_gpu,
        int width, int height,
        int64_t pts_ns,
        int32_t stream_index,
        cudaStream_t stream = nullptr);

private:
    StreamInferContext* get_stream_context(int stream_index);

    std::vector<PantographROI> extract_pantograph_rois(
        TRTEngine* vis_engine,
        const unsigned char* visible_gpu,
        int width, int height,
        cudaStream_t stream);

    std::vector<ArcFlashEvent> detect_arc_in_rois(
        TRTEngine* uv_engine,
        const unsigned char* uv_gpu,
        const unsigned char* visible_gpu,
        int width, int height,
        const std::vector<PantographROI>& rois,
        int64_t pts_ns,
        int32_t stream_index,
        cudaStream_t stream);

    std::vector<ArcFlashEvent> nms(
        std::vector<ArcFlashEvent>& events,
        float nms_thresh);

    ArcDetectorConfig config_;
    std::mutex contexts_mutex_;
    std::unordered_map<int, std::unique_ptr<StreamInferContext>> stream_contexts_;
    bool base_initialized_ = false;
};

#endif
