#include "arc_detector.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

ArcDetector::ArcDetector(const ArcDetectorConfig& config)
    : config_(config),
      visible_engine_(std::make_unique<TRTEngine>()),
      uv_engine_(std::make_unique<TRTEngine>()) {}

ArcDetector::~ArcDetector() {
    shutdown();
}

bool ArcDetector::init() {
    if (initialized_) return true;

    InferConfig vis_config;
    vis_config.engine_path = config_.visible_engine_path;
    vis_config.device_id = config_.tensorrt_device;
    vis_config.score_threshold = config_.visible_score_thresh;
    vis_config.nms_threshold = config_.nms_thresh;
    vis_config.input_h = config_.model_input_h;
    vis_config.input_w = config_.model_input_w;
    vis_config.fp16 = true;

    if (!visible_engine_->load(vis_config)) {
        fprintf(stderr, "[ArcDetector] Failed to load visible light engine\n");
        return false;
    }

    InferConfig uv_config;
    uv_config.engine_path = config_.uv_engine_path;
    uv_config.device_id = config_.tensorrt_device;
    uv_config.score_threshold = config_.uv_score_thresh;
    uv_config.nms_threshold = config_.nms_thresh;
    uv_config.input_h = config_.model_input_h;
    uv_config.input_w = config_.model_input_w;
    uv_config.fp16 = true;

    if (!uv_engine_->load(uv_config)) {
        fprintf(stderr, "[ArcDetector] Failed to load UV engine\n");
        return false;
    }

    initialized_ = true;
    fprintf(stderr, "[ArcDetector] Initialized successfully\n");
    return true;
}

void ArcDetector::shutdown() {
    if (!initialized_) return;
    visible_engine_->unload();
    uv_engine_->unload();
    initialized_ = false;
}

std::vector<ArcFlashEvent> ArcDetector::detect(
    const unsigned char* visible_gpu,
    const unsigned char* uv_gpu,
    int width, int height,
    int64_t pts_ns,
    int32_t stream_index,
    cudaStream_t stream) {
    if (!initialized_) return {};

    auto rois = extract_pantograph_rois(visible_gpu, width, height, stream);
    if (rois.empty()) {
        return {};
    }

    auto events = detect_arc_in_rois(uv_gpu, visible_gpu, width, height,
                                     rois, pts_ns, stream_index, stream);

    return nms(events, config_.nms_thresh);
}

std::vector<PantographROI> ArcDetector::extract_pantograph_rois(
    const unsigned char* visible_gpu,
    int width, int height,
    cudaStream_t stream) {
    auto detections = visible_engine_->infer(visible_gpu, width, height, 3, stream);

    std::vector<PantographROI> rois;
    for (const auto& det : detections) {
        PantographROI roi;
        roi.x1 = det.x1;
        roi.y1 = det.y1;
        roi.x2 = det.x2;
        roi.y2 = det.y2;
        roi.confidence = det.confidence;

        int pad = config_.roi_padding;
        roi.x1 = std::max(0.0f, roi.x1 - pad);
        roi.y1 = std::max(0.0f, roi.y1 - pad);
        roi.x2 = std::min(static_cast<float>(width), roi.x2 + pad);
        roi.y2 = std::min(static_cast<float>(height), roi.y2 + pad);

        rois.push_back(roi);
    }

    return rois;
}

std::vector<ArcFlashEvent> ArcDetector::detect_arc_in_rois(
    const unsigned char* uv_gpu,
    const unsigned char* visible_gpu,
    int width, int height,
    const std::vector<PantographROI>& rois,
    int64_t pts_ns,
    int32_t stream_index,
    cudaStream_t stream) {
    std::vector<ArcFlashEvent> all_events;

    void* blended_gpu = nullptr;
    size_t blend_size = static_cast<size_t>(width) * height * 3 * sizeof(float);
    cudaMalloc(&blended_gpu, blend_size);

    cuda_blend_uv_visible(visible_gpu, uv_gpu,
                          static_cast<float*>(blended_gpu),
                          width, height, config_.uv_blend_weight, stream);

    for (const auto& roi : rois) {
        int roi_w = static_cast<int>(roi.x2 - roi.x1);
        int roi_h = static_cast<int>(roi.y2 - roi.y1);

        if (roi_w < 32 || roi_h < 32) continue;

        int dst_w = config_.model_input_w;
        int dst_h = config_.model_input_h;

        float mean[3] = {0.0f, 0.0f, 0.0f};
        float std_dev[3] = {255.0f, 255.0f, 255.0f};

        void* roi_input_gpu = nullptr;
        size_t roi_input_size = static_cast<size_t>(dst_w) * dst_h * 3 * sizeof(float);
        cudaMalloc(&roi_input_gpu, roi_input_size);

        cuda_preprocess_roi_crop(
            static_cast<const unsigned char*>(blended_gpu),
            width, height,
            static_cast<int>(roi.x1), static_cast<int>(roi.y1),
            static_cast<int>(roi.x2), static_cast<int>(roi.y2),
            static_cast<float*>(roi_input_gpu),
            dst_w, dst_h,
            mean, std_dev, false, stream);

        auto detections = uv_engine_->infer(
            static_cast<const unsigned char*>(roi_input_gpu),
            dst_w, dst_h, 3, stream);

        for (const auto& det : detections) {
            ArcFlashEvent evt;
            evt.x1 = det.x1 / dst_w * roi_w + roi.x1;
            evt.y1 = det.y1 / dst_h * roi_h + roi.y1;
            evt.x2 = det.x2 / dst_w * roi_w + roi.x1;
            evt.y2 = det.y2 / dst_h * roi_h + roi.y1;
            evt.confidence = det.confidence;
            evt.pts_ns = pts_ns;
            evt.stream_index = stream_index;

            float box_w = evt.x2 - evt.x1;
            float box_h = evt.y2 - evt.y1;
            float box_area = box_w * box_h;
            evt.intensity = box_area > 0 ? det.confidence / box_area : 0.0;

            all_events.push_back(evt);
        }

        cudaFree(roi_input_gpu);
    }

    cudaFree(blended_gpu);
    return all_events;
}

std::vector<ArcFlashEvent> ArcDetector::nms(
    std::vector<ArcFlashEvent>& events, float nms_thresh) {
    if (events.empty()) return {};

    std::sort(events.begin(), events.end(),
              [](const ArcFlashEvent& a, const ArcFlashEvent& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(events.size(), false);
    for (size_t i = 0; i < events.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < events.size(); ++j) {
            if (suppressed[j]) continue;

            float ix1 = std::max(events[i].x1, events[j].x1);
            float iy1 = std::max(events[i].y1, events[j].y1);
            float ix2 = std::min(events[i].x2, events[j].x2);
            float iy2 = std::min(events[i].y2, events[j].y2);

            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;

            float area_i = (events[i].x2 - events[i].x1) *
                           (events[i].y2 - events[i].y1);
            float area_j = (events[j].x2 - events[j].x1) *
                           (events[j].y2 - events[j].y1);
            float union_area = area_i + area_j - inter;

            if (union_area > 0 && inter / union_area > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }

    std::vector<ArcFlashEvent> result;
    for (size_t i = 0; i < events.size(); ++i) {
        if (!suppressed[i]) result.push_back(events[i]);
    }
    return result;
}
