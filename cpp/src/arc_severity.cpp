#include "arc_severity.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

KalmanFilter1D::KalmanFilter1D(double process_noise, double measurement_noise) {
    state_.x = 0.0;
    state_.p = 1.0;
    state_.q = process_noise;
    state_.r = measurement_noise;
}

void KalmanFilter1D::init(double initial_value) {
    state_.x = initial_value;
    state_.p = 1.0;
    initialized_ = true;
}

double KalmanFilter1D::update(double measurement) {
    if (!initialized_) {
        init(measurement);
        return state_.x;
    }

    state_.p += state_.q;

    double k = state_.p / (state_.p + state_.r);
    state_.x = state_.x + k * (measurement - state_.x);
    state_.p = (1.0 - k) * state_.p;

    return state_.x;
}

double KalmanFilter1D::predict() const {
    return state_.x;
}

ArcLifecycleTracker::ArcLifecycleTracker(const ArcSeverityThresholds& thresholds,
                                          double iou_match_threshold,
                                          int64_t max_gap_ns,
                                          int max_tracks)
    : thresholds_(thresholds),
      iou_match_threshold_(iou_match_threshold),
      max_gap_ns_(max_gap_ns),
      max_tracks_(max_tracks) {}

std::vector<ArcLifecycleTracker::TrackUpdate> ArcLifecycleTracker::update(
    int64_t pts_ns,
    const std::vector<std::pair<double,double,double,double>>& detections,
    const std::vector<double>& energies) {
    std::vector<TrackUpdate> updates;
    std::vector<bool> detection_matched(detections.size(), false);
    std::vector<bool> track_matched(active_tracks_.size(), false);

    for (size_t t = 0; t < active_tracks_.size(); ++t) {
        auto& track = active_tracks_[t];
        double best_iou = 0.0;
        int best_det = -1;

        for (size_t d = 0; d < detections.size(); ++d) {
            if (detection_matched[d]) continue;
            double iou = compute_iou(track,
                                     detections[d].first, detections[d].second,
                                     std::get<2>(detections[d]), std::get<3>(detections[d]));
            if (iou > best_iou) {
                best_iou = iou;
                best_det = static_cast<int>(d);
            }
        }

        if (best_det >= 0 && best_iou >= iou_match_threshold_) {
            detection_matched[best_det] = true;
            track_matched[t] = true;

            track.last_box_x1 = detections[best_det].first;
            track.last_box_y1 = detections[best_det].second;
            track.last_box_x2 = std::get<2>(detections[best_det]);
            track.last_box_y2 = std::get<3>(detections[best_det]);
            track.last_pts_ns = pts_ns;
            track.frame_count++;

            double raw_energy = energies[best_det];
            track.last_instant_energy = raw_energy;
            track.smoothed_energy = track.energy_kf.update(raw_energy);

            int64_t dt_ns = pts_ns - track.start_pts_ns;
            double dt_s = static_cast<double>(dt_ns) / 1e9;
            double avg_energy = track.smoothed_energy;
            track.cumulative_energy += avg_energy * (1.0 / 30.0);

            ArcSeverityLevel prev_severity = track.severity;
            track.severity = classify_severity(track);

            TrackUpdate upd;
            upd.track_id = track.track_id;
            upd.box_x1 = track.last_box_x1;
            upd.box_y1 = track.last_box_y1;
            upd.box_x2 = track.last_box_x2;
            upd.box_y2 = track.last_box_y2;
            upd.instant_energy = track.last_instant_energy;
            upd.smoothed_energy = track.smoothed_energy;
            upd.cumulative_energy = track.cumulative_energy;
            upd.severity = track.severity;
            upd.frame_count = track.frame_count;
            upd.duration_ns = dt_ns;
            upd.is_new = false;
            upd.severity_escalated = (static_cast<int32_t>(track.severity) >
                                      static_cast<int32_t>(prev_severity));
            updates.push_back(upd);
        }
    }

    for (size_t d = 0; d < detections.size(); ++d) {
        if (detection_matched[d]) continue;

        ArcTrack new_track;
        new_track.track_id = next_track_id_++;
        new_track.stream_index = 0;
        new_track.start_pts_ns = pts_ns;
        new_track.last_pts_ns = pts_ns;
        new_track.frame_count = 1;
        new_track.last_box_x1 = detections[d].first;
        new_track.last_box_y1 = detections[d].second;
        new_track.last_box_x2 = std::get<2>(detections[d]);
        new_track.last_box_y2 = std::get<3>(detections[d]);
        new_track.last_instant_energy = energies[d];
        new_track.smoothed_energy = new_track.energy_kf.update(energies[d]);
        new_track.cumulative_energy = new_track.smoothed_energy * (1.0 / 30.0);
        new_track.severity = ArcSeverityLevel::LEVEL_1_MINOR_SPARK;
        new_track.active = true;

        active_tracks_.push_back(new_track);

        TrackUpdate upd;
        upd.track_id = new_track.track_id;
        upd.box_x1 = new_track.last_box_x1;
        upd.box_y1 = new_track.last_box_y1;
        upd.box_x2 = new_track.last_box_x2;
        upd.box_y2 = new_track.last_box_y2;
        upd.instant_energy = new_track.last_instant_energy;
        upd.smoothed_energy = new_track.smoothed_energy;
        upd.cumulative_energy = new_track.cumulative_energy;
        upd.severity = new_track.severity;
        upd.frame_count = 1;
        upd.duration_ns = 0;
        upd.is_new = true;
        upd.severity_escalated = false;
        updates.push_back(upd);
    }

    return updates;
}

void ArcLifecycleTracker::expire_old_tracks(int64_t current_pts_ns) {
    for (auto& track : active_tracks_) {
        if (track.active && (current_pts_ns - track.last_pts_ns) > max_gap_ns_) {
            track.active = false;
            completed_tracks_.push_back(track);
        }
    }

    active_tracks_.erase(
        std::remove_if(active_tracks_.begin(), active_tracks_.end(),
                       [](const ArcTrack& t) { return !t.active; }),
        active_tracks_.end());

    if (completed_tracks_.size() > static_cast<size_t>(max_tracks_)) {
        completed_tracks_.erase(
            completed_tracks_.begin(),
            completed_tracks_.begin() + (completed_tracks_.size() - max_tracks_));
    }
}

const ArcTrack* ArcLifecycleTracker::get_track(int32_t track_id) const {
    for (const auto& track : active_tracks_) {
        if (track.track_id == track_id) return &track;
    }
    for (const auto& track : completed_tracks_) {
        if (track.track_id == track_id) return &track;
    }
    return nullptr;
}

double ArcLifecycleTracker::compute_iou(const ArcTrack& track,
                                         double x1, double y1, double x2, double y2) const {
    double ix1 = std::max(track.last_box_x1, x1);
    double iy1 = std::max(track.last_box_y1, y1);
    double ix2 = std::min(track.last_box_x2, x2);
    double iy2 = std::min(track.last_box_y2, y2);

    double iw = std::max(0.0, ix2 - ix1);
    double ih = std::max(0.0, iy2 - iy1);
    double inter = iw * ih;

    double area_t = (track.last_box_x2 - track.last_box_x1) *
                    (track.last_box_y2 - track.last_box_y1);
    double area_d = (x2 - x1) * (y2 - y1);
    double union_area = area_t + area_d - inter;

    return union_area > 0 ? inter / union_area : 0.0;
}

ArcSeverityLevel ArcLifecycleTracker::classify_severity(const ArcTrack& track) const {
    int64_t duration_ns = track.last_pts_ns - track.start_pts_ns;

    if (track.cumulative_energy >= thresholds_.level5_cumulative_energy ||
        track.smoothed_energy >= thresholds_.level5_instant_energy ||
        duration_ns >= thresholds_.level5_duration_ns) {
        return ArcSeverityLevel::LEVEL_5_FATAL_ABLATION;
    }

    if (track.cumulative_energy >= thresholds_.level4_cumulative_energy ||
        duration_ns >= thresholds_.level4_duration_ns) {
        return ArcSeverityLevel::LEVEL_4_SUSTAINED_BURN;
    }

    if (track.cumulative_energy >= thresholds_.level3_cumulative_energy ||
        track.smoothed_energy >= thresholds_.level3_instant_energy) {
        return ArcSeverityLevel::LEVEL_3_INTENSE_ARC;
    }

    if (track.cumulative_energy >= thresholds_.level2_cumulative_energy) {
        return ArcSeverityLevel::LEVEL_2_STABLE_ARC;
    }

    return ArcSeverityLevel::LEVEL_1_MINOR_SPARK;
}

double EnergyEstimator::compute_instant_energy(
    double box_x1, double box_y1, double box_x2, double box_y2,
    const unsigned char* uv_data, int uv_width, int uv_height, int uv_channels) {
    int x1 = std::max(0, static_cast<int>(box_x1));
    int y1 = std::max(0, static_cast<int>(box_y1));
    int x2 = std::min(uv_width, static_cast<int>(box_x2));
    int y2 = std::min(uv_height, static_cast<int>(box_y2));

    if (x2 <= x1 || y2 <= y1) return 0.0;

    double pixel_area = static_cast<double>((x2 - x1) * (y2 - y1));
    double luminance_sum = 0.0;
    int sample_count = 0;

    for (int row = y1; row < y2; ++row) {
        const unsigned char* row_ptr = uv_data + row * uv_width * uv_channels;
        for (int col = x1; col < x2; ++col) {
            luminance_sum += static_cast<double>(row_ptr[col * uv_channels]);
            sample_count++;
        }
    }

    if (sample_count == 0) return 0.0;

    double mean_luminance = luminance_sum / sample_count;
    double normalized_luminance = mean_luminance / 255.0;
    double energy = pixel_area * normalized_luminance * normalized_luminance;

    return energy;
}

double EnergyEstimator::compute_instant_energy_gpu(
    double box_x1, double box_y1, double box_x2, double box_y2,
    const unsigned char* uv_gpu, int uv_width, int uv_height,
    cudaStream_t stream) {
    int x1 = std::max(0, static_cast<int>(box_x1));
    int y1 = std::max(0, static_cast<int>(box_y1));
    int x2 = std::min(uv_width, static_cast<int>(box_x2));
    int y2 = std::min(uv_height, static_cast<int>(box_y2));

    if (x2 <= x1 || y2 <= y1) return 0.0;

    int roi_w = x2 - x1;
    int roi_h = y2 - y1;
    size_t roi_size = static_cast<size_t>(roi_w) * roi_h;

    unsigned char* host_buf = new unsigned char[roi_size];

    for (int row = 0; row < roi_h; ++row) {
        const unsigned char* src = uv_gpu + (y1 + row) * uv_width + x1;
        cudaMemcpyAsync(host_buf + row * roi_w, src, roi_w,
                        cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    double pixel_area = static_cast<double>(roi_w * roi_h);
    double luminance_sum = 0.0;
    for (size_t i = 0; i < roi_size; ++i) {
        luminance_sum += static_cast<double>(host_buf[i]);
    }
    delete[] host_buf;

    double mean_luminance = luminance_sum / static_cast<double>(roi_size);
    double normalized_luminance = mean_luminance / 255.0;
    double energy = pixel_area * normalized_luminance * normalized_luminance;

    return energy;
}
