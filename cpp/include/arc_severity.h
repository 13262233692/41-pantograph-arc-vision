#ifndef ARC_SEVERITY_H
#define ARC_SEVERITY_H

#include <cstdint>
#include <vector>
#include <deque>
#include <cmath>
#include <cstdio>

enum class ArcSeverityLevel : int32_t {
    LEVEL_1_MINOR_SPARK = 1,
    LEVEL_2_STABLE_ARC = 2,
    LEVEL_3_INTENSE_ARC = 3,
    LEVEL_4_SUSTAINED_BURN = 4,
    LEVEL_5_FATAL_ABLATION = 5,
};

struct ArcSeverityThresholds {
    double level2_cumulative_energy;
    double level3_cumulative_energy;
    double level4_cumulative_energy;
    double level5_cumulative_energy;
    double level3_instant_energy;
    double level5_instant_energy;
    int64_t level4_duration_ns;
    int64_t level5_duration_ns;
};

struct KalmanState1D {
    double x;
    double p;
    double q;
    double r;
};

class KalmanFilter1D {
public:
    explicit KalmanFilter1D(double process_noise = 0.01, double measurement_noise = 0.1);

    void init(double initial_value);
    double update(double measurement);
    double predict() const;

    const KalmanState1D& state() const { return state_; }

private:
    KalmanState1D state_;
    bool initialized_ = false;
};

struct ArcTrack {
    int32_t track_id;
    int32_t stream_index;
    int64_t start_pts_ns;
    int64_t last_pts_ns;
    int32_t frame_count;

    double last_box_x1, last_box_y1, last_box_x2, last_box_y2;
    double last_instant_energy;
    double smoothed_energy;
    double cumulative_energy;

    KalmanFilter1D energy_kf;

    ArcSeverityLevel severity;
    bool active;

    ArcTrack()
        : track_id(-1), stream_index(-1),
          start_pts_ns(0), last_pts_ns(0), frame_count(0),
          last_box_x1(0), last_box_y1(0), last_box_x2(0), last_box_y2(0),
          last_instant_energy(0), smoothed_energy(0), cumulative_energy(0),
          energy_kf(0.01, 0.1),
          severity(ArcSeverityLevel::LEVEL_1_MINOR_SPARK),
          active(true) {}
};

class ArcLifecycleTracker {
public:
    explicit ArcLifecycleTracker(const ArcSeverityThresholds& thresholds,
                                  double iou_match_threshold = 0.3,
                                  int64_t max_gap_ns = 100000000LL,
                                  int max_tracks = 64);

    struct TrackUpdate {
        int32_t track_id;
        double box_x1, box_y1, box_x2, box_y2;
        double instant_energy;
        double smoothed_energy;
        double cumulative_energy;
        ArcSeverityLevel severity;
        int32_t frame_count;
        int64_t duration_ns;
        bool is_new;
        bool severity_escalated;
    };

    std::vector<TrackUpdate> update(
        int64_t pts_ns,
        const std::vector<std::pair<double,double,double,double>>& detections,
        const std::vector<double>& energies);

    void expire_old_tracks(int64_t current_pts_ns);

    const ArcTrack* get_track(int32_t track_id) const;

private:
    double compute_iou(const ArcTrack& track,
                       double x1, double y1, double x2, double y2) const;
    ArcSeverityLevel classify_severity(const ArcTrack& track) const;

    ArcSeverityThresholds thresholds_;
    double iou_match_threshold_;
    int64_t max_gap_ns_;
    int max_tracks_;
    int32_t next_track_id_ = 1;

    std::deque<ArcTrack> active_tracks_;
    std::vector<ArcTrack> completed_tracks_;
};

class EnergyEstimator {
public:
    static double compute_instant_energy(
        double box_x1, double box_y1, double box_x2, double box_y2,
        const unsigned char* uv_data, int uv_width, int uv_height, int uv_channels);

    static double compute_instant_energy_gpu(
        double box_x1, double box_y1, double box_x2, double box_y2,
        const unsigned char* uv_gpu, int uv_width, int uv_height,
        cudaStream_t stream);
};

#endif
