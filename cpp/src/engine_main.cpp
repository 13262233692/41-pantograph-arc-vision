#include "arc_detector.h"
#include "shm_frame_queue.h"
#include "zmq_publisher.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <cuda_runtime.h>

static std::atomic<bool> g_running{true};
static ArcDetector* g_detector = nullptr;
static ZmqPublisher* g_publisher = nullptr;
static int g_num_streams = 0;

struct StreamQueues {
    ShmFrameQueue* visible_queue;
    ShmFrameQueue* uv_queue;
};

static std::unordered_map<int, StreamQueues> g_stream_queues;
static std::mutex g_stream_mutex;

extern "C" {

int arcvision_engine_init(const char* visible_engine_path,
                          const char* uv_engine_path,
                          int device_id,
                          float visible_thresh,
                          float uv_thresh,
                          float nms_thresh,
                          const char* zmq_endpoint,
                          int num_streams) {
    cudaSetDevice(device_id);

    ArcDetectorConfig config;
    config.visible_engine_path = visible_engine_path ? visible_engine_path : "";
    config.uv_engine_path = uv_engine_path ? uv_engine_path : "";
    config.tensorrt_device = device_id;
    config.visible_score_thresh = visible_thresh;
    config.uv_score_thresh = uv_thresh;
    config.nms_thresh = nms_thresh;
    config.context_pool_size = 1;

    config.severity_thresholds.level2_cumulative_energy = 500.0;
    config.severity_thresholds.level3_cumulative_energy = 5000.0;
    config.severity_thresholds.level4_cumulative_energy = 50000.0;
    config.severity_thresholds.level5_cumulative_energy = 500000.0;
    config.severity_thresholds.level3_instant_energy = 1000.0;
    config.severity_thresholds.level5_instant_energy = 10000.0;
    config.severity_thresholds.level4_duration_ns = 3000000000LL;
    config.severity_thresholds.level5_duration_ns = 10000000000LL;

    g_detector = new ArcDetector(config);
    if (!g_detector->init()) {
        fprintf(stderr, "[Engine] Failed to initialize arc detector base\n");
        delete g_detector;
        g_detector = nullptr;
        return -1;
    }

    g_num_streams = num_streams > 0 ? num_streams : 4;
    for (int i = 0; i < g_num_streams; ++i) {
        if (!g_detector->init_stream(i)) {
            fprintf(stderr, "[Engine] Failed to initialize inference context for stream %d\n", i);
            delete g_detector;
            g_detector = nullptr;
            return -1;
        }
    }

    g_publisher = new ZmqPublisher();
    std::string endpoint = zmq_endpoint ? zmq_endpoint : "ipc:///tmp/arcvision_results";
    if (!g_publisher->bind(endpoint)) {
        fprintf(stderr, "[Engine] Failed to bind ZMQ publisher\n");
        delete g_detector;
        delete g_publisher;
        g_detector = nullptr;
        g_publisher = nullptr;
        return -2;
    }

    fprintf(stderr, "[Engine] Initialized with %d exclusive stream contexts\n", g_num_streams);
    return 0;
}

void arcvision_engine_set_stream_queues(int stream_index,
                                         void* visible_queue,
                                         void* uv_queue) {
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    g_stream_queues[stream_index] = StreamQueues{
        static_cast<ShmFrameQueue*>(visible_queue),
        static_cast<ShmFrameQueue*>(uv_queue)
    };
    fprintf(stderr, "[Engine] stream=%d: Queues registered\n", stream_index);
}

int arcvision_engine_process_frame(const unsigned char* visible_gpu,
                                   const unsigned char* uv_gpu,
                                   int width, int height,
                                   int64_t pts_ns,
                                   int32_t stream_index) {
    if (!g_detector) return -1;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto events = g_detector->detect(visible_gpu, uv_gpu, width, height,
                                     pts_ns, stream_index, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (!events.empty() && g_publisher) {
        std::vector<ArcResultMsg> msgs;
        msgs.reserve(events.size());
        for (const auto& evt : events) {
            ArcResultMsg m;
            m.x1 = evt.x1;
            m.y1 = evt.y1;
            m.x2 = evt.x2;
            m.y2 = evt.y2;
            m.confidence = evt.confidence;
            m.class_id = 1;
            m.pts_ns = evt.pts_ns;
            m.stream_index = evt.stream_index;
            m.intensity = evt.intensity;
            m.instant_energy = evt.instant_energy;
            m.smoothed_energy = evt.smoothed_energy;
            m.cumulative_energy = evt.cumulative_energy;
            m.severity_level = static_cast<int32_t>(evt.severity);
            m.track_id = evt.track_id;
            m.track_frame_count = evt.track_frame_count;
            m.track_duration_ns = evt.track_duration_ns;
            msgs.push_back(m);
        }

        int rc = g_publisher->publish_nonblock(msgs.data(), static_cast<int>(msgs.size()));
        if (rc != 0) {
            fprintf(stderr, "[Engine] stream=%d: ZMQ publish failed or would block (rc=%d)\n",
                    stream_index, rc);
        }
    }

    return static_cast<int>(events.size());
}

static void stream_detect_loop(int stream_index, int64_t sync_tolerance_ns) {
    fprintf(stderr, "[Engine] stream=%d: Starting dedicated detection thread\n", stream_index);

    StreamQueues queues;
    {
        std::lock_guard<std::mutex> lock(g_stream_mutex);
        auto it = g_stream_queues.find(stream_index);
        if (it == g_stream_queues.end()) {
            fprintf(stderr, "[Engine] stream=%d: No queues registered, exiting\n", stream_index);
            return;
        }
        queues = it->second;
    }

    FrameHeader vis_header, uv_header;
    unsigned char* vis_data = new unsigned char[MAX_FRAME_SIZE];
    unsigned char* uv_data = new unsigned char[MAX_FRAME_SIZE];

    while (g_running.load(std::memory_order_relaxed)) {
        int rc_vis = shm_queue_pop(queues.visible_queue, &vis_header, vis_data);
        if (rc_vis != 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        int rc_uv = shm_queue_pop(queues.uv_queue, &uv_header, uv_data);
        if (rc_uv != 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        int64_t pts_diff = vis_header.pts_ns - uv_header.pts_ns;
        if (pts_diff < 0) pts_diff = -pts_diff;

        if (pts_diff > sync_tolerance_ns) {
            fprintf(stderr, "[Engine] stream=%d: Frame sync mismatch %lld ns\n",
                    stream_index, static_cast<long long>(pts_diff));
            continue;
        }

        unsigned char* vis_gpu = nullptr;
        unsigned char* uv_gpu = nullptr;
        size_t vis_size = static_cast<size_t>(vis_header.width) *
                          vis_header.height * vis_header.channels;
        size_t uv_size = static_cast<size_t>(uv_header.width) *
                         uv_header.height * uv_header.channels;

        cudaMalloc(&vis_gpu, vis_size);
        cudaMalloc(&uv_gpu, uv_size);
        cudaMemcpy(vis_gpu, vis_data, vis_size, cudaMemcpyHostToDevice);
        cudaMemcpy(uv_gpu, uv_data, uv_size, cudaMemcpyHostToDevice);

        arcvision_engine_process_frame(vis_gpu, uv_gpu,
                                       vis_header.width, vis_header.height,
                                       vis_header.pts_ns,
                                       stream_index);

        cudaFree(vis_gpu);
        cudaFree(uv_gpu);
    }

    delete[] vis_data;
    delete[] uv_data;
    fprintf(stderr, "[Engine] stream=%d: Detection thread stopped\n", stream_index);
}

void arcvision_engine_run_loop(int64_t sync_tolerance_ns) {
    fprintf(stderr, "[Engine] Starting multi-stream detection with %d streams (tolerance=%lld ns)\n",
            g_num_streams, static_cast<long long>(sync_tolerance_ns));

    std::vector<std::thread> threads;
    for (int i = 0; i < g_num_streams; ++i) {
        threads.emplace_back(stream_detect_loop, i, sync_tolerance_ns);
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

void arcvision_engine_shutdown() {
    g_running.store(false, std::memory_order_relaxed);

    if (g_detector) {
        g_detector->shutdown();
        delete g_detector;
        g_detector = nullptr;
    }

    if (g_publisher) {
        g_publisher->close();
        delete g_publisher;
        g_publisher = nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(g_stream_mutex);
        g_stream_queues.clear();
    }

    fprintf(stderr, "[Engine] Shutdown complete\n");
}

}
