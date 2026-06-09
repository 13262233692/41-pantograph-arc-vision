#ifndef SHM_FRAME_QUEUE_H
#define SHM_FRAME_QUEUE_H

#include <cstdint>
#include <atomic>
#include <cstring>
#include <cuda_runtime.h>

constexpr int SHM_QUEUE_CAPACITY = 64;
constexpr int MAX_FRAME_WIDTH = 1920;
constexpr int MAX_FRAME_HEIGHT = 1080;
constexpr int MAX_FRAME_CHANNELS = 3;
constexpr int MAX_FRAME_SIZE = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * MAX_FRAME_CHANNELS;

enum class FrameType : int32_t {
    VISIBLE_LIGHT = 0,
    ULTRAVIOLET = 1,
};

struct alignas(64) FrameHeader {
    int64_t pts_ns;
    int32_t width;
    int32_t height;
    int32_t channels;
    FrameType type;
    int32_t stream_index;
    int64_t timestamp_wall;
};

struct alignas(64) FrameSlot {
    FrameHeader header;
    alignas(64) unsigned char data[MAX_FRAME_SIZE];
    std::atomic<int32_t> state;
};

enum SlotState : int32_t {
    SLOT_FREE = 0,
    SLOT_WRITING = 1,
    SLOT_READY = 2,
    SLOT_READING = 3,
};

struct ShmFrameQueueLayout {
    alignas(64) std::atomic<int64_t> write_seq;
    alignas(64) std::atomic<int64_t> read_seq;
    alignas(64) FrameSlot slots[SHM_QUEUE_CAPACITY];
};

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ShmFrameQueueLayout ShmFrameQueue;

ShmFrameQueue* shm_queue_create(size_t shm_key);
void shm_queue_destroy(ShmFrameQueue* queue, size_t shm_key);

int shm_queue_push(ShmFrameQueue* queue,
                   const FrameHeader* header,
                   const unsigned char* data);

int shm_queue_pop(ShmFrameQueue* queue,
                  FrameHeader* out_header,
                  unsigned char* out_data);

int shm_queue_push_gpu(ShmFrameQueue* queue,
                       const FrameHeader* header,
                       const unsigned char* gpu_data,
                       cudaStream_t stream);

int shm_queue_pop_batch(ShmFrameQueue* queue,
                        FrameHeader* out_headers,
                        unsigned char** out_datas,
                        int max_count,
                        int* out_count);

#ifdef __cplusplus
}
#endif

#endif
