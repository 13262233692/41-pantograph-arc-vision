#include "shm_frame_queue.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/ipc.h>
#include <fcntl.h>
#include <unistd.h>
#endif

static const int64_t NS_PER_MS = 1000000LL;

ShmFrameQueue* shm_queue_create(size_t shm_key) {
#ifdef _WIN32
    char name[64];
    snprintf(name, sizeof(name), "Local\\ArcVisionShm_%zu", shm_key);
    HANDLE hMap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                     0, sizeof(ShmFrameQueueLayout), name);
    if (!hMap) return nullptr;

    void* ptr = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(ShmFrameQueueLayout));
    if (!ptr) {
        CloseHandle(hMap);
        return nullptr;
    }

    auto* queue = static_cast<ShmFrameQueue*>(ptr);
#else
    char name[64];
    snprintf(name, sizeof(name), "/arcvision_shm_%zu", shm_key);
    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd < 0) return nullptr;

    if (ftruncate(fd, sizeof(ShmFrameQueueLayout)) != 0) {
        close(fd);
        return nullptr;
    }

    void* ptr = mmap(nullptr, sizeof(ShmFrameQueueLayout),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return nullptr;

    auto* queue = static_cast<ShmFrameQueue*>(ptr);
#endif

    queue->write_seq.store(0, std::memory_order_relaxed);
    queue->read_seq.store(0, std::memory_order_relaxed);

    for (int i = 0; i < SHM_QUEUE_CAPACITY; ++i) {
        queue->slots[i].state.store(SLOT_FREE, std::memory_order_relaxed);
    }

    return queue;
}

void shm_queue_destroy(ShmFrameQueue* queue, size_t shm_key) {
    if (!queue) return;
#ifdef _WIN32
    UnmapViewOfFile(queue);
#else
    char name[64];
    snprintf(name, sizeof(name), "/arcvision_shm_%zu", shm_key);
    munmap(queue, sizeof(ShmFrameQueueLayout));
    shm_unlink(name);
#endif
}

int shm_queue_push(ShmFrameQueue* queue,
                   const FrameHeader* header,
                   const unsigned char* data) {
    if (!queue || !header || !data) return -1;

    int64_t w = queue->write_seq.load(std::memory_order_relaxed);
    int64_t r = queue->read_seq.load(std::memory_order_acquire);

    if (w - r >= SHM_QUEUE_CAPACITY) {
        return -2;
    }

    int slot_idx = static_cast<int>(w % SHM_QUEUE_CAPACITY);
    auto& slot = queue->slots[slot_idx];

    int32_t expected = SLOT_FREE;
    if (!slot.state.compare_exchange_strong(expected, SLOT_WRITING,
                                            std::memory_order_acq_rel)) {
        return -3;
    }

    slot.header = *header;
    slot.header.timestamp_wall = std::chrono::steady_clock::now().time_since_epoch().count();

    size_t frame_size = static_cast<size_t>(header->width) * header->height * header->channels;
    frame_size = (frame_size > MAX_FRAME_SIZE) ? MAX_FRAME_SIZE : frame_size;
    memcpy(slot.data, data, frame_size);

    slot.state.store(SLOT_READY, std::memory_order_release);
    queue->write_seq.store(w + 1, std::memory_order_release);

    return 0;
}

int shm_queue_pop(ShmFrameQueue* queue,
                  FrameHeader* out_header,
                  unsigned char* out_data) {
    if (!queue || !out_header || !out_data) return -1;

    int64_t r = queue->read_seq.load(std::memory_order_relaxed);
    int64_t w = queue->write_seq.load(std::memory_order_acquire);

    if (r >= w) {
        return -2;
    }

    int slot_idx = static_cast<int>(r % SHM_QUEUE_CAPACITY);
    auto& slot = queue->slots[slot_idx];

    int32_t expected = SLOT_READY;
    if (!slot.state.compare_exchange_strong(expected, SLOT_READING,
                                            std::memory_order_acq_rel)) {
        return -3;
    }

    *out_header = slot.header;

    size_t frame_size = static_cast<size_t>(slot.header.width) *
                        slot.header.height * slot.header.channels;
    frame_size = (frame_size > MAX_FRAME_SIZE) ? MAX_FRAME_SIZE : frame_size;
    memcpy(out_data, slot.data, frame_size);

    slot.state.store(SLOT_FREE, std::memory_order_release);
    queue->read_seq.store(r + 1, std::memory_order_release);

    return 0;
}

int shm_queue_push_gpu(ShmFrameQueue* queue,
                       const FrameHeader* header,
                       const unsigned char* gpu_data,
                       cudaStream_t stream) {
    if (!queue || !header || !gpu_data) return -1;

    int64_t w = queue->write_seq.load(std::memory_order_relaxed);
    int64_t r = queue->read_seq.load(std::memory_order_acquire);

    if (w - r >= SHM_QUEUE_CAPACITY) {
        return -2;
    }

    int slot_idx = static_cast<int>(w % SHM_QUEUE_CAPACITY);
    auto& slot = queue->slots[slot_idx];

    int32_t expected = SLOT_FREE;
    if (!slot.state.compare_exchange_strong(expected, SLOT_WRITING,
                                            std::memory_order_acq_rel)) {
        return -3;
    }

    slot.header = *header;
    slot.header.timestamp_wall = std::chrono::steady_clock::now().time_since_epoch().count();

    size_t frame_size = static_cast<size_t>(header->width) * header->height * header->channels;
    frame_size = (frame_size > MAX_FRAME_SIZE) ? MAX_FRAME_SIZE : frame_size;

    cudaError_t err = cudaMemcpyAsync(slot.data, gpu_data, frame_size,
                                       cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        slot.state.store(SLOT_FREE, std::memory_order_release);
        return -4;
    }
    cudaStreamSynchronize(stream);

    slot.state.store(SLOT_READY, std::memory_order_release);
    queue->write_seq.store(w + 1, std::memory_order_release);

    return 0;
}

int shm_queue_pop_batch(ShmFrameQueue* queue,
                        FrameHeader* out_headers,
                        unsigned char** out_datas,
                        int max_count,
                        int* out_count) {
    if (!queue || !out_headers || !out_datas || !out_count) return -1;

    *out_count = 0;
    int64_t r = queue->read_seq.load(std::memory_order_relaxed);
    int64_t w = queue->write_seq.load(std::memory_order_acquire);
    int64_t available = w - r;

    int count = static_cast<int>((available < max_count) ? available : max_count);

    for (int i = 0; i < count; ++i) {
        int slot_idx = static_cast<int>((r + i) % SHM_QUEUE_CAPACITY);
        auto& slot = queue->slots[slot_idx];

        int32_t expected = SLOT_READY;
        if (!slot.state.compare_exchange_strong(expected, SLOT_READING,
                                                std::memory_order_acq_rel)) {
            break;
        }

        out_headers[i] = slot.header;

        size_t frame_size = static_cast<size_t>(slot.header.width) *
                            slot.header.height * slot.header.channels;
        frame_size = (frame_size > MAX_FRAME_SIZE) ? MAX_FRAME_SIZE : frame_size;
        memcpy(out_datas[i], slot.data, frame_size);

        slot.state.store(SLOT_FREE, std::memory_order_release);
        (*out_count)++;
    }

    queue->read_seq.store(r + *out_count, std::memory_order_release);
    return 0;
}
