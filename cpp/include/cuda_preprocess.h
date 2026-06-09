#ifndef CUDA_PREPROCESS_H
#define CUDA_PREPROCESS_H

#include <cstdint>
#include <cuda_runtime.h>

struct PreprocessParams {
    int src_width;
    int src_height;
    int dst_width;
    int dst_height;
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3] = {1.0f, 1.0f, 1.0f};
    bool swap_rb = true;
    bool letterbox = true;
};

#ifdef __cplusplus
extern "C" {
#endif

int cuda_preprocess_resize_normalize(
    const unsigned char* src_device,
    int src_width, int src_height, int src_channels,
    float* dst_device,
    int dst_width, int dst_height,
    const float* mean, const float* std_dev,
    bool swap_rb, bool letterbox,
    cudaStream_t stream);

int cuda_preprocess_roi_crop(
    const unsigned char* src_device,
    int src_width, int src_height,
    int roi_x1, int roi_y1, int roi_x2, int roi_y2,
    float* dst_device,
    int dst_width, int dst_height,
    const float* mean, const float* std_dev,
    bool swap_rb,
    cudaStream_t stream);

int cuda_uv_enhance(
    const unsigned char* uv_src_device,
    int width, int height,
    float* dst_device,
    float gain,
    cudaStream_t stream);

int cuda_blend_uv_visible(
    const unsigned char* visible_device,
    const unsigned char* uv_device,
    float* dst_device,
    int width, int height,
    float uv_weight,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
