#include "cuda_preprocess.h"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel_resize_normalize_letterbox(
    const unsigned char* __restrict__ src,
    int src_w, int src_h, int src_c,
    float* __restrict__ dst,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    bool swap_rb, bool letterbox,
    int pad_x, int pad_y, int resize_w, int resize_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_w || dy >= dst_h) return;

    int sx, sy;
    if (letterbox) {
        if (dx < pad_x || dx >= pad_x + resize_w ||
            dy < pad_y || dy >= pad_y + resize_h) {
            dst[(0 * dst_h + dy) * dst_w + dx] = 0.0f;
            dst[(1 * dst_h + dy) * dst_w + dx] = 0.0f;
            dst[(2 * dst_h + dy) * dst_w + dx] = 0.0f;
            return;
        }
        sx = static_cast<int>((dx - pad_x) * static_cast<float>(src_w) / resize_w);
        sy = static_cast<int>((dy - pad_y) * static_cast<float>(src_h) / resize_h);
    } else {
        sx = static_cast<int>(dx * static_cast<float>(src_w) / dst_w);
        sy = static_cast<int>(dy * static_cast<float>(src_h) / dst_h);
    }

    sx = min(sx, src_w - 1);
    sy = min(sy, src_h - 1);

    int src_idx = (sy * src_w + sx) * src_c;
    float r = static_cast<float>(src[src_idx + (swap_rb ? 2 : 0)]);
    float g = static_cast<float>(src[src_idx + 1]);
    float b = static_cast<float>(src[src_idx + (swap_rb ? 0 : 2)]);

    dst[(0 * dst_h + dy) * dst_w + dx] = (r - mean_r) / std_r;
    dst[(1 * dst_h + dy) * dst_w + dx] = (g - mean_g) / std_g;
    dst[(2 * dst_h + dy) * dst_w + dx] = (b - mean_b) / std_b;
}

__global__ void kernel_roi_crop_normalize(
    const unsigned char* __restrict__ src,
    int src_w, int src_h,
    int roi_x1, int roi_y1,
    float* __restrict__ dst,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    bool swap_rb)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_w || dy >= dst_h) return;

    int sx = roi_x1 + static_cast<int>(dx * static_cast<float>(roi_x1 > 0 ? 1 : 1));
    int sy = roi_y1 + static_cast<int>(dy * static_cast<float>(roi_y1 > 0 ? 1 : 1));

    sx = min(max(sx, 0), src_w - 1);
    sy = min(max(sy, 0), src_h - 1);

    int src_idx = (sy * src_w + sx) * 3;
    float r = static_cast<float>(src[src_idx + (swap_rb ? 2 : 0)]);
    float g = static_cast<float>(src[src_idx + 1]);
    float b = static_cast<float>(src[src_idx + (swap_rb ? 0 : 2)]);

    dst[(0 * dst_h + dy) * dst_w + dx] = (r - mean_r) / std_r;
    dst[(1 * dst_h + dy) * dst_w + dx] = (g - mean_g) / std_g;
    dst[(2 * dst_h + dy) * dst_w + dx] = (b - mean_b) / std_b;
}

__global__ void kernel_uv_enhance(
    const unsigned char* __restrict__ uv_src,
    int width, int height,
    float* __restrict__ dst,
    float gain)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float val = static_cast<float>(uv_src[idx]) * gain;
    val = fminf(fmaxf(val, 0.0f), 255.0f);

    dst[(0 * height + y) * width + x] = val / 255.0f;
    dst[(1 * height + y) * width + x] = val / 255.0f;
    dst[(2 * height + y) * width + x] = val / 255.0f;
}

__global__ void kernel_blend_uv_visible(
    const unsigned char* __restrict__ visible,
    const unsigned char* __restrict__ uv,
    float* __restrict__ dst,
    int width, int height,
    float uv_weight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = (y * width + x) * 3;
    int chw_idx_base = y * width + x;

    float vr = static_cast<float>(visible[pixel_idx + 2]);
    float vg = static_cast<float>(visible[pixel_idx + 1]);
    float vb = static_cast<float>(visible[pixel_idx + 0]);

    float uv_val = static_cast<float>(uv[y * width + x]);

    float ur = uv_val * 1.5f;
    float ug = uv_val * 0.2f;
    float ub = uv_val * 0.3f;

    float blend_r = vr * (1.0f - uv_weight) + ur * uv_weight;
    float blend_g = vg * (1.0f - uv_weight) + ug * uv_weight;
    float blend_b = vb * (1.0f - uv_weight) + ub * uv_weight;

    dst[(0 * height + y) * width + x] = fminf(fmaxf(blend_r / 255.0f, 0.0f), 1.0f);
    dst[(1 * height + y) * width + x] = fminf(fmaxf(blend_g / 255.0f, 0.0f), 1.0f);
    dst[(2 * height + y) * width + x] = fminf(fmaxf(blend_b / 255.0f, 0.0f), 1.0f);
}

int cuda_preprocess_resize_normalize(
    const unsigned char* src_device,
    int src_width, int src_height, int src_channels,
    float* dst_device,
    int dst_width, int dst_height,
    const float* mean, const float* std_dev,
    bool swap_rb, bool letterbox,
    cudaStream_t stream)
{
    float mr = mean ? mean[0] : 0.0f;
    float mg = mean ? mean[1] : 0.0f;
    float mb = mean ? mean[2] : 0.0f;
    float sr = std_dev ? std_dev[0] : 1.0f;
    float sg = std_dev ? std_dev[1] : 1.0f;
    float sb = std_dev ? std_dev[2] : 1.0f;

    int pad_x = 0, pad_y = 0, resize_w = dst_width, resize_h = dst_height;
    if (letterbox) {
        float scale = fminf(static_cast<float>(dst_width) / src_width,
                            static_cast<float>(dst_height) / src_height);
        resize_w = static_cast<int>(src_width * scale);
        resize_h = static_cast<int>(src_height * scale);
        pad_x = (dst_width - resize_w) / 2;
        pad_y = (dst_height - resize_h) / 2;
    }

    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x,
              (dst_height + block.y - 1) / block.y);

    kernel_resize_normalize_letterbox<<<grid, block, 0, stream>>>(
        src_device, src_width, src_height, src_channels,
        dst_device, dst_width, dst_height,
        mr, mg, mb, sr, sg, sb,
        swap_rb, letterbox,
        pad_x, pad_y, resize_w, resize_h);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] resize_normalize kernel failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int cuda_preprocess_roi_crop(
    const unsigned char* src_device,
    int src_width, int src_height,
    int roi_x1, int roi_y1, int roi_x2, int roi_y2,
    float* dst_device,
    int dst_width, int dst_height,
    const float* mean, const float* std_dev,
    bool swap_rb,
    cudaStream_t stream)
{
    float mr = mean ? mean[0] : 0.0f;
    float mg = mean ? mean[1] : 0.0f;
    float mb = mean ? mean[2] : 0.0f;
    float sr = std_dev ? std_dev[0] : 1.0f;
    float sg = std_dev ? std_dev[1] : 1.0f;
    float sb = std_dev ? std_dev[2] : 1.0f;

    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x,
              (dst_height + block.y - 1) / block.y);

    kernel_roi_crop_normalize<<<grid, block, 0, stream>>>(
        src_device, src_width, src_height,
        roi_x1, roi_y1,
        dst_device, dst_width, dst_height,
        mr, mg, mb, sr, sg, sb,
        swap_rb);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] roi_crop kernel failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int cuda_uv_enhance(
    const unsigned char* uv_src_device,
    int width, int height,
    float* dst_device,
    float gain,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    kernel_uv_enhance<<<grid, block, 0, stream>>>(
        uv_src_device, width, height, dst_device, gain);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] uv_enhance kernel failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int cuda_blend_uv_visible(
    const unsigned char* visible_device,
    const unsigned char* uv_device,
    float* dst_device,
    int width, int height,
    float uv_weight,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    kernel_blend_uv_visible<<<grid, block, 0, stream>>>(
        visible_device, uv_device, dst_device, width, height, uv_weight);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] blend_uv_visible kernel failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
