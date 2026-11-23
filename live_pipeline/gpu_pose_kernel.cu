// One precomputed OpenCV INTER_LINEAR coordinate and three RGB output values
// per thread. Source is the packed BGR frame captured by OpenCV.
__device__ __forceinline__ int sample_channel(const unsigned char* source,
                                             int width, int height, int x, int y, int channel) {
    if (static_cast<unsigned>(x) >= static_cast<unsigned>(width) ||
        static_cast<unsigned>(y) >= static_cast<unsigned>(height)) {
        return 0;
    }
    return source[(y * width + x) * 3 + channel];
}

extern "C" __global__ void rtmpose_warp_rgb_chw(const unsigned char* source,
                                              int width, int height, const int* terms,
                                              const float* lut, float* output) {
    constexpr int out_w = 288;
    constexpr int out_h = 384;
    constexpr int pixels = out_w * out_h;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= pixels) {
        return;
    }
    const int crop = blockIdx.y;
    const int col = offset % out_w;
    const int row = offset / out_w;
    const int* x_terms = terms + crop * 2 * (out_w + out_h);
    const int* y_terms = x_terms + out_w + out_h;
    const int sx_fixed = (x_terms[col] + x_terms[out_w + row]) >> 5;
    const int sy_fixed = (y_terms[col] + y_terms[out_w + row]) >> 5;
    const int x = max(-32768, min(32767, sx_fixed >> 5));
    const int y = max(-32768, min(32767, sy_fixed >> 5));
    const int dx = sx_fixed & 31;
    const int dy = sy_fixed & 31;
    const int w00 = (32 - dx) * (32 - dy);
    const int w01 = dx * (32 - dy);
    const int w10 = (32 - dx) * dy;
    const int w11 = dx * dy;
    #pragma unroll
    for (int rgb = 0; rgb < 3; ++rgb) {
        const int bgr = 2 - rgb;
        const int weighted = sample_channel(source, width, height, x, y, bgr) * w00
            + sample_channel(source, width, height, x + 1, y, bgr) * w01
            + sample_channel(source, width, height, x, y + 1, bgr) * w10
            + sample_channel(source, width, height, x + 1, y + 1, bgr) * w11;
        const int uint8_value = (weighted + 512) >> 10;
        output[(crop * 3 + rgb) * pixels + offset] = lut[rgb * 256 + uint8_value];
    }
}
