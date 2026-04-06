#include <cstdint>
#include <cuda_runtime.h>

extern "C" __global__
void apply_bias_u8(uint8_t* data, int64_t n, int bias) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int v = static_cast<int>(data[idx]) + bias;
  if (v < 0) v = 0;
  if (v > 255) v = 255;
  data[idx] = static_cast<uint8_t>(v);
}
