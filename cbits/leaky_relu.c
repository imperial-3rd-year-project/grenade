#include "leaky_relu.h"


// channels/height = 1 if matrix of shape S1D/S2D
void apply_leaky_relu_bulk(RealNum* src, const int channels, const int height, const int width, const RealNum alpha, RealNum* dst) {
  for (int offset = 0; offset < channels * height * width; offset++) {
    RealNum a = src[offset];
    dst[offset] = a < 0 ? alpha * a : a;
  }
}

