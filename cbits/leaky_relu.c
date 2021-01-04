#include "leaky_relu.h"

void leaky_relu_forward(RealNum* src, const int matrix_size, const RealNum alpha, RealNum* dst) {
  for (int offset = 0; offset < matrix_size; offset++) {
    RealNum a = src[offset];
    dst[offset] = a < 0 ? alpha * a : a;
  }
}

