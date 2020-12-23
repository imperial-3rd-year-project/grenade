#ifndef INCLUDED_RELU
#define INCLUDED_RELU
#include "type.h"

// channels/height = 1 if matrix of shape S1D/S2D
void apply_leaky_relu_bulk(RealNum* src, const int channels, const int height, const int width, const RealNum alpha, RealNum* dst);
#endif
