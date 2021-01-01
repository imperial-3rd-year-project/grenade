#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "type.h"

void add_per_channel_cpu(RealNum* src, const int channels, const int height, 
    const int width, RealNum* bias, RealNum* dst) {
    
    for (int c = 0; c < channels; c++) {
        RealNum b = bias[c];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; ++j) {
                int offset = c * height * width + i * width + j; 
                dst[offset] = src[offset] + b;
            }
        }
    }
}

void in_place_add_per_channel_cpu(RealNum* src, const int channels, const int height, 
    const int width, RealNum* bias) {
  int channel_size = height * width;

  for (int c = 0; c < channels; ++c) {
    RealNum b = bias[c];
    for (int i = 0; i < channel_size; ++i) {
      src[c * channel_size + i] += b;
    }
  }
}

void sum_over_channels_cpu(RealNum* src, const int channels, const int height, 
    const int width, RealNum* out) {
  int channel_size = height * width;
  RealNum running_sum;
  for (int c = 0; c < channels; ++c) {
    running_sum = 0;
    for (int i = 0; i < channel_size; ++i) {
      running_sum += src[c * channel_size + i];
    }
    out[c] = running_sum;
  }
}