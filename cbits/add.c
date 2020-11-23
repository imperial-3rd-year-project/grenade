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