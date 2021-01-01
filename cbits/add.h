#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "type.h"

void add_per_channel_cpu(RealNum* src, const int channels, const int height, 
    const int width, RealNum* bias, RealNum* dst);

void in_place_add_per_channel_cpu(RealNum* src, const int channels, const int height, 
    const int width, RealNum* bias);

void sum_over_channels_cpu(RealNum* src, const int channels, const int height, 
    const int width, RealNum* out);