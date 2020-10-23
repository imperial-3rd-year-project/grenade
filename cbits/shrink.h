#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "type.h"

void shrink_2d_cpu(RealNum* src, const int src_height, const int src_width, 
    const int dst_height, const int dst_width, RealNum* dst);

void shrink_2d_rgba_cpu(uint8_t* src, const int src_height, const int src_width,
    const int dst_height, const int dst_width, RealNum* dst);