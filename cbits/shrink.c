#include "shrink.h"

void shrink_2d_cpu(RealNum* src, const int src_height, const int src_width, 
    const int dst_height, const int dst_width, RealNum* dst) {
  
  int ratio_height = src_height / dst_height;
  int ratio_width = src_width / dst_width;

  int pixels_per_block = ratio_height * ratio_width;

  memset(dst, 0, dst_height * dst_width * sizeof(RealNum));

  for (int y = 0; y < dst_height; y++) {
    for (int x = 0; x < dst_width; x++) {
      RealNum s = 0;
      for (int a = 0; a < ratio_height; a++) {
        for (int b = 0; b < ratio_width; b++) {
          s += src[(src_height - (y * ratio_height + a)) * src_width + (x * ratio_width+ b)];
        }
      }

      s /= pixels_per_block;

      dst[y * dst_width + x] = 255 - s;
    }
  }
}

void shrink_2d_rgba_cpu(uint8_t* src, const int src_height, const int src_width,
    const int dst_height, const int dst_width, RealNum* dst) {
  
  int ratio_height = (src_height / dst_height);
  int ratio_width  = (src_width / dst_width);

  int pixels_per_block = ratio_height * ratio_width;

  memset(dst, 0, dst_height * dst_width * sizeof(RealNum));

  for (int y = 0; y < dst_height; y++) {
    for (int x = 0; x < dst_width; x++) {
      RealNum s = 0;

      for (int a = 0; a < ratio_height; a++) {
        for (int b = 0; b < ratio_width; b++) {
          s += (double) src[(y * ratio_height + a) * src_width * 4 + (x * ratio_width + b) * 4 + 3];
        }
      }
      
      s /= pixels_per_block;

      dst[y * dst_width + x] = s;
    }
  }
}