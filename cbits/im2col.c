#include "im2col.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>

void im2col_cpu(const RealNum* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    RealNum* data_col) {

  const int channel_size = height * width;

  for (int fitting_height = 0; fitting_height <= (height - kernel_h); fitting_height += stride_h) {
    for (int fitting_width = 0; fitting_width <= (width - kernel_w); fitting_width += stride_w) {
      for (int channel = 0; channel < channels; channel++) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = fitting_height + kernel_row;
            int input_col = fitting_width + kernel_col;
            *(data_col++) = data_im[input_row * width + input_col + channel_size * channel];
          }
        }
      }
    }
  }
}

void col2im_cpu(const RealNum* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    RealNum* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(RealNum));

  const int channel_size = height * width;

  for (int fitting_height = 0; fitting_height <= (height - kernel_h); fitting_height += stride_h) {
    for (int fitting_width = 0; fitting_width <= (width - kernel_w); fitting_width += stride_w) {
      for (int channel = 0; channel < channels; channel++) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = fitting_height + kernel_row;
            int input_col = fitting_width + kernel_col;
            data_im[input_row * width + input_col + channel_size * channel] += *(data_col++);
          }
        }
      }
    }
  }
}

inline RealNum max ( RealNum a, RealNum b ) { return a > b ? a : b; }

inline int ceil_divison ( int x, int y) { return (x + y - 1) / y; }

void pool_forwards_cpu(const RealNum* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    RealNum* data_pooled) {
  const int channel_size = height * width;

  for (int channel = 0; channel < channels; channel++) {
    for (int fitting_height = 0; fitting_height <= (height - kernel_h); fitting_height += stride_h) {
      for (int fitting_width = 0; fitting_width <= (width - kernel_w); fitting_width += stride_w) {
        // Start with the value in 0,0
        int    max_index = fitting_height * width + fitting_width + channel_size * channel;
        RealNum max_value = data_im[max_index];
        // Initial row, skipping the corner we've done
        for (int kernel_col = 1; kernel_col < kernel_w; kernel_col++) {
          int    input_row  = fitting_height;
          int    input_col  = fitting_width + kernel_col;
          int    data_index = input_row * width + input_col + channel_size * channel;
          RealNum data_value = data_im[data_index];
          max_value = max ( max_value, data_value );
        }
        // The remaining rows
        for (int kernel_row = 1; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int    input_row = fitting_height + kernel_row;
            int    input_col = fitting_width + kernel_col;
            int    data_index = input_row * width + input_col + channel_size * channel;
            RealNum data_value = data_im[data_index];
            max_value = max ( max_value, data_value );
          }
        }
        *(data_pooled++) = max_value;
      }
    }
  }
}

void pool_backwards_cpu(const RealNum* data_im, const RealNum* data_pooled,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    RealNum* data_backgrad ) {
  memset(data_backgrad, 0, height * width * channels * sizeof(RealNum));

  const int channel_size = height * width;

  for (int channel = 0; channel < channels; channel++) {
    for (int fitting_height = 0; fitting_height <= (height - kernel_h); fitting_height += stride_h) {
      for (int fitting_width = 0; fitting_width <= (width - kernel_w); fitting_width += stride_w) {
        int    max_index = fitting_height * width + fitting_width + channel_size * channel;
        RealNum max_value = data_im[max_index];
        for (int kernel_col = 1; kernel_col < kernel_w; kernel_col++) {
          int    input_row  = fitting_height;
          int    input_col  = fitting_width + kernel_col;
          int    data_index = input_row * width + input_col + channel_size * channel;
          RealNum data_value = data_im[data_index];
          if ( data_value > max_value )  {
              max_index = data_index;
              max_value = data_value;
          }
        }
        for (int kernel_row = 1; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int    input_row = fitting_height + kernel_row;
            int    input_col = fitting_width + kernel_col;
            int    data_index = input_row * width + input_col + channel_size * channel;
            RealNum data_value = data_im[data_index];
            if ( data_value > max_value )  {
              max_index = data_index;
              max_value = data_value;
            }
          }
        }
        data_backgrad[max_index] += *(data_pooled++);
      }
    }
  }
}

void same_pad_pool_forwards_cpu(const RealNum* data_im, const int channels,
    const int in_height, const int in_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_l, const int pad_t, const int pad_r, const int pad_b,
    RealNum* data_pooled) {
  
  const int out_height = ceil_divison(in_height, stride_h);
  const int out_width = ceil_divison(in_width, stride_w);

  const int in_channel_size = in_height * in_width;
  const int out_channel_size = out_height * out_width;

  for (int c = 0; c < channels; ++c) {
    for (int y = 0; y < out_height; ++y) {
      for (int x = 0; x < out_width; ++x) {
        // (x, y) of the pixel in the input channel that corresponds to the 
        // top left of the kernel that will produce this pixel in the output
        int effective_x = (x * stride_w) - pad_l;
        int effective_y = (y * stride_h) - pad_t;

        bool found_nonnan = false;
        RealNum max_value;

        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            int curr_x = effective_x + kernel_col;
            int curr_y = effective_y + kernel_row;

            if (curr_x < 0 || curr_x >= in_width || curr_y < 0 || curr_y >= in_height) {
              continue;
            } else {
              RealNum curr_pixel = data_im[in_channel_size * c + curr_y * in_width + curr_x];

              if (!found_nonnan) {
                max_value = curr_pixel;
                found_nonnan = true;
              } else {
                max_value = max( max_value, curr_pixel );
              }
            }
          }
        }

        int output_index = out_channel_size * c + y * out_width + x;

        if (found_nonnan) {
          data_pooled[output_index] = max_value;
        } else {
          data_pooled[output_index] = 0;
        }
      } 
    }
  }
}