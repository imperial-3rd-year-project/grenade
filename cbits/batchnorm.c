#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "type.h"

void batchnorm_forward_cpu(const RealNum* data, const RealNum* gamma, 
    const RealNum* beta, const RealNum* running_mean, const RealNum* running_var, RealNum epsilon,
    const int channels, const int height, const int width, RealNum* output) {

    for (int c = 0; c < channels; ++c) {
        RealNum std = sqrt(running_var[c] + epsilon);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int offset = c * height * width + width * y + x;

                output[offset] = ((data[offset] - running_mean[c]) / std) * gamma[c] + beta[c];
            }
        }
    }
}
