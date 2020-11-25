#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "type.h"

void batchnorm_forward_cpu(const RealNum* data, const RealNum* gamma, 
    const RealNum* beta, const RealNum* running_mean, const RealNum* running_var, RealNum epsilon,
    const int channels, const int height, const int width, RealNum* output);
