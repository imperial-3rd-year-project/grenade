#include <stdio.h>
#include "type.h"

void transpose_4d(RealNum* src, int* dims, int* perms, RealNum* dst) {

    int in_depth = dims[0];
    int in_channels = dims[1];
    int in_rows = dims[2];
    int in_columns = dims[3];

    int out_channels = dims[perms[1]];
    int out_rows = dims[perms[2]];
    int out_columns = dims[perms[3]];
    
    int idx[4];

    for (int h = 0; h < in_depth; ++h) {  // depth
        idx[0] = h;
        for (int i = 0; i < in_channels; ++i) { // channels
            idx[1] = i;
            for (int j = 0; j < in_rows; ++j) {  // rows
                idx[2] = j;
                for (int k = 0; k < in_columns; ++k) { // columns
                    idx[3] = k;

                    int j1 = idx[perms[0]] * out_channels * out_rows * out_columns;
                    int j2 = idx[perms[1]] * out_rows * out_columns;
                    int j3 = idx[perms[2]] * out_columns;
                    int j4 = idx[perms[3]];

                    int i1 = h * in_channels * in_rows * in_columns;
                    int i2 = i * in_rows * in_columns;
                    int i3 = j * in_columns;
                    int i4 = k;

                    dst[j1 + j2 + j3 + j4] = src[i1 + i2 + i3 + i4];
                }
            }
        }
    }
}