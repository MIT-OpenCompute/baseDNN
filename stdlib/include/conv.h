#ifndef CONV_H
#define CONV_H

#include "../../core/include/tensor.h"

// ====================================================
// Convolution Operations
// ====================================================

Tensor* tensor_conv2d(Tensor *input, Tensor *weight, Tensor *bias, size_t stride, size_t padding);
void backward_conv2d(Tensor *output);

Tensor* tensor_maxpool2d(Tensor *input, size_t kernel_size, size_t stride);
void backward_maxpool2d(Tensor *output);

Tensor* tensor_avgpool2d(Tensor *input, size_t kernel_size, size_t stride);
void backward_avgpool2d(Tensor *output);

Tensor* tensor_adaptive_avgpool2d(Tensor *input, size_t out_h, size_t out_w);
void backward_adaptive_avgpool2d(Tensor *output);

// ====================================================
// Convolution Layers
// ====================================================

typedef struct Conv2DParams {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    int use_bias; 
} Conv2DParams;

#define CONV2D(in_ch, out_ch, k, s, p, u_b)(LayerConfig){.name="conv2d", .params=&(Conv2DParams){in_ch, out_ch, k, s, p, u_b}}

typedef struct MaxPool2DParams {
    size_t kernel_size;
    size_t stride;
} MaxPool2DParams;

#define MAXPOOL2D(k, s)(LayerConfig){.name="maxpool2d", .params=&(MaxPool2DParams){k, s}}

typedef struct BatchNorm2DParams {
    size_t num_features;
    float eps;
    float momentum;
} BatchNorm2DParams;

#define BATCHNORM2D(num_f, e, m)(LayerConfig){.name="batchnorm2d", .params=&(BatchNorm2DParams){num_f, e, m}}

typedef struct AdaptiveAvgPool2DParams {
    size_t output_h;
    size_t output_w;
} AdaptiveAvgPool2DParams;

#define ADAPTIVEAVGPOOL2D(out_h, out_w)(LayerConfig){.name="adaptive_avgpool2d", .params=&(AdaptiveAvgPool2DParams){out_h, out_w}}

#endif