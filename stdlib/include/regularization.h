#ifndef SHAPE_H
#define SHAPE_H

#include "../../core/include/tensor.h"

// ====================================================
// Regularization Operations
// ====================================================

Tensor* tensor_dropout(Tensor *input, float dropout_rate);
void backward_dropout(Tensor *output);

Tensor* tensor_dropout2d(Tensor *input, float dropout_rate);
void backward_dropout2d(Tensor *output);

// ====================================================
// Regularization Layers
// ====================================================

typedef struct Dropout2DParams {
    float p;
} Dropout2DParams;

#define DROPOUT2D(p)(LayerConfig){.name="dropout2d", .params=&(Dropout2DParams){p}}

#endif