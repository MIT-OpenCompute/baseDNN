#ifndef SHAPE_H
#define SHAPE_H

#include "../../core/include/tensor.h"

// ====================================================
// Shape Operations
// ====================================================

Tensor* tensor_reshape(Tensor *input, size_t *new_shape, size_t new_ndim);
void backward_reshape(Tensor *output);

Tensor* tensor_transpose(Tensor *input, size_t dim0, size_t dim1);
void backward_transpose(Tensor *output);

Tensor* tensor_concat(Tensor **inputs, size_t num_inputs, size_t dim);
void backward_concat(Tensor *output);

Tensor** tensor_split(Tensor *input, size_t num_splits, size_t dim); 
void backward_split(Tensor **outputs, size_t num_outputs);

Tensor* tensor_squeeze(Tensor *input, size_t dim);
void backward_squeeze(Tensor *output);

// ====================================================
// Shape Layers
// ====================================================

typedef struct FlattenParams {
    size_t start_dim;
    size_t end_dim;
} FlattenParams;

#define FLATTEN(start_d, end_d)(LayerConfig){.name="flatten", .params=&(FlattenParams){start_d, end_d}}

typedef struct ReshapeParams {
    size_t *new_shape;
    size_t new_ndim;
} ReshapeParams;

#define RESHAPE(new_shape, new_ndim)(LayerConfig){.name="reshape", .params=&(ReshapeParams){new_shape, new_ndim}}
#endif