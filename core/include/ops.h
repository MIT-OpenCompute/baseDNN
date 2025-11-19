#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// ====================================================
// Elementwise Operations
// ====================================================

Tensor* tensor_add(Tensor *A, Tensor *B);
void backward_add(Tensor *C);

Tensor* tensor_sub(Tensor *A, Tensor *B);
void backward_sub(Tensor *C);

Tensor* tensor_mul(Tensor *A, Tensor *B);
void backward_mul(Tensor *C);

// ====================================================
// Linear Algebra
// ====================================================

Tensor* tensor_matmul(Tensor *A, Tensor *B);
void backward_matmul(Tensor *C);

Tensor* tensor_transpose2d(Tensor *A);
void backward_transpose2d(Tensor *C);

// ====================================================
// Activation Functions
// ====================================================

Tensor* tensor_relu(Tensor *Z);
void backward_relu(Tensor *A);

Tensor* tensor_sigmoid(Tensor *Z);
void backward_sigmoid(Tensor *A);

Tensor* tensor_tanh(Tensor *Z);
void backward_tanh(Tensor *A);

Tensor* tensor_softmax(Tensor *Z);
void backward_softmax(Tensor *A);

// ====================================================
// Loss Functions
// ====================================================

Tensor* tensor_mse(Tensor *predictions, Tensor *targets);
void backward_mse(Tensor *L);

Tensor* tensor_cross_entropy(Tensor *predictions, Tensor *targets);
void backward_cross_entropy(Tensor *L);

Tensor* tensor_binary_cross_entropy(Tensor *predictions, Tensor *targets);
void backward_binary_cross_entropy(Tensor *L);

// ====================================================
// Slice
// ====================================================

Tensor* tensor_slice(Tensor *input, size_t start, size_t end); 


#endif