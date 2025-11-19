#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "../../core/include/tensor.h"

// ====================================================
// Activations
// ====================================================

Tensor* tensor_leaky_relu(Tensor *input, float alpha);
void backward_leaky_relu(Tensor *output);

Tensor* tensor_gelu(Tensor *input); 
void backward_gelu(Tensor *output);

Tensor* tensor_swish(Tensor *input);
void backward_swish(Tensor *output);

Tensor* tensor_softplus(Tensor *input);
void backward_softplus(Tensor *output);


#endif