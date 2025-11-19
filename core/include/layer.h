#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include "ops.h"

typedef struct LayerConfig {
    const char *name;
    void *params;
} LayerConfig;

typedef struct LinearParams {
    size_t in_features;
    size_t out_features;
} LinearParams;

#define LINEAR(in_features, out_features) (LayerConfig){ .name = "linear", .params = &(LinearParams){ in_features, out_features } }
#define RELU() (LayerConfig){ .name = "relu", .params = NULL }
#define SIGMOID() (LayerConfig){ .name = "sigmoid", .params = NULL }
#define TANH() (LayerConfig){ .name = "tanh", .params = NULL }
#define SOFTMAX() (LayerConfig){ .name = "softmax", .params = NULL }

typedef struct Layer Layer;

struct Layer {
    char *name;
    Tensor *weights;
    Tensor *bias;
    Tensor *output;
    Tensor **parameters;
    size_t num_parameters;
    Tensor* (*forward)(Layer *self, Tensor *input);
    void *config_data;  // Store layer-specific configuration
    size_t config_data_size;
}; 

// Layer constructors/destructor
Layer* layer_create(LayerConfig config);
void layer_free(Layer *layer); 

// Layer operations
Tensor* layer_forward(Layer *layer, Tensor *input);

// Utilities
void layer_zero_grad(Layer *layer);
Tensor** layer_get_parameters(Layer *layer, size_t *num_params);

#endif