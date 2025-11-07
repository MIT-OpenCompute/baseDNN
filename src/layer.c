#include "trebuchet/layer.h"
#include <stdlib.h>

static Tensor* linear_forward(Layer *self, Tensor *input); 
static Tensor* relu_forward(Layer *self, Tensor *input);
static Tensor* sigmoid_forward(Layer *self, Tensor *input);
static Tensor* tanh_forward(Layer *self, Tensor *input);
static Tensor* softmax_forward(Layer *self, Tensor *input);

Layer* layer_create(LayerConfig config) {
    Layer *layer = malloc(sizeof(Layer)); 
    layer->type = config.type; 
    layer->weights = NULL;
    layer->bias = NULL;
    layer->output = NULL;
    layer->parameters = NULL;
    layer->num_parameters = 0;

    switch (config.type) {
        case LAYER_LINEAR: {
            size_t in_features = config.params.linear.in_features;
            size_t out_features = config.params.linear.out_features; 

            layer->weights = tensor_randn(2, (size_t[]){in_features, out_features}, 42);
            layer->bias = tensor_zeroes(1, (size_t[]){out_features});

            layer->parameters = malloc(2 * sizeof(Tensor*));
            layer->parameters[0] = layer->weights;
            layer->parameters[1] = layer->bias;
            layer->num_parameters = 2;

            layer->forward = linear_forward;
            break;
        } case LAYER_RELU:
            layer->forward = relu_forward;
            break;
        case LAYER_SIGMOID:
            layer->forward = sigmoid_forward;
            break;
        case LAYER_TANH:
            layer->forward = tanh_forward;
            break;
        case LAYER_SOFTMAX:
            layer->forward = softmax_forward;
            break;
        default:
            free(layer);
            return NULL;
    }
    return layer;
}