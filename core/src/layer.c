#include "../include/layer.h"
#include "../include/registry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Layer* layer_create(LayerConfig config) {
    LayerCreateFn create_fn = get_layer_create_fn(config.name);
    if (!create_fn) return NULL;
    return create_fn(&config);
}

void layer_free(Layer *layer) {
    if (!layer) return; 

    if (layer->name) free(layer->name);
    if (layer->weights) tensor_free(layer->weights);
    if (layer->bias) tensor_free(layer->bias);
    if (layer->output) tensor_free(layer->output);
    if (layer->parameters) free(layer->parameters);
    if (layer->config_data) free(layer->config_data);

    free(layer);
}

Tensor* layer_forward(Layer *layer, Tensor *input) {
    if (!layer || !layer->forward) return NULL; 
    return layer->forward(layer, input);
}

// Utilities
void layer_zero_grad(Layer *layer) {
    if (!layer) return;

    for (size_t i = 0; i < layer->num_parameters; i++) {
        if (layer->parameters[i]->grad) {
            tensor_zero_grad(layer->parameters[i]);
        }
    }
}

Tensor** layer_get_parameters(Layer *layer, size_t *num_params) {
    if (!layer || !num_params) return NULL;

    *num_params = layer->num_parameters;
    return layer->parameters;
}

