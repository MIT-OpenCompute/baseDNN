#include "../include/registry.h"
#include "../include/layer.h"
#include "../include/network.h"
#include "../include/optimizer.h"
#include "../include/ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ====================================================
// Hash Table Helpers
// ====================================================

#define REGISTRY_SIZE 64

typedef struct RegistryEntry {
    char *key;
    void *value;
    struct RegistryEntry *next;
} RegistryEntry;

typedef struct {
    RegistryEntry *buckets[REGISTRY_SIZE];
} Registry;

static unsigned int hash(const char *str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % REGISTRY_SIZE;
}

static void registry_set(Registry *reg, const char *key, void *value) {
    unsigned int idx = hash(key);
    RegistryEntry *entry = reg->buckets[idx];
    
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;
            return;
        }
        entry = entry->next;
    }
    
    entry = malloc(sizeof(RegistryEntry));
    entry->key = strdup(key);
    entry->value = value;
    entry->next = reg->buckets[idx];
    reg->buckets[idx] = entry;
}

static void* registry_get(Registry *reg, const char *key) {
    unsigned int idx = hash(key);
    RegistryEntry *entry = reg->buckets[idx];
    
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

static void registry_free(Registry *reg) {
    for (int i = 0; i < REGISTRY_SIZE; i++) {
        RegistryEntry *entry = reg->buckets[i];
        while (entry) {
            RegistryEntry *next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
        reg->buckets[i] = NULL;
    }
}

// ====================================================
// Layer Registry
// ====================================================

typedef struct {
    LayerCreateFn create_fn;
    LayerForwardFn forward_fn;
} LayerRegistryEntry;

static Registry layer_registry = {{NULL}};

void register_layer(const char *name, LayerCreateFn create_fn, LayerForwardFn forward_fn) {
    LayerRegistryEntry *entry = malloc(sizeof(LayerRegistryEntry));
    entry->create_fn = create_fn;
    entry->forward_fn = forward_fn;
    registry_set(&layer_registry, name, entry);
}

LayerCreateFn get_layer_create_fn(const char *name) {
    LayerRegistryEntry *entry = registry_get(&layer_registry, name);
    return entry ? entry->create_fn : NULL;
}

LayerForwardFn get_layer_forward_fn(const char *name) {
    LayerRegistryEntry *entry = registry_get(&layer_registry, name);
    return entry ? entry->forward_fn : NULL;
}

// ====================================================
// Operation Registry (with backend priority)
// ====================================================

typedef struct {
    OpFn op_fn;
    int priority;
} OperationRegistryEntry;

static Registry operation_registry = {{NULL}};

void register_operation(const char *name, OpFn op_fn) {
    register_operation_backend(name, op_fn, 0); // Default CPU priority
}

void register_operation_backend(const char *name, OpFn op_fn, int priority) {
    OperationRegistryEntry *existing = (OperationRegistryEntry*)registry_get(&operation_registry, name);
    
    // Only replace if new priority is higher or entry doesn't exist
    if (!existing || priority > existing->priority) {
        OperationRegistryEntry *entry = malloc(sizeof(OperationRegistryEntry));
        entry->op_fn = op_fn;
        entry->priority = priority;
        registry_set(&operation_registry, name, entry);
    }
}

OpFn get_operation_fn(const char *name) {
    OperationRegistryEntry *entry = (OperationRegistryEntry*)registry_get(&operation_registry, name);
    return entry ? entry->op_fn : NULL;
}

// ====================================================
// Tensor Operation Registry (for autograd)
// ====================================================

static Registry tensor_op_registry = {{NULL}};

void register_tensor_op(const char *name, BackwardFn backward_fn) {
    registry_set(&tensor_op_registry, name, (void*)backward_fn);
}

BackwardFn get_tensor_op_backward_fn(const char *name) {
    return (BackwardFn)registry_get(&tensor_op_registry, name);
}

// ====================================================
// Optimizer Registry
// ====================================================

typedef struct {
    OptimizerInitStateFn init_state_fn;
    OptimizerStepFn step_fn;
    OptimizerFreeStateFn free_state_fn;
} OptimizerRegistryEntry;

static Registry optimizer_registry = {{NULL}};

void register_optimizer(const char *name,
                       OptimizerInitStateFn init_state_fn,
                       OptimizerStepFn step_fn,
                       OptimizerFreeStateFn free_state_fn) {
    OptimizerRegistryEntry *entry = malloc(sizeof(OptimizerRegistryEntry));
    entry->init_state_fn = init_state_fn;
    entry->step_fn = step_fn;
    entry->free_state_fn = free_state_fn;
    registry_set(&optimizer_registry, name, entry);
}

OptimizerInitStateFn get_optimizer_init_state_fn(const char *name) {
    OptimizerRegistryEntry *entry = registry_get(&optimizer_registry, name);
    return entry ? entry->init_state_fn : NULL;
}

OptimizerStepFn get_optimizer_step_fn(const char *name) {
    OptimizerRegistryEntry *entry = registry_get(&optimizer_registry, name);
    return entry ? entry->step_fn : NULL;
}

OptimizerFreeStateFn get_optimizer_free_state_fn(const char *name) {
    OptimizerRegistryEntry *entry = registry_get(&optimizer_registry, name);
    return entry ? entry->free_state_fn : NULL;
}

// ====================================================
// Built-in Layer Implementations
// ====================================================

static Layer* linear_create(LayerConfig *config);
static Tensor* linear_forward(Layer *self, Tensor *input);
static Tensor* relu_forward(Layer *self, Tensor *input);
static Tensor* sigmoid_forward(Layer *self, Tensor *input);
static Tensor* tanh_forward(Layer *self, Tensor *input);
static Tensor* softmax_forward(Layer *self, Tensor *input);

static Layer* linear_create(LayerConfig *config) {
    LinearParams *params = (LinearParams*)config->params;
    Layer *layer = malloc(sizeof(Layer));
    layer->name = strdup(config->name);
    layer->weights = tensor_randn((size_t[]){params->in_features, params->out_features}, 2, 42);
    
    float scale = sqrtf(2.0f / (float)params->in_features);
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->data[i] *= scale;
    }
    
    layer->bias = tensor_zeroes((size_t[]){params->out_features}, 1);
    layer->output = NULL;
    layer->parameters = malloc(2 * sizeof(Tensor*));
    layer->parameters[0] = layer->weights;
    layer->parameters[1] = layer->bias;
    layer->num_parameters = 2;
    layer->forward = linear_forward;
    
    // Store config data for serialization
    layer->config_data_size = sizeof(LinearParams);
    layer->config_data = malloc(layer->config_data_size);
    memcpy(layer->config_data, params, layer->config_data_size);
    
    return layer;
}

static Layer* activation_create(LayerConfig *config) {
    Layer *layer = malloc(sizeof(Layer));
    layer->name = strdup(config->name);
    layer->weights = NULL;
    layer->bias = NULL;
    layer->output = NULL;
    layer->parameters = NULL;
    layer->num_parameters = 0;
    layer->forward = get_layer_forward_fn(config->name);
    
    // Activation layers have no config data
    layer->config_data = NULL;
    layer->config_data_size = 0;
    
    return layer;
}

static Tensor* linear_forward(Layer *self, Tensor *input) {
    if (!self || !input || !self->weights || !self->bias) return NULL;
    Tensor *Z_0 = tensor_matmul(input, self->weights);
    Tensor *Z = tensor_add(Z_0, self->bias);
    return Z;
}

static Tensor* relu_forward(Layer *self, Tensor *input) {
    return tensor_relu(input);
}

static Tensor* sigmoid_forward(Layer *self, Tensor *input) {
    return tensor_sigmoid(input);
}

static Tensor* tanh_forward(Layer *self, Tensor *input) {
    return tensor_tanh(input);
}

static Tensor* softmax_forward(Layer *self, Tensor *input) {
    return tensor_softmax(input);
}

// ====================================================
// Built-in Optimizer Implementations
// ====================================================

typedef struct {
    float learning_rate;
    float momentum;
    Tensor **velocity;
} SGDState;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    Tensor **m;
    Tensor **v;
} AdamState;

static void* sgd_init_state(Tensor **parameters, size_t num_parameters, void *params);
static void sgd_step(Optimizer *opt);
static void sgd_free_state(void *state, size_t num_parameters);

static void* adam_init_state(Tensor **parameters, size_t num_parameters, void *params);
static void adam_step(Optimizer *opt);
static void adam_free_state(void *state, size_t num_parameters);

static void* sgd_init_state(Tensor **parameters, size_t num_parameters, void *params) {
    SGDParams *p = (SGDParams*)params;
    SGDState *state = malloc(sizeof(SGDState));
    state->learning_rate = p->learning_rate;
    state->momentum = p->momentum;
    state->velocity = NULL;
    
    if (state->momentum > 0.0f) {
        state->velocity = malloc(num_parameters * sizeof(Tensor*));
        for (size_t i = 0; i < num_parameters; i++) {
            state->velocity[i] = tensor_create(parameters[i]->shape, parameters[i]->ndim);
            tensor_fill(state->velocity[i], 0.0f);
        }
    }
    return state;
}

static void sgd_step(Optimizer *opt) {
    SGDState *state = (SGDState*)opt->state;
    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i];
        if (!param->grad) continue;
        
        if (state->momentum > 0.0f) {
            for (size_t j = 0; j < param->size; j++) {
                state->velocity[i]->data[j] = state->momentum * state->velocity[i]->data[j] 
                                             - state->learning_rate * param->grad[j];
                param->data[j] += state->velocity[i]->data[j];
            }
        } else {
            for (size_t j = 0; j < param->size; j++) {
                param->data[j] -= state->learning_rate * param->grad[j];
            }
        }
    }
}

static void sgd_free_state(void *state, size_t num_parameters) {
    SGDState *s = (SGDState*)state;
    if (s->velocity) {
        for (size_t i = 0; i < num_parameters; i++) {
            tensor_free(s->velocity[i]);
        }
        free(s->velocity);
    }
    free(s);
}

static void* adam_init_state(Tensor **parameters, size_t num_parameters, void *params) {
    AdamParams *p = (AdamParams*)params;
    AdamState *state = malloc(sizeof(AdamState));
    state->learning_rate = p->learning_rate;
    state->beta1 = p->beta1;
    state->beta2 = p->beta2;
    state->epsilon = p->epsilon;
    state->t = 0;
    
    state->m = malloc(num_parameters * sizeof(Tensor*));
    state->v = malloc(num_parameters * sizeof(Tensor*));
    
    for (size_t i = 0; i < num_parameters; i++) {
        state->m[i] = tensor_create(parameters[i]->shape, parameters[i]->ndim);
        state->v[i] = tensor_create(parameters[i]->shape, parameters[i]->ndim);
        tensor_fill(state->m[i], 0.0f);
        tensor_fill(state->v[i], 0.0f);
    }
    return state;
}

static void adam_step(Optimizer *opt) {
    AdamState *state = (AdamState*)opt->state;
    state->t += 1;
    
    float bias_correction1 = 1.0f - powf(state->beta1, state->t);
    float bias_correction2 = 1.0f - powf(state->beta2, state->t);
    
    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i];
        if (!param->grad) continue;
        
        for (size_t j = 0; j < param->size; j++) {
            state->m[i]->data[j] = state->beta1 * state->m[i]->data[j] + (1.0f - state->beta1) * param->grad[j];
            state->v[i]->data[j] = state->beta2 * state->v[i]->data[j] + (1.0f - state->beta2) * param->grad[j] * param->grad[j];
            
            float m_hat = state->m[i]->data[j] / bias_correction1;
            float v_hat = state->v[i]->data[j] / bias_correction2;
            param->data[j] -= state->learning_rate * m_hat / (sqrtf(v_hat) + state->epsilon);
        }
    }
}

static void adam_free_state(void *state, size_t num_parameters) {
    AdamState *s = (AdamState*)state;
    for (size_t i = 0; i < num_parameters; i++) {
        tensor_free(s->m[i]);
        tensor_free(s->v[i]);
    }
    free(s->m);
    free(s->v);
    free(s);
}

// ====================================================
// Registry Initialization
// ====================================================

// Forward declaration for backend initialization
#ifdef HAS_WEBGPU
extern void backend_init_all(void);
#else
static void backend_init_all(void) {
    // No backends available
}
#endif

void registry_init() {
    // Register built-in layers
    register_layer("linear", linear_create, linear_forward);
    register_layer("relu", activation_create, relu_forward);
    register_layer("sigmoid", activation_create, sigmoid_forward);
    register_layer("tanh", activation_create, tanh_forward);
    register_layer("softmax", activation_create, softmax_forward);
    
    // Register built-in loss functions
    register_loss("mse", tensor_mse);
    register_loss("cross_entropy", tensor_cross_entropy);
    register_loss("binary_cross_entropy", tensor_binary_cross_entropy);
    
    // Register built-in tensor operations (backward functions)
    register_tensor_op("add", backward_add);
    register_tensor_op("sub", backward_sub);
    register_tensor_op("mul", backward_mul);
    register_tensor_op("matmul", backward_matmul);
    register_tensor_op("transpose2d", backward_transpose2d);
    register_tensor_op("relu", backward_relu);
    register_tensor_op("sigmoid", backward_sigmoid);
    register_tensor_op("tanh", backward_tanh);
    register_tensor_op("softmax", backward_softmax);
    register_tensor_op("mse", backward_mse);
    register_tensor_op("cross_entropy", backward_cross_entropy);
    register_tensor_op("binary_cross_entropy", backward_binary_cross_entropy);
    
    // Register built-in optimizers
    register_optimizer("sgd", sgd_init_state, sgd_step, sgd_free_state);
    register_optimizer("adam", adam_init_state, adam_step, adam_free_state);
    
    // Initialize available backends (WebGPU, etc.)
    backend_init_all();
}

void registry_cleanup() {
    // Free layer registry entries
    for (int i = 0; i < REGISTRY_SIZE; i++) {
        RegistryEntry *entry = layer_registry.buckets[i];
        while (entry) {
            free(entry->value);
            entry = entry->next;
        }
    }
    registry_free(&layer_registry);
    
    // Free operation registry entries
    for (int i = 0; i < REGISTRY_SIZE; i++) {
        RegistryEntry *entry = operation_registry.buckets[i];
        while (entry) {
            free(entry->value);
            entry = entry->next;
        }
    }
    registry_free(&operation_registry);
    
    // Free tensor operation registry (no extra malloc for entries)
    registry_free(&tensor_op_registry);
    
    // Free optimizer registry entries
    for (int i = 0; i < REGISTRY_SIZE; i++) {
        RegistryEntry *entry = optimizer_registry.buckets[i];
        while (entry) {
            free(entry->value);
            entry = entry->next;
        }
    }
    registry_free(&optimizer_registry);
}
