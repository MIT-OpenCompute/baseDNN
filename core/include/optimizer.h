#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include "ops.h"

typedef struct OptimizerConfig {
    const char *name;
    void *params;
} OptimizerConfig;

typedef struct SGDParams {
    float learning_rate;
    float momentum;
} SGDParams;

typedef struct AdamParams {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
} AdamParams;

#define SGD(lr, momentum) (OptimizerConfig){ .name = "sgd", .params = &(SGDParams){ lr, momentum } }
#define ADAM(lr, beta1, beta2, epsilon) (OptimizerConfig){ .name = "adam", .params = &(AdamParams){ lr, beta1, beta2, epsilon } }

typedef struct Optimizer Optimizer;

struct Optimizer {
    char *name;
    Tensor **parameters;
    size_t num_parameters;
    void (*step)(Optimizer *self);
    void (*zero_grad)(Optimizer *self);
    void (*free_state)(void *state, size_t num_parameters);
    void *state;
};

// Optimizer constructor/destructors. 
Optimizer* optimizer_create(Tensor **parameters, size_t num_parameters, OptimizerConfig config);

// Optimizer operations
void optimizer_step(Optimizer *opt);
void optimizer_zero_grad(Optimizer *opt);
void optimizer_free(Optimizer *opt); 

#endif