#include "optimizer.h"
#include "registry.h"
#include <stdlib.h>
#include <string.h> 
#include <math.h>

// Optimizer constructor
Optimizer* optimizer_create(Tensor **parameters, size_t num_parameters, OptimizerConfig config) {
    if (!parameters || num_parameters == 0) return NULL;

    OptimizerInitStateFn init_fn = get_optimizer_init_state_fn(config.name);
    OptimizerStepFn step_fn = get_optimizer_step_fn(config.name);
    OptimizerFreeStateFn free_fn = get_optimizer_free_state_fn(config.name);
    
    if (!init_fn || !step_fn || !free_fn) return NULL;

    Optimizer *opt = malloc(sizeof(Optimizer));
    if (!opt) return NULL;

    opt->name = strdup(config.name);
    opt->parameters = parameters;
    opt->num_parameters = num_parameters;
    opt->step = step_fn;
    opt->zero_grad = optimizer_zero_grad;
    opt->free_state = free_fn;
    opt->state = init_fn(parameters, num_parameters, config.params);

    if (!opt->state) {
        free(opt->name);
        free(opt);
        return NULL;
    }

    return opt;
}



void optimizer_step(Optimizer *opt) {
    if (!opt || !opt->step) return;
    opt->step(opt);
}

void optimizer_zero_grad(Optimizer *opt) {
    if (!opt) return; 

    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i]; 
        if (param->grad) {
            tensor_zero_grad(param);
        }
    }
}

void optimizer_free(Optimizer *opt) {
    if (!opt) return;

    if (opt->state && opt->free_state) {
        opt->free_state(opt->state, opt->num_parameters);
    }
    
    if (opt->name) free(opt->name);
    free(opt);
}