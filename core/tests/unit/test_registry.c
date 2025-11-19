#include "../../include/basednn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { printf("Running %s...\n", #name); test_##name(); printf("  PASSED\n"); } while(0)

// ====================================================
// Registry Tests
// ====================================================

TEST(registry_init) {
    // Already called in main, but test that it doesn't crash
    registry_cleanup();
    registry_init();
}

TEST(layer_registration_linear) {
    LayerCreateFn create_fn = get_layer_create_fn("linear");
    LayerForwardFn forward_fn = get_layer_forward_fn("linear");
    
    assert(create_fn != NULL);
    assert(forward_fn != NULL);
}

TEST(layer_registration_activations) {
    const char *activation_names[] = {"relu", "sigmoid", "tanh", "softmax"};
    
    for (int i = 0; i < 4; i++) {
        LayerCreateFn create_fn = get_layer_create_fn(activation_names[i]);
        LayerForwardFn forward_fn = get_layer_forward_fn(activation_names[i]);
        
        assert(create_fn != NULL);
        assert(forward_fn != NULL);
    }
}

TEST(loss_registration) {
    const char *loss_names[] = {"mse", "cross_entropy", "binary_cross_entropy"};
    
    for (int i = 0; i < 3; i++) {
        LossFn loss_fn = get_loss_fn(loss_names[i]);
        assert(loss_fn != NULL);
    }
}

TEST(optimizer_registration) {
    const char *opt_names[] = {"sgd", "adam"};
    
    for (int i = 0; i < 2; i++) {
        OptimizerInitStateFn init_fn = get_optimizer_init_state_fn(opt_names[i]);
        OptimizerStepFn step_fn = get_optimizer_step_fn(opt_names[i]);
        OptimizerFreeStateFn free_fn = get_optimizer_free_state_fn(opt_names[i]);
        
        assert(init_fn != NULL);
        assert(step_fn != NULL);
        assert(free_fn != NULL);
    }
}

TEST(get_nonexistent_layer) {
    LayerCreateFn create_fn = get_layer_create_fn("nonexistent_layer");
    assert(create_fn == NULL);
}

TEST(get_nonexistent_loss) {
    LossFn loss_fn = get_loss_fn("nonexistent_loss");
    assert(loss_fn == NULL);
}

TEST(get_nonexistent_optimizer) {
    OptimizerInitStateFn init_fn = get_optimizer_init_state_fn("nonexistent_optimizer");
    assert(init_fn == NULL);
}

// ====================================================
// Layer Creation via Registry Tests
// ====================================================

TEST(layer_create_via_registry) {
    Layer *layer = layer_create(LINEAR(10, 5));
    
    assert(layer != NULL);
    assert(layer->weights != NULL);
    assert(layer->bias != NULL);
    assert(layer->weights->shape[0] == 10);
    assert(layer->weights->shape[1] == 5);
    assert(layer->bias->shape[0] == 5);
    assert(layer->num_parameters == 2);
    
    layer_free(layer);
}

TEST(activation_layers_via_registry) {
    Layer *relu = layer_create(RELU());
    Layer *sigmoid = layer_create(SIGMOID());
    Layer *tanh_layer = layer_create(TANH());
    Layer *softmax = layer_create(SOFTMAX());
    
    assert(relu != NULL);
    assert(sigmoid != NULL);
    assert(tanh_layer != NULL);
    assert(softmax != NULL);
    
    assert(relu->num_parameters == 0);
    assert(sigmoid->num_parameters == 0);
    
    layer_free(relu);
    layer_free(sigmoid);
    layer_free(tanh_layer);
    layer_free(softmax);
}

// ====================================================
// Loss Functions via Registry Tests
// ====================================================

TEST(loss_mse_via_registry) {
    size_t shape[] = {3};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 1.0f; pred->data[1] = 2.0f; pred->data[2] = 3.0f;
    target->data[0] = 1.5f; target->data[1] = 2.5f; target->data[2] = 3.5f;
    
    LossFn loss_fn = get_loss_fn("mse");
    assert(loss_fn != NULL);
    
    Tensor *loss = loss_fn(pred, target);
    assert(loss != NULL);
    assert(loss->data[0] > 0.0f);
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

TEST(loss_cross_entropy_via_registry) {
    size_t shape[] = {3};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 0.7f; pred->data[1] = 0.2f; pred->data[2] = 0.1f;
    target->data[0] = 1.0f; target->data[1] = 0.0f; target->data[2] = 0.0f;
    
    LossFn loss_fn = get_loss_fn("cross_entropy");
    assert(loss_fn != NULL);
    
    Tensor *loss = loss_fn(pred, target);
    assert(loss != NULL);
    assert(loss->data[0] > 0.0f);
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

// ====================================================
// Optimizer via Registry Tests
// ====================================================

TEST(optimizer_sgd_via_registry) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 3)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.01f, 0.0f));
    
    assert(opt != NULL);
    assert(opt->num_parameters == 2); // weights + bias
    assert(opt->step != NULL);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_adam_via_registry) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 3)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, ADAM(0.001f, 0.9f, 0.999f, 1e-8f));
    
    assert(opt != NULL);
    assert(opt->num_parameters == 2);
    assert(opt->step != NULL);
    
    optimizer_free(opt);
    network_free(net);
}

// ====================================================
// Integration Tests
// ====================================================

TEST(full_network_via_registry) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(4, 8)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(8, 2)));
    
    assert(net->num_layers == 3);
    assert(net->num_parameters == 4); // 2 layers * (weights + bias)
    
    size_t input_shape[] = {2, 4};
    Tensor *input = tensor_ones(input_shape, 2);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 2);
    assert(output->shape[1] == 2);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(training_with_registry_loss) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 1)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.1f, 0.0f));
    
    size_t shape[] = {4, 2};
    Tensor *inputs = tensor_ones(shape, 2);
    
    size_t target_shape[] = {4, 1};
    Tensor *targets = tensor_ones(target_shape, 2);
    
    // This should not crash
    network_train(net, opt, inputs, targets, 2, 2, "mse", 0);
    
    tensor_free(inputs);
    tensor_free(targets);
    optimizer_free(opt);
    network_free(net);
}

// ====================================================
// Main Test Runner
// ====================================================

int main() {
    printf("=== Running Registry Tests ===\n\n");
    
    basednn_init();
    
    // Registry initialization tests
    RUN_TEST(registry_init);
    RUN_TEST(layer_registration_linear);
    RUN_TEST(layer_registration_activations);
    RUN_TEST(loss_registration);
    RUN_TEST(optimizer_registration);
    
    // Error handling tests
    RUN_TEST(get_nonexistent_layer);
    RUN_TEST(get_nonexistent_loss);
    RUN_TEST(get_nonexistent_optimizer);
    
    // Layer creation tests
    RUN_TEST(layer_create_via_registry);
    RUN_TEST(activation_layers_via_registry);
    
    // Loss function tests
    RUN_TEST(loss_mse_via_registry);
    RUN_TEST(loss_cross_entropy_via_registry);
    
    // Optimizer tests
    RUN_TEST(optimizer_sgd_via_registry);
    RUN_TEST(optimizer_adam_via_registry);
    
    // Integration tests
    RUN_TEST(full_network_via_registry);
    RUN_TEST(training_with_registry_loss);
    
    basednn_cleanup();
    
    printf("\n=== All Registry Tests Passed! ===\n");
    return 0;
}
