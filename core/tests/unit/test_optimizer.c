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
// SGD Optimizer Tests
// ====================================================

TEST(sgd_creation) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 3)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.01f, 0.0f));
    
    assert(opt != NULL);
    assert(opt->parameters != NULL);
    assert(opt->num_parameters == 2);
    assert(opt->step != NULL);
    assert(opt->zero_grad != NULL);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(sgd_step_no_momentum) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(2, 1));
    network_add_layer(net, layer);
    
    // Set initial values
    layer->weights->data[0] = 1.0f;
    layer->weights->data[1] = 2.0f;
    layer->bias->data[0] = 0.5f;
    
    // Create gradients
    layer->weights->grad = (float*)malloc(2 * sizeof(float));
    layer->bias->grad = (float*)malloc(1 * sizeof(float));
    layer->weights->grad[0] = 0.1f;
    layer->weights->grad[1] = 0.2f;
    layer->bias->grad[0] = 0.05f;
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.1f, 0.0f));
    
    optimizer_step(opt);
    
    // weights should be updated: w = w - lr * grad
    ASSERT_FLOAT_EQ(layer->weights->data[0], 1.0f - 0.1f * 0.1f);  // 0.99
    ASSERT_FLOAT_EQ(layer->weights->data[1], 2.0f - 0.1f * 0.2f);  // 1.98
    ASSERT_FLOAT_EQ(layer->bias->data[0], 0.5f - 0.1f * 0.05f);    // 0.495
    
    optimizer_free(opt);
    network_free(net);
}

TEST(sgd_with_momentum) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(2, 1));
    network_add_layer(net, layer);
    
    layer->weights->data[0] = 1.0f;
    layer->weights->data[1] = 2.0f;
    
    layer->weights->grad = (float*)malloc(2 * sizeof(float));
    layer->bias->grad = (float*)malloc(1 * sizeof(float));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.1f, 0.9f));
    
    // First step
    layer->weights->grad[0] = 0.1f;
    layer->weights->grad[1] = 0.2f;
    layer->bias->grad[0] = 0.0f;
    
    float w0_before = layer->weights->data[0];
    float w1_before = layer->weights->data[1];
    
    optimizer_step(opt);
    
    // Should have changed
    assert(layer->weights->data[0] != w0_before);
    assert(layer->weights->data[1] != w1_before);
    
    optimizer_free(opt);
    network_free(net);
}

// ====================================================
// Adam Optimizer Tests
// ====================================================

TEST(adam_creation) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(3, 2)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, ADAM(0.001f, 0.9f, 0.999f, 1e-8f));
    
    assert(opt != NULL);
    assert(opt->parameters != NULL);
    assert(opt->num_parameters == 2);
    assert(opt->step != NULL);
    assert(opt->state != NULL);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(adam_step) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(2, 1));
    network_add_layer(net, layer);
    
    layer->weights->data[0] = 1.0f;
    layer->weights->data[1] = 2.0f;
    layer->bias->data[0] = 0.5f;
    
    layer->weights->grad = (float*)malloc(2 * sizeof(float));
    layer->bias->grad = (float*)malloc(1 * sizeof(float));
    layer->weights->grad[0] = 0.1f;
    layer->weights->grad[1] = 0.2f;
    layer->bias->grad[0] = 0.05f;
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, ADAM(0.001f, 0.9f, 0.999f, 1e-8f));
    
    float w0_before = layer->weights->data[0];
    float w1_before = layer->weights->data[1];
    float b0_before = layer->bias->data[0];
    
    optimizer_step(opt);
    
    // Parameters should have changed
    assert(layer->weights->data[0] != w0_before);
    assert(layer->weights->data[1] != w1_before);
    assert(layer->bias->data[0] != b0_before);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(adam_multiple_steps) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(1, 1));
    network_add_layer(net, layer);
    
    layer->weights->data[0] = 1.0f;
    layer->bias->data[0] = 0.0f;
    
    layer->weights->grad = (float*)malloc(1 * sizeof(float));
    layer->bias->grad = (float*)malloc(1 * sizeof(float));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, ADAM(0.01f, 0.9f, 0.999f, 1e-8f));
    
    // Run multiple steps to test momentum accumulation
    for (int i = 0; i < 5; i++) {
        layer->weights->grad[0] = 0.1f;
        layer->bias->grad[0] = 0.01f;
        optimizer_step(opt);
    }
    
    // Weight should have decreased significantly
    assert(layer->weights->data[0] < 1.0f);
    
    optimizer_free(opt);
    network_free(net);
}

// ====================================================
// Optimizer Utility Tests
// ====================================================

TEST(optimizer_zero_grad) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(2, 2));
    network_add_layer(net, layer);
    
    layer->weights->grad = (float*)malloc(layer->weights->size * sizeof(float));
    layer->bias->grad = (float*)malloc(layer->bias->size * sizeof(float));
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 1.0f;
    }
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.01f, 0.0f));
    
    optimizer_zero_grad(opt);
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        ASSERT_FLOAT_EQ(layer->weights->grad[i], 0.0f);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->grad[i], 0.0f);
    }
    
    optimizer_free(opt);
    network_free(net);
}

// ====================================================
// Multi-Layer Network Optimizer Tests
// ====================================================

TEST(optimizer_multilayer) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(4, 8)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(8, 2)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, ADAM(0.001f, 0.9f, 0.999f, 1e-8f));
    
    assert(opt != NULL);
    assert(opt->num_parameters == 4); // 2 layers with weights + bias each
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_step_multilayer) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 3)));
    network_add_layer(net, layer_create(LINEAR(3, 1)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.1f, 0.0f));
    
    // Create gradients for all parameters
    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i];
        param->grad = (float*)calloc(param->size, sizeof(float));
        for (size_t j = 0; j < param->size; j++) {
            param->grad[j] = 0.1f;
        }
    }
    
    // Store initial values
    float initial_values[4];
    for (size_t i = 0; i < opt->num_parameters; i++) {
        initial_values[i] = opt->parameters[i]->data[0];
    }
    
    optimizer_step(opt);
    
    // All parameters should have changed
    for (size_t i = 0; i < opt->num_parameters; i++) {
        assert(opt->parameters[i]->data[0] != initial_values[i]);
    }
    
    optimizer_free(opt);
    network_free(net);
}

// ====================================================
// Edge Cases
// ====================================================

TEST(optimizer_free_null) {
    optimizer_free(NULL); // Should not crash
}

TEST(optimizer_step_without_gradients) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 1)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.1f, 0.0f));
    
    // Store initial values
    float w0 = net->layers[0]->weights->data[0];
    
    // Step without setting gradients (they should be NULL)
    optimizer_step(opt);
    
    // Values should remain unchanged
    ASSERT_FLOAT_EQ(net->layers[0]->weights->data[0], w0);
    
    optimizer_free(opt);
    network_free(net);
}

// ====================================================
// Main Test Runner
// ====================================================

int main() {
    printf("=== Running Optimizer Tests ===\n\n");
    
    basednn_init();
    
    // SGD tests
    RUN_TEST(sgd_creation);
    RUN_TEST(sgd_step_no_momentum);
    RUN_TEST(sgd_with_momentum);
    
    // Adam tests
    RUN_TEST(adam_creation);
    RUN_TEST(adam_step);
    RUN_TEST(adam_multiple_steps);
    
    // Utility tests
    RUN_TEST(optimizer_zero_grad);
    
    // Multi-layer tests
    RUN_TEST(optimizer_multilayer);
    RUN_TEST(optimizer_step_multilayer);
    
    // Edge cases
    RUN_TEST(optimizer_free_null);
    RUN_TEST(optimizer_step_without_gradients);
    
    basednn_cleanup();
    
    printf("\n=== All Optimizer Tests Passed! ===\n");
    return 0;
}
