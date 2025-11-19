#include "../../include/basednn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define EPSILON 1e-4f
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { printf("Running %s...\n", #name); test_##name(); printf("  PASSED\n"); } while(0)

// ====================================================
// Network Creation Tests
// ====================================================

TEST(network_create) {
    Network *net = network_create();
    
    assert(net != NULL);
    assert(net->layers != NULL);
    assert(net->num_layers == 0);
    assert(net->num_parameters == 0);
    assert(net->capacity >= 8);
    
    network_free(net);
}

TEST(network_add_layer) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(5, 3));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(3, 2));
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    assert(net->num_layers == 3);
    assert(net->layers[0] == layer1);
    assert(net->layers[1] == layer2);
    assert(net->layers[2] == layer3);
    
    network_free(net);
}

TEST(network_add_many_layers) {
    Network *net = network_create();
    
    // Add more layers than initial capacity to test resizing
    for (int i = 0; i < 20; i++) {
        network_add_layer(net, layer_create(RELU()));
    }
    
    assert(net->num_layers == 20);
    assert(net->capacity >= 20);
    
    network_free(net);
}

// ====================================================
// Network Forward Pass Tests
// ====================================================

TEST(network_forward_single_layer) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(3, 2)));
    
    size_t input_shape[] = {1, 3};
    Tensor *input = tensor_ones(input_shape, 2);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 1);
    assert(output->shape[1] == 2);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_forward_multilayer) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(4, 8)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(8, 2)));
    
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

TEST(network_forward_with_activations) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 3)));
    network_add_layer(net, layer_create(TANH()));
    network_add_layer(net, layer_create(LINEAR(3, 1)));
    network_add_layer(net, layer_create(SIGMOID()));
    
    size_t input_shape[] = {1, 2};
    Tensor *input = tensor_create(input_shape, 2);
    input->data[0] = 0.5f;
    input->data[1] = -0.5f;
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 1);
    assert(output->shape[1] == 1);
    
    // Sigmoid output should be between 0 and 1
    assert(output->data[0] > 0.0f && output->data[0] < 1.0f);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

// ====================================================
// Network Parameter Tests
// ====================================================

TEST(network_get_parameters) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(3, 4)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(4, 2)));
    
    size_t num_params;
    Tensor **params = network_get_parameters(net, &num_params);
    
    assert(params != NULL);
    assert(num_params == 4); // 2 linear layers * (weights + bias)
    
    free(params);
    network_free(net);
}

TEST(network_zero_grad) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(2, 2));
    network_add_layer(net, layer);
    
    // Create gradients
    layer->weights->grad = (float*)malloc(layer->weights->size * sizeof(float));
    layer->bias->grad = (float*)malloc(layer->bias->size * sizeof(float));
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 1.0f;
    }
    
    network_zero_grad(net);
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        ASSERT_FLOAT_EQ(layer->weights->grad[i], 0.0f);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->grad[i], 0.0f);
    }
    
    network_free(net);
}

// ====================================================
// Network Training Tests
// ====================================================

TEST(network_train_step) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 1)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.1f, 0.0f));
    
    size_t shape[] = {2, 2};
    Tensor *input = tensor_ones(shape, 2);
    
    size_t target_shape[] = {2, 1};
    Tensor *target = tensor_ones(target_shape, 2);
    
    float loss = network_train_step(net, input, target, opt, "mse");
    
    assert(loss >= 0.0f);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_epochs) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 1)));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, SGD(0.1f, 0.0f));
    
    size_t shape[] = {4, 2};
    Tensor *inputs = tensor_ones(shape, 2);
    
    size_t target_shape[] = {4, 1};
    Tensor *targets = tensor_ones(target_shape, 2);
    
    // Train for a few epochs
    network_train(net, opt, inputs, targets, 3, 2, "mse", 0);
    
    // Test should complete without crashes
    
    tensor_free(inputs);
    tensor_free(targets);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_with_cross_entropy) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 3)));
    network_add_layer(net, layer_create(SOFTMAX()));
    
    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, ADAM(0.01f, 0.9f, 0.999f, 1e-8f));
    
    size_t shape[] = {4, 2};
    Tensor *inputs = tensor_ones(shape, 2);
    
    size_t target_shape[] = {4, 3};
    Tensor *targets = tensor_create(target_shape, 2);
    
    // One-hot encoded targets
    for (size_t i = 0; i < 4; i++) {
        targets->data[i * 3 + 0] = (i % 2 == 0) ? 1.0f : 0.0f;
        targets->data[i * 3 + 1] = (i % 2 == 1) ? 1.0f : 0.0f;
        targets->data[i * 3 + 2] = 0.0f;
    }
    
    network_train(net, opt, inputs, targets, 2, 2, "cross_entropy", 0);
    
    tensor_free(inputs);
    tensor_free(targets);
    optimizer_free(opt);
    network_free(net);
}

// ====================================================
// Network Accuracy Tests
// ====================================================

TEST(network_accuracy_perfect) {
    size_t shape[] = {3, 3};
    Tensor *predictions = tensor_create(shape, 2);
    Tensor *targets = tensor_create(shape, 2);
    
    // Perfect predictions (one-hot)
    predictions->data[0] = 1.0f; predictions->data[1] = 0.0f; predictions->data[2] = 0.0f;
    predictions->data[3] = 0.0f; predictions->data[4] = 1.0f; predictions->data[5] = 0.0f;
    predictions->data[6] = 0.0f; predictions->data[7] = 0.0f; predictions->data[8] = 1.0f;
    
    targets->data[0] = 1.0f; targets->data[1] = 0.0f; targets->data[2] = 0.0f;
    targets->data[3] = 0.0f; targets->data[4] = 1.0f; targets->data[5] = 0.0f;
    targets->data[6] = 0.0f; targets->data[7] = 0.0f; targets->data[8] = 1.0f;
    
    float accuracy = network_accuracy(predictions, targets);
    
    ASSERT_FLOAT_EQ(accuracy, 1.0f);
    
    tensor_free(predictions);
    tensor_free(targets);
}

TEST(network_accuracy_partial) {
    size_t shape[] = {4, 2};
    Tensor *predictions = tensor_create(shape, 2);
    Tensor *targets = tensor_create(shape, 2);
    
    // 2 out of 4 correct
    predictions->data[0] = 0.8f; predictions->data[1] = 0.2f; // Correct (class 0)
    predictions->data[2] = 0.3f; predictions->data[3] = 0.7f; // Correct (class 1)
    predictions->data[4] = 0.6f; predictions->data[5] = 0.4f; // Wrong (predicts 0, should be 1)
    predictions->data[6] = 0.4f; predictions->data[7] = 0.6f; // Wrong (predicts 1, should be 0)
    
    targets->data[0] = 1.0f; targets->data[1] = 0.0f;
    targets->data[2] = 0.0f; targets->data[3] = 1.0f;
    targets->data[4] = 0.0f; targets->data[5] = 1.0f;
    targets->data[6] = 1.0f; targets->data[7] = 0.0f;
    
    float accuracy = network_accuracy(predictions, targets);
    
    ASSERT_FLOAT_EQ(accuracy, 0.5f); // 2 out of 4
    
    tensor_free(predictions);
    tensor_free(targets);
}

// ====================================================
// Network Save/Load Tests
// ====================================================

TEST(network_save_load) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(3, 2)));
    network_add_layer(net, layer_create(RELU()));
    
    // Set specific weights
    net->layers[0]->weights->data[0] = 1.5f;
    net->layers[0]->bias->data[0] = 0.5f;
    
    const char *filepath = "/tmp/test_network.bdnn";
    network_save(net, filepath);
    
    Network *loaded = network_load(filepath);
    
    assert(loaded != NULL);
    assert(loaded->num_layers == 2);
    ASSERT_FLOAT_EQ(loaded->layers[0]->weights->data[0], 1.5f);
    ASSERT_FLOAT_EQ(loaded->layers[0]->bias->data[0], 0.5f);
    
    network_free(net);
    network_free(loaded);
}

// ====================================================
// Network Print Tests
// ====================================================

TEST(network_print) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(10, 5)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(5, 2)));
    
    // Should not crash
    network_print(net);
    
    network_free(net);
}

// ====================================================
// Edge Cases
// ====================================================

TEST(network_free_null) {
    network_free(NULL); // Should not crash
}

TEST(network_forward_null_input) {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 2)));
    
    Tensor *output = network_forward(net, NULL);
    
    assert(output == NULL);
    
    network_free(net);
}

TEST(network_empty) {
    Network *net = network_create();
    
    size_t num_params;
    Tensor **params = network_get_parameters(net, &num_params);
    
    assert(num_params == 0);
    
    network_free(net);
}

// ====================================================
// Main Test Runner
// ====================================================

int main() {
    printf("=== Running Network Tests ===\n\n");
    
    basednn_init();
    
    // Creation tests
    RUN_TEST(network_create);
    RUN_TEST(network_add_layer);
    RUN_TEST(network_add_many_layers);
    
    // Forward pass tests
    RUN_TEST(network_forward_single_layer);
    RUN_TEST(network_forward_multilayer);
    RUN_TEST(network_forward_with_activations);
    
    // Parameter tests
    RUN_TEST(network_get_parameters);
    RUN_TEST(network_zero_grad);
    
    // Training tests
    RUN_TEST(network_train_step);
    RUN_TEST(network_train_epochs);
    RUN_TEST(network_train_with_cross_entropy);
    
    // Accuracy tests
    RUN_TEST(network_accuracy_perfect);
    RUN_TEST(network_accuracy_partial);
    
    // Save/load tests
    RUN_TEST(network_save_load);
    
    // Print test
    RUN_TEST(network_print);
    
    // Edge cases
    RUN_TEST(network_free_null);
    RUN_TEST(network_forward_null_input);
    RUN_TEST(network_empty);
    
    basednn_cleanup();
    
    printf("\n=== All Network Tests Passed! ===\n");
    return 0;
}
