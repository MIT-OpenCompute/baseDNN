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
// Linear Layer Tests
// ====================================================

TEST(linear_layer_creation) {
    Layer *layer = layer_create(LINEAR(10, 5));
    
    assert(layer != NULL);
    assert(strcmp(layer->name, "linear") == 0);
    assert(layer->weights != NULL);
    assert(layer->bias != NULL);
    assert(layer->weights->ndim == 2);
    assert(layer->weights->shape[0] == 10);
    assert(layer->weights->shape[1] == 5);
    assert(layer->bias->ndim == 1);
    assert(layer->bias->shape[0] == 5);
    assert(layer->num_parameters == 2);
    assert(layer->forward != NULL);
    
    layer_free(layer);
}

TEST(linear_layer_forward) {
    Layer *layer = layer_create(LINEAR(3, 2));
    
    // Manually set weights and bias for predictability
    layer->weights->data[0] = 1.0f; layer->weights->data[1] = 2.0f;
    layer->weights->data[2] = 3.0f; layer->weights->data[3] = 4.0f;
    layer->weights->data[4] = 5.0f; layer->weights->data[5] = 6.0f;
    
    layer->bias->data[0] = 0.1f;
    layer->bias->data[1] = 0.2f;
    
    size_t input_shape[] = {1, 3};
    Tensor *input = tensor_create(input_shape, 2);
    input->data[0] = 1.0f;
    input->data[1] = 1.0f;
    input->data[2] = 1.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    assert(output->ndim == 2);
    assert(output->shape[0] == 1);
    assert(output->shape[1] == 2);
    
    // [1, 1, 1] * [[1, 2], [3, 4], [5, 6]] + [0.1, 0.2]
    // = [9, 12] + [0.1, 0.2] = [9.1, 12.2]
    ASSERT_FLOAT_EQ(output->data[0], 9.1f);
    ASSERT_FLOAT_EQ(output->data[1], 12.2f);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(linear_layer_batch) {
    Layer *layer = layer_create(LINEAR(2, 3));
    
    size_t input_shape[] = {4, 2};
    Tensor *input = tensor_ones(input_shape, 2);
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 4); // batch size preserved
    assert(output->shape[1] == 3); // output features
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

// ====================================================
// Activation Layer Tests
// ====================================================

TEST(relu_layer) {
    Layer *layer = layer_create(RELU());
    
    assert(layer != NULL);
    assert(strcmp(layer->name, "relu") == 0);
    assert(layer->weights == NULL);
    assert(layer->bias == NULL);
    assert(layer->num_parameters == 0);
    
    size_t shape[] = {4};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = -1.0f;
    input->data[1] = 0.0f;
    input->data[2] = 1.0f;
    input->data[3] = 2.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    ASSERT_FLOAT_EQ(output->data[0], 0.0f);
    ASSERT_FLOAT_EQ(output->data[1], 0.0f);
    ASSERT_FLOAT_EQ(output->data[2], 1.0f);
    ASSERT_FLOAT_EQ(output->data[3], 2.0f);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(sigmoid_layer) {
    Layer *layer = layer_create(SIGMOID());
    
    assert(layer != NULL);
    assert(strcmp(layer->name, "sigmoid") == 0);
    assert(layer->num_parameters == 0);
    
    size_t shape[] = {1};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = 0.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    ASSERT_FLOAT_EQ(output->data[0], 0.5f);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(tanh_layer) {
    Layer *layer = layer_create(TANH());
    
    assert(layer != NULL);
    assert(strcmp(layer->name, "tanh") == 0);
    assert(layer->num_parameters == 0);
    
    size_t shape[] = {1};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = 0.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    ASSERT_FLOAT_EQ(output->data[0], 0.0f);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(softmax_layer) {
    Layer *layer = layer_create(SOFTMAX());
    
    assert(layer != NULL);
    assert(strcmp(layer->name, "softmax") == 0);
    assert(layer->num_parameters == 0);
    
    size_t shape[] = {3};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    
    // Sum should be 1.0
    float sum = 0.0f;
    for (size_t i = 0; i < 3; i++) {
        sum += output->data[i];
    }
    ASSERT_FLOAT_EQ(sum, 1.0f);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

// ====================================================
// Layer Parameter Management Tests
// ====================================================

TEST(layer_get_parameters_linear) {
    Layer *layer = layer_create(LINEAR(5, 3));
    
    size_t num_params;
    Tensor **params = layer_get_parameters(layer, &num_params);
    
    assert(params != NULL);
    assert(num_params == 2);
    assert(params[0] == layer->weights);
    assert(params[1] == layer->bias);
    
    layer_free(layer);
}

TEST(layer_get_parameters_activation) {
    Layer *layer = layer_create(RELU());
    
    size_t num_params;
    Tensor **params = layer_get_parameters(layer, &num_params);
    
    assert(num_params == 0);
    
    layer_free(layer);
}

TEST(layer_zero_grad) {
    Layer *layer = layer_create(LINEAR(3, 2));
    
    // Set requires_grad and create gradients
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    
    layer->weights->grad = (float*)malloc(layer->weights->size * sizeof(float));
    layer->bias->grad = (float*)malloc(layer->bias->size * sizeof(float));
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 1.0f;
    }
    
    layer_zero_grad(layer);
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        ASSERT_FLOAT_EQ(layer->weights->grad[i], 0.0f);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->grad[i], 0.0f);
    }
    
    layer_free(layer);
}

// ====================================================
// Edge Cases
// ====================================================

TEST(layer_free_null) {
    layer_free(NULL); // Should not crash
}

TEST(layer_forward_null_input) {
    Layer *layer = layer_create(LINEAR(3, 2));
    
    Tensor *output = layer_forward(layer, NULL);
    
    assert(output == NULL);
    
    layer_free(layer);
}

// ====================================================
// Main Test Runner
// ====================================================

int main() {
    printf("=== Running Layer Tests ===\n\n");
    
    basednn_init();
    
    // Linear layer tests
    RUN_TEST(linear_layer_creation);
    RUN_TEST(linear_layer_forward);
    RUN_TEST(linear_layer_batch);
    
    // Activation layer tests
    RUN_TEST(relu_layer);
    RUN_TEST(sigmoid_layer);
    RUN_TEST(tanh_layer);
    RUN_TEST(softmax_layer);
    
    // Parameter management tests
    RUN_TEST(layer_get_parameters_linear);
    RUN_TEST(layer_get_parameters_activation);
    RUN_TEST(layer_zero_grad);
    
    // Edge cases
    RUN_TEST(layer_free_null);
    RUN_TEST(layer_forward_null_input);
    
    basednn_cleanup();
    
    printf("\n=== All Layer Tests Passed! ===\n");
    return 0;
}
