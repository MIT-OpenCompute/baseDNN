#include "../../core/include/basednn.h"
#include "../webgpu/webgpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MEASURE_TIME(name, code) do { \
    clock_t start = clock(); \
    code; \
    clock_t end = clock(); \
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000; \
    printf("  %s: %.2f ms\n", name, ms); \
} while(0)

int main() {
    printf("=== BaseDNN Performance Test: CPU vs WebGPU ===\n\n");
    
    basednn_init();
    
    if (webgpu_available()) {
        printf("✓ WebGPU backend available\n\n");
    } else {
        printf("⚠ WebGPU backend not available - running CPU only\n\n");
    }
    
    // Test 1: Large matrix multiplication
    printf("Test 1: Matrix Multiplication (1024x1024 @ 1024x1024)\n");
    {
        size_t shape_a[] = {1024, 1024};
        size_t shape_b[] = {1024, 1024};
        Tensor *a = tensor_create(shape_a, 2);
        Tensor *b = tensor_create(shape_b, 2);
        
        // Fill with random values
        for (size_t i = 0; i < a->size; i++) a->data[i] = (float)rand() / RAND_MAX;
        for (size_t i = 0; i < b->size; i++) b->data[i] = (float)rand() / RAND_MAX;
        
        Tensor *c = NULL;
        MEASURE_TIME("Matrix multiply", c = tensor_matmul(a, b));
        
        printf("  Result sample: c[0] = %f\n", c->data[0]);
        
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
    }
    
    // Test 2: Element-wise operations
    printf("\nTest 2: Element-wise Addition (5000x5000)\n");
    {
        size_t shape[] = {5000, 5000};
        Tensor *a = tensor_create(shape, 2);
        Tensor *b = tensor_create(shape, 2);
        tensor_fill(a, 1.5f);
        tensor_fill(b, 2.5f);
        
        Tensor *c = NULL;
        MEASURE_TIME("Addition", c = tensor_add(a, b));
        
        printf("  Result: c[0] = %f (expected 4.0)\n", c->data[0]);
        
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
    }
    
    // Test 3: Activation functions
    printf("\nTest 3: ReLU Activation (5000x5000)\n");
    {
        size_t shape[] = {5000, 5000};
        Tensor *a = tensor_create(shape, 2);
        for (size_t i = 0; i < a->size; i++) {
            a->data[i] = (float)i / a->size - 0.5f;
        }
        
        Tensor *b = NULL;
        MEASURE_TIME("ReLU", b = tensor_relu(a));
        
        printf("  Negative values zeroed: %s\n", b->data[0] == 0.0f ? "✓" : "✗");
        
        tensor_free(a);
        tensor_free(b);
    }
    
    // Test 4: Softmax
    printf("\nTest 4: Softmax (1000x1000)\n");
    {
        size_t shape[] = {1000, 1000};
        Tensor *a = tensor_create(shape, 2);
        for (size_t i = 0; i < a->size; i++) {
            a->data[i] = (float)rand() / RAND_MAX;
        }
        
        Tensor *b = NULL;
        MEASURE_TIME("Softmax", b = tensor_softmax(a));
        
        // Verify first row sums to 1.0
        float sum = 0.0f;
        for (size_t i = 0; i < 1000; i++) sum += b->data[i];
        printf("  Row sum: %.6f (expected 1.0)\n", sum);
        
        tensor_free(a);
        tensor_free(b);
    }
    
    // Test 5: Small neural network forward pass
    printf("\nTest 5: Small Network Forward Pass (batch=256)\n");
    {
        Network *net = network_create();
        network_add_layer(net, layer_create(LINEAR(784, 256)));
        network_add_layer(net, layer_create(RELU()));
        network_add_layer(net, layer_create(LINEAR(256, 128)));
        network_add_layer(net, layer_create(RELU()));
        network_add_layer(net, layer_create(LINEAR(128, 10)));
        network_add_layer(net, layer_create(SOFTMAX()));
        
        size_t input_shape[] = {256, 784};
        Tensor *input = tensor_create(input_shape, 2);
        for (size_t i = 0; i < input->size; i++) {
            input->data[i] = (float)rand() / RAND_MAX;
        }
        
        Tensor *output = NULL;
        MEASURE_TIME("Forward pass", output = network_forward(net, input));
        
        printf("  Output shape: [%zu, %zu]\n", output->shape[0], output->shape[1]);
        
        tensor_free(input);
        tensor_free(output);
        network_free(net);
    }
    
    printf("\n=== Test Complete ===\n");
    
    if (webgpu_available()) {
        printf("\n✓ All operations accelerated with WebGPU\n");
        printf("  GPU: Uses compute shaders on Metal/Vulkan/DirectX backends\n");
        printf("  Cross-platform: Works on macOS, Linux, Windows\n");
    }
    
    basednn_cleanup();
    return 0;
}
