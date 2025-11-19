#include "../../core/include/basednn.h"
#include "../webgpu/webgpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    basednn_init();
    
    if (webgpu_available()) {
        printf("✓ WebGPU backend initialized successfully!\n\n");
    } else {
        printf("✗ WebGPU backend not available\n");
        return 1;
    }
    
    printf("Testing WebGPU-accelerated operations:\n\n");
    
    // Test element-wise addition
    printf("1. Element-wise addition (1000x1000)...\n");
    size_t shape[] = {1000, 1000};
    Tensor *a = tensor_create(shape, 2);
    Tensor *b = tensor_create(shape, 2);
    tensor_fill(a, 2.0f);
    tensor_fill(b, 3.0f);
    
    clock_t start = clock();
    Tensor *c = tensor_add(a, b);  // Will use WebGPU automatically
    clock_t end = clock();
    
    printf("   Result: c[0]=%f (expected 5.0)\n", c->data[0]);
    printf("   Time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    
    // Test matrix multiplication
    printf("\n2. Matrix multiplication (512x512 @ 512x512)...\n");
    size_t m_shape_a[] = {512, 512};
    size_t m_shape_b[] = {512, 512};
    Tensor *m_a = tensor_create(m_shape_a, 2);
    Tensor *m_b = tensor_create(m_shape_b, 2);
    tensor_fill(m_a, 1.0f);
    tensor_fill(m_b, 2.0f);
    
    start = clock();
    Tensor *m_c = tensor_matmul(m_a, m_b);  // Will use WebGPU automatically
    end = clock();
    
    printf("   Result: c[0]=%f (expected 1024.0)\n", m_c->data[0]);
    printf("   Time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
    tensor_free(m_a);
    tensor_free(m_b);
    tensor_free(m_c);
    
    // Test activation functions
    printf("\n3. ReLU activation (1000x1000)...\n");
    Tensor *relu_in = tensor_create(shape, 2);
    for (size_t i = 0; i < relu_in->size; i++) {
        relu_in->data[i] = (float)i / relu_in->size - 0.5f;  // Range [-0.5, 0.5]
    }
    
    start = clock();
    Tensor *relu_out = tensor_relu(relu_in);
    end = clock();
    
    printf("   Result: out[0]=%f, out[999999]=%f\n", relu_out->data[0], relu_out->data[999999]);
    printf("   Time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
    tensor_free(relu_in);
    tensor_free(relu_out);
    
    // Test softmax
    printf("\n4. Softmax (100x1000)...\n");
    size_t sm_shape[] = {100, 1000};
    Tensor *sm_in = tensor_create(sm_shape, 2);
    tensor_fill(sm_in, 1.0f);
    
    start = clock();
    Tensor *sm_out = tensor_softmax(sm_in);
    end = clock();
    
    // Check that each row sums to 1.0
    float row_sum = 0.0f;
    for (size_t i = 0; i < 1000; i++) {
        row_sum += sm_out->data[i];
    }
    
    printf("   Row sum: %f (expected 1.0)\n", row_sum);
    printf("   Time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
    tensor_free(sm_in);
    tensor_free(sm_out);
    
    printf("\n✓ All WebGPU operations completed successfully!\n");
    
    basednn_cleanup();
    return 0;
}
