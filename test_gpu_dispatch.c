#include "core/include/basednn.h"
#include "backend/webgpu/webgpu_backend.h"
#include <stdio.h>

int main() {
    printf("Initializing...\n");
    basednn_init();
    
    printf("WebGPU available: %d\n", webgpu_available());
    
    // Simple test
    size_t shape[] = {10, 10};
    Tensor *a = tensor_create(shape, 2);
    Tensor *b = tensor_create(shape, 2);
    tensor_fill(a, 2.0f);
    tensor_fill(b, 3.0f);
    
    printf("Before add operation\n");
    Tensor *c = tensor_add(a, b);
    printf("After add operation\n");
    printf("Result: c[0] = %f (expected 5.0)\n", c->data[0]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    
    basednn_cleanup();
    return 0;
}
