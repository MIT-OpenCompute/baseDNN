#include "../../core/include/basednn.h"
#include "../webgpu/webgpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void stress_gpu() {
    printf("\n=== GPU Stress Test - Watch Activity Monitor GPU History ===\n\n");
    
    if (!webgpu_available()) {
        printf("ERROR: WebGPU not available!\n");
        return;
    }
    
    printf("Starting intensive GPU workload...\n");
    printf("This will run for 30 seconds with continuous GPU operations.\n");
    printf("Open Activity Monitor > Window > GPU History to see GPU usage.\n\n");
    
    sleep(2);
    
    time_t start_time = time(NULL);
    int iteration = 0;
    
    while (time(NULL) - start_time < 30) {
        iteration++;
        
        // Large matrix multiplication - this should show GPU activity
        size_t shape_a[] = {2048, 2048};
        size_t shape_b[] = {2048, 2048};
        Tensor *a = tensor_create(shape_a, 2);
        Tensor *b = tensor_create(shape_b, 2);
        
        // Fill with data
        for (size_t i = 0; i < a->size; i++) {
            a->data[i] = (float)(rand() % 100) / 100.0f;
        }
        for (size_t i = 0; i < b->size; i++) {
            b->data[i] = (float)(rand() % 100) / 100.0f;
        }
        
        // GPU matrix multiply
        Tensor *c = tensor_matmul(a, b);
        
        // Multiple element-wise operations
        Tensor *d = tensor_add(a, a);
        Tensor *e = tensor_mul(d, d);
        Tensor *f = tensor_relu(e);
        Tensor *g = tensor_sigmoid(f);
        
        // Report progress
        if (iteration % 5 == 0) {
            printf("Iteration %d: matmul result[0] = %f\n", iteration, c->data[0]);
            fflush(stdout);
        }
        
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
        tensor_free(d);
        tensor_free(e);
        tensor_free(f);
        tensor_free(g);
    }
    
    printf("\n=== Stress Test Complete ===\n");
    printf("Completed %d iterations\n", iteration);
}

int main() {
    basednn_init();
    
    if (webgpu_available()) {
        printf("✓ WebGPU backend initialized\n");
        stress_gpu();
    } else {
        printf("✗ WebGPU backend not available\n");
        return 1;
    }
    
    basednn_cleanup();
    return 0;
}
