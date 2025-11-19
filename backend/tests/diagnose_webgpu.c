#include "../../core/include/basednn.h"
#include "../webgpu/webgpu_backend.h"
#include <wgpu.h>
#include <stdio.h>

int main() {
    printf("=== WebGPU Backend Diagnostics ===\n\n");
    
    basednn_init();
    
    if (!webgpu_available()) {
        printf("ERROR: WebGPU not initialized!\n");
        return 1;
    }
    
    printf("✓ WebGPU initialized\n\n");
    
    // Get device info
    WGPUDevice device = webgpu_get_device();
    if (!device) {
        printf("ERROR: No device!\n");
        return 1;
    }
    
    printf("Device obtained: %p\n", (void*)device);
    
    // Try to get adapter features/limits
    printf("\nAttempting simple GPU operation...\n");
    
    size_t shape[] = {100, 100};
    Tensor *a = tensor_create(shape, 2);
    Tensor *b = tensor_create(shape, 2);
    tensor_fill(a, 1.0f);
    tensor_fill(b, 2.0f);
    
    printf("Running tensor_add...\n");
    Tensor *c = tensor_add(a, b);
    printf("Result: %f (expected 3.0)\n", c->data[0]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    
    printf("\n=== Diagnostics Complete ===\n");
    printf("Note: wgpu-native v0.19 may default to software rendering.\n");
    printf("The library is working but may need backend configuration.\n");
    
    basednn_cleanup();
    return 0;
}
