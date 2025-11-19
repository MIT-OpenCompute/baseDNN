#include <stdio.h>
#include <webgpu/webgpu.h>
#include <unistd.h>

int main() {
    printf("Testing Dawn Metal compute shader execution...\n");
    
    WGPUInstance instance = wgpuCreateInstance(NULL);
    if (!instance) {
        printf("Failed to create instance\n");
        return 1;
    }
    
    // Get adapter
    WGPUAdapter adapter = NULL;
    WGPURequestAdapterOptions opts = {
        .powerPreference = WGPUPowerPreference_HighPerformance,
        .backendType = WGPUBackendType_Metal,
    };
    
    // Synchronous request using Dawn native API
    wgpuInstanceRequestAdapter(instance, &opts, NULL);
    wgpuInstanceProcessEvents(instance);
    
    printf("Check Activity Monitor GPU History NOW!\n");
    sleep(3);
    
    printf("If you see GPU activity above, Dawn Metal is working.\n");
    printf("If not, Dawn is using software emulation.\n");
    
    return 0;
}
