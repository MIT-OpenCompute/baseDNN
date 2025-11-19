#include "../../include/basednn.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef HAS_WEBGPU
#include "../../../backend/webgpu/webgpu_backend.h"
#endif

// ./build/mnist

uint32_t read_uint32(FILE *f) {
    uint32_t val;
    fread(&val, 4, 1, f);
    return __builtin_bswap32(val);
}

void load_mnist_images(const char *path, Tensor **images, int *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Error opening %s\n", path); exit(1); }
    
    read_uint32(f);
    *count = read_uint32(f);
    int rows = read_uint32(f);
    int cols = read_uint32(f);
    
    size_t shape[] = {*count, rows * cols};
    *images = tensor_create(shape, 2);
    
    for (int i = 0; i < *count * rows * cols; i++) {
        uint8_t pixel;
        fread(&pixel, 1, 1, f);
        (*images)->data[i] = pixel / 255.0f;
    }
    fclose(f);
}

void load_mnist_labels(const char *path, Tensor **labels, int count) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Error opening %s\n", path); exit(1); }
    
    read_uint32(f);
    read_uint32(f);
    
    size_t shape[] = {count, 10};
    *labels = tensor_create(shape, 2);
    tensor_fill(*labels, 0.0f);
    
    for (int i = 0; i < count; i++) {
        uint8_t label;
        fread(&label, 1, 1, f);
        (*labels)->data[i * 10 + label] = 1.0f;
    }
    fclose(f);
}

int main() {
    printf("[MAIN] Starting MNIST...\n");
    fflush(stdout);
    basednn_init();
    
    #ifdef HAS_WEBGPU
    if (webgpu_available()) {
        printf("WebGPU GPU acceleration enabled\n");
    } else {
        printf("WebGPU not available, using CPU\n");
    }
    #else
    printf("Using CPU backend\n");
    #endif
    
    Tensor *train_images, *train_labels;
    Tensor *test_images, *test_labels;
    int train_count, test_count;
    
    printf("\nLoading MNIST data...\n");
    load_mnist_images("../core/tests/full/data/train-images-idx3-ubyte", &train_images, &train_count);
    load_mnist_labels("../core/tests/full/data/train-labels-idx1-ubyte", &train_labels, train_count);
    load_mnist_images("../core/tests/full/data/t10k-images-idx3-ubyte", &test_images, &test_count);
    load_mnist_labels("../core/tests/full/data/t10k-labels-idx1-ubyte", &test_labels, test_count);

    printf("Train: %d images, Test: %d images\n", train_count, test_count);
    
    int n_train = train_count < 5000 ? train_count : 5000;
    int n_test = test_count < 1000 ? test_count : 1000;
    
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(784, 256)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(256, 128)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(128, 10)));
    network_add_layer(net, layer_create(SOFTMAX()));

    Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, ADAM(0.005f, 0.9f, 0.999f, 1e-8f));    train_images->shape[0] = n_train;
    train_labels->shape[0] = n_train;
    test_images->shape[0] = n_test;
    test_labels->shape[0] = n_test;
    
    printf("\nTraining on %d samples...\n", n_train);
    network_train(net, opt, train_images, train_labels, 3, 64, "cross_entropy", 1);
    
    printf("\nEvaluating...\n");
    Tensor *predictions = network_forward(net, test_images);
    float accuracy = network_accuracy(predictions, test_labels);
    printf("Test Accuracy: %.2f%%\n", accuracy * 100.0f);
    
    tensor_free(train_images);
    tensor_free(train_labels);
    tensor_free(test_images);
    tensor_free(test_labels);
    tensor_free(predictions);
    optimizer_free(opt);
    network_free(net);
    
    basednn_cleanup();
    return 0;
}
