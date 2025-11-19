#ifndef BASEDNN_H
#define BASEDNN_H

#include "tensor.h"
#include "ops.h"
#include "registry.h"
#include "layer.h"
#include "network.h"
#include "optimizer.h"

// Initialize the registry with built-in layers, losses, and optimizers
// Call this once at the start of your program
static inline void basednn_init() {
    registry_init();
}

// Cleanup registry resources
// Call this at the end of your program
static inline void basednn_cleanup() {
    registry_cleanup();
}

#endif