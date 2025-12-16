#ifndef BASEDNN_H
#define BASEDNN_H

#include "tensor.h"
#include "ops.h"
#include "registry.h"
#include "layer.h"
#include "network.h"
#include "optimizer.h"

static inline void basednn_init() {
    registry_init();
}


static inline void basednn_cleanup() {
    registry_cleanup();
}

#endif
