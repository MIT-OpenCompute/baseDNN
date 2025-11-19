#include "../../include/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-5f
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { printf("Running %s...\n", #name); test_##name(); printf("  PASSED\n"); } while(0)

// ====================================================
// Tensor Creation Tests
// ====================================================

TEST(tensor_create) {
    size_t shape[] = {2, 3};
    Tensor *t = tensor_create(shape, 2);
    
    assert(t != NULL);
    assert(t->ndim == 2);
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);
    assert(t->size == 6);
    assert(t->data != NULL);
    assert(t->grad == NULL);
    assert(t->requires_grad == 0);
    assert(t->owns_data == 1);
    
    tensor_free(t);
}

TEST(tensor_zeroes) {
    size_t shape[] = {3, 2};
    Tensor *t = tensor_zeroes(shape, 2);
    
    assert(t != NULL);
    for (size_t i = 0; i < t->size; i++) {
        ASSERT_FLOAT_EQ(t->data[i], 0.0f);
    }
    
    tensor_free(t);
}

TEST(tensor_ones) {
    size_t shape[] = {2, 2};
    Tensor *t = tensor_ones(shape, 2);
    
    assert(t != NULL);
    for (size_t i = 0; i < t->size; i++) {
        ASSERT_FLOAT_EQ(t->data[i], 1.0f);
    }
    
    tensor_free(t);
}

TEST(tensor_randn) {
    size_t shape[] = {10, 10};
    Tensor *t = tensor_randn(shape, 2, 42);
    
    assert(t != NULL);
    
    // Check that values are roughly normally distributed
    float sum = 0.0f;
    for (size_t i = 0; i < t->size; i++) {
        sum += t->data[i];
    }
    float mean = sum / t->size;
    
    // Mean should be close to 0 (within 0.5 for 100 samples)
    assert(fabsf(mean) < 0.5f);
    
    tensor_free(t);
}

// ====================================================
// Tensor Utilities Tests
// ====================================================

TEST(tensor_fill) {
    size_t shape[] = {3, 3};
    Tensor *t = tensor_create(shape, 2);
    
    tensor_fill(t, 5.5f);
    
    for (size_t i = 0; i < t->size; i++) {
        ASSERT_FLOAT_EQ(t->data[i], 5.5f);
    }
    
    tensor_free(t);
}

TEST(tensor_copy) {
    size_t shape[] = {2, 3};
    Tensor *t1 = tensor_create(shape, 2);
    
    for (size_t i = 0; i < t1->size; i++) {
        t1->data[i] = (float)i;
    }
    
    Tensor *t2 = tensor_copy(t1);
    
    assert(t2 != NULL);
    assert(t2 != t1);
    assert(t2->data != t1->data);
    assert(t2->ndim == t1->ndim);
    assert(t2->size == t1->size);
    
    for (size_t i = 0; i < t1->size; i++) {
        ASSERT_FLOAT_EQ(t2->data[i], t1->data[i]);
    }
    
    tensor_free(t1);
    tensor_free(t2);
}

// ====================================================
// Autograd Tests
// ====================================================

TEST(tensor_set_requires_grad) {
    size_t shape[] = {2, 2};
    Tensor *t = tensor_create(shape, 2);
    
    assert(t->requires_grad == 0);
    
    tensor_set_requires_grad(t, 1);
    assert(t->requires_grad == 1);
    
    tensor_set_requires_grad(t, 0);
    assert(t->requires_grad == 0);
    
    tensor_free(t);
}

TEST(tensor_zero_grad) {
    size_t shape[] = {3, 2};
    Tensor *t = tensor_create(shape, 2);
    
    t->grad = (float*)malloc(t->size * sizeof(float));
    for (size_t i = 0; i < t->size; i++) {
        t->grad[i] = (float)i;
    }
    
    tensor_zero_grad(t);
    
    for (size_t i = 0; i < t->size; i++) {
        ASSERT_FLOAT_EQ(t->grad[i], 0.0f);
    }
    
    tensor_free(t);
}

TEST(tensor_backward_simple) {
    size_t shape[] = {1};
    Tensor *t = tensor_create(shape, 1);
    t->data[0] = 5.0f;
    
    tensor_set_requires_grad(t, 1);
    tensor_backward(t);
    
    assert(t->grad != NULL);
    ASSERT_FLOAT_EQ(t->grad[0], 1.0f);
    
    tensor_free(t);
}

// ====================================================
// Shape Tests
// ====================================================

TEST(tensor_different_shapes) {
    size_t shape1[] = {5};
    size_t shape2[] = {3, 4};
    size_t shape3[] = {2, 3, 4};
    
    Tensor *t1 = tensor_create(shape1, 1);
    Tensor *t2 = tensor_create(shape2, 2);
    Tensor *t3 = tensor_create(shape3, 3);
    
    assert(t1->ndim == 1 && t1->size == 5);
    assert(t2->ndim == 2 && t2->size == 12);
    assert(t3->ndim == 3 && t3->size == 24);
    
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
}

// ====================================================
// Edge Cases
// ====================================================

TEST(tensor_free_null) {
    tensor_free(NULL);  // Should not crash
}

TEST(tensor_single_element) {
    size_t shape[] = {1, 1};
    Tensor *t = tensor_create(shape, 2);
    
    assert(t != NULL);
    assert(t->size == 1);
    
    t->data[0] = 42.0f;
    ASSERT_FLOAT_EQ(t->data[0], 42.0f);
    
    tensor_free(t);
}

TEST(tensor_large) {
    size_t shape[] = {100, 100};
    Tensor *t = tensor_create(shape, 2);
    
    assert(t != NULL);
    assert(t->size == 10000);
    
    tensor_fill(t, 1.5f);
    
    for (size_t i = 0; i < t->size; i++) {
        ASSERT_FLOAT_EQ(t->data[i], 1.5f);
    }
    
    tensor_free(t);
}

// ====================================================
// Main Test Runner
// ====================================================

int main() {
    printf("=== Running Tensor Tests ===\n\n");
    
    // Creation tests
    RUN_TEST(tensor_create);
    RUN_TEST(tensor_zeroes);
    RUN_TEST(tensor_ones);
    RUN_TEST(tensor_randn);
    
    // Utilities tests
    RUN_TEST(tensor_fill);
    RUN_TEST(tensor_copy);
    
    // Autograd tests
    RUN_TEST(tensor_set_requires_grad);
    RUN_TEST(tensor_zero_grad);
    RUN_TEST(tensor_backward_simple);
    
    // Shape tests
    RUN_TEST(tensor_different_shapes);
    
    // Edge cases
    RUN_TEST(tensor_free_null);
    RUN_TEST(tensor_single_element);
    RUN_TEST(tensor_large);
    
    printf("\n=== All Tensor Tests Passed! ===\n");
    return 0;
}
