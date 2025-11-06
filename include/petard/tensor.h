#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int rows, cols;
    float *d;
} mat;


// Creation and Deletion
mat empty(int rows, int cols);
mat zeros(int rows, int cols);
mat ones(int rows, int cols);
mat randn(int rows, int cols, unsigned int *seed, float scale);
void free(mat *m);

// Copy and Assignment
void copy(mat *dst, const mat *src);
void fill(mat *t, float value);

// Reshape and View
int reshape(mat *t, int rows, int cols);
mat flatten(mat *t);

// Indexing 
mat index(const mat *src, int row0, int rows, int col0, int cols); 
float get(const mat *t, int row, int col); 
void set(mat *t, int row, int col, float value); 
void td_get_row(const mat *t, int row, mat *out);
void td_set_row(mat *t, int row, const mat *in);

#ifdef __cplusplus
}
#endif