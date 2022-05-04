#pragma once

#include <omp.h>
#include <cmath>
#include <algorithm>

// Base version of algorithm used as baseline for the measurements
void Cholesky_Decomposition_Base(double* A, double* L, int n) {
    memset(L, 0, sizeof(double) * n * n);
    for (int i = 0; i < n; i++) {
        L[i * n + i] = A[i * n + i];

        for (int k = 0; k < i; k++) {
            L[i * n + i] -= L[i * n + k] * L[i * n + k];
        }
        L[i * n + i] = sqrt(L[i * n + i]);

        for (int j = i + 1; j < n; j++) {
            L[j * n + i] = A[j * n + i];
            for (int k = 0; k < i; k++) {
                L[j * n + i] -= L[i * n + k] * L[j * n + k];
            }
            L[j * n + i] /= L[i * n + i];
        }
    }
}

// Base version of algorithm but paralleled
void Cholesky_Decomposition_Base_Parallel(double* A, double* L, int n) {
    memset(L, 0, sizeof(double) * n * n);
    for (int i = 0; i < n; i++) {
        L[i * n + i] = A[i * n + i];

        for (int k = 0; k < i; k++) {
            L[i * n + i] -= L[i * n + k] * L[i * n + k];
        }
        L[i * n + i] = sqrt(L[i * n + i]);

#pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            L[j * n + i] = A[j * n + i];
            for (int k = 0; k < i; k++) {
                L[j * n + i] -= L[i * n + k] * L[j * n + k];
            }
            L[j * n + i] /= L[i * n + i];
        }
    }
}

void Cholesky_Decomposition_By_Blocks_Body(double* A, double* L, int n, int row_index,
                                           int column_index,
                                      int block_size) {
    if (row_index + block_size > n) {
        block_size = n - row_index;
    }

    auto A11 = [&](const int i, const int j) -> double& {
        const int index = (row_index * n + column_index) + i * n + j;
        return A[index];
    };
    auto L11 = [&](const int i, const int j) -> double& {
        const int index = (row_index * n + column_index) + i * n + j;
        return L[index];
    };

    // Perform Cholesky Decomposition for the first block
    // A11 = L11 * L11T
    // Find L11
    for (int i = 0; i < block_size; i++) {
        L11(i, i) = A11(i, i);

        for (int k = 0; k < i; k++) {
            L11(i, i) -= L11(i, k) * L11(i, k);
        }

        L11(i, i) = sqrt(L11(i, i));

        //#pragma omp parallel // for num_threads(4)
        for (int j = i + 1; j < block_size; j++) {
            L11(j, i) = A11(j, i);
            for (int k = 0; k < i; k++) {
                L11(j, i) -= L11(i, k) * L11(j, k);
            }
            L11(j, i) /= L11(i, i);
        }
    }

    if (row_index + block_size >= n) {
        return;
    }

    auto A21 = [&](const int i, const int j) -> double& {
        const int index = ((row_index + block_size) * n + column_index) + i * n + j;
        return A[index];
    };
    auto L21 = [&](const int i, const int j) -> double& {
        const int index = (row_index + block_size + i) * n + column_index + j;
        return L[index];
    };
    const int A21_rows_count = n - row_index - block_size;
    // A21 = L21 * L11T
    // Find L21
    for (int i = 0; i < A21_rows_count; i++) {
        for (int j = 0; j < block_size; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L21(i, k) * L11(j, k);
            }
            L21(i, j) = (A21(i, j) - sum) / L11(j, j);
        }
    }

    auto A22 = [&](const int i, const int j) -> double& {
        const int index = (row_index + block_size + i) * n + column_index + block_size + j;
        return A[index];
    };
    const int A22_rows_count = A21_rows_count;
    const int A22_columns_count = n - column_index - block_size;

    auto L21T = [&](const int i, const int j) -> double& { return L21(j, i); };

    // A22~ = A22 - L21 * L21T
    // Modify A
    for (int i = 0; i < A22_rows_count; i++) {
        for (int j = 0; j < A22_columns_count; j++) {
            for (int k = 0; k < block_size; k++)
                A22(i, j) -= L21(i, k) * L21T(k, j);
        }
    }

    Cholesky_Decomposition_By_Blocks_Body(A, L, n, row_index + block_size, column_index + block_size,
                                     block_size);
}

// Cholesky Decomposition by blocks
void Cholesky_Decomposition(double* A, double* L, int n) {
    constexpr int block_size = 3;
    memset(L, 0, sizeof(double) * n * n);

    Cholesky_Decomposition_By_Blocks_Body(A, L, n, 0, 0, block_size);
}
