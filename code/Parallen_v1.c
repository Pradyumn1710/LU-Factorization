#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 50  // Minimum matrix size (can be changed to larger)

double **allocate_matrix(int n) {
    double **A = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        A[i] = (double *)malloc(n * sizeof(double));
    }
    return A;
}

void free_matrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
}

// Print the matrix (for debugging)
void print_matrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.3f ", A[i][j]);
        }
        printf("\n");
    }
}

// OpenMP Parallel LU Decomposition with Partial Pivoting
void lu_decomposition(double **A, int n, int *P) {
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    for (int i = 0; i < n; i++) {
        // Pivoting (Find the row with the max absolute value in column i)
        int pivot = i;
        for (int j = i + 1; j < n; j++) {
            if (fabs(A[j][i]) > fabs(A[pivot][i])) {
                pivot = j;
            }
        }

        // Swap rows in A and P
        if (pivot != i) {
            double *temp = A[i];
            A[i] = A[pivot];
            A[pivot] = temp;

            int tmp = P[i];
            P[i] = P[pivot];
            P[pivot] = tmp;
        }

        // LU Factorization using OpenMP
        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            A[j][i] /= A[i][i]; // Compute L
            for (int k = i + 1; k < n; k++) {
                A[j][k] -= A[j][i] * A[i][k]; // Compute U
            }
        }
    }
}

int main() {
    int n = N;  // Set matrix size
    double **A = allocate_matrix(n);
    int *P = (int *)malloc(n * sizeof(int));

    // Fill matrix with random values
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 100 + 1;  // Random values from 1 to 100
        }
    }

    printf("Original Matrix:\n");
    print_matrix(A, n);

    double start = omp_get_wtime();  // Start time
    lu_decomposition(A, n, P);
    double end = omp_get_wtime();  // End time

    printf("\nLU Decomposed Matrix:\n");
    print_matrix(A, n);
    printf("\nTime Taken: %f seconds\n", end - start);

    free_matrix(A, n);
    free(P);
    return 0;
}
