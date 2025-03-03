#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 10  // Change this to your required matrix size

void read_matrix_from_file(const char *filename, double **A, double *b, int n) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Skip the first line (comment)
    char buffer[256];
    fgets(buffer, sizeof(buffer), file);

    // Read matrix A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fscanf(file, "%lf", &A[i][j]);
        }
    }

    // Read vector b
    for (int i = 0; i < n; i++) {
        fscanf(file, "%lf", &b[i]);
    }

    fclose(file);
}

// Allocate memory for an n x n matrix
double **allocate_matrix(int n) {
    double **A = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        A[i] = (double *)malloc(n * sizeof(double));
    }
    return A;
}

// Free allocated memory for matrix A
void free_matrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
}

// Print matrix for debugging
void print_matrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.3f ", A[i][j]);
        }
        printf("\n");
    }
}

// LU Decomposition with Partial Pivoting (No explicit singularity check)
void lu_decomposition(double **A, int n, int *P) {
    for (int i = 0; i < n; i++) {
        P[i] = i;  // Initialize permutation vector
    }

    for (int i = 0; i < n; i++) {
        // **Pivoting: Find the row with the maximum absolute value in column i**
        int pivot = i;
        for (int j = i + 1; j < n; j++) {
            if (fabs(A[j][i]) > fabs(A[pivot][i])) {
                pivot = j;
            }
        }

        // **Swap rows in A and update permutation array P**
        if (pivot != i) {
            double *temp = A[i];
            A[i] = A[pivot];
            A[pivot] = temp;

            int temp_idx = P[i];
            P[i] = P[pivot];
            P[pivot] = temp_idx;
        }

        // **LU Factorization using OpenMP for parallel computation**
        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            A[j][i] /= A[i][i];  // Compute L (lower triangular matrix)
            for (int k = i + 1; k < n; k++) {
                A[j][k] -= A[j][i] * A[i][k];  // Compute U (upper triangular matrix)
            }
        }
    }
}

int main() {
    int n = N;  // Set matrix size
    double **A = allocate_matrix(n);
    double *b = (double *)malloc(n * sizeof(double));
    int *P = (int *)malloc(n * sizeof(int));

    // Read matrix A and vector b from file
    read_matrix_from_file("system.txt", A, b, n);

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
