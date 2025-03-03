#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_MATRICES 100    // Total number of LU factorizations
#define MATRIX_SIZE 10      // Matrix dimension
#define NUM_THREADS 16       // Number of threads (adjust as needed)

double **allocate_matrix();
void free_matrix(double **A);
void copy_matrix(double **dest, double **src);
void lu_decomposition(double **A);
void process_matrix(double **A, double *b, double *x);
void read_matrix_from_file(const char *filename, double **A, double *b, int n);
void forwardSubstitution(int n, double **L, double *b, double *y);
void backwardSubstitution(int n, double **U, double *y, double *x);

int main() {
    int n = MATRIX_SIZE;
    double **A = allocate_matrix();
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    read_matrix_from_file("system.txt", A, b, n);
    omp_set_num_threads(NUM_THREADS);
    double thread_times[NUM_THREADS] = {0};
    double total_start = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double thread_start = omp_get_wtime();

        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_MATRICES; i++) {
            process_matrix(A, b, x);
        }
        thread_times[tid] = omp_get_wtime() - thread_start;
    }

    double total_time = omp_get_wtime() - total_start;
    printf("\nTotal execution time: %.4f seconds\n", total_time);
    for (int t = 0; t < NUM_THREADS; t++) {
        printf("Thread %d time: %.4f seconds\n", t, thread_times[t]);
    }

    free_matrix(A);
    free(b);
    free(x);
    return 0;
}

void process_matrix(double **A, double *b, double *x) {
    double **copy = allocate_matrix();
    copy_matrix(copy, A);
    lu_decomposition(copy);
    
    double y[MATRIX_SIZE];
    forwardSubstitution(MATRIX_SIZE, copy, b, y);
    backwardSubstitution(MATRIX_SIZE, copy, y, x);
    
    free_matrix(copy);
}

void lu_decomposition(double **A) {
    for (int k = 0; k < MATRIX_SIZE; k++) {
        for (int i = k+1; i < MATRIX_SIZE; i++) {
            A[i][k] /= A[k][k];
            for (int j = k+1; j < MATRIX_SIZE; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

void forwardSubstitution(int n, double **L, double *b, double *y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
        y[i] /= L[i][i];
    }
}

void backwardSubstitution(int n, double **U, double *y, double *x) {
    #pragma omp parallel for
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
}

double **allocate_matrix() {
    double **A = (double **)malloc(MATRIX_SIZE * sizeof(double *));
    for (int i = 0; i < MATRIX_SIZE; i++) {
        A[i] = (double *)malloc(MATRIX_SIZE * sizeof(double));
    }
    return A;
}

void free_matrix(double **A) {
    for (int i = 0; i < MATRIX_SIZE; i++) free(A[i]);
    free(A);
}

void copy_matrix(double **dest, double **src) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

void read_matrix_from_file(const char *filename, double **A, double *b, int n) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    char buffer[256];
    fgets(buffer, sizeof(buffer), file);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fscanf(file, "%lf", &A[i][j]);
        }
    }
    for (int i = 0; i < n; i++) {
        fscanf(file, "%lf", &b[i]);
    }
    fclose(file);
}
