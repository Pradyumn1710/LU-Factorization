#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 50  // Define the matrix size as 50

// ---------------------------
// Helper Functions
// ---------------------------

// Allocate a square matrix of size n x n.
double **allocate_matrix(int n)
{
    double **A = (double **)malloc(n * sizeof(double *));
    if (!A)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    for (int i = 0; i < n; i++)
    {
        A[i] = (double *)malloc(n * sizeof(double));
        if (!A[i])
        {
            printf("Memory allocation failed at row %d!\n", i);
            exit(1);
        }
    }
    return A;
}

// Free the allocated matrix.
void free_matrix(double **A, int n)
{
    for (int i = 0; i < n; i++)
    {
        free(A[i]);
    }
    free(A);
}

// Print the matrix.
void print_matrix(double **A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%8.3f ", A[i][j]);
        }
        printf("\n");
    }
}

// Copy one matrix to another.
void copy_matrix(double **dest, double **src, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dest[i][j] = src[i][j];
        }
    }
}

// ---------------------------
// Sequential LU Decomposition with Partial Pivoting
// ---------------------------
void lu_decomposition_seq(double **A, int n, int *P)
{
    for (int i = 0; i < n; i++)
    {
        P[i] = i;
    }

    for (int i = 0; i < n; i++)
    {
        int pivot = i;
        double maxVal = fabs(A[i][i]);

        for (int j = i + 1; j < n; j++)
        {
            if (fabs(A[j][i]) > maxVal)
            {
                maxVal = fabs(A[j][i]);
                pivot = j;
            }
        }

        if (pivot != i)
        {
            double *temp = A[i];
            A[i] = A[pivot];
            A[pivot] = temp;

            int tmp = P[i];
            P[i] = P[pivot];
            P[pivot] = tmp;
        }

        for (int j = i + 1; j < n; j++)
        {
            double multiplier = A[j][i] / A[i][i];
            A[j][i] = multiplier;

            for (int k = i + 1; k < n; k++)
            {
                A[j][k] -= multiplier * A[i][k];
            }
        }
    }
}

// ---------------------------
// OpenMP Parallel LU Decomposition with Partial Pivoting
// ---------------------------
void lu_decomposition_omp(double **A, int n, int *P)
{
    for (int i = 0; i < n; i++)
    {
        P[i] = i;
    }

    for (int i = 0; i < n; i++)
    {
        int pivot = i;
        double maxVal = fabs(A[i][i]);

        for (int j = i + 1; j < n; j++)
        {
            if (fabs(A[j][i]) > maxVal)
            {
                maxVal = fabs(A[j][i]);
                pivot = j;
            }
        }

        if (pivot != i)
        {
            double *temp = A[i];
            A[i] = A[pivot];
            A[pivot] = temp;

            int tmp = P[i];
            P[i] = P[pivot];
            P[pivot] = tmp;
        }

// Parallelizing row updates
#pragma omp parallel for
        for (int j = i + 1; j < n; j++)
        {
            double multiplier = A[j][i] / A[i][i];
            A[j][i] = multiplier;

            for (int k = i + 1; k < n; k++)
            {
                A[j][k] -= multiplier * A[i][k];
            }
        }
    }
}

// ---------------------------
// Main Function: Test the LU Decomposition
// ---------------------------
int main()
{
    int n = N;  // Now correctly setting the size to 50

    // Allocate matrices
    double **A = allocate_matrix(n);
    double **A_parallel = allocate_matrix(n);

    // Fill the matrix
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = (i + 1) * (j + 1);
        }
    }

    // Copy the matrix for OpenMP version
    copy_matrix(A_parallel, A, n);

    // Permutation vectors
    int *P_seq = (int *)malloc(n * sizeof(int));
    int *P_omp = (int *)malloc(n * sizeof(int));

    // Sequential LU Decomposition
    printf("Sequential LU Decomposition:\n");
    lu_decomposition_seq(A, n, P_seq);
    print_matrix(A, n);

    // OpenMP Parallel LU Decomposition
    printf("\nOpenMP Parallel LU Decomposition:\n");
    // lu_decomposition_omp(A_parallel, n, P_omp);
    // print_matrix(A_parallel, n);

    // Free memory
    free_matrix(A, n);
    free_matrix(A_parallel, n);
    free(P_seq);
    free(P_omp);

    return 0;
}
