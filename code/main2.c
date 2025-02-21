/*
    Author - Pradyumn shirsath
    compile- gcc main2.c -o main2
    run- ./main2
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 50  // Define the matrix size as 50

// Allocate a square matrix of size n x n.
double **allocate_matrix(int n)
{
    double **A = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
    {
        A[i] = (double *)malloc(n * sizeof(double));
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

// Simple LU Decomposition with Partial Pivoting
void lu_decomposition(double **A, int n, int *P)
{
    for (int i = 0; i < n; i++)
    {
        P[i] = i;
    }

    for (int i = 0; i < n; i++)
    {
        int pivot = i;
        for (int j = i + 1; j < n; j++)
        {
            if (fabs(A[j][i]) > fabs(A[pivot][i]))
            {
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
            A[j][i] /= A[i][i];
            for (int k = i + 1; k < n; k++)
            {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }
}

int main()
{
    int n = N;
    double **A = allocate_matrix(n);
    int *P = (int *)malloc(n * sizeof(int));

    // Fill the matrix with sample values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (rand() % 100) + 1;  // Random numbers from 1 to 100
        }
    }

    printf("Original Matrix:\n");
    print_matrix(A, n);

    lu_decomposition(A, n, P);

    printf("\nLU Decomposed Matrix:\n");
    print_matrix(A, n);

    free_matrix(A, n);
    free(P);
    return 0;
}
