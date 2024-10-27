#include <stdlib.h>
#include <stdio.h>

#include "mpi.h"
#include "program.h"

#define max(a, b) a > b ? a : b

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int nprocs = 0, myrank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int dims[2];
    MPI_Dims_create(nprocs, 2, dims);
    int nprow = dims[0]; // cartesian direction 0
    int npcol = dims[1]; // cartesian direction 1
    // Get a default BLACS context

    int context = 0;
    Cblacs_get(-1, 0, &context);
    // Initialize the BLACS context
    char order = 'R';
    Cblacs_gridinit(&context, &order, nprow, npcol);
    int myrow = 0, mycol = 0;
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);
    printf("myrow: %d, mycol: %d\n", myrow, mycol);
    // Computation of local matrix size
    int zero = 0;
    int m = 100, n = 100; // Размеры "глобальной матрицы"
    int mb = 1, nb = 1; //
    int mloc = max(1, numroc_(&m, &mb, &myrow, &zero, &nprow));
    int nloc = max(1, numroc_(&n, &nb, &mycol, &zero, &npcol));
    printf("mloc: %d, nloc: %d\n", mloc, nloc);

    double* A = malloc(mloc * nloc * sizeof(*A));
    // // Descriptor
    int info = 0;
    int* descA = malloc(9 * sizeof(*descA)); // Specified as: an array of (at least) length 9, containing fullword integers.

    descinit_(descA, &m, &n, &mb, &nb, &zero, &zero, &context, &mloc, &info); // mloc -> LLDA
    //
    // Some operations on matrix A
    //
    free(A);
    // Close BLACS environment
    Cblacs_gridexit(context);
    Cblacs_exit(0);
    // MPI_Finalize();
    return 0;
}
