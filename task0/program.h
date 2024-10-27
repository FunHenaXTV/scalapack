extern void Cblacs_pinfo(int* mypnum, int* nprocs);
extern void Cblacs_get(int context, int request, int* value);
extern int Cblacs_gridinit(int* context, char* order, int np_row, int np_col);
extern void Cblacs_gridinfo(int context, int* np_row, int* np_col, int* my_row, int* my_col);
extern void Cblacs_gridexit(int context);
extern void Cblacs_exit(int error_code);
extern void Cblacs_barrier(int context, char* scope);

int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

void descinit_(int * desca, int * m, int * n, int * mb, int * nb,
int * irsrc, int * icsrc, int * context,int * llda, int * info);
void pdgesv_(int * n, int * nrhs, double * A, int * ia, int * ja,
    int * desca, int * ipiv, double * b, int * ib, int * jb, int * descb,
    int * info);
void pdelset_(double * A, int * i, int * j, int * desca, double * alpha);
void pdlaprnt_(int * m, int * n, double * A, int * ia, int * ja,
    int * desca, int * irprnt, int * icprn, char * cmatnm, int * nout,
    double * work);
