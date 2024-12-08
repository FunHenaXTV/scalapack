#pragma once

#ifndef SCALAPACK_H
#define SCALAPACK_H

#include <complex>
typedef std::complex<float> complex_s;
typedef std::complex<double> complex_d;

extern "C" {
	void Cblacs_pinfo(int* mypnum, int* nprocs);
	void Cblacs_get(int context, int request, int* value);
	int  Cblacs_gridinit(int* context, char * order, int np_row, int np_col);
	void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
	void Cblacs_gridexit(int context);
	void Cblacs_exit(int error_code);
	void Cblacs_gridmap(int* context, int* map, int ld_usermap, int np_row, int np_col);

	void Cblacs_pcoord(int, int, int *, int *);
	void Cdgerv2d(int, int, int, double *, int, int, int);
	void Cdgesd2d(int, int, int, double *, int, int, int);
	void Czgerv2d(int, int, int, complex_d *, int, int, int);
	void Czgesd2d(int, int, int, complex_d *, int, int, int);

	int  indxl2g_(int* indxloc, int* nb, int* iproc, int* isrcproc, int* nprocs);

	int  npreroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
	int  numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

	void descinit_(int *idescal, int *m, int *n, int *mb, int *nb, int *dummy1 , int *dummy2 , int *icon, int *procRows, int *info);

	void psgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *ia, int *ja, int *desca, float *s, float *u, int *iu, int *ju, int *descu, float *vt, int *ivt, int *jvt, int *descvt, float *work, int *lwork, int *info);
	void pdgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *ia, int *ja, int *desca, double *s, double *u, int *iu, int *ju, int *descu, double *vt, int *ivt, int *jvt, int *descvt, double *work, int *lwork, int *info);
	void pcgesvd_(char *jobu, char *jobvt, int *m, int *n, complex_s *a, int *ia, int *ja, int *desca, float *s, complex_s *u, int *iu, int *ju, int *descu, complex_s *vt, int *ivt, int *jvt, int *descvt, complex_s *work, int *lwork, float *rwork, int *info);
	void pzgesvd_(char *jobu, char *jobvt, int *m, int *n, complex_d *a, int *ia, int *ja, int *desca, double *s, complex_d *u, int *iu, int *ju, int *descu, complex_d *vt, int *ivt, int *jvt, int *descvt, complex_d *work, int *lwork, double *rwork, int *info);

    void pdgemm_(char*, char*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
	void pzgemm_(char*, char*, int*, int*, int*, double*, complex_d*, int*, int*, int*, complex_d*, int*, int*, int*, double*, complex_d*, int*, int*, int*);

	void pzheev_(const char *jobz, const char *uplo, const int *n, complex_d *a, const int *ia, const int *ja, const int *desca, double *w, complex_d *z, const int *iz, const int *jz, const int *descz, complex_d *work, const int *lwork, double *rwork, const int *lrwork, int *info);
}

#endif

#ifndef TASK1LIB_H
#define TASK1LIB_H

int GetDensityMatricesDiagonalsSequence(int argc, char** argv);

#endif
