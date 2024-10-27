SCALAPACK_LIB = /home/henax/scalapack/scalapack-2.0.2

C_FLAGS = -L${SCALAPACK_LIB} -lscalapack -lgfortran # -L${LAPACK_LIB} -llapack -L${BLAS_LIB} -lblas
CC = mpicc

all: program run

program:
	${CC} program.c ${C_FLAGS} -o ./program.out

run:
	mpirun -n 4 ./program.out

clean:
	rm -rf *.out *.o
