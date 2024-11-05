SCALAPACK_LIB = /home/henax/scalapack/scalapack-2.0.2

C_FLAGS = -L${SCALAPACK_LIB} -lscalapack -lgfortran -llapack -lblas
CC = mpicxx

all: program run

program:
	${CC} program.cpp ${C_FLAGS} -o ./program.out

run:
	mpirun -n 4 ./program.out

clean:
	rm -rf *.out *.o
