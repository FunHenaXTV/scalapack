SCALAPACK_LIB = /home/henax/scalapack/scalapack-2.0.2

CC = mpicxx
C_FLAGS = -L../task1 -ltask1lib -L${SCALAPACK_LIB} -lscalapack -lgfortran -llapack -lblas

all: build_lib program run_all

build_lib:
	make -f ../task1/Makefile

program:
	${CC} program.cpp ${C_FLAGS} -o ./program.out

run_all:
	mpirun -np 4 ./program.out 4 4x4.txt 0.1 4x4 5
	mpirun -np 4 ./program.out 4 4x1.txt 0.1 4x4 5
	mpirun -np 4 ./program.out 4 1x1.txt 0.1 4x4 5

run_4x4:
	mpirun -np 4 ./program.out 4 4x4.txt 0.1 4x4 5

run_4x1:
	mpirun -np 4 ./program.out 4 4x1.txt 0.1 4x4 5

run_1x1:
	mpirun -np 4 ./program.out 4 1x1.txt 0.1 4x4 5

clean:
	rm -rf *.out *.o *.so
