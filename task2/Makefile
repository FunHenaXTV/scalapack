SCALAPACK_LIB = /home/nastyapion/prac423/scalapack-2.0.2

C_FLAGS = -L${SCALAPACK_LIB} -lscalapack -lgfortran -llapack -lblas
CC = mpicxx

all: program run

program:
	${CC} task2.cpp ${C_FLAGS} -o ./program.out

run:
	mpirun -n 4 ./program.out 4 a.txt w.txt 0 2 2

clean:
	rm -rf *.out *.o