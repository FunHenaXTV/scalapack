SCALAPACK_LIB = /home/nastyapion/prac423/scalapack-2.0.2

C_FLAGS = -L${SCALAPACK_LIB} -lscalapack -lgfortran -llapack -lblas
CC = mpicxx

all: program run

program:
	${CC} program.cpp ${C_FLAGS} -o ./program.out

run:
	 mpirun -n 4 ./program.out 4 4_4.txt 0.1 4_4 5

clean:
	rm -rf *.out *.o
