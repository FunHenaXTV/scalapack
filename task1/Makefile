SCALAPACK_LIB = /home/henax/scalapack/scalapack-2.0.2

CURRENT_DIR := /home/henax/scalapack/task1/

C_FLAGS = -L${SCALAPACK_LIB} -lscalapack -lgfortran -llapack -lblas
CC = mpicxx
LIB_NAME = $(CURRENT_DIR)libtask1lib.a
SOURCES = $(CURRENT_DIR)task1lib.cpp
OBJECTS = $(CURRENT_DIR)task1lib.o
AR = ar
ARFLAGS = rcs

all: $(LIB_NAME)

$(LIB_NAME): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB_NAME) $(OBJECTS)

$(CURRENT_DIR)task1lib.o: $(CURRENT_DIR)task1lib.cpp
	$(CC) $(CFLAGS) -c $(CURRENT_DIR)task1lib.cpp -o $(CURRENT_DIR)task1lib.o

clean:
	rm -rf *.out *.o *.a ../task1/*.o ../task1/*.a
