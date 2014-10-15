## C++ compiler
CC = g++
#CC = clang++

### BLAS and LAPACK
BLAS_LIB = blas
LAPACK_LIB = lapack


#CCFLAGS = -O -std=c++11 -Wall
CCFLAGS = -O -Wall
CCLIBS = -l${BLAS_LIB} -l${LAPACK_LIB}

#
APP = new_bfss
HEAD = gpr.hpp bsa.hpp optimize.hpp
SRC = ${HEAD} main_fss.cc
all: ${APP}
new_bfss: main_fss.cc ${HEAD} Makefile
	${CC} $(CCFLAGS) -o new_bfss main_fss.cc $(CCLIBS)
clean:
	@rm -rf ${APP}
doc:
	doxygen Etc/Doxyfile 
	open Etc/Doxygen/html/index.html
test:
	./new_bfss Sample/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 1 1 1 1 1 1 > test.op 2>test.log
	./new_bfss -c Sample/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 1 1 1 1 1 1 > test_mc.op 2>test_mc.log
