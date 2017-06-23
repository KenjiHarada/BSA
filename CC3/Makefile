## GNU compiler with blas and lapack
CC = g++
CCFLAGS = -Wall -O
CCLIBS = -lblas -llapack

## GNU compiler with Intel MKL
# CC = g++
# CCFLAGS = -O -Wall
# CCLIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

## Intel compiler with mkl
# CC = icpc
# CCFLAGS = -O1
# CCLIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

## Intel compiler with sequential mkl
# CC = icpc
# CCFLAGS = -O1 -Wall -mkl=sequential

## Intel compiler with parallel mkl
# CC = icpc
# CCFLAGS = -O1 -Wall -mkl=parallel

APP = new_bfss
HEAD = gpr.hpp bsa.hpp optimize.hpp
SRC = ${HEAD} main_fss.cc
all: ${APP}
new_bfss: main_fss.cc ${HEAD} Makefile
	${CC} $(CCFLAGS) -o new_bfss main_fss.cc $(CCLIBS)
clean:
	@rm -rf ${APP}
	@rm -rf FMT
code:
	rm -rf FMT
	mkdir FMT
	for f in ${SRC};do echo $$f; clang-format -style="{Standard: Cpp03}" $$f > FMT/$$f;done
	cp Makefile FMT

test: new_bfss
	./new_bfss Sample/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 1 1 1 1 1 1 > test1.op 2>test1.log
	./new_bfss -d 2 Test_fss2D/sample.dat 1 1e-2 1 1 1 3e-3 1 1.5 0 0 >test2.op 2>test2.log
	./new_bfss -c Sample/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 1 1 1 1 1 1 > test1_mc.op 2>test1_mc.log
	./new_bfss -c -d 2 Test_fss2D/sample.dat 1 1e-2 1 1 1 3e-3 1 1.5 0 0 >test2_mc.op 2>test2_mc.log

plot:
	gnuplot plot.gp
