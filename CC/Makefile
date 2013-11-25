## Director of GSL Library. Set the directory of your GSL liblary.
GSL_DIR = /usr/local/lib
## Name of BLAS library. If you use another blas, replace it.
BLAS_LIB = gslblas
## C++ compiler
CC = g++

### For GSL and BLAS by Macports
GSL_DIR = /opt/local
BLAS_LIB = blas

#
CCFLAGS = -O -I${GSL_DIR}/include -L${GSL_DIR}/lib
CCLIBS = -lgsl -l${BLAS_LIB}
#
APP = bfss bfss_mc bfss_c bfss_c_mc bext bcross
all: ${APP}
bfss: main_fss.cc gpr.hpp gpr_bsa.hpp
	${CC} $(CCFLAGS) -o bfss main_fss.cc $(CCLIBS)
bfss_mc: main_fss.cc gpr.hpp gpr_bsa.hpp
	${CC} $(CCFLAGS) -DMAIN_MC -o bfss_mc main_fss.cc $(CCLIBS)
bfss_c: main_fss.cc gpr.hpp gpr_bsa.hpp
	${CC} $(CCFLAGS) -DFSS_C -o bfss_c main_fss.cc $(CCLIBS)
bfss_c_mc: main_fss.cc gpr.hpp gpr_bsa.hpp
	${CC} $(CCFLAGS) -DFSS_C -DMAIN_MC -o bfss_c_mc main_fss.cc $(CCLIBS)
bext: main_extrapolate.cc gpr_extrapolate.hpp gpr.hpp
	${CC} -DGPR_NLOG $(CCFLAGS) -o bext main_extrapolate.cc $(CCLIBS)
bcross: main_cross.cc gpr_extrapolate.hpp gpr.hpp
	${CC} -DGPR_NLOG $(CCFLAGS) -o bcross main_cross.cc $(CCLIBS)
clean:
	@rm -rf ${APP} *.o test.op test_mc.op test.log test_mc.log Etc/Doxygen
test:
	./bfss Sample/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 1 1 1 1 1 1 > test.op 2>test.log
	./bfss_mc Sample/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 1 1 1 1 1 1 > test_mc.op 2>test_mc.log
test_c:
	./bfss_c Sample/sample-c.dat 1 0.28 1 0.9 0 0 1 1 1 2 1 1 1 1 1 1 > test_c.op 2>test_c.log
	./bfss_c_mc Sample/sample-c.dat 1 0.28 1 0.9 0 0 1 1 1 2 1 1 1 1 1 1 > test_c_mc.op 2>test_c_mc.log
doc:
	doxygen Etc/Doxyfile 
code:
	for f in *.cc *.h *.hpp;do uncrustify -c Etc/my.cfg $$f; mv $$f.uncrustify $$f;done
