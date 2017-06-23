set term x11
set title "Original data (Ising-square-Binder.dat)"
plot for [L in "64 128 256"] "Sample/Ising-square-Binder.dat" i 0 u ($1 == L ? $2:1/0):3:4 w e t L
pause -1
set title "FSS result (Ising-square-Binder.dat)"
plot for [L in "64 128 256"] "test1.op" i 0 u ($4 == L ? $1:1/0):2:3 w e t L, "test1.op" i 1 u 1:2 w d t "scaling func."
pause -1
set title "FSS result with MC (Ising-square-Binder.dat)"
plot for [L in "64 128 256"] "test1_mc.op" i 0 u ($4 == L ? $1:1/0):2:3 w e t L, "test1_mc.op" i 1 u 1:2 w d t "scaling func."
pause -1
set title "Original data (sample.dat)"
splot for [L in "64 128 256"] "Test_fss2D/sample.dat" i 0 u ($1 == L ? $2:1/0):3:4 t L
pause -1
set title "FSS result (sample.dat)"
splot for [L in "64 128 256"] "test2.op" i 0 u ($5 == L ? $1:1/0):2:3 t L, "test2.op" i 1 u 1:2:3 w d t "scaling func."
pause -1
set title "FSS result with MC (sample.dat)"
splot for [L in "64 128 256"] "test2_mc.op" i 0 u ($5 == L ? $1:1/0):2:3 t L, "test2_mc.op" i 1 u 1:2:3 w d t "scaling func."
pause -1
