* Scaling form
  A(L, x_1, x_2, ...) = L^c_0 f[ (x_1-g_1)L^c_1, ... ]

* Parameter list
  (g_1, c_1, ..., g_n, c_n, c_0)

* Data format
  An input data file consists of data lines of which format is as follows:

  [L] [x_1] ... [x_n] [A] [dA]

  Here, [dA] is the error of [A].

* Usage

./new_bfss -d [n] [data file] [mask_1] [g_1] [mask_2] [c_1] ... [mask_{2n-1}] [g_n] [mask_{2n}] [c_n] [mask_{2n+1}] [c_0]

[n]: the number of arguments of scaling function "f".
[data file]: the name of data file.
[mask_i]: if the value is 0, we fix the value of the fallowing parameter.
[g_i], [c_i]: the initial value of a parameter.
