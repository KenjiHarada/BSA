cx(x,L)=(x-tc)*L**inu
cy(y,L)=y/(1+a/L**w)
# tc=0.285766
# inu=1.00341
# a=5.08
# w=0.165293

tc=0.3
inu=1
a=2
w=1
plot "test_c.op" i 0 u 1:2:3 t "Bayesian FSS" w e, "test_c.op" i 0 u (cx($5, $4)):(cy($6, $4)):(cy($7, $4)) t "Exact FSS" w e
