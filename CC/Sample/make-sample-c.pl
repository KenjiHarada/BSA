#!/usr/bin/perl
#
# Data: A(t,L) = (8 + X + X^2 / 16) (1+2/L)
#            X = (t - 0.3) L
#       t = [0.2:0.4]
#       L = 16, 32, 64
#
# Answer: T_c = 0.3, c_1 = 1, c_2 = 0, a = 2, w = 1
#
print "# Sample Data of FSS with correction to scaling\n";
print "#\n";
print "# Comment lines are started by the character '#'\n";
print "#\n";
print "# Format of data file of \"main_fss.cc\" as follows:\n";
print "#\n";
print "# Answer: T_c = 0.3, c_1 = 1, c_2 = 0, a = 2, w = 1\n";
print "#\n";
print "# L T A(T,L) Error\n";
foreach $l (16, 32, 64){
    for($x0 = -3; $x0 <= 3; $x0 += 0.2){
        $x = $x0 + rand(0.2)-0.125;
        $t = $x / $l + 0.3;
        $a = (8 + $x + $x**2/16.0) * (1+2/$l) + rand(0.2)-0.1;
        $e = 0.1;
        printf("%g %g %g %g\n", $l, $t, $a, $e);
    }
}
