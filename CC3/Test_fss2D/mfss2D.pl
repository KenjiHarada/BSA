#!/usr/bin/perl
use Math::Trig 'pi';
sub gauss {
    my $func = sqrt(-2.0*log(1-rand()))*cos(2*pi*rand());
    return $func;
}
my $N=30;
my $dz=2e-2;
my $c1 = 1;
my $c2 = 1.5;
my $XC = 1e-2;
my $YC = 3e-3;
my $c0 = 0;
foreach my $L (64, 128, 256){
    for(my $i=0;$i < $N; ++$i){
	my $x = (rand()-0.5);
	my $y = (rand()-0.5)*2;
	my $r2 = $x*$x+$y*$y;
	my $z = exp(-$r2/2);
	my $X = $x*($L**(-$c1)) + $XC;
	my $Y = $y*($L**(-$c2)) + $YC;
	my $dz0 = $dz * (1 + gauss()*1e-1);
	my $Z = ($z + gauss()*$dz0) * ($L**$c0);
	my $DZ = $dz0 * ($L**$c0);
	printf("%g %g %g %g %g\n", $L, $X, $Y, $Z, $DZ);
    }
}
