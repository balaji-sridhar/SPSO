from Act_Pyswarm import pso
from Act_Pyswarm import var_pso
import numpy as np
import math
import csv
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum


def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

def con(x):
    x1 = x[0]
    x2 = x[1]
    return [-(x1 + 0.25)**2 + 0.75*x2]



michalewicz_m = .5
def michalewicz( x ):  # mich.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( sin(x) * sin( j * x**2 / pi ) ** (2 * michalewicz_m) )

def rastrigin(position):
    err = 0.0
    for i in range(len(position)):
        xi = position[i]
        err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return err

def griewank( x, fr=4000 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s = sum( x**2 )
    p = prod( cos( x / sqrt(j) ))
    return s/fr - p + 1

#xopt, fopt = pso(banana, lb, ub, f_ieqcons=con, debug=True)
output_file_name = "PSO_Comparison.csv"
output_file = open(output_file_name, 'w+', newline='')
writer = csv.writer(output_file)
writer.writerow(['Segmented_Min', 'NonSegmented_Min', 'Segmented_Win', 'NonSegmented_Win', 'Delta'])

def schwefel( x ):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829*n - sum( x * sin( sqrt( abs( x ))))


dim = 100
upper_bound = 5.12
lower_bound = - upper_bound

lb = np.ones(dim) * lower_bound
ub = np.ones(dim) * upper_bound


for i in range(10):
    #print(pso(rastrigin, lb, ub, maxiter=500, debug=False, omega=0.729, phip=1.49445, phig=1.49445, swarmsize=500))
    output = var_pso(rastrigin, lb, ub, maxiter=600, debug=False, omega=0.729, phip=1.49445, phig=1.49445, swarmsize=1000,segments = 5)
    print(output)
    writer.writerow(output)
    # Optimum should be around x=[0.5, 0.76] with banana(x)=4.5 and con(x)=0

output_file.close()