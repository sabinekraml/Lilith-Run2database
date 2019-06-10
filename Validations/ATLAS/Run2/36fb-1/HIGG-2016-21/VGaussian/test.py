import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.optimize as optimize
from scipy.special import gamma
from scipy.special import factorial

def g(z):
    labda = 4.5
    return -2*((z-4)*sp.log(labda)-sp.log(gamma(z+1))+sp.log(factorial(4)))-1
    # return (labda**(z-4)*factorial(4))+2*(gamma(z+1))


sigp = fsolve(g,8) - 4
sigm = 4 - fsolve(g,1)
print(4-sigm, sigp+4)
print(sigm, sigp)

def f(t):
    return np.exp(-t*(sigm + sigp)) - (1 - t*sigm)/(1 + t*sigp)

A = fsolve(f,0.2)
print(A)
gam = A
nu = 1/(2*(gam*sigp - np.log(1 + gam*sigp)))
rho = nu*gam
print("%.30f, %.30f" % (rho,nu))