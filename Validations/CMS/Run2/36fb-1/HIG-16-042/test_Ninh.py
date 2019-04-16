# gamma = solve_bifurcation_f_gamma(sigm,sigp,1000) # number of bifurcation attemps e.g. N = 1000
import numpy as np
def solve_bifurcation_f_gamma(m, p, N):
    a = 0.
    b = 1.0/m
    for i in range(N):
        x = (a+b)/2
        if np.exp(-x*(m+p)) <= (1-m*x)/(1+p*x):
            a = x
        else:
            b = x
    if (a == 0):
        raise LikelihoodComputationError(
                'Cannot find a non-trivial root for Poisson approximation')
    return a
gamma=solve_bifurcation_f_gamma(0.65,8,1000)
print(gamma)