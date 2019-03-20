from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# Open the 1D grid files

# Choose VBF - ggH
f = open('HIGG-2016-21_VBF-1d-Grid.txt', 'r')
# f = open('HIGG-2016-21_ggH-1d-Grid.txt', 'r')

fig = plt.figure()
# Plot the grids
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]
plt.plot(x,y,'.',markersize=3, color = 'blue',label="Observed")

# test
def func(X, alpha, cen2, L):
    z = X[:]
    return -2*(-alpha*(z - cen2) + L*np.log(np.abs(1 + alpha*(z - cen2)/L)))

def fit5para(xr, yr):
    guess = (1, 1, 1)
    AR, pcov = optimize.curve_fit(func, xr, yr, guess)
    return AR

ff = fit5para(x, y)
print(ff[0])
alpha, cen2, nu = ff[0], ff[1], ff[2]
gam = alpha/nu

def g1(sig2p):
    return gam*sig2p - np.log(1 + gam*sig2p) - 1/(2*nu)

A = fsolve(g1,1)
sig2p = A
print(A)

def g2(sig2m):
    return np.exp(-gam*(sig2m + sig2p)) - (1 - gam*sig2m)/(1 + gam*sig2p)
A = fsolve(g2,1)
print(A)


# Input data

# Data from Fig. 12
# ggF
cen1, sig1m, sig1p = 0.81, 0.18, 0.19
# VBF
# cen2, sig2m, sig2p = 2, 0.5, 0.6

# Slightly Fixed the data from Fig. 12
# VBF
cen2, sig2m, sig2p = 2.03, 0.52, 0.61


def f(t):
    # Choose VBF - ggH
    return np.exp(-t*(sig2m + sig2p)) - (1 - t*sig2m)/(1 + t*sig2p)
    # return np.exp(-t*(sig1m + sig1p)) - (1 - t*sig1m)/(1 + t*sig1p)

A = fsolve(f,1.001)
print(A)

gam = A

# Choose VBF - ggH
L = 1/(2*(gam*sig2p - np.log(1 + gam*sig2p)))
# L = 1/(2*(gam*sig1p - np.log(1 + gam*sig1p)))
alpha = L*gam

# Choose VBF - ggH
x = np.arange(1.3,3.1,0.005)
alpha, cen2, L = ff[0], ff[1], ff[2]
y2 = -2*(-alpha*(x - cen2) + L*np.log(1 + alpha*(x - cen2)/L))
# x = np.arange(0.57,1.08,0.005)
# y2 = -2*(-alpha*(x - cen1) + L*np.log(1 + alpha*(x - cen1)/L))


plt.plot(x,y2,'-',markersize=2, color = 'g',label="Barlow's Poisson Appx.")

# Choose VBF - ggH
plt.xlabel(r'$\mu_{ZZ}^{VBF}$', fontsize=20)
# plt.xlabel(r'$\mu_{ZZ}^{ggH}$', fontsize=20)


plt.ylabel(r'-2 Loglikelihood', fontsize=20)
plt.title("$\mu$ from ATLAS-HIGG-2016-22 (Poisson)")
plt.legend(loc='upper right', fontsize=12)

fig.set_tight_layout(True)

# Choose VBF - ggH
# fig.savefig('mu_VBF_1D_Poisson.pdf')
# fig.savefig('mu_VBF_1D_Poisson-fixed.pdf')
# fig.savefig('mu_ggF_1D_Poisson.pdf')
fig.savefig('mu_VBF_1D_Poisson-fixed-test.pdf')

print(L)