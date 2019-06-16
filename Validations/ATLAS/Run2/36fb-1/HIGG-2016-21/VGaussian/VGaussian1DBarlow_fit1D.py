from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# Open the 1D grid files

# Choose VBF - ggH - ttH - VH
# f = open('HIGG-2016-21_VBF-1d-Grid.txt', 'r')
# f = open('HIGG-2016-21_ggH-1d-Grid.txt', 'r')
# f = open('HIGG-2016-21_ttH-1d-Grid.txt', 'r')
f = open('HIGG-2016-21_VH-1d-Grid.txt', 'r')

# Plot the grids
fig = plt.figure()
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]
plt.plot(x,y,'.',markersize=3, color = 'blue',label="Observed")

# Find parameters
def func(X, cen2, sig1, sig2):
    z = X[:]
    return (z-cen2)**2/(sig1+sig2*(z-cen2))

def fit5para(xr, yr):

    # VBF
    # guess = (2, 0.6, 0.5)

    # ggH
    # guess = (0.81, 0.2, 0.2)

    # ttH
    # guess = (0.6, 0.2, 0.2)

    # VH
    guess = (0.81, 0.2, 0.2)

    AR, pcov = optimize.curve_fit(func, xr, yr, guess)
    return AR

ff = fit5para(x, y)
cen2,sig1,sig2 = ff[0], ff[1], ff[2]
print("\n Parameters:")
print("cen2, sig1, sig2 =", ff[0], ",", ff[1], ",", ff[2])

# Find central value and uncertainties from those parameters
print("\n Central value and uncertainties:")
def g1(y):
    return y**2 - sig2*y - sig1

a2 = fsolve(g1,1.5)
print("sig2p =", a2)
a2 = fsolve(g1,0)
print("sig2m =", a2)

# Choose VBF - ggH - ttH - VH

# VBF
# x = np.arange(1.2,3.2,0.005)
# ggH
# x = np.arange(0.55,1.1,0.005)
# ttH
# x = np.arange(-0.1,1.5,0.005)
# VH
x = np.arange(-0.3,2,0.005)

y2 = (x-cen2)**2/(sig1+sig2*(x-cen2))



plt.plot(x,y2,'-',markersize=2, color = 'g',label="Barlow's V. Gaussian Appx.")

# Choose VBF - ggH
# plt.xlabel(r'$\mu_{ZZ}^{VBF}$', fontsize=20)
plt.xlabel(r'$\mu_{ZZ}^{ggH}$', fontsize=20)
# plt.xlabel(r'$\mu_{ZZ}^{ttH}$', fontsize=20)
# plt.xlabel(r'$\mu_{ZZ}^{VH}$', fontsize=20)

plt.ylabel(r'-2 Loglikelihood', fontsize=20)
plt.title("$\mu$ from ATLAS-HIGG-2016-21 (Poisson)")
plt.legend(loc='upper right', fontsize=12)

fig.set_tight_layout(True)

# Choose VBF - ggH - ttH - VH

# fig.savefig('mu_VBF_1D_VGaussian.pdf')
# fig.savefig('mu_ggH_1D_VGaussian.pdf')
# fig.savefig('mu_ttH_1D_Poisson.pdf')
# fig.savefig('mu_VH_1D_Poisson.pdf')
plt.show()