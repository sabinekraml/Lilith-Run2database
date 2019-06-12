from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import matplotlib
import sys

# Read Grids
f = np.genfromtxt('HIGG_2016-07_Llh-2d-Grid95.txt')
x = f[:,0]
y = f[:,1]

# Find parameters
def func(X, p):
    z1, z2 = X[:,0], X[:,1]
    # z10, z20 = 11.4011976,	0.490450205
    # z10, z20 = 1.096269, 0.60549408
    z10, z20 = 1.10, 0.62
    sig1p, sig1m, sig2p, sig2m = 0.21, 0.20, 0.36, 0.35
    V1 = sig1p * sig1m
    V1e = sig1p - sig1m
    V2 = sig2p * sig2m
    V2e = sig2p - sig2m
    V1f = V1 + V1e * (z1 - z10)
    V2f = V2 + V2e * (z2 - z20)
    L2t = 1 / (1 - p ** 2) * (
            (z1 - z10) ** 2 / V1f - 2 * p * (z1 - z10) * (z2 - z20) / np.sqrt(V1f * V2f) + (z2 - z20) ** 2 / V2f)
    return L2t

def fit5para(xr, yr):
    A = []
    B = []
    for i in range(0, len(xr)):
        A.append([xr[i], yr[i]])
        # B.append(2.30)
        B.append(5.99)
    A = np.array(A)
    B = np.array(B)
    guess = (0)

    AR, pcov = optimize.curve_fit(func, A, B, guess)
    return AR

ff = fit5para(x, y)
p = ff
print("Corr =", ff)

#sys.exit("end here")

# Open the full grid files
expdata = np.genfromtxt('HIGG_2016-07_Llh-2d-GridFull.txt')
xExp = expdata[:,0]
yExp = expdata[:,1]

# Make validation plot

matplotlib.rcParams['xtick.major.pad'] = 15
matplotlib.rcParams['ytick.major.pad'] = 15

# Loglikelihood Calculation
def Loglikelihood(z1,z2):
    # z10, z20 = 11.4011976,	0.490450205
    # z10, z20 = 1.096269,    0.60549408
    z10, z20 = 1.10, 0.62
    sig1p, sig1m, sig2p, sig2m = 0.21, 0.20, 0.36, 0.35
    V1 = sig1p*sig1m
    V1e = sig1p - sig1m
    V2 = sig2p * sig2m
    V2e = sig2p - sig2m
    V1f = V1 + V1e*(z1-z10)
    V2f = V2 + V2e * (z2 - z20)
    L2t = 1/(1-p**2)*((z1-z10)**2/V1f-2*p*(z1-z10)*(z2-z20)/np.sqrt(V1f*V2f)+(z2-z20)**2/V2f)
    return L2t

# Grid calculations
xmax, ymax = 2, 2
(a, b) = (0.01, 0.01)
x2 = np.arange(0.25, xmax, a, dtype=np.float)
y2 = np.arange(-0.5, ymax, b, dtype=np.float)
X2, Y2 = np.meshgrid(x2, y2)
Z = Loglikelihood(X2,Y2)
print("Minimum Log likelihood =",Z.min())
plt.contour(X2,Y2,Z,[0.00001,2.30,5.99])



# Plot
matplotlib.rcParams['xtick.major.pad'] = 15
matplotlib.rcParams['ytick.major.pad'] = 15
fig = plt.figure()

plt.contourf(X2,Y2,Z,[10**(-5),2.3,5.99],colors=['0.5','0.75'])
plt.plot(xExp,yExp,'.',markersize=4, color = 'blue', label="ATLAS official")
plt.plot([1],[1], '*', c='k', ms=6, label="SM")

plt.xlim((0.25,2))
plt.ylim((-0.5,2))
plt.minorticks_on()
plt.tick_params(labelsize=20, length=14, width=2)
plt.tick_params(which='minor', length=7, width=1.2)

plt.legend(loc='upper right', fontsize=12)
plt.xlabel(r'$\mu_{ZZ}^{ggF}$', fontsize=20)
plt.ylabel(r'$\mu_{ZZ}^{VBF}$', fontsize=20)
plt.title("$\mu$ from ATLAS-HIGG-2016-07 (VGaussian, 95% CL, fitted corr)")
fig.set_tight_layout(True)

# Save fig
fig.savefig('mu_ggH-VBF_2D_VGaussian-fitted-corr-95CL.pdf')
