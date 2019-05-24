import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.optimize as optimize

# ---------- 2D Continuous Poisson Dist. with Negative Correlation ----------
# ---------- (ggF marginal - VBF dependent) ----------

# Input data

# # correlation
# corr = -0.27
#
#
# # # Data from Fig. 12
# # # ggF
# # cen1, sig1m, sig1p = 0.81, 0.18, 0.19
# # # VBF
# # cen2, sig2m, sig2p = 2, 0.5, 0.6
#
# # Fit uncertainties from Aux. Fig. 23
# # ggF
cen1, sig1m, sig1p = 0.809361060171, 0.186957153857, 0.176141692478
# # VBF
cen2, sig2m, sig2p = 2.03669874783, 0.606157414018, 0.526259988038


# Calculate parameters for continuous distribution


def g1(x):
    return np.exp(-x*(sig1m + sig1p)) - (1 - x*sig1m)/(1 + x*sig1p)


gam1 = fsolve(g1,1)
nu1 = 1/(2*(gam1*sig1p - np.log(1 + gam1*sig1p)))
alpha1 = nu1*gam1


def g2(x):
    return np.exp(-x*(sig2m + sig2p)) - (1 - x*sig2m)/(1 + x*sig2p)


gam2 = fsolve(g2,1)
nu2 = 1/(2*(gam2*sig2p - np.log(1 + gam2*sig2p)))
alpha2 = nu2*gam2

# # print(gam1,gam2)
# # Solving 2D Poisson parameters for negative correlation
#
#
# def f(x):
#     return corr*np.sqrt(nu1*nu2*(1+nu2*(np.exp(nu1*x**2)-1)))-nu1*nu2*x
#
# A = fsolve(f,0)
# alpha = np.float(np.log(1+A))

# Calculate Loglikelihood

f = open('HIGG_2016-21_Llh-2d-Grid95.txt', 'r')

# Plot the grids
fig = plt.figure()
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]


def func(X, alpha1, alpha2, nu1, nu2, alpha):
    z1, z2 = X[:,0], X[:,1]
    A = np.exp(alpha)-1
    L2t1 = - alpha1 * (z1 - cen1) + nu1 * np.log(1 + alpha1 * (z1 - cen1) / nu1)
    L2t2a = - (alpha2 * (z2 - cen2) + nu2) * np.exp(alpha * nu1 - A * (alpha1 * (z1 - cen1) + nu1))
    L2t2b = - nu2 * np.exp((alpha - A) * nu1)
    L2t2c = nu2 * np.log(L2t2a/L2t2b)
    L2t2 = L2t2a - L2t2b + L2t2c
    L2t = -2 * (L2t1 + L2t2)
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
    guess = (62, 8, 125, 22, -0.005)
    AR, pcov = optimize.curve_fit(func, A, B, guess)
    return AR

ff = fit5para(x, y)
alpha1, alpha2, nu1, nu2, alpha = ff[0], ff[1], ff[2], ff[3], ff[4]

x = np.exp(alpha)-1
corr = nu1*nu2*x/np.sqrt(nu1*nu2*(1+nu2*(np.exp(nu1*x**2)-1)))
print("correlation =", corr)


gam1 = alpha1/nu1
gam2 = alpha2/nu2

print("\n Central value and uncertainties:")
def g1(sig1p):
    return gam1*sig1p - np.log(1 + gam1*sig1p) - 1/(2*nu1)

sig1p = fsolve(g1,1)
print("sig1p =",sig1p[0])

def g2(sig1m):
    return np.exp(-gam1*(sig1m + sig1p)) - (1 - gam1*sig1m)/(1 + gam1*sig1p)
sig1m = fsolve(g2,1)

print("sig1m =", sig1m[0])
print("cen1 =", cen1)


print("\n Central value and uncertainties:")
def s1(sig2p):
    return gam2*sig2p - np.log(1 + gam2*sig2p) - 1/(2*nu2)

sig2p = fsolve(s1,1)
print("sig2p =",sig2p[0])

def s2(sig2m):
    return np.exp(-gam2*(sig2m + sig2p)) - (1 - gam2*sig2m)/(1 + gam2*sig2p)
sig2m = fsolve(s2,1)

print("sig1m =", sig2m[0])
print("cen1 =", cen2)

A = np.exp(alpha)-1

def Loglikelihood(z1,z2):
    L2t1 = - alpha1 * (z1 - cen1) + nu1 * np.log(1 + alpha1 * (z1 - cen1) / nu1)
    L2t2a = - alpha2 * (z2 - cen2 + 1 / gam2) * np.exp(alpha * nu1 - A * alpha1 * (z1 - cen1 + 1/gam1))
    L2t2b = - alpha2 * ( 1 / gam2) * np.exp(alpha * nu1 - A * alpha1 * ( 1 / gam1))
    L2t2c = nu2 * np.log(L2t2a/L2t2b)
    L2t2 = L2t2a - L2t2b + L2t2c
    L2t = -2 * (L2t1 + L2t2)
    return L2t


# Discrete Poisson Distribution
xmax, ymax = 4, 10
x = np.arange(0, xmax, 0.02, dtype=np.float)
y = np.arange(0, ymax, 0.02, dtype=np.float)
X, Y = np.meshgrid(x, y)
Z = Loglikelihood(Y,X)

# Check for minimum ML
print(Z.min())

# read data for official 68% and 95% CL contours
expdata = np.genfromtxt('HIGG_2016-21_Llh-2d-Grid.txt')
xExp = expdata[:,0]
yExp = expdata[:,1]

# # Make validation plot

matplotlib.rcParams['xtick.major.pad'] = 15
matplotlib.rcParams['ytick.major.pad'] = 15

fig = plt.figure()

plt.contourf(Y,X,Z,[10**(-5),2.3,5.99],colors=['0.5','0.75'])
plt.plot(xExp,yExp,'.',markersize=4, color = 'blue', label="ATLAS official")
plt.plot([1],[1], '*', c='k', ms=6, label="SM")

plt.xlim((0.3,1.5))
plt.ylim((0,4.5))
plt.minorticks_on()
plt.tick_params(labelsize=20, length=14, width=2)
plt.tick_params(which='minor', length=7, width=1.2)

plt.legend(loc='upper right', fontsize=12)
plt.xlabel(r'$\mu_{ZZ}^{ggF}$', fontsize=20)
plt.ylabel(r'$\mu_{ZZ}^{VBF}$', fontsize=20)
plt.title("$\mu$ ATLAS-HIGG-2016-21 (Poisson, 68% CL, fitted corr)")
# # plt.title("$\mu$ ATLAS-HIGG-2016-21 (Poisson, provided 1D, corr = -0.27)")
#
fig.set_tight_layout(True)
# # fig.savefig('HIGG-2016-21-mu-Poisson-Fig12.pdf')
# fig.savefig('HIGG-2016-21-mu-Poisson-fit1d-Poisson-AuxFig23.pdf')
# # fig.savefig('mu_VBF_ggH_2D_Poisson-ggH-marginal-fixed.pdf')
# fig.savefig('HIGG-2016-21-mu-Poisson-fit2d-Poisson-Fig15_68.pdf')

plt.show()