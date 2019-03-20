import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ---------- 2D Continuous Poisson Dist. with Negative Correlation ----------
# ---------- (ggF marginal - VBF dependent) ----------

# Input data

# Data from Fig. 12

# # Htautau
# corr = -0.32
# # ggF
# cen1, sig1m, sig1p = 1.05, 0.47, 0.53
# # VBF
# cen2, sig2m, sig2p = 1.12, 0.43, 0.45

# # HWW
# corr = -0.21
# # ggF
# cen1, sig1m, sig1p = 1.35, 0.19, 0.21
# # VBF
# cen2, sig2m, sig2p = 0.28, 0.60, 0.64

# # HZZ
# corr = -0.41
# # ggF
# cen1, sig1m, sig1p = 1.22, 0.21, 0.23
# # VBF
# cen2, sig2m, sig2p = -0.09, 0.76, 1.02

# # Hgaga
# corr = -0.28
# # ggF
# cen1, sig1m, sig1p = 1.16, 0.18, 0.21
# # VBF
# cen2, sig2m, sig2p = 0.67, 0.46, 0.59

# Hmumu
corr = -0.54
# ggF
cen1, sig1m, sig1p = 0.31, 1.79, 1.80
# VBF
cen2, sig2m, sig2p = 2.72, 7.03, 7.12


# Slightly Fixed the data from Fig. 12
# VBF
# cen2, sig2m, sig2p = 2.03, 0.52, 0.61


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

# Solving 2D Poisson parameters for negative correlation


def f(x):
    return corr*np.sqrt(nu1*nu2*(1+nu2*(np.exp(nu1*x**2)-1)))-nu1*nu2*x

A = fsolve(f,0)
alpha = np.float(np.log(1+A))

# Calculate Loglikelihood


def Loglikelihood(z1,z2):
    L2t1 = - alpha1 * (z1 - cen1) + nu1 * np.log(1 + alpha1 * (z1 - cen1) / nu1)
    L2t2a = - alpha2 * (z2 - cen2 + 1 / gam2) * np.exp(alpha * nu1 - A * alpha1 * (z1 - cen1 + 1/gam1))
    L2t2b = - alpha2 * ( 1 / gam2) * np.exp(alpha * nu1 - A * alpha1 * ( 1 / gam1))
    L2t2c = nu2 * np.log(L2t2a/L2t2b)
    L2t2 = L2t2a - L2t2b + L2t2c
    L2t = -2 * (L2t1 + L2t2)
    return L2t


# Discrete Poisson Distribution
y = np.arange(cen1-nu1/alpha1+0.01, cen1+sig1p+5, 0.02, dtype=np.float)
x = np.arange(cen2-nu2/alpha2+0.01, cen2+sig2p+5, 0.02, dtype=np.float)
X, Y = np.meshgrid(x, y)
Z = Loglikelihood(Y,X)

# Check for minimum ML
print(Z.min())

# Print data to file
# g = open('VBF-ggH_tautau_f_Poisson-ggH-marginal.txt', 'w')
# g = open('VBF-ggH_WW_f_Poisson-ggH-marginal.txt', 'w')
# g = open('VBF-ggH_ZZ_f_Poisson-ggH-marginal.txt', 'w')
# g = open('VBF-ggH_gaga_f_Poisson-ggH-marginal.txt', 'w')
g = open('VBF-ggH_mumu_f_Poisson-ggH-marginal.txt', 'w')

# g = open('VBF-ggH_ZZ_f_Poisson-ggH-marginal-fixed.txt', 'w')
for i in range (len(x)):
    for j in range (len(y)):
        g.write(str(y[j]) + " " + str(x[i]) + " " + str(Z[j][i]) + '\n')
g.close

# read data for official 68% and 95% CL contours
# expdata = np.genfromtxt('HIGG_2016-21_Llh-2d-Grid.txt')
# xExp = expdata[:,0]
# yExp = expdata[:,1]

# Make validation plot

matplotlib.rcParams['xtick.major.pad'] = 15
matplotlib.rcParams['ytick.major.pad'] = 15

fig = plt.figure()

plt.contourf(Y,X,Z,[10**(-5),2.3,5.99],colors=['0.5','0.75'])
# plt.plot(xExp,yExp,'.',markersize=4, color = 'blue', label="ATLAS official")
plt.plot([1],[1], '*', c='k', ms=6, label="SM")

plt.xlim((-7, cen2+sig2p+5))
plt.ylim((-7, cen1+sig1p+5))
plt.minorticks_on()
plt.tick_params(labelsize=20, length=14, width=2)
plt.tick_params(which='minor', length=7, width=1.2)

plt.legend(loc='upper right', fontsize=16)
plt.xlabel(r'$\mu({\rm ggF},ZZ)$', fontsize=20)
plt.ylabel(r'$\mu({\rm VBF},ZZ)$', fontsize=20)
plt.title("ATLAS-HIGG-2017-31 (2D Poisson dist.)", fontsize=20)

fig.set_tight_layout(True)
# fig.savefig('mu_VBF-ggH_tautau_2D_Poisson-ggH-marginal.pdf')
# fig.savefig('mu_VBF-ggH_WW_2D_Poisson-ggH-marginal.pdf')
# fig.savefig('mu_VBF-ggH_ZZ_2D_Poisson-ggH-marginal.pdf')
# fig.savefig('mu_VBF-ggH_gaga_2D_Poisson-ggH-marginal.pdf')
fig.savefig('mu_VBF-ggH_mumu_2D_Poisson-ggH-marginal.pdf')

# fig.savefig('mu_VBF_ggH_2D_Poisson-ggH-marginal-fixed.pdf')
