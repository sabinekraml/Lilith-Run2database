import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ---------- 2D Continuous Poisson Dist. with Negative Correlation ----------
# ---------- (ggF marginal - VBF dependent) ----------

# Input data

# correlation
corr = -0.41

# # Data from Table 9
# # ggF
# cen1, sig1m, sig1p = 1.11, np.sqrt(0.2**2 + 0.06**2 + 0.04**2), np.sqrt(0.22**2+0.07**2+0.04**2)
# # VBF
# cen2, sig2m, sig2p = 4, np.sqrt(1.4**2 + 2*0.3**2), np.sqrt(1.7**2 + 2*0.3**2)

# Slightly Fixed the data from Table 9
# ggF
cen1, sig1m, sig1p = 1.11+0.01, np.sqrt(0.2**2 + 0.06**2 + 0.04**2)+0.01, np.sqrt(0.22**2+0.07**2+0.04**2)+0.01
# VBF
cen2, sig2m, sig2p = 4.0-0.1, np.sqrt(1.4**2 + 2*0.3**2), np.sqrt(1.7**2 + 2*0.3**2)

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
xmax, ymax = 12, 3
x = np.arange(0, xmax, 0.02, dtype=np.float)
y = np.arange(0, ymax, 0.02, dtype=np.float)
X, Y = np.meshgrid(x, y)
Z = Loglikelihood(Y,X)

# Check for minimum ML
print(Z.min())

# Print data to file
# g = open('VBF-ggH_ZZ_f_Poisson-ggH-marginal.txt', 'w')
g = open('VBF-ggH_ZZ_f_Poisson-ggH-marginal-fixed.txt', 'w')
for i in range (len(x)):
    for j in range (len(y)):
        g.write(str(x[i]) + " " + str(y[j]) + " " + str(Z[j][i]) + '\n')
g.close

# read data for official 68% and 95% CL contours
expdata = np.genfromtxt('HIGG_2016-22_Llh-2d-Grid.txt')
xExp = expdata[:,0]
yExp = expdata[:,1]

# Make validation plot

matplotlib.rcParams['xtick.major.pad'] = 15
matplotlib.rcParams['ytick.major.pad'] = 15

fig = plt.figure()

plt.contourf(X,Y,Z,[10**(-5),2.3,5.99],colors=['0.5','0.75'])
plt.plot(xExp,yExp,'.',markersize=4, color = 'blue', label="ATLAS official")
plt.plot([1],[1], '*', c='k', ms=6, label="SM")

plt.xlim((0,10))
plt.ylim((0,2.5))
plt.minorticks_on()
plt.tick_params(labelsize=20, length=14, width=2)
plt.tick_params(which='minor', length=7, width=1.2)

plt.legend(loc='upper right', fontsize=16)
plt.xlabel(r'$\mu({\rm VBF},ZZ)$', fontsize=20)
plt.ylabel(r'$\mu({\rm ggF},ZZ)$', fontsize=20)
plt.title("ATLAS-HIGG-2016-22 (2D Poisson dist.)", fontsize=20)

fig.set_tight_layout(True)
# fig.savefig('mu_VBF_ggH_2D_Poisson-ggH-marginal.pdf')
fig.savefig('mu_VBF_ggH_2D_Poisson-ggH-marginal-fixed.pdf')