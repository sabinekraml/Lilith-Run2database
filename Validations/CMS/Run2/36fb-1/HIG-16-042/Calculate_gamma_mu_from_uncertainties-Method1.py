from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt


# Input data

# Data from Fig. 12

# ggH
# cen, sigm, sigp = 1.38, 0.24, 0.21
# VBF
cen, sigm, sigp = 0.29, 0.29, 0.66
# ZH
# cen, sigm, sigp = 1.00, 1.00, 1.57
# WH
# cen, sigm, sigp = 3.27, 1.70, 1.88


def f(t):
    return np.exp(-t*(sigm + sigp)) - (1 - t*sigm)/(1 + t*sigp)

gam = fsolve(f,1/sigm)
# print(A)

# Choose VBF - ggH
nu= 1/(2*(gam*sigp - np.log(1 + gam*sigp)))
# nu= 1/(2*(gam*sig1p - np.log(1 + gam*sig1p)))
alpha = nu*gam

print(gam[0])
print(alpha[0])
print(nu[0])
print(cen)

# Choose VBF - ggH
x = np.arange(cen-sigm-1,cen+sigp+2,0.005)
y2 = -2*(-alpha*(x - cen) + nu*np.log(1 + alpha*(x - cen)/nu))
# x = np.arange(0.57,1.08,0.005)
# y2 = -2*(-alpha*(x - cen1) + L*np.log(1 + alpha*(x - cen1)/L))

fig = plt.figure()
plt.plot(x,y2,'-',markersize=2, color = 'g',label="Barlow's Poisson Appx.")

# Choose VBF - ggH
plt.xlabel(r'$\mu_{ZZ}^{VBF}$', fontsize=20)
# plt.xlabel(r'$\mu_{ZZ}^{ggH}$', fontsize=20)


plt.ylabel(r'-2 Loglikelihood', fontsize=20)
plt.title("$\mu$ from CMS-HIG-16-042 (Poisson)")
plt.legend(loc='upper right', fontsize=12)

fig.set_tight_layout(True)
plt.show()
# Choose VBF - ggH
# fig.savefig('mu_VBF_1D_Poisson.pdf')

# fig.savefig('mu_ggF_1D_Poisson.pdf')
# fig.savefig('test.png')