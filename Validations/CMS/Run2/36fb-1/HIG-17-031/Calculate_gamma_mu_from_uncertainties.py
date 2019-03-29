from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

# Open the 1D grid files

fig = plt.figure()
# Plot the grids
# text = [[float(num) for num in line.split()] for line in f]
# f.close()
# text = np.array(text)
# textt = np.transpose(text)
# x = textt[0]
# y = textt[1]
# plt.plot(x,y,'.',markersize=3, color = 'blue',label="Observed")

# Input data

# Data from Fig. 12

## ggH
# bb
# cen2, sig2m, sig2p = 2.51, 2.01, 2.43

## WH
# bb
# cen2, sig2m, sig2p = 1.73, 0.68, 0.70
# WW
# cen2, sig2m, sig2p = 3.91, 2.01, 2.26
# ZZ
# cen2, sig2m, sig2p = 0.00, 0.00, 2.33
# gaga
# cen2, sig2m, sig2p = 3.76, 1.35, 1.48

## ZH
# bb
# cen2, sig2m, sig2p = 0.99, 0.45, 0.47
# WW
# cen2, sig2m, sig2p = 0.96, 1.46, 1.81
# ZZ
# cen2, sig2m, sig2p = 0.00, 0.00, 4.26
# gaga
# cen2, sig2m, sig2p = 0.00, 0.00, 1.14

## ttH
# bb
# cen2, sig2m, sig2p = 0.91, 0.43, 0.45
# tautau
# cen2, sig2m, sig2p = 0.23, 0.88, 1.03
# WW
# cen2, sig2m, sig2p = 1.60, 0.59, 0.65
# ZZ
# cen2, sig2m, sig2p = 0.00, 0.00, 1.50
# gaga
cen2, sig2m, sig2p = 2.18, 0.75, 0.88



# VBF
# cen2, sig2m, sig2p = 0.8, 0.5, 0.6
# VH
# cen2, sig2m, sig2p = 2.4, 1, 1.1
# ttH
# cen2, sig2m, sig2p = 2.2, 0.8, 0.9


def f(t):
    # Choose VBF - ggH
    return np.exp(-t*(sig2m + sig2p)) - (1 - t*sig2m)/(1 + t*sig2p)
    # return np.exp(-t*(sig1m + sig1p)) - (1 - t*sig1m)/(1 + t*sig1p)

gam = fsolve(f,1)
# print(A)

# Choose VBF - ggH
nu= 1/(2*(gam*sig2p - np.log(1 + gam*sig2p)))
# nu= 1/(2*(gam*sig1p - np.log(1 + gam*sig1p)))
alpha = nu*gam

print(cen2)
print(alpha[0])
print(nu[0])


# Choose VBF - ggH
x = np.arange(cen2-sig2m-2,cen2+sig2p+2,0.005)
y2 = -2*(-alpha*(x - cen2) + nu*np.log(1 + alpha*(x - cen2)/nu))
# x = np.arange(0.57,1.08,0.005)
# y2 = -2*(-alpha*(x - cen1) + L*np.log(1 + alpha*(x - cen1)/L))


plt.plot(x,y2,'-',markersize=2, color = 'g',label="Barlow's Poisson Appx.")

# Choose VBF - ggH
plt.xlabel(r'$\mu_{ZZ}^{VBF}$', fontsize=20)
# plt.xlabel(r'$\mu_{ZZ}^{ggH}$', fontsize=20)


plt.ylabel(r'-2 Loglikelihood', fontsize=20)
plt.title("$\mu$ from CMS-HIG-17-031 (Poisson)")
plt.legend(loc='upper right', fontsize=12)

fig.set_tight_layout(True)
plt.show()
# Choose VBF - ggH
# fig.savefig('mu_VBF_1D_Poisson.pdf')

# fig.savefig('mu_ggF_1D_Poisson.pdf')