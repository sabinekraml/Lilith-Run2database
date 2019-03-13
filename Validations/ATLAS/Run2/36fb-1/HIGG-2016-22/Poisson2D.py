from sympy.solvers import solve
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import poisson
from sympy import Symbol
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

# Open the 1D grid files
f = open('A:\Paper Sabine\DATA PYTHON MU\Mu\HIGG_16-022_68-95.txt', 'r')

# Plot the grids
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]
plt.plot(x,y,'.',markersize=3, color = 'red', label="Exp")

# Set maximum bins
labda1, labda2 = 25.55, 7.61
# Correlation
corr = -0.41

# Solving 2D Poisson parameters
x = Symbol('x')
A = solve(corr*sp.sqrt(labda1*labda2*(1+labda2*(sp.exp(labda1*x**2)-1)))-labda1*labda2*x, x)
alpha1 = np.float(np.log(1+np.float(A[1])))
xbeta1 = np.float(np.log(labda2)-A[1]*labda1)

# Loglikelihood Calculation
def Loglikelihood(z1,z2):
    labda2 = np.exp(xbeta1+alpha1*z1)
    P = labda1**z1*np.exp(-labda1)/gamma(z1+1)*labda2**z2*np.exp(-labda2)/gamma(z2+1)
    L2t = -2*np.log(P)
    return L2t

# Discrete Poisson Distribution
xmax, ymax = 35, 90
L1, L2 = 25, 7
Logmax = Loglikelihood(L1,L2)
x = np.arange(0, xmax, 1, dtype=np.float)
y = np.arange(0, ymax, 1, dtype=np.float)
X, Y = np.meshgrid(x, y)
Z = Loglikelihood(Y,X)-Logmax

# Check for minimum ML
print(Z.min())

# Fix Central values

# a = 3.917193/L2
# b = 1.117641/L1
a = 3.98706897/L2
b = 1.11016949/L1

# Interpolation for Continuous distribution
x3 = np.arange(0, a*xmax, a, dtype=np.float)
y3 = np.arange(0, b*ymax, b, dtype=np.float)
A = interpolate.RectBivariateSpline(y3, x3, Z, kx=4, ky=4, s=0)
X3, Y3 = np.meshgrid(x3, y3)

L = np.zeros((len(y3), len(x3)))
for i in range(len(y3)):
    for j in range(len(x3)):
        L[i][j] = A(y3[i],x3[j])

plt.contour(X3,Y3,L,[0.0001,2.30,5.99])

# Print data
g = open('2D_CL_Poisson.txt', 'w')
for i in range (len(x3)):
    for j in range (len(y3)):
        g.write(str(x3[i]) + " " + str(y3[j]) + " " + str(L[j][i]) + '\n')
g.close

# Plot
plt.legend(loc='upper right', fontsize=12)
plt.xlabel(r'$\mu_{ZZ}^{VBF}$', fontsize=30)
plt.ylabel(r'$\mu_{ZZ}^{ggF}$', fontsize=30)
plt.title("$\mu$ from ATLAS-HIGG-2016-22 (Poisson)")

plt.show()