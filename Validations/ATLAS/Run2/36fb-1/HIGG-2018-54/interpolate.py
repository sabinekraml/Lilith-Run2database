from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy import interpolate

# Open the 1D grid files
f = open('HIGG-2018-54_VBF-1d-Grid.txt', 'r')


# Read & Plot the grids
fig = plt.figure()
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]
plt.plot(x,y,'.',markersize=3, color = 'blue',label="Observed")

# Interpolation function (Try different k, s):
Lxy = interpolate.UnivariateSpline(x, y, k=3, s=0)

# Range of interpolation function (Try to vary the x1 range):
x1 = np.arange(-0.41,1.5,0.005)
# x1 = np.arange(-0.21,0.71,0.005)
y1 = Lxy(x1)

# Plot interpolation
plt.plot(x1,y1,'-',markersize=2, color = 'g',label="Interpolate Appx.")

plt.xlabel(r'$\mu_{invi}^{VBF}$', fontsize=20)
plt.ylabel(r'-2 Loglikelihood', fontsize=20)
plt.title("$\mu$ from ATLAS-HIGG-2018-54 (Interpolate)")
plt.legend(loc='upper right', fontsize=12)
fig.set_tight_layout(True)

fig.savefig('Interpolate.pdf')
plt.show()