import numpy as np
import matplotlib.pyplot as plt

# Open the 1D grid files
f = open('HIGG_2016-22_Llh-2d-Grid.txt', 'r')

# Plot the grids
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]
plt.plot(x,y,'.',markersize=3, color = 'red', label="Exp")

# Loglikelihood Calculation
def Loglikelihood(z1,z2):
    # (z10, sig1p, sig1m, z20, sig2p, sig2m) = (3.82356639105, 1.7319741945, 1.40305529427, 1.11225919501, 0.245219850472, 0.219324916401)
    (z10, sig1p, sig1m, z20, sig2p, sig2m) = (4, 1.75, 1.46, 1.11, 0.23, 0.21)
    # (z10, sig1p, sig1m, z20, sig2p, sig2m) = (3.8587855175, 2.05910747, 1.74865865, 1.11435560099, 0.27109778, 0.2461435)
    p = -0.41
    V1 = sig1p*sig1m
    V1e = sig1p - sig1m
    V2 = sig2p * sig2m
    V2e = sig2p - sig2m
    V1f = V1 + V1e*(z1-z10)
    V2f = V2 + V2e * (z2 - z20)
    L2t = 1/(1-p**2)*((z1-z10)**2/V1f-2*p*(z1-z10)*(z2-z20)/np.sqrt(V1f*V2f)+(z2-z20)**2/V2f)
    return L2t

# Grid calculations
xmax, ymax = 11, 3
(a, b) = (0.01, 0.01)
x2 = np.arange(0, xmax, a, dtype=np.float)
y2 = np.arange(0, ymax, b, dtype=np.float)
X2, Y2 = np.meshgrid(x2, y2)
Z = Loglikelihood(X2,Y2)
print(Z.min())
plt.contour(X2,Y2,Z,[0.00001,2.30,5.99])


# Print data
g = open('2D_CL_VGaussian.txt', 'w')
for i in range (len(x2)):
    for j in range (len(y2)):
        g.write(str(x2[i]) + " " + str(y2[j]) + " " + str(Z[j][i]) + '\n')
g.close


# Plot
plt.legend(loc='upper right', fontsize=12)
plt.xlabel(r'$\mu_{ZZ}^{VBF}$', fontsize=30)
plt.ylabel(r'$\mu_{ZZ}^{ggF}$', fontsize=30)
plt.title("$\mu$ from ATLAS-HIGG-2016-22 (Variable Gaussian)")

plt.show()