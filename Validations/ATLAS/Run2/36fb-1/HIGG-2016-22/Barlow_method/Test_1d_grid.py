import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib
from scipy.stats import poisson
from scipy.special import gamma

# Open the 1D grid files
f = open('HIGG-2016-22_ggH-1d-Grid.txt', 'r')

# Plot the grids
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]
plt.plot(x,y,'.',markersize=3, color = 'blue',label="Observed mu")

g = open('HIGG-2016-22_ggH-1d-norm-cr-Grid.txt', 'r')

text = [[float(num) for num in line.split()] for line in g]
g.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]/1.18
y = textt[1]
plt.plot(x,y,'.',markersize=3, color = 'r',label="Normalized Cross Section")

plt.legend(loc='upper right', fontsize=12)
plt.show()
