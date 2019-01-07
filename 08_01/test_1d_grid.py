import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib

# Open the 1D grid files
f = open('Lilith-Run2database/08_01/grid_HIGG-2016-22_VBF_ZZ.txt', 'r')
#f = open('Lilith-Run2database/08_01/grid_HIGG-2016-22_ggH_ZZ.txt', 'r')

# Plot the grids
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]
plt.plot(x,y,'.',markersize=3, color = 'blue')

# Calculate Lxy

# Use either np.poly1d or scipy.interpolate.UnivariateSpline
Lxy = interpolate.UnivariateSpline(x, y, k = 4, s = 1)
#Lxy = np.poly1d(np.polyfit(x, y, 4))

xs = np.linspace(-1, 11, 200)
plt.plot(xs, Lxy(xs), 'g', lw=1)
plt.show()
print(Lxy)