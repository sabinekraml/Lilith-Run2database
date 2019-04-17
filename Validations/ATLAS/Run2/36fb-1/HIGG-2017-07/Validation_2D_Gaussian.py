import numpy as np
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt


def fitEllipse(x, y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a


def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.arctan(2*b/(a-c))/2


# test
center = np.array([1.11239096,  1.21142269])
A1, B1, U1 = 3.2993942,  1.50299369,   3.77001839
# center = np.array([1.767, 1.383])
# A1, B1, U1 = 0.881, 0.875, 3.891

a7 = A1*center[0]**2 + 2*B1*center[0]*center[1] + U1*center[1]**2 - 2.3
a = [A1, 2*B1, U1, -2*A1*center[0]-2*B1*center[1], -2*B1*center[0]-2*U1*center[1], a7]
a72 = A1*center[0]**2 + 2*B1*center[0]*center[1] + U1*center[1]**2 - 5.99
a2 = [A1, 2*B1, U1, -2*A1*center[0]-2*B1*center[1], -2*B1*center[0]-2*U1*center[1], a72]

#a = [0.88107842388532587, 1.7502743925252628, 3.8908361183900975, -5.5360786456836228, -13.859077749518059, 12.179163669102653]
#a = [-0.04461328, -0.08862489, -0.19701193,  0.28031855,  0.70175242, -0.61669021]
print(a)

phi = ellipse_angle_of_rotation(a)
axes = ellipse_axis_length(a)

phi2 = ellipse_angle_of_rotation(a2)
axes2 = ellipse_axis_length(a2)

print("center = ",  center)
print("angle of rotation = ",  phi)
print("axes = ", axes)

a, b = axes
a2, b2 = axes2
arc = 0.8
R = np.arange(0,arc*np.pi*3, 0.01)
xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
xx2 = center[0] + a2*np.cos(R)*np.cos(phi2) - b2*np.sin(R)*np.sin(phi2)
yy2 = center[1] + a2*np.cos(R)*np.sin(phi2) + b2*np.sin(R)*np.cos(phi2)



f = open('HIGG-2017-07_Llh-2d-Grid68.txt', 'r')
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x = textt[0]
y = textt[1]

f = open('HIGG-2017-07_Llh-2d-Grid95.txt', 'r')
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
x2 = textt[0]
y2 = textt[1]


from pylab import *
fig = plt.figure()
plot(x,y,'.',markersize=4,label="Exp CL", color = 'blue')
plot(x2,y2,'.',markersize=4, color = 'blue')
# plot(1.18,1.19,'bo',markersize=3,label="Exp best-fit", color = 'blue')
plot(xx,yy,label="Gaussian CL", color = 'red')
plot(xx2,yy2, color = 'red')
plot(center[0],center[1], 'bo',markersize=3,label="Gaussian best-fit", color = 'red')

# center = np.array([1.207, 1.07])
# A1, B1, U1 = 32.96, 4.894, 4.143
# # center = np.array([1.767, 1.383])
# # A1, B1, U1 = 0.881, 0.875, 3.891
#
# a7 = A1*center[0]**2 + 2*B1*center[0]*center[1] + U1*center[1]**2 - 2.3
# a = [A1, 2*B1, U1, -2*A1*center[0]-2*B1*center[1], -2*B1*center[0]-2*U1*center[1], a7]
# a72 = A1*center[0]**2 + 2*B1*center[0]*center[1] + U1*center[1]**2 - 5.99
# a2 = [A1, 2*B1, U1, -2*A1*center[0]-2*B1*center[1], -2*B1*center[0]-2*U1*center[1], a72]

#a = [0.88107842388532587, 1.7502743925252628, 3.8908361183900975, -5.5360786456836228, -13.859077749518059, 12.179163669102653]
#a = [-0.04461328, -0.08862489, -0.19701193,  0.28031855,  0.70175242, -0.61669021]
print(a)

# phi = ellipse_angle_of_rotation(a)
# axes = ellipse_axis_length(a)
#
# phi2 = ellipse_angle_of_rotation(a2)
# axes2 = ellipse_axis_length(a2)
#
# a, b = axes
# a2, b2 = axes2
# arc = 0.8
# R = np.arange(0,arc*np.pi*3, 0.01)
# xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
# yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
# xx2 = center[0] + a2*np.cos(R)*np.cos(phi2) - b2*np.sin(R)*np.sin(phi2)
# yy2 = center[1] + a2*np.cos(R)*np.sin(phi2) + b2*np.sin(R)*np.cos(phi2)
#
# plot(xx,yy,label="Lilith's CL", color = 'green')
# plot(xx2,yy2, color = 'green')
# plot(center[0],center[1], 'bo',markersize=3, label="Lilith's best-fit", color = 'green')
legend(loc='upper right', fontsize=12)
plt.xlabel(r'$\mu_{\tau\tau}^{ggH}$', fontsize=30)
plt.ylabel(r'$\mu_{\tau\tau}^{VBF}$', fontsize=30)
plt.title("ATLAS-HIGG-2017-07 (2D Poisson dist.)", fontsize=20)

fig.set_tight_layout(True)
fig.savefig('mu_VBF_ggH_2D_Poisson.pdf')
print(a, b)



