###############################################################
#
# Lilith routine example
#
# To execute from /Lilith-1.X root folder
#
# Constraints on reduced couplings
# in the (CV, CF) model. CV: common coupling to EW gauge bosons,
# CF: common coupling for fermions.
# The loop-induced effective couplings CGa and Cg are
# determined from CV and CF.
#
# 1-dimensional likelihood profiles are obtained from a
# profile-likelihood analysis
#
# Use the libraries matplotlib, iminuit
#
###############################################################

import sys, os
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from iminuit import Minuit
import matplotlib
import numpy as np

lilith_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(lilith_dir)
sys.path.append('../..')
import lilith

######################################################################
# Parameters
######################################################################

# Exprimental results
myexpinput = "data/latest.list"
# Lilith precision mode
myprecision = "BEST-QCD"
# Output
plottitle = "results/CVCF_1dprofile.pdf"
# Higgs mass to test
myhmass = 125.
# Number of grid steps in each of the two dimensions (squared grid)
grid_subdivisions = 100
# Output file
output = "results/CVCF_2d.out"
# Output plot
outputplot = "results/CVCF_2d.pdf"
# Range of the scan
CF_min = 0.
CF_max = 3.
CV_min = 0.
CV_max = 3.

verbose = False
timer = False


######################################################################
# * usrXMLinput: generate XML user input
# * getL:        -2LogL for a given (CV, CF) point
######################################################################

def usrXMLinput(mh=125., CV=1., CF=1., precision="BEST-QCD"):
    """generate XML input from reduced couplings
       the gamma-gamma and/or gluon-gluon couplings are
       evaluated as functions of the fermionic and bosonic
       reduced couplings """

    myInputTemplate = """<?xml version="1.0"?>

<lilithinput>

<reducedcouplings>
  <mass>%(mh)s</mass>

  <C to="tt">%(CF)s</C>
  <C to="bb">%(CF)s</C>
  <C to="tautau">%(CF)s</C>
  <C to="cc">%(CF)s</C>
  <C to="gg">%(CF)s</C>
  <C to="WW">%(CV)s</C>
  <C to="ZZ">%(CV)s</C>

  <extraBR>
    <BR to="invisible">0.</BR>
    <BR to="undetected">0.</BR>
  </extraBR>

  <precision>%(precision)s</precision>
</reducedcouplings>

</lilithinput>
"""
    myInput = {'mh': mh, 'CV': CV, 'CF': CF, 'precision': precision}

    return myInputTemplate % myInput


def getL(CV, CF):
    myXML_user_input = usrXMLinput(mh=myhmass, CV=CV, CF=CF, precision=myprecision)
    lilithcalc.computelikelihood(userinput=myXML_user_input)
    return lilithcalc.l


######################################################################
# Calculations
######################################################################

# Initialize a Lilith object
lilithcalc = lilith.Lilith(verbose, timer)
# Read experimental data
lilithcalc.readexpinput(myexpinput)

print "***** initializing (CV, CF) model fit *****"
# Initialize the fit; parameter starting values and limits
m = Minuit(getL, CV=1, limit_CV=(CV_min, CV_max), CF=1, limit_CF=(CF_min, CF_max), print_level=0, errordef=1, error_CV=1, error_CF=1)

print "***** performing (CV, CF) model fit *****"
# Fit the model
m.migrad()
# Display parameter values at the best-fit point
print "\nBest-fit point of the (CV, CF) model: "
print "CV =", m.values["CV"], ", CF =", m.values["CF"], "\n"

# Prepare output
fresults = open(output, 'w')

m2logLmin = 10000
max = -1

print "***** running scan *****"

for CV in np.linspace(CV_min, CV_max, grid_subdivisions):
    fresults.write('\n')
    for CF in np.linspace(CF_min, CF_max, grid_subdivisions):
        myXML_user_input = usrXMLinput(myhmass, CV=CV, CF=CF, precision=myprecision)
        lilithcalc.computelikelihood(userinput=myXML_user_input)
        m2logL = lilithcalc.l
        if m2logL < m2logLmin:
            m2logLmin = m2logL
            CVmin = CV
            CFmin = CF
        fresults.write('%.5f    ' % CV + '%.5f    ' % CF + '%.5f     ' % m2logL + '\n')

fresults.close()

print "***** scan finalized *****"

######################################################################
# Plot routine
######################################################################


print "***** plotting *****"

# Preparing plot
matplotlib.rcParams['xtick.major.pad'] = 15
matplotlib.rcParams['ytick.major.pad'] = 15

fig = plt.figure()
ax = fig.add_subplot(111)

plt.minorticks_on()
plt.tick_params(labelsize=20, length=14, width=2)
plt.tick_params(which='minor', length=7, width=1.2)

# Getting the data
data = np.genfromtxt(output)

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Substracting the -2LogL minimum to form Delta(-2LogL)
z2 = []
for z_el in z:
    z2.append(z_el - z.min())

# Interpolating the grid
xi = np.linspace(x.min(), x.max(), grid_subdivisions)
yi = np.linspace(y.min(), y.max(), grid_subdivisions)

X, Y = np.meshgrid(xi, yi)
Z = griddata(x, y, z2, xi, yi, interp="linear")


# Plotting the 68%, 95% and 99.7% CL contours
ax.contour(xi, yi, Z, [2.3, 5.99, 11.83], linewidths=[2.5, 2.5, 2.5], colors=["#B22222", "#FF8C00", "#FFD700"])
cax = ax.imshow(Z, vmin=0, vmax=20, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], \
                aspect=(CV_max - CV_min) / (CF_max - CF_min), cmap=plt.get_cmap("rainbow"))

# Title, labels, color bar...
plt.title("  Lilith-" + str(lilith.__version__) + ", DB " + str(lilithcalc.dbversion), fontsize=14.5, ha="left")
plt.xlabel(r'$C_V$', fontsize=25)
plt.ylabel(r'$C_F$', fontsize=25)

plt.tight_layout()
cbar = fig.colorbar(cax)
cbar.set_label(r"$\Delta(-2\log L)$", fontsize=25)

#Plot experiment data

f = open('Lilith-Run2database/08_01/ATLAS_HIGG-2016-22_18b_68.txt', 'r')
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
locx = textt[0]
locy = textt[1]
plt.plot(locx,locy,'.',markersize=3, color = 'blue')

f = open('Lilith-Run2database/08_01/ATLAS_HIGG-2016-22_18b_95.txt', 'r')
text = [[float(num) for num in line.split()] for line in f]
f.close()
text = np.array(text)
textt = np.transpose(text)
locx = textt[0]
locy = textt[1]
plt.plot(locx,locy,'.',markersize=3, color = 'blue')


# Saving figure (.pdf)
plt.savefig(outputplot)



