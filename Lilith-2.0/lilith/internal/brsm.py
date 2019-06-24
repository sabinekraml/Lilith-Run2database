##########################################################################
#
#  This file is part of Lilith
#  made by J. Bernon and B. Dumont
#  extended by LE Duc Ninh (LDN, leducninh@gmail.com)
#
#  Web page: http://lpsc.in2p3.fr/projects-th/lilith/
#
#  In case of questions email bernon@lpsc.in2p3.fr 
#
#
#    Lilith is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Lilith is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Lilith.  If not, see <http://www.gnu.org/licenses/>.
#
##########################################################################

import os
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

wdir = '/'.join(os.path.realpath(__file__).split("/")[:-1])+'/Grids/'

def getBRfunctions(spline_deg=3):
    #### SM BR(h --> ff) ####
    BR_ferm_file = open(wdir+'BR_fermions.dat')
    BR_ferm_file.next()
    BR_ferm_file.next()
    ferm_grid = {"bb": [], "tautau": [], "cc": [], "mumu": [], "ss": []}
    hmass = []
    for line in BR_ferm_file:
        line = line.strip("\n").split()
        hmass.append(float(line[0]))
        ferm_grid["bb"].append(float(line[1]))
        ferm_grid["tautau"].append(float(line[4]))
        ferm_grid["cc"].append(float(line[10]))
        ferm_grid["mumu"].append(float(line[7]))
        ferm_grid["ss"].append(float(line[13]))

    BR_bb = UnivariateSpline(hmass, ferm_grid["bb"], k=spline_deg, s=0)
    BR_tautau = UnivariateSpline(hmass, ferm_grid["tautau"], k=spline_deg, s=0)
    BR_cc = UnivariateSpline(hmass, ferm_grid["cc"], k=spline_deg, s=0)
    BR_mumu = UnivariateSpline(hmass, ferm_grid["mumu"], k=spline_deg, s=0)
    BR_ss = UnivariateSpline(hmass, ferm_grid["ss"], k=spline_deg, s=0)

    BR_ferm_file.close()

    #### SM BR(h --> VV) ####
    BR_gauge_file = open(wdir+'BR_gauge.dat')
    BR_gauge_file.next()
    BR_gauge_file.next()
    gauge_grid = {"gg": [], "gammagamma": [], "Zgamma": [], "WW": [], "ZZ": []}
    hmass = []
    for line in BR_gauge_file:
        line = line.strip("\n").split()
        hmass.append(float(line[0]))
        gauge_grid["gg"].append(float(line[1]))
        gauge_grid["gammagamma"].append(float(line[4]))
        gauge_grid["Zgamma"].append(float(line[7]))
        gauge_grid["WW"].append(float(line[10]))
        gauge_grid["ZZ"].append(float(line[13]))

    BR_gg = UnivariateSpline(hmass, gauge_grid["gg"], k=spline_deg, s=0)
    BR_gammagamma = UnivariateSpline(hmass, gauge_grid["gammagamma"], k=spline_deg, s=0)
    BR_Zgamma = UnivariateSpline(hmass, gauge_grid["Zgamma"], k=spline_deg, s=0)
    BR_WW = UnivariateSpline(hmass, gauge_grid["WW"], k=spline_deg, s=0)
    BR_ZZ = UnivariateSpline(hmass, gauge_grid["ZZ"], k=spline_deg, s=0)
    BR_gauge_file.close()

    BR = {"bb": BR_bb, "cc": BR_cc, "tautau": BR_tautau, "mumu": BR_mumu,
          "ss": BR_ss, "WW": BR_WW, "ZZ": BR_ZZ, "gammagamma": BR_gammagamma, "gg": BR_gg,
          "Zgamma": BR_Zgamma}

    return BR

# begin LDN added
def geteffVVHfunctions(sqrts, spline_deg=3):
    #### SM efficiencies h --> WW & h --> ZZ ####
    WHZH_xsec_file = open(wdir+'WH_ZH_ZHgg_VBF_xsec'+str(sqrts)+'.dat')
    eff_grid = {"WH": [], "ZH": [], "ZHgg": [], "VBF": [], "ZHall": [], "VH": []}
    hmass = []
    for line in WHZH_xsec_file:
        line = line.strip("\n").split()
        hmass.append(float(line[0]))
        eff_grid["WH"].append(float(line[1])/(float(line[1])+float(line[2])+float(line[3])))
        eff_grid["ZH"].append(float(line[2])/(float(line[2])+float(line[3])))
        eff_grid["ZHgg"].append(float(line[3])/(float(line[2])+float(line[3])))
        eff_grid["ZHall"].append((float(line[2])+float(line[3]))/(float(line[1])+float(line[2])+float(line[3])))
        eff_grid["VH"].append((float(line[1])+float(line[2])+float(line[3]))/(float(line[1])+float(line[2])+float(line[3])+float(line[4])))
        eff_grid["VBF"].append(float(line[4])/(float(line[1])+float(line[2])+float(line[3])+float(line[4])))    

    eff_WH = UnivariateSpline(hmass, eff_grid["WH"], k=spline_deg, s=0)
    eff_ZH = UnivariateSpline(hmass, eff_grid["ZH"], k=spline_deg, s=0)
    eff_ZHgg = UnivariateSpline(hmass, eff_grid["ZHgg"], k=spline_deg, s=0)
    eff_ZHall = UnivariateSpline(hmass, eff_grid["ZHall"], k=spline_deg, s=0)
    eff_VH = UnivariateSpline(hmass, eff_grid["VH"], k=spline_deg, s=0)
    eff_VBF = UnivariateSpline(hmass, eff_grid["VBF"], k=spline_deg, s=0)
    WHZH_xsec_file.close()

    effWZ = {"eff_WH": eff_WH, "eff_ZH": eff_ZH, "eff_ZHgg": eff_ZHgg, "eff_VBF": eff_VBF, "eff_ZHall": eff_ZHall, "eff_VH": eff_VH}

    return effWZ

def getefftopfunctions(sqrts, spline_deg=3):
    #### SM efficiencies pp > tHq, tHW, ttH ####
    top_xsec_file = open(wdir+'tHq_tHW_ttH_xsec'+str(sqrts)+'.dat')
    eff_grid = {"tHq": [], "tHW": [], "tH": [], "ttH": []}
    hmass = []
    for line in top_xsec_file:
        line = line.strip("\n").split()
        hmass.append(float(line[0]))
        eff_grid["tHq"].append(float(line[1])/(float(line[1])+float(line[2])))
        eff_grid["tHW"].append(float(line[2])/(float(line[1])+float(line[2])))
        eff_grid["tH"].append((float(line[1])+float(line[2]))/(float(line[1])+float(line[2])+float(line[3])))
        eff_grid["ttH"].append(float(line[3])/(float(line[1])+float(line[2])+float(line[3])))

    eff_tHq = UnivariateSpline(hmass, eff_grid["tHq"], k=spline_deg, s=0)
    eff_tHW = UnivariateSpline(hmass, eff_grid["tHW"], k=spline_deg, s=0)
    eff_tH = UnivariateSpline(hmass, eff_grid["tH"], k=spline_deg, s=0)
    eff_ttH = UnivariateSpline(hmass, eff_grid["ttH"], k=spline_deg, s=0)
    top_xsec_file.close()

    efftop = {"eff_tHq": eff_tHq, "eff_tHW": eff_tHW, "eff_tH": eff_tH, "eff_ttH": eff_ttH}

    return efftop
# end LDN added
