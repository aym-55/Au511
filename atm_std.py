#!/usr/bin/python
# coding: utf-8 
#########################
## Atmosphere standard ##
## et constantes       ##
## physiques           ##
#########################


from __future__ import unicode_literals
from matplotlib.pyplot import *
import control 
from control.matlab import *
from math import *
from scipy.interpolate import interp1d
from pylab import *
from matplotlib.widgets import Slider
import numpy as np
import scipy.interpolate

# acclération de la pesanteur (m/s^2) 
g0=9.81
#rayon de la Terre (m)
Rayon_terre=6370e3
# TP pilotage automatique
ft=0.3048 #feet to meters conversion

deg=pi/180.0
ussa76ut86=np.loadtxt('ussa76ut86.txt')
#    1            2 3 4  5                6             7                           8
# altitude (km)          temperature (K)  Pression (Pa) masse volumique (kg/m^3)  vitesse du son (m/s)




geoaltus76=ussa76ut86[:,0]*1000.0
# earth radius for geopotential altitude calculation
RE=6356000
# rayon moyen de la Terre (m)
R0=6356.766*1000

altus76=RE*geoaltus76/(RE+geoaltus76)
Tus76=ussa76ut86[:,4]
Pus76=ussa76ut86[:,5]
rhous76=ussa76ut86[:,6]
aus76=ussa76ut86[:,7]

# densité de l'air (kg/m**3) 	
rhoInterp=scipy.interpolate.interp1d(altus76,rhous76)
# vitesse du son (m/s)
aInterp=scipy.interpolate.interp1d(altus76,aus76)


def get_cte_atm(z):
    # altitude géopotentielle (m)
    hgeo=R0*z/(R0+z)
    # densité de l'air à l'altitude courante (kg/m**3)
    rho=rhoInterp(hgeo)
    # vitesse du son à l'altitude courante (m/s)
    a=aInterp(hgeo)
    return hgeo,rho,a

if(0):
    #rc('text', usetex=False)
    figure(1)
    plot(altus76,rhous76,'b',geoaltus76,rhous76,'r:',lw=2)
    grid(True)
    xlabel(r"Altitude (m)")
    ylabel(r"Densité de l'air (kg/m$^3$)")
    
    figure(2)
    plot(altus76,aus76,'b',geoaltus76,aus76,'r:',lw=2)
    grid(True)
    xlabel(r"Altitude (m)")
    ylabel(r"Vitesse du son (m/s)")

    figure(3)
    plot(altus76,altus76-geoaltus76,'b',lw=2)
    grid(True)
    xlabel('Geometric altitude')
    ylabel('Geometric altitude - geopotential altitude')
    show()
#end
