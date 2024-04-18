# Capstone-Project-
#Simulation of action potentials utilizing the Hodgkin-Huxley Model.
#Shared w Lab Partner
#Part 1 consists of writing the 4 differential equations of the model

from scipy.optimize import bisect
import numpy as np
import scipy as sc
from scipy.optimize import fsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.constants as cnst
%matplotlib inline

def alpha_rtconsts(V): #Defines the alpha values for n, m, and h
    num_n = 10 - V
    num_m = 25 - V
    
    ah = 0.07*np.exp(-V/20)
    am = 0.1*(num_m)/(np.exp(num_m/10)-1)
    an = 0.01*(num_n)/(np.exp(num_n/10)-1)

    return ah, am, an

def beta_rtconsts(V): #Defines the beta values for n, m, and h, 
    num = 30-V
    
    bh = 1/(np.exp(num/10)+1)
    bm = 4*np.exp(-V/18)
    bn  = 0.125*np.exp(-V/80)

    return bh, bm, bn

def hod_hux(V, n, m, h, Iext): #This defines all the Hodgkin-huxley equations
    
    #The following values are determined from previous experimental data.
    gNa = 120 #mS/cm^2 (maximal Sodium conductance)
    gK = 36 #mS/cm^2 (maximal Potassium conductance)
    gL = 0.3 #mS/cm^2 (Leak conductance)
    ENa = 50 #mV (sodium equilibrium potential)
    EK = -77 #mV (potassium equilibrium potential)
    EL = -54.4 #mV (leak equilibrium potential)
    Cm = 1 #mu*F/cm^2 (Membrane capacitance)

    ah, am, an = alpha_rtconsts(V) 
    bh, bm, bn = beta_rtconsts(V)

    Ina = gNa*(m**3)*h*(V - ENa)
    Ik = gK*(n**4)*(V - EK)
    Il = gL*(V - EL)

    dVdt = (Iext - Ina - Ik - Il)/Cm #Defines the function for membrane potential w/ respect to time (dV_m/dt).

    dndt = an*(1-n) - bn*n #Defines the Potassium activation gating variable (dn/dt)
    dmdt = am*(1-m) - bm*m #Defines the Sodium activation gating variable (dm/dt)
    dhdt = ah*(1-h) - bh*h #Defines the Sodium inactivation gating variable (dh/dt)
    
    return dVdt, dndt, dmdt, dhdt
    
def m_vals(am, bm): #defines the value of m (sodium activation)
    return am/(am + bm)

def n_vals(an, bn): #defines the value of n (for potassium)
    return an/(an + bn)

def h_vals(ah, bh): #defines the value of h (Sodium inactivation)
    return ah/(ah + bh)

#the returned values are alpha, beta for h, m, and n
ah, am, an = alpha_rtconsts( , , )
bh, bm, bn = beta_rtconsts( , , )

#Input alpha and beta values to get the values for m, n, and h.
m = m_vals(am, bm)
n = n_vals(an, bn)
h = h_vals(ah, bh)

#We need the rest potential for the axon to start with (-65 mV).
#Define initial values for n, m and h using our rest potential.
#We need to define the external current Iext which depends on time. 
#Time can be defines using linspace(0, 50, N), 0 to 50 milliseconds with N points.
#We need V, m, n, and h values. which we can use from scipy library 'odeint'.
#We need odeint to integrate the four diff eqs.
#edit the code if you want
