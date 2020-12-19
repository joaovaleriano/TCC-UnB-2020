#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *************************************************************************
#
# Physics Institute, University of Brasília, Brasil
# João Pedro Valeriano Miranda
# 
# Program related to calculations of my Undergraduate Thesis
# 
# Simulation of the discretized Langevin Equation with harmonic potential
# and noise constant by intervals.
# *************************************************************************

# Useful packages
import numpy as np # Arrays
import matplotlib.pyplot as plt # Plotting
from scipy import linalg # Linear Algebra methods

# Plot font size
plt.rcParams.update({"font.size":30})

# Arrays to store position and velocity
x = np.zeros((1000,2,10000))

dt = 0.01 # # Size of constant noise interval
k=1 # Harmonic potential coefficeint
b=1 # Drag coefficient
m=1 # Mass

# System matrix
A = np.array([[0, 1],[-k/m, -b/m]])

# Noise (normalized by square root of time step) divided by mass
eps0 = np.random.uniform(-1, 1, (1000,2,10000))/np.sqrt(dt)/m

# Initial condition
eps0[:,0,:] = 0

##################### Transformations for discretization ######################

expA = linalg.expm(A*dt)

d, p = linalg.eig(A)
d = np.diag(d)
p1 = linalg.inv(p)

eps = np.matmul(np.matmul(np.linalg.inv(A),expA-np.eye(2)),eps0)

a1, a2 = np.matmul(np.linalg.inv(A),expA-np.eye(2))[:,1]

for i in range(eps0.shape[0]):
    
    eps[i] = np.matmul(np.linalg.inv(A),expA-np.eye(2))[:,1].reshape((2,1)) * eps0[i,1]

##############################################################################

# Evolution of the system
for i in range(x.shape[0]-1):
    
    x[i+1] = np.matmul(expA,x[i]) + eps[i+1]

################ Plotting position and velocity distributions ################

plt.figure(figsize=(30,23))

plt.subplot(2,2,1)
h, bins = np.histogram(x[-1,0], 30, density=True)
plt.fill_between(bins[:-1], h, lw=0, step="post", label="Simulação")
t = np.linspace(np.min(x[-1,0]), np.max(x[-1,0]), 1000)
plt.plot(t, np.exp(-t**2/2/np.var(x[-1,0]))/np.std(x[-1,0])/np.sqrt(2*np.pi), color="red", lw=6, label="Ajuste Gaussiano")
plt.xlabel(r"$x_{1}$")
plt.ylabel(r"Distribuição limite de $x_{1}$")
plt.legend(loc="center right")
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))


plt.subplot(2,2,2)
h, bins = np.histogram(x[-1,1], 30, density=True)
plt.fill_between(bins[:-1], h, lw=0, step="post", label="Simulação")
t = np.linspace(np.min(x[-1,1]), np.max(x[-1,1]), 1000)
plt.plot(t, np.exp(-t**2/2/np.var(x[-1,1]))/np.std(x[-1,1])/np.sqrt(2*np.pi), color="red", lw=6, label="Ajuste Gaussiano")
plt.xlabel(r"$x_{2}$")
plt.ylabel(r"Distribuição limite de $x_{2}$")
plt.legend(loc="center right")
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

##############################################################################

# Discrete time array 
n = np.linspace(0, x.shape[0]-1, x.shape[0])

# Plot variances evolution
if (b == 0):
    
    ########## Variance predictions (calculated with Mathematica: OS_fluid.nb) ############
    var1 = np.var(eps0[:,1,:])*( (p[0,0]*(a1*p1[0,0]+a2*p1[0,1]))**2*(1-np.exp(2*n*d[0,0]*dt))/(1-np.exp(2*d[0,0]*dt)) +
             (p[0,1]*(a1*p1[1,0]+a2*p1[1,1]))**2*(1-np.exp(2*n*d[1,1]*dt))/(1-np.exp(2*d[1,1]*dt)) +
             2*(p[0,0]*(a1*p1[0,0]+a2*p1[0,1]))*(p[0,1]*(a1*p1[1,0]+a2*p1[1,1]))*n )
    
    var2 = np.var(eps0[:,1,:])*( (p[1,0]*(a1*p1[0,0]+a2*p1[0,1]))**2*(1-np.exp(2*n*d[0,0]*dt))/(1-np.exp(2*d[0,0]*dt)) +
             (p[1,1]*(a1*p1[1,0]+a2*p1[1,1]))**2*(1-np.exp(2*n*d[1,1]*dt))/(1-np.exp(2*d[1,1]*dt)) +
             2*(p[1,0]*(a1*p1[0,0]+a2*p1[0,1]))*(p[1,1]*(a1*p1[1,0]+a2*p1[1,1]))*n )
     
    # Plotting
    plt.subplot(2,2,3)
    plt.plot(np.var(x[:,0,:], axis=-1), lw=6, label="Simulação")
    plt.plot(var1, linestyle="--", color="red", lw=6, label="Previsão")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\left\langle x_{1}[n] \right\rangle$")
    plt.legend()
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    
    plt.subplot(2,2,4)
    plt.plot(np.var(x[:,1,:], axis=-1), lw=6, label="Simulação")
    plt.plot(var2, linestyle="--", color="red", lw=6, label="Previsão")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\left\langle x_{2}[n] \right\rangle$")
    plt.legend()
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

else:
    
    ########## Variance predictions (calculated with Mathematica) ############
    var1 = np.var(eps0[:,1,:])*( (p[0,0]*(a1*p1[0,0]+a2*p1[0,1]))**2*(1-np.exp(2*n*d[0,0]*dt))/(1-np.exp(2*d[0,0]*dt)) +
              (p[0,1]*(a1*p1[1,0]+a2*p1[1,1]))**2*(1-np.exp(2*n*d[1,1]*dt))/(1-np.exp(2*d[1,1]*dt)) +
              2*(p[0,0]*(a1*p1[0,0]+a2*p1[0,1]))*(p[0,1]*(a1*p1[1,0]+a2*p1[1,1]))*(1-np.exp(n*np.trace(d)*dt))/(1-np.exp(np.trace(d)*dt)) )
    
    var1_lim = np.var(eps0[:,1,:])*( (p[0,0]*(a1*p1[0,0]+a2*p1[0,1]))**2/(1-np.exp(2*d[0,0]*dt)) +
             (p[0,1]*(a1*p1[1,0]+a2*p1[1,1]))**2/(1-np.exp(2*d[1,1]*dt)) +
             2*(p[0,0]*(a1*p1[0,0]+a2*p1[0,1]))*(p[0,1]*(a1*p1[1,0]+a2*p1[1,1]))*n )
    
    var2 = np.var(eps0[:,1,:])*( (p[1,0]*(a1*p1[0,0]+a2*p1[0,1]))**2*(1-np.exp(2*n*d[0,0]*dt))/(1-np.exp(2*d[0,0]*dt)) +
              (p[1,1]*(a1*p1[1,0]+a2*p1[1,1]))**2*(1-np.exp(2*n*d[1,1]*dt))/(1-np.exp(2*d[1,1]*dt)) +
              2*(p[1,0]*(a1*p1[0,0]+a2*p1[0,1]))*(p[1,1]*(a1*p1[1,0]+a2*p1[1,1]))*(1-np.exp(n*np.trace(d)*dt))/(1-np.exp(np.trace(d)*dt)) )
    
    var2_lim = np.var(eps0[:,1,:])*( (p[1,0]*(a1*p1[0,0]+a2*p1[0,1]))**2/(1-np.exp(2*d[0,0]*dt)) +
             (p[1,1]*(a1*p1[1,0]+a2*p1[1,1]))**2/(1-np.exp(2*d[1,1]*dt)) +
             2*(p[1,0]*(a1*p1[0,0]+a2*p1[0,1]))*(p[1,1]*(a1*p1[1,0]+a2*p1[1,1]))*n )
    
    # Plotting
    plt.subplot(2,2,3)
    plt.plot(np.var(x[:,0,:], axis=-1), lw=6, label="Simulação")
    plt.plot(var1, linestyle="--", color="red", lw=6, label="Previsão")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\left\langle x_{1}[n] \right\rangle$")
    plt.legend()
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    
    plt.subplot(2,2,4)
    plt.plot(np.var(x[:,1,:], axis=-1), lw=6, label="Simulação")
    plt.plot(var2, linestyle="--", color="red", lw=6, label="Previsão")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\left\langle x_{2}[n] \right\rangle$")
    plt.legend()
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

# plt.savefig(r"OS_fluid_%s_%s_%s_%s.pdf" % (b,k,m,dt), format="pdf", bbox_inches="tight")
