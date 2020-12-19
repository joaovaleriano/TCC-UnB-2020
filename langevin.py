#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *************************************************************************
#
# Physics Institute, University of Brasília, Brasil
# João Pedro Valeriano Miranda
# 
# Program related to calculations of my Undergraduate Thesis
# 
# Simulation of the discretized Langevin Equation with noise constant
# by intervals.
# *************************************************************************

# Useful packages
import numpy as np # Arrays
import matplotlib.pyplot as plt # Plotting
from scipy import linalg # Linear Algebra methods

# Plot font size
plt.rcParams.update({"font.size":30})

# Arrays to store position and velocity
x = np.zeros((1000,10000))
v = np.zeros((1000,10000))

dt=1 # Size of constant noise interval
b=2 # Drag coefficient
m=1 # Mass

# Noise (normalized by square root of time step)
eps0 = np.random.uniform(-1,1,v.shape)/np.sqrt(dt)

# Discretization transformation on noise
eps = eps0 * (1-np.exp(-b/m*dt))/b

# System evolution
for i in range(x.shape[0]-1):
    
    v[i+1] = np.exp(-b/m*dt)*v[i] + eps[i+1]
    x[i+1] = x[i] + (1-np.exp(-b/m*dt))/(b/m)*v[i] + (dt-(1-np.exp(-b/m*dt))/(b/m))/b*eps0[i+1]

###############################################################################

# Plotting final distributions and variance evolutions for position and velocity

# Final position distribution
plt.subplot(2,2,1)
h, bins = np.histogram(x[-1], 30, density=True)
plt.fill_between(bins[:-1], h, lw=0, step="post", label="Simulação")
t = np.linspace(np.min(x[-1]), np.max(x[-1]), 1000)
plt.plot(t, np.exp(-t**2/2/np.var(x[-1]))/np.std(x[-1])/np.sqrt(2*np.pi), color="red", lw=6, label="Ajuste Gaussiano")
plt.xlabel(r"$x_{1}$")
plt.ylabel(r"Distribuição limite de $x_{1}$")
plt.legend(loc="center right")
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

# Final velocity distribution
plt.figure(figsize=(30,23))
plt.subplot(2,2,2)
h, bins = np.histogram(v[-1], 30, density=True)
plt.fill_between(bins[:-1], h, lw=0, step="post", label="Simulação")
t = np.linspace(np.min(v[-1]), np.max(v[-1]), 1000)
plt.plot(t, np.exp(-t**2/2/np.var(v[-1]))/np.std(v[-1])/np.sqrt(2*np.pi), color="red", lw=6, label="Ajuste Gaussiano")
plt.xlabel(r"$x_{2}$")
plt.ylabel(r"Distribuição limite de $x_{2}$")
plt.legend(loc="center right")
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

# Position variance evolution
plt.subplot(2,2,3)
plt.plot(np.var(x, axis=-1), lw=6, label="Simulação")
t = np.linspace(0, x.shape[0]-1, x.shape[0])
svar = ( ((1-np.exp(-b/m*dt))/b)**2*(1-np.exp(-2*t*b/m*dt))/(1-np.exp(-2*b/m*dt)) - 2*dt/b/m*(1-np.exp(-b*t/m*dt)) +
        (dt/m)**2*t )*np.var(eps0)*(m/b)**2
plt.plot(t, svar, "--", color="red", lw=6, label="Previsão")
plt.plot(( ((1-np.exp(-b/m*dt))/b)**2/(1-np.exp(-2*b/m*dt)) - 2*dt/b/m +
        (dt/m)**2*t )*np.var(eps0)*(m/b)**2, linestyle=":", lw=6, color="black", label=r"$t \rightarrow \infty$")
plt.xlabel(r"$n$")
plt.ylabel(r"$\left\langle x_{1}[n] \right\rangle$")
plt.legend()
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

# Velocity variance evolution
plt.subplot(2,2,4)
plt.plot(np.var(v, axis=-1), lw=6, label="Simulação")
t = np.linspace(0, v.shape[0]-1, v.shape[0])
svar = (1-np.exp(-2*b/m*t*dt))/(1-np.exp(-2*b/m*dt))*(1-np.exp(-b/m*dt))**2/b**2*np.var(eps0)
plt.plot(t, svar, "--", color="red", lw=6, label=r"Previsão")
plt.plot(np.ones(len(t))*np.var(eps)/(1-np.exp(-2*b/m*dt)), color="black", ls=":", lw=6, label=r"$t \rightarrow \infty$")
plt.xlabel(r"$n$")
plt.ylabel(r"$\left\langle x_{2}[n] \right\rangle$")
plt.legend()
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

plt.savefig(r"results_langevin.pdf", format="pdf", bbox_inches="tight")
