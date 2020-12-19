# TCC-UnB-2020
Scripts related to my undergraduate final project on discrete time stochastic processes as a Physics Student at Universidade de Brasília (UnB)

## PID omega coefficients
Function to calculate the coefficients of the omega function power series from the statistical moments of a random variable for the sum of identically distributed reduced random variables.

**PID_omega_coeffs.py** calculates new coefficients, which are saved to **Kcoeffs_list.txt**, so they never have to be calculated again.

## Simulations
We simulate the discretized langevin equation on **langevin.py**, which plots the final position and velocity distributions and their variances evolutions. **OS_fluid.py** does the same, considering the addition of a harmonic potential to the langevin equation. The necessary calculations are a bit involved and are developed with Mathematica (**OS_fluid.nb**).
