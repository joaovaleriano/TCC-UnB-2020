#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *************************************************************************
#
# Physics Institute, University of Brasília, Brasil
# João Pedro Valeriano Miranda
# 
# Program related to calculations of my Undergraduate Thesis
# 
# Function to calculate the coefficients of the omega function power series
# from the statistical moments of a random variable for the sum of
# identically distributed reduced random variables.
#
# As it calculates the expression for each coefficient, the script saves it
# to a record file so we don't have to calculate it again, as it may take
# a long time.
#
# It may also calculate the coefficients for a specific distribution if
# the moments are provided.
# *************************************************************************

import numpy as np
from sympy import * # Symbolic math package.
from IPython.display import display, Latex

# Function to calculate the coefficients of the omega function power series
# 'n' is the number of coefficients to be calculated, and 'm' the list of
# statistical moments for the calculation for a spefic distribution if
# desired.

def K(n,m=[]): 
    
    # Opens file with saved expressions to save time
    Kcoeffs_file = open("Kcoeffs_list.txt", "r+")
    Kcoeffs = Kcoeffs_file.read().split("\n")[:-1]
    
    z = symbols("z") # Characteristic Function z variable.
    w = Function("omega") # omega function.
    nu = IndexedBase("nu") # Random Variable's Statistical Moments.
    psi = exp(-z**2/2*(1+w(z))) # Characteristic Function.
    
    ks = np.empty((n), dtype="object") # omega coefficients list
    
    i = 0 # Index of the coeff. to be calculated.
    j = 3 # Index of the derivative to be applied to 'psi', 'j' starts from 3
          # as it is the first derivative which is useful to calculate a coeff.
    
    while (ks[i] == None):
        
        # If we have the expression for K_i saved, we use it.
        if (i < len(Kcoeffs)):
            ks[i] = sympify(Kcoeffs[i], locals={"nu":IndexedBase("nu")})
            i += 1
        
        # If we don't have the expression yet, we calculatate it.
        else:
            
            # We know that w(0)=0 and put z=0 as the coefficients of a power series
            # are related to derivatives at zero.
            dpsi = diff(psi, z, j).subs(z,0).subs(w(0),0) # Expression of psi derivative.
            
            # Substitute known coefficients of lower order compared to the one to be calculated.
            for l in range(0, i):
                
                dpsi = dpsi.subs(diff(w(z),z,l+1).subs(z,0),I**(l+1)*ks[l])
            
            if ("omega" in str(dpsi)): # Make sure that there is a derivative in the expression
                                       # that will allow the calculation of a coeff.
                                       
                # Isolate the derivative associated to the coeff. to be calculated
                # and assigns it to the coefficients list.
                ks[i] = (-I)**(i+1)*solve(dpsi-I**(j)*nu[j], diff(w(z),z,i+1).subs(z,0))[0]
                i += 1 # As we calculate a coeff. we can jump to the next one to be calculated.
                
        if (i == n): # Stop when all the n first coefficients are calculated.
            break
        
        j += 1 # Take next derivative of 'psi' to calculate following coefficients.
    
    # Display beautifully the coefficients calculated.
    for i in range(len(ks)):
        display(Latex("$${} = {}$$".format(latex("K_{%i}" % (i+1)), latex(ks[i]))))        
        
        # Save new expressions to expressions record file.
        if (i >= len(Kcoeffs)):
            Kcoeffs_file.write(str(ks[i]))
            Kcoeffs_file.write("\n")
        
    Kcoeffs_file.close()
        
    # Substitutes the statistical moments, if provided.
    if (len(m) >= j):
        
        for i in range(len(ks)):
            
            for k in range(3,j+1):
        
                ks[i] = ks[i].subs(nu[k], m[k-1])
        
            ks[i] = np.float32(ks[i])
        
    return ks # Return coefficients list.

## Example
#K(3)