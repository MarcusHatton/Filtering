# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:38:23 2024

@author: marcu
"""

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import glob
import os
import pickle
#from os import system
import sys as system

def getFourierTrans(model, u):
    """
    Returns the 1D discrete fourier transform of the variable u along the x-direction
    ready for the power spectrum method.
    Parameters
    ----------
    intPlot : object
        interactivePlot object containing all the simulation data, normally the final instance
    u : ndarray
        Two dimensional array of the variable we want the power spectrum of
    Returns
    -------
    uhat : array (N,)
        Fourier transform of u
    """
    nx, ny = model.domain_vars['nx'], model.domain_vars['ny']
    NN = nx // 2
    uhat = np.zeros((NN, ny), dtype=np.complex_)

    #print(nx, ny)
    #print(u.shape)

    for k in range(NN):
        for y in range(ny):
            # Sum over all x adding to uhat
            for i in range(nx):
                #print(np.exp(-(2*np.pi*1j*k*i)/nx))
                uhat[k, y] += u[i, y] * np.exp(-(2*np.pi*1j*k*i)/nx)

    return uhat / nx

def getPowerSpectrumSq(model, u):
    """
    Returns the integrated power spectrum of the variable u, up to the Nyquist frequency = nx/2
    Parameters
    ----------
    intPlot : object
        interactivePlot object containing all the simulation data, normally the final instance
    u : ndarray
        Two dimensional array of the variable we want the power spectrum of
    """
    NN = model.domain_vars['nx'] // 2
    #dy = model.domain_vars['dy']
    ny = model.domain_vars['ny']
    dy = 0.00125
    uhat = getFourierTrans(model, u)
    P = np.zeros(NN)

    print(uhat.shape)
    print(dy)

    for k in range(NN):
        for j in range(ny):
            P[k] += (np.absolute(uhat[k, j])**2) * dy

    P = P / np.sum(P)
    return P

def GetKESF(model):
    """
    Retrieves and computes the kinetic energy density for each frame in a single fluid animation.
    Parameters
    ----------
    anim : object
        animation class containing all user def variables
    frame : Array
        Frame from the animation class containing all user def variables at the time we want
    """
    Nx = model.domain_vars['nx'] // 2
    vx = model.prim_vars['v1'][2]
    vy = model.prim_vars['v2'][2]
    rho = model.prim_vars['rho'][2]
    vsq = vx**2 + vy**2
    W = 1 / np.sqrt(1 - vsq)
    KE = rho * W * (W-1) ## this expression ??
    print(KE)
    return KE

if __name__ == "__main__":
    

    # Create and setup micromodel
    with open('./Output/t_998_1002/800x1600/Pickles/IdealHydro2D.pickle', 'rb') as filehandle:
        model = pickle.load(filehandle)
    
    ISSpecIdeal = 0
    ISSpecIdeal = getPowerSpectrumSq(model, GetKESF(model))

    with open('./KESpec_IHD_800x1600.pickle', 'wb') as filehandle:
        pickle.dump(ISSpecIdeal, filehandle)
    
    fig = plt.figure(figsize=(4.0,3.0), dpi=1200, layout='constrained')
    Nx = model.domain_vars['nx'] // 2
    plt.loglog(np.arange(1, Nx+1), np.arange(1, Nx+1)*ISSpecIdeal)
    #plt.loglog([1, Nx], [ISSpecIdeal[0], ISSpecIdeal[0]*(Nx**(-5/3))], 'k--')
    #plt.loglog([5, 60], [ISSpecIdeal[20], ISSpecIdeal[20]*(12**(-5/3))], 'k--')
    plt.loglog(np.arange(1, Nx+1), (np.arange(1, Nx+1)**(5/3))*np.arange(1, Nx+1)*ISSpecIdeal, 'k--')
    plt.annotate(r'$k^{-5/3}$', xy=(20, 0.01), fontsize=15)
    plt.ylabel(r"$k|P_{T}(k)|^2$", {'fontsize':'large'})
    plt.xlabel(r'$k$')
    plt.savefig('./800x1600_Testing_Pickle_t=10.pdf')
    plt.close()































