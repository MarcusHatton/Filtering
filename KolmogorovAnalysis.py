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

def getFourierTrans(intPlot, u):
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
    nx, ny = intPlot['Domain'].attrs['nx'][0], intPlot['Domain'].attrs['ny'][0]
    NN = nx // 2
    uhat = np.zeros((NN, ny), dtype=np.complex_)

    for k in range(NN):
        for y in range(ny):
            # Sum over all x adding to uhat
            for i in range(nx):
                uhat[k, y] += u[i, y] * np.exp(-(2*np.pi*1j*k*i)/nx)

    return uhat / nx

def getPowerSpectrumSq(intPlot, u):
    """
    Returns the integrated power spectrum of the variable u, up to the Nyquist frequency = nx/2
    Parameters
    ----------
    intPlot : object
        interactivePlot object containing all the simulation data, normally the final instance
    u : ndarray
        Two dimensional array of the variable we want the power spectrum of
    """
    NN = intPlot['Domain'].attrs['nx'][0] // 2
    dy = intPlot['Domain'].attrs['dy'][0]
    uhat = getFourierTrans(intPlot, u)
    P = np.zeros(NN)

    for k in range(NN):
        for j in range(intPlot['Domain'].attrs['ny'][0]):
            P[k] += (np.absolute(uhat[k, j])**2) * dy

    P = P / np.sum(P)
    return P

def GetKESF(frame):
    """
    Retrieves and computes the kinetic energy density for each frame in a single fluid animation.
    Parameters
    ----------
    anim : object
        animation class containing all user def variables
    frame : Array
        Frame from the animation class containing all user def variables at the time we want
    """
    Nx = frame['Domain'].attrs['nx'][0] // 2
    vx = frame['Primitive/v1'][:]
    vy = frame['Primitive/v2'][:]
    rho = frame['Primitive/rho']
    vsq = vx**2 + vy**2
    W = 1 / np.sqrt(1 - vsq)
    KE = rho * W * (W-1) ## this expression ??

    TE = gamma*m_0*c**2 = rho
    KE = gamma*m_0*c**2 - m_0*c**2 - e*n*c**2  = (gamma-1)*m_0*c**2 - e*n*c**2 = (W-1)*n - e*n

    return KE

if __name__ == "__main__":
    
    fs = []
    n_files = 5
    for n in range(n_files):
        #fs.append(h5py.File(f'./Filtering/KHI/Ideal/t_1998_2002/dp_800x1600x0_{n}.hdf5', 'r'))
        fs.append(h5py.File(f'/mainfs/filtering/Data/KHI/Ideal/t_2998_3002/800x1600/dp_800x1600x0_{n}.hdf5', 'r'))
    # print(fs)
    
    ISSpecIdeal = 0
    f = fs[2]
    ISSpecIdeal = getPowerSpectrumSq(f, GetKESF(f))
    
    Nx = f['Domain'].attrs['nx'][0] // 2
    plt.loglog(np.arange(1, Nx+1), np.arange(1, Nx+1)*ISSpecIdeal)
    #plt.loglog([5.0, 60.0], [5.0*10**-3, (5.0*10**-3)*(12.0**(-5/3))], 'k--')
    plt.loglog(np.arange(1, Nx+1), (np.arange(1, Nx+1)**(5/3))*np.arange(1, Nx+1)*ISSpecIdeal, 'k--')
    plt.annotate(r'$k^{-5/3}$', xy=(20, 0.01), fontsize=15)
    plt.ylabel(r"$k|P_{T}(k)|^2$", {'fontsize':'large'})
    plt.xlabel(r'$k$')
    plt.savefig('./KESpec_800x1600_t=30.pdf')
    plt.close()
































