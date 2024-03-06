# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:21:02 2022

@author: mjh1n20
"""

from multiprocessing import Process, Pool
import os
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer
import h5py
from scipy.interpolate import interpn
from scipy.optimize import root, minimize
#from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp, quad
import cProfile, pstats, io
import math

class Base(object):

    @staticmethod
    def Mink_dot(vec1,vec2):
        """
        Parameters:
        -----------
        vec1, vec2 : list of floats (or np.arrays)

        Return:
        -------
        mink-dot (cartesian) in 1+n dim
        """
        if len(vec1) != len(vec2):
            print("The two vectors passed to Mink_dot are not of same dimension!")

        dot = -vec1[0]*vec2[0]
        for i in range(1,len(vec1)):
            dot += vec1[i] * vec2[i]
        return dot
  
    @staticmethod
    def get_rel_vel(spatial_vels):
        """
        Build unit vectors starting from spatial components
        Needed as this will enter the minimization procedure

        Parameters:
        ----------
        spatial_vels: list of floats

        Returns:
        --------
        list of floats: the d+1 vector, normalized wrt Mink metric
        """
        W = 1 / np.sqrt(1-np.sum(spatial_vels**2))
        return W * np.insert(spatial_vels,0,1.0)

    @staticmethod
    def project_tensor(vector1_wrt, vector2_wrt, to_project):
        return np.inner(vector1_wrt,np.inner(vector2_wrt,to_project))
    
    @staticmethod
    def orthogonal_projector(u, metric):
        return metric + np.outer(u,u)    


    """
    A pair of functions that work in conjuction (thank you stack overflow).
    find_nearest returns the closest value to 'value' in 'array',
    find_nearest_cell then takes this closest value and returns its indices.
    Should now work for any dimensional data.
    """
    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]

    @staticmethod   
    def find_nearest_cell(point, points):
        if len(points) != len(point):
            print("find_nearest_cell: The length of the coordinate vector\
                   does not match the length of the coordinates.")
        positions = []
        for dim in range(len(point)):
            positions.append(Base.find_nearest(points[dim], point[dim]))
        return [np.where(points[i] == positions[i])[0][0] for i in range(len(positions))]
    
    @staticmethod
    def clip_flat(self, data, mod=False, log=False):
        if mod and log:
            for i in range(data.shape[0]):
                data[i] = np.abs(data[i])
                data[i] = np.log10(data[i])
        if log and not mod:
            for i in range(data.shape[0]):
                sign = np.sign(data[i])
                print(sign)
                data[i] = sign*np.log10(np.abs(data[i]))
        if mod and not log:
            for i in range(data.shape[0]):
                data[i] = np.abs(data[i])
        #return data

    @staticmethod
    def clip(self, data, mod=False, log=False):
        if mod and log:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data[i,j] = np.abs(data[i,j])
                    data[i,j] = np.log10(data[i,j])
        if log and not mod:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data[i,j] = np.log10(data[i,j])
        if mod and not log:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data[i,j] = np.abs(data[i,j])
        #return data

    @staticmethod
    def trim_flat(self, y_data, x_data, x_min_max, y_min_max):
        x_min, x_max = x_min_max[0], x_min_max[1]
        y_min, y_max = y_min_max[0], y_min_max[1]
        i = 0
        while i < y_data.shape[0]:
            if x_data[i] < x_min or x_data[i] > x_max\
            or y_data[i] < y_min or y_data[i] > y_max:
                y_data = np.delete(y_data, i)
                x_data = np.delete(x_data, i)
            elif x_data[i] > x_max:
                y_data = np.delete(y_data, i)
                x_data = np.delete(x_data, i)
            else:
                i += 1
        return y_data, x_data

    @staticmethod
    def trim(self, y_data, x_data, x_minimum):
        for i in range(y_data.shape[0]):
            for j in range(y_data.shape[1]):
                if x_data[i,j] < x_minimum:
                    y_data = np.delete(y_data, [i,j])
                    x_data = np.delete(x_data, [i,j])
        #return y_data, x_data
        
    @staticmethod
    def getFourierTrans(u, nx, ny):
        """
        Returns the 1D discrete fourier transform of the variable u along both 
        the x and y directions ready for the power spectrum method.
        Parameters
        ----------
        u : ndarray
            Two dimensional array of the variable we want the power spectrum of
        Returns
        -------
        uhat : array (N,)
            Fourier transform of u
        """
       
        NN = nx // 2
        uhat_x = np.zeros((NN, ny), dtype=np.complex_)
    
        for k in range(NN):
            for y in range(ny):
                # Sum over all x adding to uhat
                for i in range(nx):
                    uhat_x[k, y] += u[i, y] * np.exp(-(2*np.pi*1j*k*i)/nx)

        NN = ny // 2
        uhat_y = np.zeros((NN, nx), dtype=np.complex_)
        
        for k in range(NN):
            for x in range(nx):
                # Sum over all y adding to uhat
                for i in range(ny):
                    uhat_y[x, k] += u[x, i] * np.exp(-(2*np.pi*1j*k*i)/ny)

        return (uhat_x / nx), (uhat_y / ny) 
    
    
    def profile(self, fnc):
        """A decorator that uses cProfile to profile a function"""
        def inner(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = fnc(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            return retval
        return inner
