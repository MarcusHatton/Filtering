# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:02:05 2023

@author: marcu
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle
import seaborn as sns
from scipy.optimize import curve_fit

class CoefficientAnalysis(object):
    
    def __init__(self, visualizer):
        """
        Nothing as yet...

        Returns
        -------
        Also nothing...
        
        """
        self.visualizer = visualizer # need to re-structure this... or do I

    def RegPlot(self, model1, model2, y_var_str, x_var_str, t, x_range, y_range,
                  interp_dims, method, y_component_indices, x_component_indices,\
                  save_fig=False, save_dir='', x_trim=False, x_trim_vals=[-np.inf,np.inf],\
                  y_trim=False, y_trim_vals=[-np.inf,np.inf], order=1, seaborn_log_fit=False, robust=False,\
                  x_mod=False, x_log=False, y_mod=False, y_log=False, manual_fit=''):

	# Get raw data from models
        y_data, points = \
            self.visualizer.get_var_data(model1, y_var_str, t, x_range, y_range, interp_dims, method, y_component_indices)
        x_data, points = \
            self.visualizer.get_var_data(model2, x_var_str, t, x_range, y_range, interp_dims, method, x_component_indices)

        # Flatten it for ease of use
        y_data = y_data.flatten()
        x_data = x_data.flatten()

        # Trim the data's extrema
        if x_trim or y_trim:
            y_data, x_data = Base.trim_flat(y_data, x_data, x_min_max=x_trim_vals, y_min_max=y_trim_vals)

        # Take modulus and/or log of the data
        if y_mod or y_log:
            Base.clip_flat(y_data, mod=y_mod, log=y_log)
        if x_mod or x_log:
            Base.clip_flat(x_data, mod=x_mod, log=x_log)

        #robust = True
        #if sum((order > 1, robust, logx)) > 1:
        #    robust = False
        #    raise ValueError("Mutually exclusive regression options.")

        fig = plt.figure(figsize=(16,16))
        sns.regplot(x=x_data, y=y_data, color="#4CB391", order=order, logx=seaborn_log_fit)#, robust=robust, x_estimator=x_estimator, x_bins=x_bins)

        # Fitting a logarithmic function - TESTING
        if manual_fit == 'logarithmic':
            def func(x, p1, p2, p3, p4):
                return p1*np.log(p2*x + p3) + p4

            popt, pcov = curve_fit(func, x_data, y_data, p0=(1.0,1.0,0.0,-5.0))
            p1, p2, p3, p4 = popt[0], popt[1], popt[2], popt[3]
            #print(x_var_str)
            #print(p1, p2, p3, p4)

            x_curve=np.linspace(np.min(x_data),np.max(x_data),1000)
            y_curve=func(x_curve,p1,p2,p3,p4)
            plt.plot(x_curve, y_curve,'r', linewidth=5)

        # Fitting a logarithmic function - TESTING
        if manual_fit == 'linear':
            def func(x, p1, p2):
                return p1*x + p2
            #print(x_data, y_data)
            popt, pcov = curve_fit(func, x_data, y_data, p0=(1.0,1.0))
            p1, p2= popt[0], popt[1]
            #print(x_var_str)
            #print(p1, p2)

            x_curve=np.linspace(np.min(x_data),np.max(x_data),1000)
            y_curve=func(x_curve,p1,p2)
            plt.plot(x_curve, y_curve,'r', linewidth=5)

        # Fitting a logarithmic function - TESTING
        if manual_fit == 'power_law':
            def func(x, p1, p2, p3):
                return p1*x**p3 + p2

            popt, pcov = curve_fit(func, x_data, y_data, p0=(1.0,1.0,1.0))
            p1, p2, p3 = popt[0], popt[1], popt[2]
            #print(x_var_str)
            #print(p1, p2, p3)

            x_curve=np.linspace(np.min(x_data),np.max(x_data),1000)
            y_curve=func(x_curve,p1,p2,p3)
            plt.plot(x_curve, y_curve,'r', linewidth=5)


        plt.title(y_var_str+'('+x_var_str+')')
        fig.tight_layout()
        if save_fig:
            save_dir += 'RegPlot_'
            save_dir += y_var_str+str(y_component_indices)+'_'+x_var_str+str(x_component_indices)+'_'
            save_dir += method
            save_dir += '.pdf'
            plt.savefig(save_dir)
            plt.close(fig)
        else:
            print('Showing a figure...')
            plt.show()


    def JointPlot(self, model1, model2, y_var_str, x_var_str, t, x_range, y_range,\
                  interp_dims, method, y_component_indices, x_component_indices,
                  save_fig=False, save_dir='', clip=False, x_trim=False, x_trim_vals=[-np.inf,np.inf],
                  y_trim=False, y_trim_vals=[-np.inf,np.inf]):

        y_data, points = \
            self.visualizer.get_var_data(model1, y_var_str, t, x_range, y_range, interp_dims, method, y_component_indices)
        x_data, points = \
            self.visualizer.get_var_data(model2, x_var_str, t, x_range, y_range, interp_dims, method, x_component_indices)

        y_data = y_data.flatten()
        x_data = x_data.flatten()
        
        if clip:
            Base.clip_flat(x_data, mod=True, log=True)
            Base.clip_flat(y_data, mod=True, log=True)

        if x_trim or y_trim:
            y_data, x_data = Base.trim_flat(y_data, x_data, x_min_max=x_trim_vals, y_min_max=y_trim_vals)

        fig = plt.figure(figsize=(16,16))
        sns.jointplot(x=x_data, y=y_data, kind="hex", color="#4CB391")
        plt.title(y_var_str+'('+x_var_str+')')
        fig.tight_layout()
        if save_fig:
            save_dir += 'JointPlot_'
            save_dir += y_var_str+str(y_component_indices)+'_'+x_var_str+str(x_component_indices)+'_'
            save_dir += method
            save_dir += '.pdf'
            plt.savefig(save_dir)
            plt.close(fig)
        else:
            print('Showing a figure...')
            plt.show()

        
    def DistributionPlot(self, model, var_str, t, x_range, y_range, interp_dims, 
                         method, component_indices, save_fig=False, save_dir='', clip=False):

        data_to_plot, points = \
            self.visualizer.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)

        if clip:
            Base.clip(data_to_plot, mod=True, log=True)

        fig = plt.figure(figsize=(16,16))
        sns.displot(data_to_plot)
        plt.title(var_str)
        fig.tight_layout()
        if save_fig:
            save_dir += 'DisPlot_'
            save_dir += var_str+str(component_indices)+'_'
            save_dir += method+'_'
            save_dir += model.get_model_name()
            save_dir += '.pdf'
            plt.savefig(save_dir)
            plt.close(fig)
        else:
            print('Showing a figure...')
            plt.show()
            
    def FourierAnalysis(self, model, t, x_range, y_range, interp_dims, 
                         method, component_indices, save_fig=False, save_dir='', clip=False):
        
        # Gather relevant data for KE power spectrum
        W, points = self.visualizer.get_var_data(model, 'W', t, x_range, y_range, interp_dims, method, component_indices)
        U0, points = self.visualizer.get_var_data(model, 'U', t, x_range, y_range, interp_dims, method, component_indices=(0,))
        rho, points = self.visualizer.get_var_data(model, 'vx', t, x_range, y_range, interp_dims, method, component_indices)
        
        # Calculate the Kinetic Energy
        KE = rho * W * (W - 1)

        if not 'Ideal' in locals():
            Ideal = Anim('Ideal/HighRes/Data/TimeSeries/UserDef/')
            idealT = Ideal.t.index(min(Ideal.t, key=lambda x : abs(x-3.0)))
            Nideal = Ideal.final.c['nx'] // 2
            KESpecIdeal = getPowerSpectrumSq(Ideal.final, GetKESF(Ideal, Ideal.frame[idealT]))
    
        ### Model Power Spectrum
    
        fig, axs = plt.subplots(1, 1, sharex=True)
        fig.set_size_inches(6,3)
        fig.tight_layout()
    
        # Kinetic energy density power
        axs.loglog(np.arange(1, Nideal+1), np.arange(1, Nideal+1)*KESpecIdeal, label=r'$Single \ Fluid \ Ideal$')
        axs.loglog(np.arange(1, Nresistive+1), np.arange(1, Nresistive+1)*KESpecResistive, label=r'$Single \ Fluid \ Resistive$')
        axs.loglog(np.arange(1, NtwoFluid+1), np.arange(1, NtwoFluid+1)*KESpecTwoFluid, label=r'$Two \ Fluid \ Resistive$')
        axs.set_ylabel(r"$k|P_{T}(k)|^2$", {'fontsize':'large'})
        axs.set_xlabel(r'$k$')
        axs.loglog([3, 94.868], [7*10**-2, 7*10**(-2 - 1.5*5/3)], 'k--')
        axs.annotate(r'$k^{-5/3}$', xy=(40, 0.01), fontsize=15)
        axs.set_xlim([1, Nideal])
        axs.legend(loc='lower left')
    
    #     plt.savefig('Figures/KineticEnergyPowerSpectrum.eps', format='eps', dpi=1200, bbox_inches='tight')
        plt.show()


        
    def getFourierTrans(u, nx, ny):
        """
        Returns the 1D discrete fourier transform of the variable u along the x-direction
        ready for the power spectrum method.
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
        uhat = np.zeros((NN, ny), dtype=np.complex_)
    
        for k in range(NN):
            for y in range(ny):
                # Sum over all x adding to uhat
                for i in range(nx):
                    uhat[k, y] += u[i, y] * np.exp(-(2*np.pi*1j*k*i)/nx)
        return uhat / nx        

    def getPowerSpectrumSq(u, nx, ny, dy):
        """
        Returns the integrated power spectrum of the variable u, up to the Nyquist frequency = nx/2
        Parameters
        ----------
        u : ndarray
            Two dimensional array of the variable we want the power spectrum of
        """
        NN = nx // 2
        uhat = getFourierTrans(u, nx, ny)
        P = np.zeros(NN)
    
        for k in range(NN):
            for j in range(ny):
                P[k] += (np.absolute(uhat[k, j])**2) * dy
    
        P = P / np.sum(P)
        return P        
            
    def GetKESF(frame):
        """
        Retrieves and computes the kinetic energy density for each frame in a single fluid animation.
        Parameters
        ----------
        """
        vx = frame[anim.variables.index("vx\n"), 4:-4, 4:-4, 0]
        vy = frame[anim.variables.index("vy\n"), 4:-4, 4:-4, 0]
        rho = frame[anim.variables.index("rho\n"), 4:-4, 4:-4, 0]
        vsq = vx**2 + vy**2
        W = 1 / np.sqrt(1 - vsq)
        KE = rho * W * (W-1)
    
        return KE        
        


        
    

















