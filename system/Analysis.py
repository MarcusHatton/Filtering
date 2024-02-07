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
                  y_trim=False, y_trim_vals=[-np.inf,np.inf], order=1, logx=False, robust=False,\
                  x_mod=False, x_log=False, y_mod=False, y_log=False, log_fit=False):

        y_data, points = \
            self.visualizer.get_var_data(model1, y_var_str, t, x_range, y_range, interp_dims, method, y_component_indices)
        x_data, points = \
            self.visualizer.get_var_data(model2, x_var_str, t, x_range, y_range, interp_dims, method, x_component_indices)

        y_data = y_data.flatten()
        x_data = x_data.flatten()

        if x_mod or x_log:
            self.clip_flat(x_data, mod=x_mod, log=x_log)
        if y_mod or y_log:
            self.clip_flat(y_data, mod=y_mod, log=y_log)

        if x_trim or y_trim:
            y_data, x_data = self.trim_flat(y_data, x_data, x_min_max=x_trim_vals, y_min_max=y_trim_vals)

        #robust = True
        #if sum((order > 1, robust, logx)) > 1:
        #    robust = False
        #    raise ValueError("Mutually exclusive regression options.")

        # Fitting a logarithmic function - TESTING
        def func(x, p1, p2, p3, p4):
            return p1*np.log(p2*x + p3) + p4
 
        popt, pcov = curve_fit(func, x_data, y_data, p0=(1.0,1.0,0.0,-5.0))
        p1, p2, p3, p4 = popt[0], popt[1], popt[2], popt[3]
        print(x_var_str)
        print(p1, p2, p3, p4)
 
        x_curve=np.linspace(np.min(x_data),np.max(x_data),1000)
        y_curve=func(x_curve,p1,p2,p3,p4)

        fig = plt.figure(figsize=(16,16))
        sns.regplot(x=x_data, y=y_data, color="#4CB391", order=order, logx=logx)#, robust=robust, x_estimator=x_estimator, x_bins=x_bins)

        if log_fit == True:
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

        if log_fit == True:
            plt.plot(x_curve, y_curve,'r', linewidth=5)
            plt.savefig('./Testing.pdf')

        

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
            self.clip_flat(x_data, mod=True, log=True)
            self.clip_flat(y_data, mod=True, log=True)

        if x_trim or y_trim:
            y_data, x_data = self.trim_flat(y_data, x_data, x_min_max=x_trim_vals, y_min_max=y_trim_vals)

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
            self.clip(data_to_plot, mod=True, log=True)

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

    def clip_flat(self, data, mod=False, log=False):
        if mod and log:
            for i in range(data.shape[0]):
                data[i] = np.abs(data[i])
                data[i] = np.log10(data[i])
        if log and not mod:
            for i in range(data.shape[0]):
                data[i] = np.log10(data[i])
        if mod and not log:
            for i in range(data.shape[0]):
                data[i] = np.abs(data[i])
        #return data

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

    def trim(self, y_data, x_data, x_minimum):
        for i in range(y_data.shape[0]):
            for j in range(y_data.shape[1]):
                if x_data[i,j] < x_minimum:
                    y_data = np.delete(y_data, [i,j])
                    x_data = np.delete(x_data, [i,j])
        #return y_data, x_data

















