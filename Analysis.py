# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:02:05 2023

@author: marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle
import seaborn as sns

class CoefficientAnalysis(object):
    
    def __init__(self, visualizer):
        """
        Nothing as yet...

        Returns
        -------
        Also nothing...
        
        """
        self.visualizer = visualizer # need to re-structure this... or do I
        
    # def JointPlot(self, model, y_var_str, x_var_str, t, x_range, y_range,\
    #               interp_dims, method, y_component_indices, x_component_indices,
    #               save_fig=False, save_dir=''):

    #     y_data_to_plot, points = \
    #         self.visualizer.get_var_data(model, y_var_str, t, x_range, y_range, interp_dims, method, y_component_indices)
    #     x_data_to_plot, points = \
    #         self.visualizer.get_var_data(model, x_var_str, t, x_range, y_range, interp_dims, method, x_component_indices)
    #     fig = plt.figure(figsize=(16,16))
    #     sns.jointplot(x=x_data_to_plot.flatten(), y=y_data_to_plot.flatten(), kind="hex", color="#4CB391")
    #     plt.title(y_var_str+'('+x_var_str+')')
    #     fig.tight_layout()
    #     if save_fig:
    #         plt.savefig(save_dir)
    #     plt.show()

    def JointPlot(self, meso_model, micro_model, y_var_str, x_var_str, t, x_range, y_range,\
                  interp_dims, method, y_component_indices, x_component_indices,
                  save_fig=False, save_dir='', clip=False):

        y_data_to_plot, points = \
            self.visualizer.get_var_data(meso_model, y_var_str, t, x_range, y_range, interp_dims, method, y_component_indices)
        x_data_to_plot, points = \
            self.visualizer.get_var_data(micro_model, x_var_str, t, x_range, y_range, interp_dims, method, x_component_indices)
            
        if clip:
            self.clip(x_data_to_plot, mod=True, log=True)
            self.clip(y_data_to_plot, mod=True, log=True)
            
        fig = plt.figure(figsize=(16,16))
        sns.jointplot(x=x_data_to_plot.flatten(), y=y_data_to_plot.flatten(), kind="hex", color="#4CB391")
        plt.title(y_var_str+'('+x_var_str+')')
        fig.tight_layout()
        if save_fig:
            plt.savefig(save_dir)
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
            plt.savefig(save_dir)
        plt.show()

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
        return data


















