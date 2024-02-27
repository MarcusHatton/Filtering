import sys
# import os
sys.path.append('../master_files/')
import pickle
import configparser
import json
import time
import math

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ##################################################################
    # #CORRELATION PLOTS: RESIDUALS VS CORRESPONDING CLOSURE INGREDIENTS
    # ##################################################################
    
    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # LOADING MESO AND MICRO MODELS 
    pickle_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Filenames']['meso_pickled_filename']
    MesoModelLoadFile = pickle_directory + meso_pickled_filename

    print('================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    statistical_tool = CoefficientsAnalysis()  
    correlation_ranges = json.loads(config['Ranges_for_analysis']['ranges'])
    x_range = correlation_ranges['x_range']
    y_range = correlation_ranges['y_range']
    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]
    model_points = meso_model.domain_vars['Points']
    saving_directory = config['Directories']['figures_dir']

    print('Producing correlation plot for pi_res')
    x = meso_model.meso_vars['shear_sq']
    y = meso_model.meso_vars['pi_res_sq']
    data = [x,y]    
    data = statistical_tool.trim_dataset(data, ranges, model_points)
    preprocess_data = {"value_ranges": [[None, None], [None, None]], "log_abs": [1, 1]}
    data = statistical_tool.preprocess_data(data, preprocess_data)
    data = statistical_tool.extract_randomly(data, 10000)
    x, y = data[0], data[1]
    xlabel = r'$\log(\tilde{\sigma}_{ab}\tilde{\sigma}^{ab})$'
    ylabel = r'$\log(\tilde{\pi}_{ab}\tilde{\pi}^{ab})$'
    g2=statistical_tool.visualize_correlation(x, y, xlabel=xlabel, ylabel=ylabel)
    saving_directory = config['Directories']['figures_dir']
    filename = '/pi_aniso_correlation.pdf'
    plt.savefig(saving_directory + filename, format='pdf')
    print(f'Finished correlation plot for pi_res, saved as {filename}\n\n')


    print('Producing correlation plot for q_res')
    x = meso_model.meso_vars['q_res_sq']
    y = meso_model.meso_vars['Theta_sq']
    data = [x,y]    
    data = statistical_tool.trim_dataset(data, ranges, model_points)
    preprocess_data = {"value_ranges": [[None, None], [None, None]], "log_abs": [1, 1]}
    data = statistical_tool.preprocess_data(data, preprocess_data)
    data = statistical_tool.extract_randomly(data, 10000)
    x, y = data[0], data[1]
    xlabel = r'$\log(\tilde{q}_a\tilde{q}^a)$'
    ylabel = r'$\log(\tilde{\Theta}_a \tilde{\Theta}^a)$'
    model_points = meso_model.domain_vars['Points']
    g2=statistical_tool.visualize_correlation(x, y, xlabel=xlabel, ylabel=ylabel)
    saving_directory = config['Directories']['figures_dir']
    filename = '/q_res_correlation.pdf'
    plt.savefig(saving_directory + filename, format='pdf')
    print(f'Finished correlation plot for q_res, saved as {filename}\n\n')


    print('Producing correlation plot for Pi_res')
    x = np.power(meso_model.meso_vars['Pi_res'], 2)
    y = np.power(meso_model.meso_vars['exp_tilde'], 2)
    data = [x,y]    
    data = statistical_tool.trim_dataset(data, ranges, model_points)
    preprocess_data = {"value_ranges": [[None, None], [None, None]], "log_abs": [1, 1]}
    data = statistical_tool.preprocess_data(data, preprocess_data)
    data = statistical_tool.extract_randomly(data, 10000)
    x, y = data[0], data[1]
    xlabel = r'$\log(\tilde{\Pi}^2)$'
    ylabel = r'$\log(\tilde{\theta}^2)$'
    model_points = meso_model.domain_vars['Points']
    g2=statistical_tool.visualize_correlation(x, y, xlabel=xlabel, ylabel=ylabel)
    saving_directory = config['Directories']['figures_dir']
    filename = '/Pi_res_correlation.pdf'
    plt.savefig(saving_directory + filename, format='pdf')
    print(f'Finished correlation plot for Pi_res, saved as {filename}\n\n')

    
    # #############################################################
    # PLOTTING RESIDUALS VS CORRESPONDING CLOSURE INGREDIENTS 
    # #############################################################

    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    visualizer = Plotter_2D()  

    vars_strs = ['zeta', 'exp_tilde', 'Pi_res']
    norms = ['mysymlog', 'symlog', 'log']
    cmaps = ['seismic', 'seismic', 'plasma']
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Bulk_viscosity.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished bulk viscosity')

    vars_strs = ['eta', 'shear_sq', 'pi_res_sq']
    norms = ['mysymlog', 'log', 'log']
    cmaps = ['seismic', 'plasma', 'plasma']
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Shear_viscosity.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished shear viscosity')

    vars_strs = ['kappa', 'Theta_sq', 'q_res_sq']
    norms = ['mysymlog', 'log', 'log']
    cmaps = ['seismic', 'plasma', 'plasma']
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Heat_conductivity.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished heat conductivity')


    