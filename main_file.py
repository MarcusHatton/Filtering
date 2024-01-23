# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:45:31 2023

@author: marcu
"""

from system.FileReaders import *
from system.Filters_TC import *
from system.MicroModels import *
from system.MesoModels import *
from system.Visualization import *
from system.Analysis import *

import sys
import json
import configparser

from multiprocessing import Process, Pool
import numpy as np

if __name__ == '__main__':

    req_config_args = {"HDF5_FileDirectory": None,
                       "MicroModel": None,
                       "MesoModel": None}
    optional_config_args = {"FigureOutputDirectory":None,
                   "DomainVariables": None,
                   "FilterCoordinateRanges": None,
                   "PickleIODirectory": None,
                   "MicroPlottingRanges": None,
                   "MesoPlottingRanges": None}
    #"LoadDumpMicroMesoPickles": None,
    
    if len(sys.argv) == 1:
        print(f"You must pass arguments from either the command line or a config file.")
        print(f"The {len(req_config_args)} necessary arguments are:")
        print(req_config_args.keys())
        raise Exception()

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    for key in req_config_args.keys():
        req_config_args[key] = json.loads(config['Required'][key])

    for key in optional_config_args.keys():
        optional_config_args[key] = json.loads(config['Optional'][key])
        
    # Pickling Options    
    #LoadMicroModelFromPickleFile = optional_config_args["LoadDumpMicroMesoPickles"]["LoadMicro"]
    LoadMicroModelFromPickleFile = True
    MicroModel = req_config_args["MicroModel"] # 'IdealHydro2D.pickle'

    DumpMicroModelToPickleFile = False
    #MicroModelPickleDumpFile = 'IdealHydro2D.pickle'

    LoadMesoModelFromPickleFile = True
    MesoModel = req_config_args["MesoModel"] # 'NonIdealHydro2D.pickle'
    
    DumpMesoModelToPickleFile = False
    #MesoModelPickleDumpFile = 'NonIdealHydro2D.pickle'
    
    PickleDir = optional_config_args["PickleIODirectory"]

    # Start timing    
    CPU_start_time = time.process_time()

    # Read in data from file
    #HDF5_Directory = '../../../../../scratch/mjh1n20/Filtering_Data/KH/Ideal/t_49_50/2em1_1em1_1/'
    #HDF5_Directory = '../../../../../scratch/mjh1n20/Filtering_Data/KH/Testing/'
    #HDF5_Directory = '../../../METHOD_Marcus/Examples/IS_CE/KH2D_speedtesting/2d/Filtering/'
    FileReader = METHOD_HDF5(req_config_args["HDF5_FileDirectory"], optional_config_args["DomainVariables"])
    #FileReader = METHOD_HDF5('../../Filtering/Data/KH/Ideal/t_998_1002/')

    # Create and setup micromodel
    if LoadMicroModelFromPickleFile:
        with open(PickleDir+MicroModel+'.pickle', 'rb') as filehandle:
            micro_model = pickle.load(filehandle) 
    else:
        micro_model = IdealHydro_2D()
        FileReader.read_in_data(micro_model) 
        micro_model.setup_structures()

    if DumpMicroModelToPickleFile:
        with open(PickleDir+micro_model.get_model_name()+'.pickle', 'wb') as filehandle:
            pickle.dump(micro_model, filehandle)  

    # Create visualizer for plotting micro data
    visualizer = Plotter_2D()
    fig_save_dir = optional_config_args["FigureOutputDirectory"]
    #visualizer.plot_vars(micro_model, ['v1'], t=micro_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                     interp_dims=(20,40), method='raw_data', components_indices=[()], save_fig=True, save_dir=fig_save_dir+'v1.pdf')

    # visualizer.plot_vars(micro_model, ['v2'], t=micro_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])

    # visualizer.plot_vars(micro_model, ['n'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])        

    # visualizer.plot_vars(micro_model, ['T'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])   

    # visualizer.plot_vars(micro_model, ['p'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])   

    # visualizer.plot_vars(micro_model, ['rho'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])   

    # visualizer.plot_vars(micro_model, ['p','rho','n'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', components_indices=[(),(),()], save_fig=True, save_dir='Output/p_rho_n.pdf')         
    # visualizer.plot_vars(micro_model, ['W','v1','v2'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', components_indices=[(),(),()], save_fig=True, save_dir='Output/W_v1_v2.pdf')  
    # visualizer.plot_vars(micro_model, ['T','s','h'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', components_indices=[(),(),()], save_fig=True, save_dir='Output/T_s_h.pdf')  

        # visualizer.plot_vars(micro_model, ['BC'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', components_indices=[(1,)])

    # visualizer.plot_vars(micro_model, ['v1','v2','n'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='interpolate', components_indices=[(),(),()])
    # visualizer.plot_vars(micro_model, ['BC'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='interpolate', components_indices=[(1,)])

    t_range = optional_config_args["FilterCoordinateRanges"]["t_range"]
    x_range = optional_config_args["FilterCoordinateRanges"]["x_range"]
    y_range = optional_config_args["FilterCoordinateRanges"]["y_range"]
    n_txy_pts = optional_config_args["FilterCoordinateRanges"]["n_txy_pts"]

    micro_t_to_plot = optional_config_args["MicroPlottingRanges"]["t_point"] # t_range[0]
    x_range_plotting = optional_config_args["MicroPlottingRanges"]["x_range"]
    y_range_plotting = optional_config_args["MicroPlottingRanges"]["y_range"]

    visualizer.plot_vars(micro_model, ['T'], t=micro_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                         interp_dims=(20,40), method='raw_data', components_indices=[()], save_fig=True, save_dir=fig_save_dir)#+'v1.pdf')

    visualizer.plot_vars(micro_model, ['v1'], t=micro_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                         interp_dims=(20,40), method='raw_data', components_indices=[()], save_fig=True, save_dir=fig_save_dir)#+'v1.pdf')

    visualizer.plot_vars(micro_model, ['v1','v2','n'], t=micro_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='raw_data', components_indices=[(),(),()], save_fig=True, save_dir=fig_save_dir)#+'v1_v2_n.pdf')
    visualizer.plot_vars(micro_model, ['BC'], t=micro_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='raw_data', components_indices=[(1,)], save_fig=True, save_dir=fig_save_dir)#+'BC[1].pdf')

    visualizer.plot_vars(micro_model, ['v1','v2','n'], t=micro_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='interpolate', components_indices=[(),(),()], save_fig=True, save_dir=fig_save_dir)#+'v1.pdf')
    visualizer.plot_vars(micro_model, ['BC'], t=micro_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='interpolate', components_indices=[(1,)], save_fig=True, save_dir=fig_save_dir)

    # Create the observer-finder and filter
    ObsFinder = FindObs_drift_root(micro_model,box_len=0.001)
    Filter = spatial_box_filter(micro_model,filter_width=0.002)
    
    CPU_start_time = time.process_time()
    coord_ranges = [t_range,x_range,y_range]

    # Create MesoModel and find special observers
    if LoadMesoModelFromPickleFile:
        with open(PickleDir+MesoModel+'.pickle', 'rb') as filehandle:
            meso_model = pickle.load(filehandle) 
    else:
        meso_model = NonIdealHydro2D(micro_model, ObsFinder, Filter)
        meso_model.find_observers(n_txy_pts, coord_ranges)
        meso_model.setup_variables()
        meso_model.filter_micro_variables()
        meso_model.calculate_dissipative_coefficients()
        
    # CPU_start_time = time.process_time()
    # meso_model.find_observers(num_points, coord_range, spacing)
    # meso_model.find_observers(num_points, coord_range)

    # print(f'\nElapsed CPU time for observer-finding is {time.process_time() - CPU_start_time}\
    #       with {np.product(num_points)} and {filter.n_filter_points**filter.spatial_dim} points per face\n')

    # Having found observers, setup MesoModel
    #meso_model.setup_variables()
    #meso_model.filter_micro_variables()
    #meso_model.calculate_dissipative_coefficients()

    if DumpMesoModelToPickleFile:
        with open(PickleDir+meso_model.get_model_name()+'.pickle', 'wb') as filehandle:
            pickle.dump(meso_model, filehandle) 

    meso_t_to_plot = optional_config_args["MesoPlottingRanges"]["t_point"]

    # Plot MesoModel variables
    visualizer.plot_vars(meso_model, ['U'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(1,)], save_fig=True, save_dir=fig_save_dir)#+'U.pdf')
            
    #visualizer.plot_var_model_comparison([micro_model, meso_model], 'SET', \
    #                                      t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                  interp_dims=(20,40), method='raw_data', component_indices=(1,2))

    # visualizer.plot_vars(meso_model, ['U'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='interpolate', components_indices=[(1,)])
            
    # visualizer.plot_var_model_comparison([micro_model, meso_model], 'SET', \
    #                                       t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='interpolate', component_indices=(1,2))
        
    # Analyse coefficients of the MesoModel
    analyzer = CoefficientAnalysis(visualizer)

    """
    # Zeta
    visualizer.plot_vars(meso_model, ['Zeta'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', components_indices=[()],\
                       save_fig=True, save_dir=fig_save_dir)
    
    analyzer.DistributionPlot(meso_model, 'Zeta', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', component_indices=(),\
                      clip=True, save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, meso_model, 'Zeta', 'U', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(), x_component_indices=(0,),\
                       save_fig=True, save_dir=fig_save_dir)

    """

    # # Kappa    
    visualizer.plot_vars(meso_model, ['Kappa'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', components_indices=[(1,)],\
                       save_fig=True, save_dir=fig_save_dir)

    visualizer.plot_vars(meso_model, ['Kappa'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', components_indices=[(2,)],\
                       save_fig=True, save_dir=fig_save_dir)

    visualizer.plot_vars(meso_model, ['Kappa_scalar'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', components_indices=[()],\
                       save_fig=True, save_dir=fig_save_dir)
        
    analyzer.DistributionPlot(meso_model, 'Kappa', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                  interp_dims=(20,40), method='raw_data', component_indices=(1,), \
                  clip=True, save_fig=True, save_dir=fig_save_dir)

    analyzer.DistributionPlot(meso_model, 'Kappa', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                  interp_dims=(20,40), method='raw_data', component_indices=(2,), \
                  clip=True, save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, meso_model, 'Kappa', 'T~', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', y_component_indices=(1,), x_component_indices=(),\
                       save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, meso_model, 'Kappa', 'T~', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', y_component_indices=(2,), x_component_indices=(),\
                       save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, meso_model, 'Kappa_scalar', 'N', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(), x_component_indices=(),\
                       save_fig=True, save_dir=fig_save_dir, x_trim=False, x_trim_vals=[0.2,0.8])

    analyzer.JointPlot(meso_model, meso_model, 'Omega', 'T~', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(1,),\
                       x_component_indices=(), save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, meso_model, 'Omega', 'T~', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(2,),\
                       x_component_indices=(), save_fig=True, save_dir=fig_save_dir)


    # Eta
#    visualizer.plot_vars(meso_model, ['pi'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
#                      interp_dims=(20,40), method='raw_data', components_indices=[(1,2)],\
#                      save_fig=True, save_dir=fig_save_dir)

    visualizer.plot_vars(meso_model, ['Sigma'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(1,2)],\
                      save_fig=True, save_dir=fig_save_dir)

    visualizer.plot_vars(meso_model, ['Eta'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(1,2)],\
                      save_fig=True, save_dir=fig_save_dir)
        
    analyzer.DistributionPlot(meso_model, 'Eta', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', component_indices=(1,2), clip=True,\
                      save_fig=True, save_dir=fig_save_dir)

#    analyzer.JointPlot(meso_model, meso_model, 'Eta', 'N', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
#                       interp_dims=(20,40), method='raw_data', y_component_indices=(1,2), x_component_indices=(),\
#                       save_fig=True, save_dir=fig_save_dir)

    visualizer.plot_vars(meso_model, ['N'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', components_indices=[()],\
                       save_fig=True, save_dir=fig_save_dir)

    visualizer.plot_vars(meso_model, ['Eta_scalar'], t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', components_indices=[()],\
                       save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, meso_model, 'Eta', 'N', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(1,2), x_component_indices=(),\
                       save_fig=True, save_dir=fig_save_dir, x_trim=True, x_trim_vals=[0.2,1e3])

    analyzer.JointPlot(meso_model, meso_model, 'Eta_scalar', 'N', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(), x_component_indices=(),\
                       save_fig=True, save_dir=fig_save_dir, x_trim=True, x_trim_vals=[0.2,np.inf], y_trim=True, y_trim_vals=[0.01,np.inf])

    analyzer.JointPlot(meso_model, meso_model, 'Eta', 'T~', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(1,2), x_component_indices=(),\
                       save_fig=True, save_dir=fig_save_dir, x_trim=True, x_trim_vals=[2,8])

    analyzer.RegPlot(meso_model, meso_model, 'Eta', 'T~', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(1,2), x_component_indices=(),\
                       save_fig=True, save_dir=fig_save_dir, x_trim=True, x_trim_vals=[2,8], order=5, logx=False)

    analyzer.JointPlot(meso_model, meso_model, 'Eta_scalar', 'T~', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(), x_component_indices=(),\
                       save_fig=True, save_dir=fig_save_dir, x_trim=False, x_trim_vals=[-np.inf,np.inf], y_trim=True, y_trim_vals=[1e-5,np.inf])

    analyzer.JointPlot(meso_model, meso_model, 'Eta', 'U', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(1,2), x_component_indices=(0,),\
                       save_fig=True, save_dir=fig_save_dir, x_trim=True, x_trim_vals=[-np.inf, 1.10])

    analyzer.RegPlot(meso_model, meso_model, 'Eta', 'U', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                       interp_dims=(20,40), method='raw_data', y_component_indices=(1,2), x_component_indices=(0,),\
                       save_fig=True, save_dir=fig_save_dir, x_trim=True, x_trim_vals=[-np.inf, 1.10], order=10, logx=False)

    analyzer.JointPlot(meso_model, meso_model, 'Eta', 'U', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(1,2),\
                       x_component_indices=(1,), save_fig=True, save_dir=fig_save_dir)

    analyzer.RegPlot(meso_model, meso_model, 'Eta', 'U', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(1,2),\
                       x_component_indices=(1,), save_fig=True, save_dir=fig_save_dir, order=1)

    analyzer.JointPlot(meso_model, meso_model, 'Eta', 'U', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(1,2),\
                       x_component_indices=(2,), save_fig=True, save_dir=fig_save_dir, x_trim=True, x_trim_vals=[-0.1,0.1])

    analyzer.RegPlot(meso_model, meso_model, 'Eta', 'U', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(1,2),\
                       x_component_indices=(2,), save_fig=True, save_dir=fig_save_dir, x_trim=True, x_trim_vals=[-0.1,0.1], order=1)

    analyzer.JointPlot(meso_model, meso_model, 'Sigma', 'N', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(1,2),\
                       x_component_indices=(), save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, meso_model, 'pi', 'T~', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(1,2),\
                       x_component_indices=(), save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, meso_model, 'pi', 'N', t=meso_t_to_plot, x_range=x_range_plotting,\
                       y_range=y_range_plotting, interp_dims=(20,40), method='raw_data', y_component_indices=(1,2),\
                       x_component_indices=(), save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, micro_model, 'Eta', 'W', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='interpolate', y_component_indices=(1,2), x_component_indices=(),\
                      clip=True, save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, micro_model, 'Eta', 'v1', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='interpolate', y_component_indices=(1,2), x_component_indices=(),\
                      clip=True, save_fig=True, save_dir=fig_save_dir)

    analyzer.JointPlot(meso_model, micro_model, 'Eta', 'v2', t=meso_t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='interpolate', y_component_indices=(1,2), x_component_indices=(),\
                      clip=True, save_fig=True, save_dir=fig_save_dir)
        
    print(f'Total elapsed CPU time for finding is {time.process_time() - CPU_start_time}.')
    print(f'Time per gridpoint of the MesoModel is {(time.process_time() - CPU_start_time)/np.prod(np.array(n_txy_pts))}.')
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
