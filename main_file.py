# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:45:31 2023

@author: marcu
"""

from multiprocessing import Process, Pool
from FileReaders import *
from Filters_TC import *
from MicroModels import *
from MesoModels import *
from Visualization import *
from Analysis import *
import sys
import json

if __name__ == '__main__':

    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

    HDF5_Directory = str(sys.argv[1])
    comm_line_dom_vars = json.loads(sys.argv[2])
    comm_line_filtering_ranges = json.loads(sys.argv[3])

    # Pickling Options
    LoadMicroModelFromPickleFile = True
    MicroModelPickleLoadFile = 'IdealHydro2D.pickle'

    DumpMicroModelToPickleFile = False
    MicroModelPickleDumpFile = 'IdealHydro2D.pickle'

    LoadMesoModelFromPickleFile = True
    MesoModelPickleLoadFile = 'NonIdealHydro2D.pickle'
    
    DumpMesoModelToPickleFile = False
    MesoModelPickleDumpFile = 'NonIdealHydro2D.pickle'
    
    # Start timing    
    CPU_start_time = time.process_time()

    # Read in data from file
    #HDF5_Directory = '../../../../../scratch/mjh1n20/Filtering_Data/KH/Ideal/t_49_50/2em1_1em1_1/'
    HDF5_Directory = '../../../../../scratch/mjh1n20/Filtering_Data/KH/Testing/'
    #HDF5_Directory = '../../../METHOD_Marcus/Examples/IS_CE/KH2D_speedtesting/2d/Filtering/'
    FileReader = METHOD_HDF5(HDF5_Directory, comm_line_dom_vars)
    #FileReader = METHOD_HDF5('../../Filtering/Data/KH/Ideal/t_998_1002/')

    # Create and setup micromodel
    if LoadMicroModelFromPickleFile:
        with open(MicroModelPickleLoadFile, 'rb') as filehandle:
            micro_model = pickle.load(filehandle) 
    else:
        micro_model = IdealHydro_2D()
        FileReader.read_in_data(micro_model) 
        micro_model.setup_structures()

    if DumpMicroModelToPickleFile:
        with open(MicroModelPickleDumpFile, 'wb') as filehandle:
            pickle.dump(micro_model, filehandle)  

    # Create visualizer for plotting micro data
    visualizer = Plotter_2D()
    # visualizer.plot_vars(micro_model, ['v1'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()], save_fig=True, save_dir='Output/v1.pdf')

    # visualizer.plot_vars(micro_model, ['v2'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])

    # visualizer.plot_vars(micro_model, ['n'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])        

    # visualizer.plot_vars(micro_model, ['T'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])   

    # visualizer.plot_vars(micro_model, ['p'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])   

    # visualizer.plot_vars(micro_model, ['rho'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='raw_data', components_indices=[()])   

    visualizer.plot_vars(micro_model, ['p','rho','n'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(),(),()], save_fig=True, save_dir='Output/p_rho_n.pdf')         
    visualizer.plot_vars(micro_model, ['W','v1','v2'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(),(),()], save_fig=True, save_dir='Output/W_v1_v2.pdf')  
    visualizer.plot_vars(micro_model, ['T','s','h'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(),(),()], save_fig=True, save_dir='Output/T_s_h.pdf')  

        # visualizer.plot_vars(micro_model, ['BC'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', components_indices=[(1,)])

    # visualizer.plot_vars(micro_model, ['v1','v2','n'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='interpolate', components_indices=[(),(),()])
    # visualizer.plot_vars(micro_model, ['BC'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                       interp_dims=(20,40), method='interpolate', components_indices=[(1,)])

    t_range = comm_line_filtering_ranges["t_range"]
    x_range = comm_line_filtering_ranges["x_range"]
    y_range = comm_line_filtering_ranges["y_range"]
    n_txy_pts = comm_line_filtering_ranges["n_txy_pts"]

    t_to_plot = t_range[0]
    x_range_plotting = x_range
    y_range_plotting = y_range

    visualizer.plot_vars(micro_model, ['v1','v2','n'], t=t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='raw_data', components_indices=[(),(),()])
    visualizer.plot_vars(micro_model, ['BC'], t=t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='raw_data', components_indices=[(1,)])

    visualizer.plot_vars(micro_model, ['v1','v2','n'], t=t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='interpolate', components_indices=[(),(),()])
    visualizer.plot_vars(micro_model, ['BC'], t=t_to_plot, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='interpolate', components_indices=[(1,)])

    # Create the observer-finder and filter
    ObsFinder = FindObs_drift_root(micro_model,box_len=0.001)
    Filter = spatial_box_filter(micro_model,filter_width=0.002)
    
    CPU_start_time = time.process_time()
    coord_ranges = [t_range,x_range,y_range]
    #coord_range = [[49.45,49.55],x_range_plotting, y_range_plotting]
    #num_points = [2,10,20]
    # spacing = 2

    # Create MesoModel and find special observers
    if LoadMesoModelFromPickleFile:
        with open(MesoModelPickleLoadFile, 'rb') as filehandle:
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
    meso_model.setup_variables()
    meso_model.filter_micro_variables()
    meso_model.calculate_dissipative_coefficients()

    if DumpMesoModelToPickleFile:
        with open(MesoModelPickleDumpFile, 'wb') as filehandle:
            pickle.dump(meso_model, filehandle) 

    # Plot MesoModel variables
    # visualizer.plot_vars(meso_model, ['U'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', components_indices=[(1,)])
            
    # visualizer.plot_var_model_comparison([micro_model, meso_model], 'SET', \
    #                                       t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', component_indices=(1,2))

    # visualizer.plot_vars(meso_model, ['U'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='interpolate', components_indices=[(1,)])
            
    # visualizer.plot_var_model_comparison([micro_model, meso_model], 'SET', \
    #                                       t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='interpolate', component_indices=(1,2))
        
    # Analyse coefficients of the MesoModel
    analyzer = CoefficientAnalysis(visualizer)
    # Zeta
    # visualizer.plot_vars(meso_model, ['Zeta'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', components_indices=[()])
        
    # analyzer.DistributionPlot(meso_model, 'Zeta', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', component_indices=())

    # analyzer.JointPlot(meso_model, 'Zeta', 'U', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', y_component_indices=(), x_component_indices=(0,))

    # # Kappa    
    # visualizer.plot_vars(meso_model, ['Kappa'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', components_indices=[(2,)])
        
    # analyzer.DistributionPlot(meso_model, 'Kappa', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #               interp_dims=(20,40), method='raw_data', component_indices=(2,))

    # analyzer.JointPlot(meso_model, 'Kappa', 'T~', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', y_component_indices=(2,), x_component_indices=())

    # Eta
    visualizer.plot_vars(meso_model, ['Eta'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(1,2)], save_fig=True, save_dir='Output/eta.pdf')
        
    analyzer.DistributionPlot(meso_model, 'Eta', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', component_indices=(1,2), clip=True)

    # analyzer.JointPlot(meso_model, 'Eta', 'U', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', y_component_indices=(1,2), x_component_indices=(0,))

    analyzer.JointPlot(meso_model, micro_model, 'Eta', 's', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='interpolate', y_component_indices=(1,2), x_component_indices=(), clip=True)
        
    print(f'Total elapsed CPU time for finding is {time.process_time() - CPU_start_time}.')
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
