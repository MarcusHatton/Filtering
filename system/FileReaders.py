# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:00 2023
@author: Thomas
"""

import h5py
import glob
import numpy as np

class METHOD_HDF5(object):

    def __init__(self, directory, comm_line_dom_vars=None):
        """
        Set up the list of files (from hdf5) and dictionary with dataset names
        in the hdf5 file. 
        Parameters
        ----------
        directory: string 
            the filenames in the directory have to be incremental (sorted is used)
        """
        self.comm_line_dom_vars = comm_line_dom_vars

        hdf5_filenames = sorted( glob.glob(directory+str('*.hdf5')))
        self.hdf5_files = []
        for filename in hdf5_filenames:
            self.hdf5_files.append(h5py.File(filename,'r'))
        self.num_files = len(self.hdf5_files)

        self.hdf5_keys = dict.fromkeys(list(self.hdf5_files[0].keys())) 
        for key in self.hdf5_keys: 
            self.hdf5_keys[key] = list(self.hdf5_files[0][key].keys())

    def get_hdf5_keys(self):
        return self.hdf5_keys

    def read_in_data(self, micro_model):   
        """
        Store data from files into micro_model 
        Parameters
        ----------
        micro_model: class MicroModel 
            strs in micromodel have to be the same as hdf5 files output from METHOD.
        """ 
        for prim_var_str in  micro_model.prim_vars:
            try: 
                for counter in range(self.num_files):
                    micro_model.prim_vars[prim_var_str].append( self.hdf5_files[counter]["Primitive/"+prim_var_str][:] )
                    # The [:] is for returning the arrays not the dataset
                micro_model.prim_vars[prim_var_str]  = np.array(micro_model.prim_vars[prim_var_str],dtype=float)
            except KeyError:
                print(f'{prim_var_str} is not in the hdf5 dataset: check Primitive/')
        

        for aux_var_str in  micro_model.aux_vars:
            try: 
                for counter in range(self.num_files):
                    micro_model.aux_vars[aux_var_str].append( self.hdf5_files[counter]["Auxiliary/"+aux_var_str][:] )
                micro_model.aux_vars[aux_var_str] = np.array(micro_model.aux_vars[aux_var_str], dtype=float)
            except KeyError:
                print(f'{aux_var_str} is not in the hdf5 dataset: check Auxiliary/')
 
        # As METHOD saves endTime, the time variables (and points) need to be dealt with separately
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else:
                    micro_model.domain_vars[dom_var_str] = int( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError:
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/.')
                print(f'Seaching command-line dictionary of domain variables for {dom_var_str}.')
                try:
                    micro_model.domain_vars[dom_var_str] = int(self.comm_line_dom_vars[dom_var_str])
                    print('Found!\n')
                except KeyError:
                    print(f'{dom_var_str} is not in the command-line dictionary either.\n')

        for dom_var_str in micro_model.domain_float_strs:
            try: 
                if dom_var_str in ['tmin', 'tmax']:
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = float( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')
                print(f'Seaching command-line dictionary of domain variables for {dom_var_str}.')
                try:
                    micro_model.domain_vars[dom_var_str] = float(self.comm_line_dom_vars[dom_var_str])
                    print('Found!\n')
                except KeyError:
                    print(f'{dom_var_str} is not in the command-line dictionary either.\n')

        for dom_var_str in micro_model.domain_array_strs: 
            try: 
                if dom_var_str in ['t','points']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = self.hdf5_files[0]['Domain/' + dom_var_str][:]
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')
                print(f'Will try to calculate it from other domain vars.\n')

        # for dom_var_str in micro_model.domain_vars:
        #     try: 
        #         if dom_var_str in ['t','nt','tmin','tmax','points']:
        #             pass
        #         if dom_var_str in ['x', 'y']:
        #             micro_model.domain_vars[dom_var_str] =  self.hdf5_files[0]['Domain/' + dom_var_str][:]
        #         if dom_var_str in ['nx', 'ny']:
        #             micro_model.domain_vars[dom_var_str] =  int(self.hdf5_files[0]['Domain/' + dom_var_str][:]
        #         else:
        #             micro_model.domain_vars[dom_var_str] =  self.hdf5_files[0]['Domain/' + dom_var_str][:]
        #     except KeyError:
        #         print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        if bool(micro_model.domain_vars['nx']) and bool(micro_model.domain_vars['xmin']) and bool(micro_model.domain_vars['xmax']):
            print('Calculating x-grid using nx, xmin and xmax')
            micro_model.domain_vars['x'] = np.linspace(micro_model.domain_vars['xmin'], micro_model.domain_vars['xmax'], micro_model.domain_vars['nx'])

        if bool(micro_model.domain_vars['ny']) and bool(micro_model.domain_vars['ymin']) and bool(micro_model.domain_vars['ymax']):
            print('Calculating y-grid using ny, ymin and ymax')
            micro_model.domain_vars['y'] = np.linspace(micro_model.domain_vars['ymin'], micro_model.domain_vars['ymax'], micro_model.domain_vars['ny'])

        micro_model.domain_vars['nt'] = self.num_files
        try:
            for counter in range(self.num_files):
                micro_model.domain_vars['t'].append( float(self.hdf5_files[counter]['Domain/endTime'][:]))
        except KeyError:
            print('Time data cannot be found from the HDF5 files.\n')
            print('Will try to calculate it from command-line data instead.')
            try:
                micro_model.domain_vars['t'] = np.linspace(self.comm_line_dom_vars["tmin"],self.comm_line_dom_vars["tmax"],micro_model.domain_vars['nt'])
                print('Done!\n')
            except KeyError:
                print(f'tmin and tmax are not in the command-line dictionary either.\n')

        micro_model.domain_vars['t'] = np.array(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmin'] = np.amin(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmax'] = np.amax(micro_model.domain_vars['t'])
        micro_model.domain_vars['points'] = [micro_model.domain_vars['t'], micro_model.domain_vars['x'], \
                                             micro_model.domain_vars['y']]

    
if __name__ == '__main__':

    from MicroModels import * 

    FileReader = METHOD_HDF5('./Data/test_res100/')
    MicroModel = IdealMHD_2D()

    # print(FileReader.get_hdf5_keys())
    # FileReader.micro_model_compatibility(MicroModel)
    FileReader.read_in_data(MicroModel)
    for str in MicroModel.domain_vars:
        print(str + '  ',type(MicroModel.domain_vars[str]),' ', MicroModel.domain_vars[str], '\n')
    