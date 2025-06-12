#!/usr/bin/env python

"""
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Created by: Jean-Sébastien Giroux, Date: 2024-05-22
"""

#.**************************************************************************************************************************************.#
#                                                          Code description                                                              #
#.**************************************************************************************************************************************.#
"""
    TASK NAME - Calculate the gradient of the selected variable (UV) between levels that are given to the program. 

     STATUS - experimental

     DESCRIPTION - TODO

    Before running this script you have to download the following items:
    . ~spst900/spooki/use_nb_master_python.dot
    
    Call example:
    ./TestGradient --levels 76696048,95370590,95353779 --inputFile  /home/spb001/site5/archive_data/operation.ensemble.ens.regmodel/2024040700_000_000 --outputFile TestFichier3Out.std
    ***IMPORTANT**** YOU MUST RUN THIS SCRIPT ON THE TERMINAL, DO NOT USE VSCODE OR ANY OTHER PROGRAMMING LANGUAGE SOFTWARE.
"""  

#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import fnmatch
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append('/home/jfg000/ss5/SuperResolution/Data_Processing/A_crop_interpolate')
import crop as cr


#.**************************************************************************************************************************************.#
#                                                             Functions                                                                  #
#.**************************************************************************************************************************************.#
class Calculate_gradient_uv():
    def std_uv(self, GRADIENTS_INPUT_DATA_UV, PATH_EXTRACT_UV, path_input_std, file_std):
        """
        Get UV values for REPS data for 24h data
        """
        def process_file(i, file):
            print(f'The {i+1} analysed file is: {file}')
            # Copy the file name from DATA to CROP_DATA
            UV_STD_DATA = os.path.join(GRADIENTS_INPUT_DATA_UV, file_std[i])

            print(f'The file is : {file}')
            print(f'The destination folder is: {UV_STD_DATA}\n')
            # Call the editfst script to put the data on the same grid and interpolate input data.
            subprocess.check_call([os.path.join(PATH_EXTRACT_UV, 'calculate_uv.sh'), file, UV_STD_DATA])

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(process_file, i, file) for i, file in enumerate(path_input_std)]
            for future in futures:
                future.result()

    def get_std_uv(self, PATH_EXTRACT_UV, input_data, INPUT_DATA_UV):
        """
        Extract UV value from standard file for the 24h data. 
        """
        #Load the initial data. 
        files_input = [file for file in os.listdir(input_data)
               if os.path.isfile(os.path.join(input_data, file)) and
               not fnmatch.fnmatch(file, '*.tgz') and
               not fnmatch.fnmatch(file, '*.sh') and
               file[11:14] == '024']
        # for file in files_input:
        #     print(f'File is : {file}')
        # print('\n')

        #Put the data in chronological order
        cropping = cr.Crop()
        files_input.sort(key=cropping.get_sort_key_input)
        # for file in files_input:
        #     print(f'File is : {file}')
        # print('\n')

        path_input_std = []
        file_std = []
        for file in files_input:
            file_std.append(file)
            file_name = os.path.join(input_data, file)
            path_input_std.append(file_name)
            print(f'File in chronological order are: {file}')
            print(f'The path of the input temporal variable are: {file_name}')
        print(f'The length of the initial files_input array is {len(path_input_std)}')

        #Call the function that will extract UV from REPS data
        self.std_uv(INPUT_DATA_UV, PATH_EXTRACT_UV, path_input_std, file_std)

#.**************************************************************************************************************************************.#
#                                                                 Main                                                                   #
#.**************************************************************************************************************************************.#
if __name__ == '__main__':
    #List of constants 
    PATH_EXTRACT_UV = '/home/jfg000/ss5/SuperResolution/Data_Processing/A00_calculate_gradients'
    INPUT_DATA_REPS = '/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel/'  # '/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel/avril2024_18Z' 
    VARIABLE_NAME_ARRAY_INPUT_dict = {'UV': [75597472, 95369342, 95364364, 95357866, 95349708, 95339883]}  
    INPUT_DATA_UV = '/home/jfg000/ss5/Data/input/initial_data_reps/uv_values'

    if not os.path.exists(INPUT_DATA_UV):
        os.makedirs(INPUT_DATA_UV)

    calculate_gt_uv = Calculate_gradient_uv()
    calculate_gt_uv.get_std_uv(PATH_EXTRACT_UV, INPUT_DATA_REPS, INPUT_DATA_UV)