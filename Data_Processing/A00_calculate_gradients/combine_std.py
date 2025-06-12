#!/usr/bin/env python

""" 
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Created by: Jean-Sébastien Giroux, Date: 2024-05-06
"""

#.**************************************************************************************************************************************.#
#                                                          Code description                                                              #
#.**************************************************************************************************************************************.#
"""
    TASK NAME - Combine the standards files together. 
                .Input data: Combine spatio-temporal variables, gradient variable (if used) and topographie variable

     STATUS - experimental

     DESCRIPTION - Take as input the cropped input data, the gradients variables that have been calculated on the cropped input (if the user
                    want to use them for the Neural Network training) and the domaine topographie information. These variables are all going
                    to be combined together and the combinaison will create a new standard file. It is important to put the variables first, 
                    then the gradients and finally the topographie. Order matter, because the neural network architecture need to access the 
                    certain variables types at different stages of the training. 

    Terminal call example: 
    Logic for one folder: editfst -s /home/jfg000/ss5/Data/input/domaine/quicktest_cropping/crop_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE/2022050100_006_000 /home/jfg000/ss5/Data/topography/east_canada_sequential_train_domaine1/crop/HR_input_from_HRDPS -d /home/jfg000/ss5/Data/input/domaine/quicktest_cropping/comb_std_input_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE/output -i 0
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
class Combine_standard_files():
    def combine_std(self, PATH_COMSTD, path_input_std, path_input_uu, path_input_vv, path_input_tt, path_input_uv, file_std):
        """
        Combine standard file together using editfst interne function: https://wiki.cmc.ec.gc.ca/wiki/Editfst_f
        """
        def process_file(i, file):
            print(f'The {i+1} analysed file is: {file}')
            # Copy the file name from DATA to CROP_DATA
            #open(file, 'a').close()
            COMBINING_STD_DATA_INPUT_F = COMBINING_STD_DATA_INPUT + '/' + file_std[i]
            COMBINING_STD_DATA_INPUT_F_TEMP = COMBINING_STD_DATA_INPUT_TEMP + '/' + file_std[i]

            print(f'The file is : {file}')
            print(f'The destination folder is: {COMBINING_STD_DATA_INPUT_F}\n')
            # Call the editfst script to put the data on the same grid and interpolate input data.
            subprocess.check_call([os.path.join(PATH_COMSTD, 'combine_std.sh'), file, path_input_uu[i], path_input_vv[i], path_input_tt[i], path_input_uv[i], COMBINING_STD_DATA_INPUT_F_TEMP, COMBINING_STD_DATA_INPUT_F])

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(process_file, i, file) for i, file in enumerate(path_input_std)]
            for future in futures:
                future.result()

    def get_combine_std(self, PATH_COMSTD, INPUT_24H_STD, INPUT_24H_GRAD_UU, INPUT_24H_GRAD_VV, INPUT_24H_GRAD_TT, INPUT_24H_GRAD_UV, PATH_COMBINED_STD, COMBINING_STD_DATA_INPUT, COMBINING_STD_DATA_INPUT_TEMP):
        #Load the input 24 hours uu gradient standard file and put them in chronological order in an array. 
        files_input_grad_uu = [file for file in os.listdir(INPUT_24H_GRAD_UU)
        if os.path.isfile(os.path.join(INPUT_24H_GRAD_UU, file)) and
        not fnmatch.fnmatch(file, '*.tgz') and
        not fnmatch.fnmatch(file, '*.sh') and
        file[11:14] == '024']
        cropping = cr.Crop()
        files_input_grad_uu.sort(key=cropping.get_sort_key_input)
        path_input_uu = []
        for file in files_input_grad_uu:
            file_name = os.path.join(INPUT_24H_GRAD_UU, file)
            path_input_uu.append(file_name)
            print(f'File in chronological order are: {file}')
            print(f'The path of the input temporal variable are: {file_name}')
        print(f'The length of the initial files_input array is {len(path_input_uu)}')

        #Load the input 24 hours vv gradient standard file and put them in chronological order in an array. 
        files_input_grad_vv = [file for file in os.listdir(INPUT_24H_GRAD_VV)
        if os.path.isfile(os.path.join(INPUT_24H_GRAD_VV, file)) and
        not fnmatch.fnmatch(file, '*.tgz') and
        not fnmatch.fnmatch(file, '*.sh') and
        file[11:14] == '024']
        cropping = cr.Crop()
        files_input_grad_vv.sort(key=cropping.get_sort_key_input)
        path_input_vv = []
        for file in files_input_grad_vv:
            file_name = os.path.join(INPUT_24H_GRAD_VV, file)
            path_input_vv.append(file_name)
            print(f'File in chronological order are: {file}')
            print(f'The path of the input temporal variable are: {file_name}')
        print(f'The length of the initial files_input array is {len(path_input_vv)}')

        #Load the input 24 hours tt gradient standard file and put them in chronological order in an array. 
        files_input_grad_tt = [file for file in os.listdir(INPUT_24H_GRAD_TT)
        if os.path.isfile(os.path.join(INPUT_24H_GRAD_TT, file)) and
        not fnmatch.fnmatch(file, '*.tgz') and
        not fnmatch.fnmatch(file, '*.sh') and
        file[11:14] == '024']
        cropping = cr.Crop()
        files_input_grad_tt.sort(key=cropping.get_sort_key_input)
        path_input_tt = []
        for file in files_input_grad_tt:
            file_name = os.path.join(INPUT_24H_GRAD_TT, file)
            path_input_tt.append(file_name)
            print(f'File in chronological order are: {file}')
            print(f'The path of the input temporal variable are: {file_name}')
        print(f'The length of the initial files_input array is {len(path_input_tt)}')

        #Load the input 24 hours uv gradient standard file and put them in chronological order in an array. 
        files_input_grad_uv = [file for file in os.listdir(INPUT_24H_GRAD_UV)
        if os.path.isfile(os.path.join(INPUT_24H_GRAD_UV, file)) and
        not fnmatch.fnmatch(file, '*.tgz') and
        not fnmatch.fnmatch(file, '*.sh') and
        file[11:14] == '024']
        cropping = cr.Crop()
        files_input_grad_uv.sort(key=cropping.get_sort_key_input)
        path_input_uv = []
        for file in files_input_grad_uv:
            file_name = os.path.join(INPUT_24H_GRAD_UV, file)
            path_input_uv.append(file_name)
            print(f'File in chronological order are: {file}')
            print(f'The path of the input temporal variable are: {file_name}')
        print(f'The length of the initial files_input array is {len(path_input_uv)}')

        #Load the input 24 hours standard file and put them in chronological order in an array. 
        files_input_std = [file for file in os.listdir(INPUT_24H_STD)
        if os.path.isfile(os.path.join(INPUT_24H_STD, file)) and
        not fnmatch.fnmatch(file, '*.tgz') and
        not fnmatch.fnmatch(file, '*.sh') and
        file[11:14] == '024']
        cropping = cr.Crop()
        files_input_std.sort(key=cropping.get_sort_key_input)
        #Validate that the file is in the other folders (Do it because there are more data on Simon-Philippe path and it needs to be deleted)
        for i, file in enumerate(files_input_std):
            if file not in files_input_grad_uu:
                files_input_std.pop(i)
        path_input_std = []
        file_std = []
        for file in files_input_std:
            file_std.append(file)
            file_name = os.path.join(INPUT_24H_STD, file)
            path_input_std.append(file_name)
            print(f'File in chronological order are: {file}')
            print(f'The path of the input temporal variable are: {file_name}')
        print(f'The length of the initial files_input array is {len(path_input_std)}')

        #Validate that the data is the same everywhere:
        assert files_input_std == files_input_grad_uu == files_input_grad_vv == files_input_grad_tt == files_input_grad_uv, "The file are not the same, there is an issue in your input data."

        #Combine standard file together.
        self.combine_std(PATH_COMSTD, path_input_std, path_input_uu, path_input_vv, path_input_tt, path_input_uv, file_std)

        print(f'Combining the files have work succesfully, you can now convert the file from std file to numpy!')

#.**************************************************************************************************************************************.#
#                                                                 Main                                                                   #
#.**************************************************************************************************************************************.#
if __name__ == '__main__':
    #List of variables used to run the script
    PATH_COMSTD = '/home/jfg000/ss5/SuperResolution/Data_Processing/A00_calculate_gradients'
    INPUT_24H_STD = '/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel'
    INPUT_24H_GRAD_UU = '/home/jfg000/ss5/Data/input/initial_data_reps/gradient/uu'
    INPUT_24H_GRAD_VV = '/home/jfg000/ss5/Data/input/initial_data_reps/gradient/vv'
    INPUT_24H_GRAD_TT = '/home/jfg000/ss5/Data/input/initial_data_reps/gradient/tt'
    INPUT_24H_GRAD_UV = '/home/jfg000/ss5/Data/input/initial_data_reps/gradient/uv'
    PATH_COMBINED_STD = '/home/jfg000/ss5/SuperResolution/Data_Processing' #Path where is located combine_std.sh file 
    COMBINING_STD_DATA_INPUT = '/home/jfg000/ss5/Data/input/initial_data_reps/combined'
    COMBINING_STD_DATA_INPUT_TEMP = '/home/jfg000/ss5/Data/input/initial_data_reps/combined_temp'

    if not os.path.exists(COMBINING_STD_DATA_INPUT):
        os.makedirs(COMBINING_STD_DATA_INPUT)
    if not os.path.exists(COMBINING_STD_DATA_INPUT_TEMP):
        os.makedirs(COMBINING_STD_DATA_INPUT_TEMP)
    
    combStd = Combine_standard_files()
    combStd.get_combine_std(PATH_COMSTD, INPUT_24H_STD, INPUT_24H_GRAD_UU, INPUT_24H_GRAD_VV, INPUT_24H_GRAD_TT, INPUT_24H_GRAD_UV, PATH_COMBINED_STD, COMBINING_STD_DATA_INPUT, COMBINING_STD_DATA_INPUT_TEMP)