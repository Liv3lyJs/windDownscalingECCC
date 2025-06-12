#!/usr/bin/env python

""" 
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Created by: Jean-Sébastien Giroux, Date: 2024-02-09
"""

#.**************************************************************************************************************************************.#
#                                                          Code description                                                              #
#.**************************************************************************************************************************************.#
"""
    TASK NAME - Do the whole data processing in a single python script. This will allow the user to only select the data he wants to create
                and the whole data processing process will get done. 

     STATUS - experimental

     DESCRIPTION - TODO
"""     

#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import argparse
import os
import sys

#Change base on the project location. To allow it working with your computer project. 
PROJECT_PATH = '/home/jfg000/ss5/SuperResolution'
DATA_PROJECT_PATH = '/home/jfg000/ss5/data_superResolution'

PATH_A_CROP_INTERPOLATE = f'{PROJECT_PATH}/Data_Processing/A_crop_interpolate'
PATH_B_CONVERT_NPY = f'{PROJECT_PATH}/Data_Processing/B_convert_npy'
PATH_C_NORMALIZE = f'{PROJECT_PATH}/Data_Processing/C_normalize'

#Add the path used in the script.
sys.path.append(PATH_A_CROP_INTERPOLATE)
sys.path.append(PATH_B_CONVERT_NPY)
sys.path.append(PATH_C_NORMALIZE)
sys.path.append('/home/jfg000/ss5/SuperResolution/Data_Processing')

#Add the script where class and method will be used. 
# import calculate_gradients as cg
import input_label_toNpy as conv_npy
import crop as cr
import normalize as norm
import normalize_domain as norm_dom

#.**************************************************************************************************************************************.#
#                                                             Functions                                                                  #
#.**************************************************************************************************************************************.#
def argparser():
    """
    Take the user input in the concatenate_input_topo.sh script. This input will then be used in this code as variables and constants. 
    INPUT
    :none
    OUTPUT
    :Param parser.parse_args(): Used to parse the user input with a variable or a constant. 
    """
    # Argparse initialisation to allow external input to be used in the code
    parser = argparse.ArgumentParser(description="Arguments used in the code that are called in the bash file or the terminal")

    parser.add_argument('--DOMAINE_NAME', type=str, help='Name of the domain that is currently being analysed.')
    parser.add_argument('--WIND', type=str, help='Target wind variable. Can be ether UU, VV or UV')
    parser.add_argument('--VARIABLE_NAME', type=str, help='Name of the variables that are currently being analysed')
    parser.add_argument('--TOPOGRAPHIE_NAME', type=str, help='Topography variables that are curently being used.')
    parser.add_argument('--VARIABLE_NAME_ARRAY_INPUT',  nargs='+', help='Array of variables that are used as input variables.')
    parser.add_argument('--VARIABLE_NAME_ARRAY_TOPO',  nargs='+', help='Array of variables that are ussed as topography input variables.')
    parser.add_argument('--TOPOGRAPHIE', action='store_true', help='Flag to know if the user is doing the domaine topography.')
    parser.add_argument('--PREDICTED_VALUE', action='store_true', help='Flag used when the user want to extract the predicted value. In this case UV.')
    parser.add_argument('--NORM_DOMAINE', action='store_true', help='Flag used when the user want to normalize the domain.')
    parser.add_argument('--TEST', action='store_true', help='Flag to know if the user is doing a test.')
    parser.add_argument('--INPUT_DATA_LOCATION', type=str, help='Names of the input data location.')
    parser.add_argument('--LABEL_DATA_LOCATION', type=str, help='Name of the label data location.')
    parser.add_argument('--START_HOUR_INPUT', type=int, help='Starting time of the analyse for input.')
    parser.add_argument('--END_HOUR_INPUT', type=int, help='Ending time of the analyse for input.')
    parser.add_argument('--START_HOUR_LABEL', type=int, help='Starting time of the analyse for label.')
    parser.add_argument('--END_HOUR_LABEL', type=int, help='Ending time of the analyse for label.')
    parser.add_argument('--DOMAINE_INPUT', type=str, help='The domaine input.')
    parser.add_argument('--DOMAINE_LABEL', type=str, help='The domaine label.')
    parser.add_argument('--DO_GRADIENTS', action='store_true', help='Flag used when calculating the gradients of variables. By defaut, it does UU, VV and TT.')
    parser.add_argument('--INTERPOLATION_TYPE', type=str, help='Name of the interpolation being executed. It will be used as the name to store values.')
    parser.add_argument('--DATA_TYPE', type=str, help='Name of the type of data being done. Can be either domaine_creation or data_used_in_neural_network')

    return parser.parse_args()

#.**************************************************************************************************************************************.#
#                                                                 Main                                                                   #
#.**************************************************************************************************************************************.#
if __name__ == '__main__':
    #Parse the argument
    args = argparser()

    DOMAINE_NAME = args.DOMAINE_NAME
    WIND = args.WIND
    VARIABLE_NAME = args.VARIABLE_NAME
    TOPOGRAPHIE_NAME = args.TOPOGRAPHIE_NAME
    VARIABLE_NAME_ARRAY_INPUT = args.VARIABLE_NAME_ARRAY_INPUT
    VARIABLE_NAME_ARRAY_TOPO = args.VARIABLE_NAME_ARRAY_TOPO
    TOPOGRAPHIE = args.TOPOGRAPHIE
    PREDICTED_VALUE = args.PREDICTED_VALUE
    NORM_DOMAINE = args.NORM_DOMAINE
    TEST = args.TEST
    INPUT_DATA_LOCATION = args.INPUT_DATA_LOCATION
    LABEL_DATA_LOCATION = args.LABEL_DATA_LOCATION
    START_HOUR_INPUT = args.START_HOUR_INPUT
    END_HOUR_INPUT = args.END_HOUR_INPUT
    START_HOUR_LABEL = args.START_HOUR_LABEL
    END_HOUR_LABEL = args.END_HOUR_LABEL
    DOMAINE_INPUT = args.DOMAINE_INPUT
    DOMAINE_LABEL = args.DOMAINE_LABEL
    DO_GRADIENTS = args.DO_GRADIENTS
    INTERPOLATION_TYPE = args.INTERPOLATION_TYPE
    DATA_TYPE = args.DATA_TYPE
    
    #Cropping constants list
    INPUT_DATA = f"{DATA_PROJECT_PATH}/{INPUT_DATA_LOCATION}" 
    LABEL_DATA = f"{DATA_PROJECT_PATH}/{LABEL_DATA_LOCATION}" 
    DOMAINE_INPUT = f'{DATA_PROJECT_PATH}/{DOMAINE_INPUT}'  #Grid_REPS_Gaspesie_fornointerp
    DOMAINE_LABEL = f'{DATA_PROJECT_PATH}/{DOMAINE_LABEL}' #Grid_meso_Gaspesie_fornointerp
    CROP_INPUT_DATA = f'{DATA_PROJECT_PATH}/input/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/crop_{WIND}_{VARIABLE_NAME}'
    CROP_LABEL_DATA = f'{DATA_PROJECT_PATH}/label/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/crop_{WIND}_{VARIABLE_NAME}'
    PGSM_PATH = f'{PROJECT_PATH}/Data_Processing/A_crop_interpolate'
    TOPO_PATH = '/fs/site5/eccc/cmd/x/spb001/storage_geophy/PREP_GEOPHY-3.3/Geophy_REPSgridHRDPS/HRDPS/Nat/geoREPSgridHRDPS/2p5/Output_GPX/data' #"/fs/site5/eccc/cmd/x/spb001/SuperResolution/get_HRinput_from_HRDPS"
    CROP_TOPO_DATA = f'{DATA_PROJECT_PATH}/topography/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/crop'
    TOPO_STD_NANE = 'HR_input_from_HRDPS_gridREPS' #"HR_input_from_HRDPS"   
    CROP_PREDICTED_VALUE_DATA = f'{DATA_PROJECT_PATH}/input/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/crop_predicted_{WIND}_{VARIABLE_NAME}'

    #Convert to numpy constant list
    NPY_TOPO_DATA = f'{DATA_PROJECT_PATH}/topography/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/topo'   
    DOTOPO = False
    NPY_INPUT_DATA=f"{DATA_PROJECT_PATH}/input/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/{WIND}_{VARIABLE_NAME}_data"
    NPY_LABEL_DATA=f"{DATA_PROJECT_PATH}/label/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/{WIND}_{VARIABLE_NAME}_data"
    NPY_SKIP_DATA=f"{DATA_PROJECT_PATH}/input/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/skip_connection_{WIND}_{VARIABLE_NAME}_data"
    VARIABLE_NAME_ARRAY_LABEL = [f"{WIND}"]

    #Normalize constant list
    NORMALIZE_INPUT_DATA = f"{DATA_PROJECT_PATH}/input/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/{WIND}_{VARIABLE_NAME}_data"
    NORMALIZE_LABEL_DATA = f"{DATA_PROJECT_PATH}/label/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/{WIND}_{VARIABLE_NAME}_data"
    NORMALIZE_SKIP_DATA = f"{DATA_PROJECT_PATH}/input/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/skip_connection_{WIND}_{VARIABLE_NAME}_data"
    NORMALIZE_TOPO_DATA = f"{DATA_PROJECT_PATH}/topography/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/{TOPOGRAPHIE_NAME}_data"
    PATH_SAVE_MEAN = f'{DATA_PROJECT_PATH}/input/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/{WIND}_{VARIABLE_NAME}_data/mean'
    PATH_SAVE_STD = f'{DATA_PROJECT_PATH}/input/domaine/{INTERPOLATION_TYPE}/{DATA_TYPE}/{DOMAINE_NAME}/{WIND}_{VARIABLE_NAME}_data/std'

    #Create folder location. 
    #Folder used for cropping
    if not os.path.exists(CROP_INPUT_DATA):
        os.makedirs(CROP_INPUT_DATA)
    if not os.path.exists(CROP_LABEL_DATA):
        os.makedirs(CROP_LABEL_DATA)
    if TOPOGRAPHIE:
        if not os.path.exists(CROP_TOPO_DATA):
            os.makedirs(CROP_TOPO_DATA)
    if PREDICTED_VALUE:
        if not os.path.exists(CROP_PREDICTED_VALUE_DATA):
            os.makedirs(CROP_PREDICTED_VALUE_DATA)
    #Folder used to convert from standard file to numpy
    if not os.path.exists(NPY_INPUT_DATA):
        os.makedirs(NPY_INPUT_DATA)
    if not os.path.exists(NPY_LABEL_DATA):
        os.makedirs(NPY_LABEL_DATA)
    if not os.path.exists(NPY_SKIP_DATA):
        os.makedirs(NPY_SKIP_DATA)
    if not os.path.exists(NPY_TOPO_DATA):
        os.makedirs(NPY_TOPO_DATA)
    NPY_INPUT_DATA = f'{NPY_INPUT_DATA}/input' #input all_dom
    NPY_LABEL_DATA = f'{NPY_LABEL_DATA}/label' #label all_dom
    if PREDICTED_VALUE:
        NPY_SKIP_DATA = f'{NPY_SKIP_DATA}/skip' #skip all_dom
    if TOPOGRAPHIE:
        NPY_TOPO_DATA = f'{NPY_TOPO_DATA}/tg' #tg all_dom

    if not os.path.exists(NORMALIZE_INPUT_DATA):
        os.makedirs(NORMALIZE_INPUT_DATA)
    if not os.path.exists(NORMALIZE_LABEL_DATA):
        os.makedirs(NORMALIZE_LABEL_DATA)

    NORMALIZE_INPUT_DATA = f'{NORMALIZE_INPUT_DATA}/normalize'  # normalize_05
    NORMALIZE_LABEL_DATA = f'{NORMALIZE_LABEL_DATA}/normalize'
    if PREDICTED_VALUE:
        if not os.path.exists(NORMALIZE_SKIP_DATA):
            os.makedirs(NORMALIZE_SKIP_DATA)
        NORMALIZE_SKIP_DATA = f'{NORMALIZE_SKIP_DATA}/normalize'
    if TOPOGRAPHIE:
        if not os.path.exists(NORMALIZE_TOPO_DATA):
            os.makedirs(NORMALIZE_TOPO_DATA)
        NORMALIZE_TOPO_DATA = f'{NORMALIZE_TOPO_DATA}/normalize'

    #Dict for variable 
    VARIABLE_NAME_ARRAY_INPUT_dict = {'UV': [75597472, 95369342, 95364364, 95357866],                      # , 95349708, 95339883     
                                      'UU': [75597472, 95369342, 95364364, 95357866],
                                      'VV': [75597472, 95369342, 95364364, 95357866],
                                      'TT': [76696048, 95370590, 95366850, 95361109],                      # , 95353779, 95344783
                                      'UV_G': [-1],
                                      'UU_G': [-1],
                                      'VV_G': [-1],
                                      'TT_G': [-1],                      
                                      'P0': [-1],
                                      'PN' : [-1], 
                                      'H': [-1],
                                      'CX': [-1],
                                      'SD': [60268832],
                                      'WGE': [-1],
                                      }   
    
    VARIABLE_NAME_ARRAY_LABEL_dict = {'VV': [75597472]
                                      }
    
    VARIABLE_NAME_ARRAY_TOPO_dict = {'ME': [-1],
                                     'MG': [-1],
                                     'Z0LC': [-1]} #Z0LC Z0

    """
    A - Do the croping of original input, label and domaine data.
        Before going to B, you need to complete croping of all input, label and domain you want to use 
        in the as data to train the Neural Network with. 
    """
    # #Create an object of class crop
    # cropping = cr.Crop()
    # #Call the method to do cropping from crop class.
    # cropping.get_crop(INPUT_DATA, LABEL_DATA, DOMAINE_INPUT, DOMAINE_LABEL, PREDICTED_VALUE, CROP_PREDICTED_VALUE_DATA, CROP_INPUT_DATA, CROP_LABEL_DATA, PGSM_PATH,
    #                 START_HOUR_INPUT, END_HOUR_INPUT, START_HOUR_LABEL, END_HOUR_LABEL, TOPOGRAPHIE, TOPO_PATH, CROP_TOPO_DATA, TOPO_STD_NANE)

    """
    B - Convert the standard file to numpy file, that will be used to train the Neural Network with.
    """
    # #Create an object of class input_label_toNpy
    # convert_to_npy = conv_npy.ConvertToNPY()
    # print('The program is going to to the convertion from standard file to numpy file.')
    # #Call the method to do convertion to numpy from ConvertToNPY class.
    # convert_to_npy.get_convertion(CROP_INPUT_DATA, CROP_LABEL_DATA, CROP_PREDICTED_VALUE_DATA, NPY_INPUT_DATA, NPY_LABEL_DATA, NPY_SKIP_DATA, VARIABLE_NAME_ARRAY_INPUT_dict, VARIABLE_NAME_ARRAY_LABEL_dict,
    #                    TOPOGRAPHIE, PREDICTED_VALUE, CROP_TOPO_DATA, NPY_TOPO_DATA, TOPO_STD_NANE, VARIABLE_NAME_ARRAY_TOPO_dict, DOTOPO)

    """
    C- Standardize the numpy file using only the input information. The target information will not be available later when doing inference with
        the model, so only use the input information. Note that the input and label will be standardize between value of -1 and 1. 
        The standardization is done on each variable individually and on each images individually. 
    """
    #Create an object of class normalize
    normalize = norm.Normalize()
    print('The program is doing the noramlization of the data.')
    NPY_INPUT_DATA = f'{NPY_INPUT_DATA}.npy'
    NPY_LABEL_DATA = f'{NPY_LABEL_DATA}.npy'
    NPY_SKIP_DATA = f'{NPY_SKIP_DATA}.npy'
    NPY_TOPO_DATA = f'{NPY_TOPO_DATA}.npy'
    #Call the method to normalize the input and target data.
    normalize.get_normalization(NPY_INPUT_DATA, NPY_LABEL_DATA, NPY_SKIP_DATA, NPY_TOPO_DATA, NORMALIZE_INPUT_DATA, NORMALIZE_LABEL_DATA, NORMALIZE_SKIP_DATA, NORMALIZE_TOPO_DATA, VARIABLE_NAME_ARRAY_INPUT,
                                 VARIABLE_NAME_ARRAY_LABEL, TEST, PATH_SAVE_MEAN, PATH_SAVE_STD, TOPOGRAPHIE, PREDICTED_VALUE, INTERPOLATION_TYPE)

    print('Data processing is complete and have work successfully. Go train that network now.')
