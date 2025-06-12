#!/usr/bin/env python

""" 
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Created by: Jean-Sébastien Giroux, Date: 2023-11-21
    Inspired by: Kevin Gauthier, Data: 2019
"""


#.**************************************************************************************************************************************.#
#                                                          Code description                                                              #
#.**************************************************************************************************************************************.#
"""
    TASK NAME - Convert input and label data to numpy format to allow the Neural Network to use data more efficently.  

     STATUS - experimental

     DESCRIPTION - This script will take the input and label crop result and convert these results into Numpy format to allow the Neural
                   Network to work efficently with the data. 
"""     


#.******************tr********************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import argparse
import csv
import os
import numpy as np 
import rpnpy.librmn.all as rmn
import sys
from datetime import datetime, timedelta

sys.path.append('/home/jfg000/ss5/SuperResolution/Data_Processing/A_crop_interpolate')
import crop as cr


#.**************************************************************************************************************************************.#
#                                                             Variables                                                                  #
#.**************************************************************************************************************************************.#
DOTOPO = False #Flag that will be set to true after the input and label convertion to numpy. 


#.**************************************************************************************************************************************.#
#                                                             Functions                                                                  #
#.**************************************************************************************************************************************.#
class ConvertToNPY():
    def readStdFile(self, fileName, VARIABLE_NAME_ARRAY):
        """
        Read a standard file 
        INPUT:
        :param fileName: Path where is located the file the program will open.   
        RETURN:
        :param readData: The data that was in the opened standard file that the program just oppened. 
        """
        #Try to open the file
        fileIdIn  = rmn.fstopenall(fileName)

        try:
            #Define an empty array to read the data. 
            readData = []
            #Itterate through every caracters of variableNameArray array.
            for (variableName_key, variableName_value)  in VARIABLE_NAME_ARRAY.items():
                print(f'The variable is: {variableName_key}')
                print(f'The ip1 adress is: {variableName_value}')
                for value in variableName_value:
                    #Get the list of records to copy #Be careful, it might inverse the image.
                    keylist = rmn.fstinl(fileIdIn, nomvar=variableName_key, ip1=value)
                    # print(f'Key list is now: {keylist}')

                    for key in keylist:
                        #Read record data and meta from fileNameIn
                        readData.append(rmn.fstluk(key))
        except:
            sys.stderr.write("Could not write the file: %s\n" % (fileName))
            pass

        finally:
            #Properly close files even if an error occured above
            rmn.fstcloseall(fileIdIn)

            return readData

    def keepOnlyDataFromReadFile(self, array):
        """
        This function iterates over a list (or array) of dictionaries and modifies each element 
        by extracting and keeping only the data associated with the key 'd'. It assumes that 
        each element in the input list is a dictionary containing the key 'd'. 

        INPUT
        :param array: list of dictionaries, where each dictionary contains the key 'd'

        OUTPUT
        :return: list where each element is the value from the original dictionaries corresponding to the key 'd'
        """
        for i in range(len(array)):
            # print(array[i].keys())
            array[i] = array[i]["d"]
        
        return array
    
    def check_dimensions(data):
        for i, sublist in enumerate(data):
            for j, array in enumerate(sublist):
                if isinstance(array, np.ndarray):
                    x, y = array.shape
                    if x != 128 or y != 128:
                        print(f"Array at data[{i}][{j}] does not have dimensions 128x128. Shape: {array.shape}")

    def convert_to_npy(self, data, CROP_DATA, np_data, VARIABLE_NAME_ARRAY, TOPOGRAPHIE, doTopo):
        """
        INPUT
        :Param data: 
        :Param np_data: 
        :Constant VARIABLE_NAME_ARRAY:  
        OUTPUT
        :Param np_data: 
        """
        # print('The program started the convertion to nunpy.')
        if not doTopo:
            #Loop throw all the elements of CROP_INPUT_LABEL_DATA.
            for i, fieldnames in enumerate(data):
                print(f'The current file number is : {i}')
                print(f'The program is converting file: {fieldnames}')
                #Creating the path of the current data.
                filepath = os.path.join(CROP_DATA, fieldnames)
                #Call function to read the standard file input of the data. 
                oneFile = self.readStdFile(filepath, VARIABLE_NAME_ARRAY)
                #Call function to keep only data from readfile
                self.keepOnlyDataFromReadFile(oneFile)
                np_data.append(oneFile)
                # print(f'The program finished the convertion for data {filepath}')
                
        elif doTopo and TOPOGRAPHIE:
            #Creating the path of the current data.
            filepath = os.path.join(CROP_DATA, data)
            #Call function to read the standard file input of the data. 
            oneFile = self.readStdFile(filepath, VARIABLE_NAME_ARRAY)
            #Call function to keep only data from readfile
            self.keepOnlyDataFromReadFile(oneFile)
            np_data.append(oneFile)

        return np_data
    
    def check_homogeneous_shape(data):
        shapes = [np.shape(item) for item in data]
        unique_shapes = set(shapes)
        
        if len(unique_shapes) > 1:
            print("Inhomogeneous shapes found:")
            for shape in unique_shapes:
                print(shape)
            return False
        else:
            print("All elements have the same shape:", unique_shapes)
            return True
    
    def get_convertion(self, CROP_INPUT_DATA, CROP_LABEL_DATA, CROP_PREDICTED_VALUE_DATA, NPY_INPUT_DATA, NPY_LABEL_DATA, NPY_SKIP_DATA, VARIABLE_NAME_ARRAY_INPUT, VARIABLE_NAME_ARRAY_LABEL,
                       TOPOGRAPHIE, PREDICTED_VALUE, CROP_TOPO_DATA, NPY_TOPO_DATA, TOPO_STD_NANE, VARIABLE_NAME_ARRAY_TOPO, DOTOPO):
        if TOPOGRAPHIE: 
            #Put the flag to true to tell the program the input and label have been convert to npy. 
            DOTOPO  = True
            #Initialize the numpy topography data array.
            np_topo_data = []
            #Convert from standard file to numpy for topo data. 
            np_topo_data = self.convert_to_npy(TOPO_STD_NANE, CROP_TOPO_DATA, np_topo_data, VARIABLE_NAME_ARRAY_TOPO, TOPOGRAPHIE, DOTOPO)
            # print(f'The length is of label array: {len(np_topo_data)}')
            #Save the topography data to numpy format. 
            np.save(NPY_TOPO_DATA, np_topo_data)
        elif PREDICTED_VALUE:
            #Create an objet of class cropping
            cropping = cr.Crop()
            #Put the input data in chronological order to allow to associate input and label correctly.
            print(f'The path is: {CROP_PREDICTED_VALUE_DATA}')
            files_input_croped = os.listdir(CROP_PREDICTED_VALUE_DATA)
            files_input_croped.sort(key=cropping.get_sort_key_input)
            # for file in files_input_croped:
            #     print(f'File in chronological order are: {file}')
            # print(f'The length of the initial files_input array is {len(files_input_croped)}')

            #Initialize the numpy input data arrray  
            np_input_data = []
            #Convert from standard file to numpy for input data. 
            np_input_data = self.convert_to_npy(files_input_croped, CROP_PREDICTED_VALUE_DATA, np_input_data, VARIABLE_NAME_ARRAY_LABEL, TOPOGRAPHIE, DOTOPO)
            # self.check_dimensions(np_input_data)
            # print(f'The length is of input array: {len(np_input_data)}')
            # for i, element in enumerate(np_input_data):
            #     print(f'The len of the {i+1} element is: {len(element)}')
            #Save the input data to numpy format
            np.save(NPY_SKIP_DATA, np_input_data)
            loading = np.load(f'{NPY_SKIP_DATA}.npy')
            print(f'The loading shape is : {loading.shape}')
        else:
            #Create an objet of class cropping
            cropping = cr.Crop()
            #Put the input data in chronological order to allow to associate input and label correctly.
            files_input_croped = os.listdir(CROP_INPUT_DATA)
            files_input_croped.sort(key=cropping.get_sort_key_input)
            # for file in files_input_croped:
            #     print(f'File in chronological order are: {file}')
            # print(f'The length of the initial files_input array is {len(files_input_croped)}')
            #Put the label data in chronological order to allow to associate input and label correctly.
            files_label_croped = os.listdir(CROP_LABEL_DATA) 
            files_label_croped.sort(key=cropping.get_sort_key_label)
            #Error handeling. This file was created for no reasons. need to remove it... #TODO find the reason why this file is being created. 
            if '.make_xml.sh.swp' in files_label_croped:
                files_label_croped.remove('.make_xml.sh.swp')

            # for file in files_label_croped:
            #     print(f'File in chronological order are: {file}')
            # print(f'The length of the initial files_label array is {len(files_label_croped)}')

            #Verifying if the index are well done. (TEST can be deleted later...)
            for i, file in enumerate(files_label_croped): 
                print(f'This is the input file: {files_input_croped[i]} associate with this label file: {file}.')
            #Verification that the function work well. 
            print(f'The function works: {self.Test_DateVerrification(files_input_croped, files_label_croped)}')

            # #Test with a small data size TODO delete this...
            # files_input_croped = files_input_croped[:30]
            # files_label_croped = files_label_croped[:30]

            # #Initialize the numpy input data arrray  
            # np_input_data = []
            # #Convert from standard file to numpy for input data. 
            # np_input_data = self.convert_to_npy(files_input_croped, CROP_INPUT_DATA, np_input_data, VARIABLE_NAME_ARRAY_INPUT, TOPOGRAPHIE, DOTOPO)
            # # print(f'The length is of input array: {len(np_input_data)}')
            # # for i, element in enumerate(np_input_data):
            # #     print(f'The len of the {i+1} element is: {len(element)}')
            # #Save the input data to numpy format
            # np.save(NPY_INPUT_DATA, np_input_data)

            #Initialize the numpy label data arrray  
            np_label_data = []
            #Convert from standard file to numpy for label data. 
            np_label_data = self.convert_to_npy(files_label_croped, CROP_LABEL_DATA, np_label_data, VARIABLE_NAME_ARRAY_LABEL, TOPOGRAPHIE, DOTOPO)
            # print(f'The length is of input array: {len(np_label_data)}')
            #Save the label data to numpy format.
            np.save(NPY_LABEL_DATA, np_label_data)
        
        print('The numpy convertion has worked successfully.')


    #.**************************************************************************************************************************************.#
    #                                                                Tests                                                                   #
    #.**************************************************************************************************************************************.#
    def Test_DateVerrification(self, files_input, files_label): 
        #Normalize the input and label name.
        normalized_inputs = [None] * len(files_input)
        normalized_labels = [None] * len(files_label)
        for i, files in enumerate(files_input):
            #Verification of the model pass curently being analysed. 
            new_files = int(files[12:14])
            if int(files[8:10]) == 0:
                correction = 0
                new_files = new_files + correction
            elif int(files[8:10]) == 6:
                correction = 6
                new_files = new_files + correction
            elif int(files[8:10]) == 12:
                correction = 12
                new_files = new_files + correction
            elif int(files[8:10]) == 18:
                correction = 18
                date_format = "%Y%m%d%H"
                actual_date = files[:8] + files[12:14]
                date_obj = datetime.strptime(actual_date, date_format)
                new_date_obj = date_obj + timedelta(hours=correction)
                new_files = new_date_obj.strftime(date_format)
            
            if len(str(new_files)) == 1 and not int(files[8:10]) == 18:
                new_files = str(0) + str(new_files)
                normalized_inputs[i] = files[:8] + str(new_files)
            elif  int(files[8:10]) == 18:
                normalized_inputs[i] = new_files
            else:
                normalized_inputs[i] = files[:8] + str(new_files)
        #     print(f'The normalized label is: {normalized_labels[i]}')
        # print('\n')
        #Convert the initial value depending of the original value. 
        for i, files in enumerate(files_label): 
            normalized_labels[i] = files[:8] + str(files[8:10])
            # print(f'label: {normalized_labels[i]}')
        # print(f'The length of the label is: {len(normalized_labels)}')
        for inputs in normalized_inputs:
            if not(inputs in normalized_labels):
                print(f'This file: {inputs} is not in normalized_labels.')
                sys.exit('Error occurred, the inputs files are not the same than the labels file')
        for labels in normalized_labels:
            if not(labels in normalized_inputs):
                print(f'This file: {labels} is not in normalized_inputs.')
                sys.exit('Error occurred, the labels files are not the same than the inputs file')
        return True

#.**************************************************************************************************************************************.#
#                                                                 Main                                                                   #
#.**************************************************************************************************************************************.#
if __name__ == '__main__':
    #Create an object of class crop
    convert_to_npy = ConvertToNPY()

    #Give the parse to a variable
    INPUT_DATA_LOCATION = '/input/initial_data_new/test_sequential'
    LABEL_DATA_LOCATION = '/label/initial_data_new/east_canada_sequential/test_sequential'
    WIND = 'UV'
    VARIABLE_NAME='UU_VV_TT_P0_PN_H_CX_SD_WGE'
    PREDICTED_VALUE = True

    PROJECT_PATH = '/home/jfg000/ss5/SuperResolution'
    DATA_PROJECT_PATH = '/home/jfg000/ss5/Data'
    DOMAINE_INPUT='prep_domaine/Grid_REPS_Est_Canada_Dom1_fornointerp_big'
    DOMAINE_LABEL='prep_domaine/Grid_meso_Est_Canada_Dom1_fornointerp_big'
    DOMAINE_NAME = 'east_canada_squential_train_allpasses_domaine5'

    INPUT_DATA = f"{DATA_PROJECT_PATH}/{INPUT_DATA_LOCATION}" 
    LABEL_DATA = f"{DATA_PROJECT_PATH}/{LABEL_DATA_LOCATION}" 
    DOMAINE_INPUT = f'{DATA_PROJECT_PATH}/{DOMAINE_INPUT}'  #Grid_REPS_Gaspesie_fornointerp
    DOMAINE_LABEL = f'{DATA_PROJECT_PATH}/{DOMAINE_LABEL}' #Grid_meso_Gaspesie_fornointerp
    NPY_INPUT_DATA=f"{DATA_PROJECT_PATH}/input/domaine_new/{DOMAINE_NAME}/{WIND}_{VARIABLE_NAME}_data"
    NPY_LABEL_DATA=f"{DATA_PROJECT_PATH}/label/domaine_new/{DOMAINE_NAME}/{WIND}_{VARIABLE_NAME}_data"
    NPY_SKIP_DATA=f"{DATA_PROJECT_PATH}/input/domaine_new/{DOMAINE_NAME}/skip_connection_{WIND}_{VARIABLE_NAME}_data"

    if not os.path.exists(NPY_INPUT_DATA):
        os.makedirs(NPY_INPUT_DATA)
    if not os.path.exists(NPY_LABEL_DATA):
        os.makedirs(NPY_LABEL_DATA)
    if not os.path.exists(NPY_SKIP_DATA):
        os.makedirs(NPY_SKIP_DATA)

    CROP_INPUT_DATA = f'{DATA_PROJECT_PATH}/input/domaine_new/{DOMAINE_NAME}/crop_predicted_{WIND}_{VARIABLE_NAME}'
    CROP_LABEL_DATA = f'{DATA_PROJECT_PATH}/label/domaine_new/{DOMAINE_NAME}/crop_predicted_{WIND}_{VARIABLE_NAME}'
    CROP_PREDICTED_VALUE_DATA = f'{DATA_PROJECT_PATH}/input/domaine_new/{DOMAINE_NAME}/crop_predicted_{WIND}_{VARIABLE_NAME}'
    NPY_INPUT_DATA = f'{NPY_INPUT_DATA}/input'
    NPY_LABEL_DATA = f'{NPY_LABEL_DATA}/label'
    NPY_SKIP_DATA = f'{NPY_SKIP_DATA}/skip'

    #Dict for variable 
    VARIABLE_NAME_ARRAY_INPUT_dict = {'UV': [75597472, 95369342, 95364364, 95357866, 95349708, 95339883],                          
                                      'UU': [75597472, 95369342, 95364364, 95357866, 95349708, 95339883],
                                      'VV': [75597472, 95369342, 95364364, 95357866, 95349708, 95339883],
                                      'TT': [76696048, 95370590, 95366850, 95361109, 95353779, 95344783],                              
                                      'P0': [-1],
                                      'PN' : [-1], 
                                      'H': [-1],
                                      'CX': [-1],
                                      'SD': [60268832],
                                      'WGE': [-1],
                                      }   
    
    VARIABLE_NAME_ARRAY_LABEL_dict = {'UV': [75597472]
                                      }
    
    VARIABLE_NAME_ARRAY_TOPO_dict = {'ME': [-1],
                                     'MG': [-1],
                                     'Z0': [-1]}

    TOPOGRAPHIE = False
    CROP_TOPO_DATA = f'{DATA_PROJECT_PATH}/topography/{DOMAINE_NAME}/crop'
    NPY_TOPO_DATA=f'{DATA_PROJECT_PATH}/topography/{DOMAINE_NAME}/topo'
    TOPO_STD_NANE = "HR_input_from_HRDPS"  

    # CROP_INPUT_DATA = '/home/jfg000/ss5/Data/input/domaine/east_canada_squential_test_allpasses_domaine1/crop_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE'
    # CROP_LABEL_DATA = '/home/jfg000/ss5/Data/label/domaine/east_canada_squential_test_allpasses_domaine1/crop_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE'

    convert_to_npy.get_convertion(CROP_INPUT_DATA, CROP_LABEL_DATA, CROP_PREDICTED_VALUE_DATA, NPY_INPUT_DATA, NPY_LABEL_DATA, NPY_SKIP_DATA, VARIABLE_NAME_ARRAY_INPUT_dict, VARIABLE_NAME_ARRAY_LABEL_dict,
                       TOPOGRAPHIE, PREDICTED_VALUE, CROP_TOPO_DATA, NPY_TOPO_DATA, TOPO_STD_NANE, VARIABLE_NAME_ARRAY_TOPO_dict, DOTOPO)
