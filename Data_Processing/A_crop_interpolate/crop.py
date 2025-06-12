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
    TASK NAME - Crop input and label data to work on the same specific location for the input and label data. The input and label grids are not
                the same so it is important to create a strategie to be at the same exact location for both grids.  

     STATUS - experimental

     DESCRIPTION - This script will take as an input the input and label data. The script will also take the domain the data are going to be at 
                   as an input. The script will then crop the data to allow the user to obrain input and label data into the same domaine. Because 
                   the input and label are not at the same resolution, the input will be interpolate to allow the data to be at the same resolution, 
                   so it will be possible to feed the Neural Network with that data.  
"""     

#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import argparse
import csv 
import fnmatch
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

#.**************************************************************************************************************************************.#
#                                                             Functions                                                                  #
#.**************************************************************************************************************************************.#
class Crop():
    def crop(self, files, data, crop_data, PGSM_PATH, TOPOGRAPHIE, DOMAINE, name):
        """
        Take as a input the data of the analyse and crop the data through a specified region for the two grid. 
        The crop is done for the input and calda data to the chosen region by the DOMAINE. This function will 
        call the internal script pgsm.sh to do the crop and will interpolate the data of the input data with 
        the nearest neighbor methodology.
        """
        def process_file(file):
            target_file = os.path.join(crop_data, file)
            # print(f'The target file is {target_file}')
            # Copy the file name from DATA to CROP_DATA
            open(target_file, 'a').close()
            # Call the pgsm script to put the data on the same grid and interpolate input data
            # print(f'The first function argument is : {os.path.join(data, file)}')
            # print(f'The second function argument is : {DOMAINE}')
            # print(f'The last function argument is : {os.path.join(crop_data, file)}')
            subprocess.check_call([os.path.join(PGSM_PATH, name), os.path.join(data, file), DOMAINE, os.path.join(crop_data, file)])

        if TOPOGRAPHIE:
            target_file = os.path.join(crop_data, files)
            print(f'The target file is: {target_file}')
            # Copy the file name from DATA to CROP_DATA
            open(target_file, 'a').close()
            # print(f'The first function argument is : {os.path.join(data, files)}')
            # print(f'The second function argument is : {DOMAINE}')
            # print(f'The last function argument is : {os.path.join(crop_data, files)}')
            # Call the pgsm script to put the data on the same grid and interpolate input data
            subprocess.check_call([os.path.join(PGSM_PATH, name), os.path.join(data, files), DOMAINE, os.path.join(crop_data, files)])
        else:
            # Use ThreadPoolExecutor to parallelize the processing of files
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(process_file, file) for file in files]
                for future in futures:
                    future.result()

    def get_sort_key_input(self, filename):
        """
        Put the input data into chronological order. From the latest date to the most recent date. 
        Input
        :Param filename: Names of the file that is currently being analysed.
        Output
        :Param sort_key: Names of the file in chronological order. 
        """
        datetime_part = filename[:10]
        prediction_part = filename[11:14]

        sort_key = datetime_part + prediction_part
        return sort_key
    
    def get_sort_key_input_passes(self, filename):
        """
        Generates a sort key for organizing filenames first by the 'pass' value
        and then chronologically by date and time.
        Input
        :param filename: Name of the file that is currently being analysed.
        Output
        :return sort_key: A string that represents the sort key constructed from the filename.
        """
        # Extract date and time segments from the filename
        year = filename[0:4]
        month = filename[4:6]
        day = filename[6:8]
        pass_val = filename[8:10]  # This is the "pass" used to group files.
        hour = filename[11:14]  # Assuming this is the correct index for the hour part based on your description.

        # Construct the sort key: Group by 'pass', then sort by year, month, day, hour
        sort_key = f"{pass_val}_{year}{month}{day}{hour}"
        return sort_key

    def get_sort_key_label(self, filename):
        """
        Put the input data into chronological order. From the latest date to the most recent date. 
        Input
        :Param filename: Names of the file that is currently being analysed.
        Output
        :Param sort_key: Names of the file in chronological order.
        """
        datetime_part = filename[:10]

        return datetime_part

    def select_hours(self, files_input, files_label, START_HOUR_INPUT, END_HOUR_INPUT, START_HOUR_LABEL, END_HOUR_LABEL):
        """
        Select the range of hours that are going to be selected to train the Neural Network with. The team select
        a specify START_HOUR and END_HOUR for every dates in the original data. The goal of this is to allow the NN
        to perform on what it is necessary for it to do, which is predict wind downscalling. The idea is to disqualify
        all the data that are not going to help achieving that goal. 
        Input
        :Param files_input: Sorted input file name.
        :Param files_label: Sorted label file name.
        :Param START_HOUR: The time at which you want to start to analyse the data.
        :Param END_HOUR: The time at which you want to end to analyse the data.
        Output
        :Param new_inputs: Sorted input file name with specified hours.
        :Param new_labels: Sorted label file name with specified hours. 

        """
        #Verification of the model pass curently being analysed. 
        #Do the assignation for the label correctly if it is the case. 
        # Pass    INPUT (REPS)   Target (Mesoanalyse)
        # 00Z     4-9            4-9
        # 06Z     4-9            10-15
        # 12Z     4-9            16-21
        # 18Z     4-9            22-23 and next day 00-03

        new_inputs = []
        for i, date in enumerate(files_input):
            if i != 0: #TODO Verify why this error occurs. temporary patch..
                if int(date[12:14]) >= int(START_HOUR_INPUT) and int(date[12:14]) <= int(END_HOUR_INPUT):
                    # print(f'The new input is: {date}')
                    new_inputs.append(date) 

        #Normalize the input and label name.
        normalized_inputs = [None] * len(new_inputs)
        normalized_labels = [None] * len(files_label)
        for i, files in enumerate(new_inputs):
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
        #     print(f'input: {normalized_inputs[i]}')
        # print('\n')
        # print(f'The length of the input is: {len(normalized_inputs)}')

        #Convert the initial value depending of the original value. 
        for i, files in enumerate(files_label): 
            normalized_labels[i] = files[:8] + str(files[8:10])
            # print(f'label: {normalized_labels[i]}')
        # print(f'The length of the label is: {len(normalized_labels)}')

        # #Put the files input in pass order.
        # files_input.sort(key=self.get_sort_key_input_passes)

        new_labels_normalized = []
        #Verifies if the new_input is in the label file
        for i, input_date in enumerate(normalized_inputs):
            if input_date in normalized_labels:
                new_labels_normalized.append(input_date) 

        new_input_normalized = []
        for i, input_date in enumerate(normalized_labels):
            if input_date in normalized_inputs:
                new_input_normalized.append(input_date) 

        new_labels_normalized.sort(key=self.get_sort_key_input)
        new_input_normalized.sort(key=self.get_sort_key_label)

        #Put the name back to the original format.
        new_labels = []
        for file in new_labels_normalized:
            original_name = file[:10] + '_000'
            new_labels.append(original_name)

        new_inputs_1 = []
        for i, file in enumerate(new_input_normalized):
            if 4 <= int(new_labels_normalized[i][8:10]) <=  9:
                passes = '00'
                hour = file[8:10]
                original_name = file[:8] + passes + '_0' + hour + '_000'
            elif 10 <=  int(new_labels_normalized[i][8:10]) <= 15:
                passes = '06'
                hour = str(int(file[8:10]) - 6)
                hour = '0' + hour
                original_name = file[:8] + passes + '_0' + hour + '_000'
            elif 16 <=  int(new_labels_normalized[i][8:10]) <= 21:
                passes = '12'
                hour = str(int(file[8:10]) - 12)
                hour = '0' + hour
                original_name = file[:8] + passes + '_0' + hour + '_000'
            elif 22 <=  int(new_labels_normalized[i][8:10]) <= 24 or 0 <= int(new_labels_normalized[i][8:10]) <= 3:
                passes = 18
                date_format = "%Y%m%d%H"
                date_obj = datetime.strptime(new_labels_normalized[i], date_format)
                new_date_obj = date_obj - timedelta(hours=passes)
                new_date = new_date_obj.strftime(date_format)
                original_name = new_date[:8] + str(passes) + '_0' + new_date[8:10] + '_000'
            
            new_inputs_1.append(original_name)

        return new_inputs_1, new_labels

    def DateVerrification(self, files_input, files_label):
        """
        Verify if the same date and hour exist in the file and if so, put the value in an array 
        that will be used to itterate over the file later. 
        INPUT
        :Param files_input: Sorted input file name with specified hours. 
        :Param files_label: Sorted label file name with specified hours. 
        OUTPUT
        :Param verificated_files_input: Sorted input file name that match the label.
        :Param verificated_files_label: Sorted label file name that match the input. 
        """
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
        #     print(f'input: {normalized_inputs[i]}')
        # print('\n')
        # print(f'The length of the input is: {len(normalized_inputs)}')

        #Convert the initial value depending of the original value. 
        for i, files in enumerate(files_label): 
            normalized_labels[i] = files[:8] + str(files[8:10])
            # print(f'label: {normalized_labels[i]}')
        # print(f'The length of the label is: {len(normalized_labels)}')

        verificated_files_input = []
        verificated_files_label = []
        #Verify if the input file exist in the label file.
        for i, input_date in enumerate(normalized_inputs):
            if input_date in normalized_labels:
                verificated_files_input.append(files_input[i]) 
        #Verify if the label file exist in the input file. 
        for i, label_date in enumerate(normalized_labels):
            if label_date in normalized_inputs: 
                verificated_files_label.append(files_label[i])

        print(f'The length of verification input is {len(verificated_files_input)} and verification label is {len(verificated_files_label)}')

        return verificated_files_input, verificated_files_label

    def argparser(self):
        """
        Take the user input in the crop.sh script. This input will then be used in this code as variables and constants. 
        INPUT
        :none
        OUTPUT
        :Param parser.parse_args(): Used to parse the user input with a variable or a constant. 
        """
        # Argparse initialisation to allow external input to be used in the code
        parser = argparse.ArgumentParser(description="Arguments used in the code that are called in the bash file or the terminal")

        parser.add_argument('--INPUT_DATA', type=str, help='Path where are located the input data.')
        parser.add_argument('--LABEL_DATA', type=str, help='Path where are located the label data.')
        parser.add_argument('--DOMAINE', type=str, help='Path to the script that put data into the same domaine (Internal script).')
        parser.add_argument('--CROP_INPUT_DATA_UV', type=str, help='Path where will be store the crop input data for UV.')
        parser.add_argument('--CROP_INPUT_DATA', type=str, help='Path where will be store the crop input data.')
        parser.add_argument('--CROP_LABEL_DATA', type=str, help='Path where will be store the crop label data.')
        parser.add_argument('--PGSM_PATH', type=str, help='Path where is located the PGSM script.')
        parser.add_argument('--START_HOUR_INPUT', type=int, help='Starting time of the analyse for input.')
        parser.add_argument('--END_HOUR_INPUT', type=int, help='Ending time of the analyse for input.')
        parser.add_argument('--START_HOUR_LABEL', type=int, help='Starting time of the analyse for label.')
        parser.add_argument('--END_HOUR_LABEL', type=int, help='Ending time of the analyse for label.')
        parser.add_argument('--TOPOGRAPHIE', action='store_true', help='Flag used if the user whant to crop the topographie of the input standard file.')
        parser.add_argument('--TOPO_PATH', type=str, help='The path where the topographie is located. ')
        parser.add_argument('--CROP_TOPO_DATA', type=str, help='Path where will be store the crop topographie data.')
        parser.add_argument('--TOPO_STD_NANE', type=str, help='Name of the standard file that represent the topographie.')

        return parser.parse_args()

    def get_crop(self, INPUT_DATA, LABEL_DATA, DOMAINE_INPUT, DOMAINE_LABEL, PREDICTED_VALUE, CROP_PREDICTED_VALUE_DATA, CROP_INPUT_DATA, CROP_LABEL_DATA, PGSM_PATH,
                    START_HOUR_INPUT, END_HOUR_INPUT, START_HOUR_LABEL, END_HOUR_LABEL, TOPOGRAPHIE, TOPO_PATH, CROP_TOPO_DATA, TOPO_STD_NANE):
        #Put the input data in chronological order to allow to associate input and label correctly.
        files_input = [file for file in os.listdir(INPUT_DATA) if os.path.isfile(os.path.join(INPUT_DATA, file)) and not fnmatch.fnmatch(file, '*.tgz') and not fnmatch.fnmatch(file, '*.sh')]
        files_input.sort(key=self.get_sort_key_input)
        # for file in files_input:
        #     print(f'File in chronological order are: {file}')
        # print(f'The length of the initial files_input array is {len(files_input)}')
        #Put the label data in chronological order to allow to associate input and label correctly.
        files_label = [file for file in os.listdir(LABEL_DATA) if os.path.isfile(os.path.join(LABEL_DATA, file)) and not fnmatch.fnmatch(file, '*.tgz') and not fnmatch.fnmatch(file, '*.sh')]
        files_label.sort(key=self.get_sort_key_label)
        # for file in files_label:
        #     print(f'File in chronological order are: {file}')
        # print(f'The length of the initial files_label array is {len(files_label)}')

        #Choose the hours of the prediction
        files_input, files_label = self.select_hours(files_input, files_label, START_HOUR_INPUT, END_HOUR_INPUT, START_HOUR_LABEL, END_HOUR_LABEL)
        print(f'The length of input file is : {len(files_input)}, and the length of file label is : {len(files_label)}')
        files_input.sort(key=self.get_sort_key_input)
        files_label.sort(key=self.get_sort_key_label)
        for i in range(len(files_input)):
            print(f'The input file is: {files_input[i]} associate with the target: {files_label[i]}')
        #Verify that the dates and time are the same for each file. (Really important for the Neural Network). 
        files_input, files_label = self.DateVerrification(files_input, files_label)

        #Verification that the function work well. 
        print(f'The function works: {self.Test_DateVerrification(files_input, files_label)}')
        if TOPOGRAPHIE:
            #Crop and interpolate if necessary the input and calda data
            self.crop(TOPO_STD_NANE, TOPO_PATH, CROP_TOPO_DATA, PGSM_PATH, TOPOGRAPHIE, DOMAINE_INPUT, name='pgsm_topo.sh')
        elif PREDICTED_VALUE:
            print(f'We are doing the skip connection. ')
            self.crop(files_input, INPUT_DATA, CROP_PREDICTED_VALUE_DATA, PGSM_PATH, TOPOGRAPHIE, DOMAINE_INPUT, name='pgsm_skip.sh')
        else:       
            #Crop and interpolate if necessary the input and caldas data
            # print(f'{files_input}\n, {INPUT_DATA}\n, {CROP_INPUT_DATA}\n, {PGSM_PATH}\n, {TOPOGRAPHIE}\n, {DOMAINE_INPUT}\n')
            #Crop spatio temporal input
            self.crop(files_input, INPUT_DATA, CROP_INPUT_DATA, PGSM_PATH, TOPOGRAPHIE, DOMAINE_INPUT, name='pgsm_input.sh')
            self.crop(files_label, LABEL_DATA, CROP_LABEL_DATA, PGSM_PATH, TOPOGRAPHIE, DOMAINE_LABEL, name='pgsm_target.sh')

        print('The cropping have succeded.')
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
    cropping = Crop()

    #Parse the argument
    args = cropping.argparser()

    #Give the parse to a variable
    INPUT_DATA_LOCATION = '/input/initial_data_new/train_sequential'
    LABEL_DATA_LOCATION = '/label/initial_data_new/east_canada_sequential/train_sequential'
    WIND = 'UV'
    VARIABLE_NAME='UU_VV_TT_P0_PN_H_CX_SD_WGE'

    PROJECT_PATH = '/home/jfg000/ss5/SuperResolution'
    DATA_PROJECT_PATH = '/home/jfg000/ss5/Data'
    DOMAINE_INPUT='prep_domaine/Grid_REPS_Est_Canada_Dom1_fornointerp_big'
    DOMAINE_LABEL='prep_domaine/Grid_meso_Est_Canada_Dom1_fornointerp_big'

    DOMAINE_NAME = 'east_canada_squential_train_allpasses_domaine1'
    INPUT_DATA = f"{DATA_PROJECT_PATH}/{INPUT_DATA_LOCATION}" 
    LABEL_DATA = f"{DATA_PROJECT_PATH}/{LABEL_DATA_LOCATION}" 
    DOMAINE_INPUT = f'{DATA_PROJECT_PATH}/{DOMAINE_INPUT}'  #Grid_REPS_Gaspesie_fornointerp
    DOMAINE_LABEL = f'{DATA_PROJECT_PATH}/{DOMAINE_LABEL}' #Grid_meso_Gaspesie_fornointerp
    CROP_INPUT_DATA_UV = f'{DATA_PROJECT_PATH}/input/domaine/{DOMAINE_NAME}/cropUV_{WIND}_{VARIABLE_NAME}'
    CROP_INPUT_DATA = f'{DATA_PROJECT_PATH}/input/domaine/{DOMAINE_NAME}/crop_{WIND}_{VARIABLE_NAME}'
    CROP_LABEL_DATA = f'{DATA_PROJECT_PATH}/label/domaine/{DOMAINE_NAME}/crop_{WIND}_{VARIABLE_NAME}'
    CROP_PREDICTED_VALUE_DATA = f'{DATA_PROJECT_PATH}/input/domaine/{DOMAINE_NAME}/crop_predicted_{WIND}_{VARIABLE_NAME}'
    PGSM_PATH=f'{PROJECT_PATH}/Data_Processing/A_crop_interpolate'
    START_HOUR_INPUT = 4
    END_HOUR_INPUT = 9
    START_HOUR_LABEL = 4
    END_HOUR_LABEL = 9
    TOPOGRAPHIE = False
    PREDICTED_VALUE = True
    TOPO_PATH = "/fs/site5/eccc/cmd/x/spb001/SuperResolution/get_HRinput_from_HRDPS"
    CROP_TOPO_DATA = f'{DATA_PROJECT_PATH}/topography/{DOMAINE_NAME}/crop'
    TOPO_STD_NANE = "HR_input_from_HRDPS"

    #Folder used for cropping
    if not os.path.exists(CROP_INPUT_DATA_UV):
        os.makedirs(CROP_INPUT_DATA_UV)
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

    #Call the method to do cropping from crop class.
    cropping.get_crop(INPUT_DATA, LABEL_DATA, DOMAINE_INPUT, DOMAINE_LABEL, PREDICTED_VALUE, CROP_PREDICTED_VALUE_DATA, CROP_INPUT_DATA, CROP_LABEL_DATA, PGSM_PATH,
                    START_HOUR_INPUT, END_HOUR_INPUT, START_HOUR_LABEL, END_HOUR_LABEL, TOPOGRAPHIE, TOPO_PATH, CROP_TOPO_DATA, TOPO_STD_NANE)
