#!/usr/bin/env python
 
import numpy as np 
import os
import psutil

"""
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Created by: Jean-Sébastien Giroux, Date: 2024-05-21
    Description: This script combine multiples domaines data in a way that allow to train the network in a chronological order. This script put all the data in chronological order
                 with multiples domains at the same time. This script only work for 5 domains, but you can modify it as you wish to add more or less. It could also be optimized to 
                 allow the used to select the number of domains to combine and the code will automatically do it. At the moment the code put the data in chronological order and one 
                 domain after the oder. For exemple :
                 Data = 2023-03-21 3pm dom 1, 2023-03-21 3pm dom 2, 2023-03-21 3pm dom 3, 2023-03-21 3pm dom 4, 2023-03-21 3pm dom 5, 
                 Data = 2023-03-21 4pm dom 1, 2023-03-21 4pm dom 2, 2023-03-21 4pm dom 3, 2023-03-21 4pm dom 4, 2023-03-21 4pm dom 5, 
                 and so on. 
                 - This script can put the domains data together for multiples domains if do_domains is True. The domain data are every spatio-temporal value used to predict the outcome of the network
                 for example UU, VV, TT, UV, P0, PN...
                 - It can also put the topographie data together if do_topo is True. topographie variables are ME, MG, Z0 at the moment. Do not worries if you see only 5 data. It is normal, in Neural_Network
                   the code will put the missing values to allow to have a topo variable associate with each inputs. 
                 - Finally it can combine skip connection together if do_skip is True. Do skip is the predicted variable which is at the moment UV. 

"""
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024 * 1024):.2f} GB")


if __name__ == '__main__':
    #List of variables used in the program
    do_domains = True
    do_topo = False
    do_skip = False
    domaine1 = '1'
    domaine2 = '3'
    domaine3 = '6'
    domaine4 = '7'
    domaine5 = '10'
    domaine6 = '11'
    domaine7 = '12'
    domaine8 = '13'
    domaine9 = '14'
    domaine10 = '15'
    data_name = 'data_used_in_neural_network_cad_10_VV'
    interpolation_type_input = 'non_interpolation'  # nearestNeighbour_interpolation      bilinear_interpolation     non_interpolation
    interpolation_type_target = 'non_interpolation_multi_target'
    data_used = 'UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    if do_domains:
        #If you havent change the name of the created folder, you should only change these variables names. 
        path_location_inputs = f'/home/jfg000/ss5/data_superResolution/input/domaine/{interpolation_type_input}/domaine_creation'
        path_location_target = f'/home/jfg000/ss5/data_superResolution/label/domaine/{interpolation_type_target}/domaine_creation'
        path_location_result_inputs = f'/home/jfg000/ss5/data_superResolution/input/domaine/{interpolation_type_target}/{data_name}'
        path_location_result_target = f'/home/jfg000/ss5/data_superResolution/label/domaine/{interpolation_type_target}/{data_name}'
    if do_topo:
        path_location_topo = f'/home/jfg000/ss5/data_superResolution/topography/{interpolation_type_input}/domaine_creation'
        path_location_result_topo = f'/home/jfg000/ss5/data_superResolution/topography/{interpolation_type_target}/{data_name}'
    if do_skip:
        path_location_inputs_skip = f'/home/jfg000/ss5/data_superResolution/input/domaine/{interpolation_type_target}/domaine_creation'
        path_location_result_inputs_skip = f'/home/jfg000/ss5/data_superResolution/input/domaine/{interpolation_type_target}/{data_name}'

    #Do not change this section if you are using 5 domains and havent change the path during the first data processing steps. 
    if do_domains:
        #Input domains
        path_dom1_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine1}/{data_used}/input.npy'
        path_dom1_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine1}/{data_used}/input.npy'
        path_dom1_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine1}/{data_used}/input.npy'

        path_dom2_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine2}/{data_used}/input.npy'
        path_dom2_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine2}/{data_used}/input.npy'
        path_dom2_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine2}/{data_used}/input.npy'

        path_dom3_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine3}/{data_used}/input.npy'
        path_dom3_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine3}/{data_used}/input.npy'
        path_dom3_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine3}/{data_used}/input.npy'

        path_dom4_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine4}/{data_used}/input.npy'
        path_dom4_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine4}/{data_used}/input.npy'
        path_dom4_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine4}/{data_used}/input.npy'

        path_dom5_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine5}/{data_used}/input.npy'
        path_dom5_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine5}/{data_used}/input.npy'
        path_dom5_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine5}/{data_used}/input.npy'

        path_dom6_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine6}/{data_used}/input.npy'
        path_dom6_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine6}/{data_used}/input.npy'
        path_dom6_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine6}/{data_used}/input.npy'

        path_dom7_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine7}/{data_used}/input.npy'
        path_dom7_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine7}/{data_used}/input.npy'
        path_dom7_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine7}/{data_used}/input.npy'

        path_dom8_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine8}/{data_used}/input.npy'
        path_dom8_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine8}/{data_used}/input.npy'
        path_dom8_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine8}/{data_used}/input.npy'

        path_dom9_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine9}/{data_used}/input.npy'
        path_dom9_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine9}/{data_used}/input.npy'
        path_dom9_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine9}/{data_used}/input.npy'

        path_dom10_input_train = f'{path_location_inputs}/east_canada_squential_train_allpasses_domaine{domaine10}/{data_used}/input.npy'
        path_dom10_input_valid = f'{path_location_inputs}/east_canada_squential_valid_allpasses_domaine{domaine10}/{data_used}/input.npy'
        path_dom10_input_test = f'{path_location_inputs}/east_canada_squential_test_allpasses_domaine{domaine10}/{data_used}/input.npy'

        paths_inputs_train = [path_dom1_input_train, path_dom2_input_train, path_dom3_input_train, path_dom4_input_train, path_dom5_input_train, path_dom6_input_train, path_dom7_input_train, path_dom8_input_train, path_dom9_input_train, path_dom10_input_train]
        paths_inputs_valid = [path_dom1_input_valid, path_dom2_input_valid, path_dom3_input_valid, path_dom4_input_valid, path_dom5_input_valid, path_dom6_input_valid, path_dom7_input_valid, path_dom8_input_valid, path_dom9_input_valid, path_dom10_input_valid]
        paths_inputs_test = [path_dom1_input_test, path_dom2_input_test, path_dom3_input_test, path_dom4_input_test, path_dom5_input_test, path_dom6_input_test, path_dom7_input_test, path_dom8_input_test, path_dom9_input_test, path_dom10_input_test]

        #Inputs targets
        path_dom1_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine1}/{data_used}/label.npy'
        path_dom1_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine1}/{data_used}/label.npy'
        path_dom1_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine1}/{data_used}/label.npy'

        path_dom2_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine2}/{data_used}/label.npy'
        path_dom2_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine2}/{data_used}/label.npy'
        path_dom2_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine2}/{data_used}/label.npy'

        path_dom3_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine3}/{data_used}/label.npy'
        path_dom3_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine3}/{data_used}/label.npy'
        path_dom3_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine3}/{data_used}/label.npy'

        path_dom4_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine4}/{data_used}/label.npy'
        path_dom4_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine4}/{data_used}/label.npy'
        path_dom4_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine4}/{data_used}/label.npy'

        path_dom5_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine5}/{data_used}/label.npy'
        path_dom5_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine5}/{data_used}/label.npy'
        path_dom5_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine5}/{data_used}/label.npy'

        path_dom6_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine6}/{data_used}/label.npy'
        path_dom6_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine6}/{data_used}/label.npy'
        path_dom6_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine6}/{data_used}/label.npy'

        path_dom7_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine7}/{data_used}/label.npy'
        path_dom7_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine7}/{data_used}/label.npy'
        path_dom7_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine7}/{data_used}/label.npy'

        path_dom8_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine8}/{data_used}/label.npy'
        path_dom8_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine8}/{data_used}/label.npy'
        path_dom8_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine8}/{data_used}/label.npy'

        path_dom9_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine9}/{data_used}/label.npy'
        path_dom9_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine9}/{data_used}/label.npy'
        path_dom9_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine9}/{data_used}/label.npy'

        path_dom10_target_train = f'{path_location_target}/east_canada_squential_train_allpasses_domaine{domaine10}/{data_used}/label.npy'
        path_dom10_target_valid = f'{path_location_target}/east_canada_squential_valid_allpasses_domaine{domaine10}/{data_used}/label.npy'
        path_dom10_target_test = f'{path_location_target}/east_canada_squential_test_allpasses_domaine{domaine10}/{data_used}/label.npy'

        paths_target_train = [path_dom1_target_train, path_dom2_target_train, path_dom3_target_train, path_dom4_target_train, path_dom5_target_train, path_dom6_target_train, path_dom7_target_train, path_dom8_target_train, path_dom9_target_train, path_dom10_target_train]
        pahs_target_valid = [path_dom1_target_valid, path_dom2_target_valid, path_dom3_target_valid, path_dom4_target_valid, path_dom5_target_valid, path_dom6_target_valid, path_dom7_target_valid, path_dom8_target_valid, path_dom9_target_valid, path_dom10_target_valid]
        paths_target_test = [path_dom1_target_test, path_dom2_target_test, path_dom3_target_test, path_dom4_target_test, path_dom5_target_test, path_dom6_target_test, path_dom7_target_test, path_dom8_target_test, path_dom9_target_test, path_dom10_target_test]

        #Result path: 
        path_dom_input_train = f'{path_location_result_inputs}/east_canada_squential_train_allpasses_domaine/{data_used}/'
        path_dom_input_valid = f'{path_location_result_inputs}/east_canada_squential_valid_allpasses_domaine/{data_used}/'
        path_dom_input_test = f'{path_location_result_inputs}/east_canada_squential_test_allpasses_domaine/{data_used}/'
        path_dom_target_train = f'{path_location_result_target}/east_canada_squential_train_allpasses_domaine/{data_used}/'
        path_dom_target_valid = f'{path_location_result_target}/east_canada_squential_valid_allpasses_domaine/{data_used}/'
        path_dom_target_test = f'{path_location_result_target}/east_canada_squential_test_allpasses_domaine/{data_used}/'

        #Create the path if it doesnt exist
        if not os.path.exists(path_dom_input_train):
            os.makedirs(path_dom_input_train)
        if not os.path.exists(path_dom_input_valid):
            os.makedirs(path_dom_input_valid)
        if not os.path.exists(path_dom_input_test):
            os.makedirs(path_dom_input_test)
        if not os.path.exists(path_dom_target_train):
            os.makedirs(path_dom_target_train)
        if not os.path.exists(path_dom_target_valid):
            os.makedirs(path_dom_target_valid)
        if not os.path.exists(path_dom_target_test):
            os.makedirs(path_dom_target_test)

        #Create the name of the npy file that will be saved at the choosed path 
        path_dom_input_train = f'{path_location_result_inputs}/east_canada_squential_train_allpasses_domaine/{data_used}/all_dom' #normalized
        path_dom_input_valid = f'{path_location_result_inputs}/east_canada_squential_valid_allpasses_domaine/{data_used}/all_dom'
        path_dom_input_test = f'{path_location_result_inputs}/east_canada_squential_test_allpasses_domaine/{data_used}/all_dom'
        path_dom_target_train = f'{path_location_result_target}/east_canada_squential_train_allpasses_domaine/{data_used}/all_dom'
        path_dom_target_valid = f'{path_location_result_target}/east_canada_squential_valid_allpasses_domaine/{data_used}/all_dom'
        path_dom_target_test = f'{path_location_result_target}/east_canada_squential_test_allpasses_domaine/{data_used}/all_dom'

        #Initialize dictionary 
        input_data_train = {}
        input_data_valid = {}
        input_data_test = {}
        target_data_train = {}
        target_data_valid = {}
        target_data_test = {}

        #Loading data into the created dictionary
        for i, path in enumerate(paths_inputs_train):
            print(f'Loading domaine {i+1}')
            input_data_train[f'dom_{i+1}'] = np.load(path)
            print(f"Shape of input_data_train['dom_{i+1}']: {input_data_train[f'dom_{i+1}'].shape}")
            input_data_valid[f'dom_{i+1}'] = np.load(paths_inputs_valid[i])
            input_data_test[f'dom_{i+1}'] = np.load(paths_inputs_test[i])
            target_data_train[f'dom_{i+1}'] = np.load(paths_target_train[i])
            target_data_valid[f'dom_{i+1}'] = np.load(pahs_target_valid[i])
            target_data_test[f'dom_{i+1}'] = np.load(paths_target_test[i])

            print_memory_usage()

        #Concatenate domaine tegether
        concatenated_input_train = np.empty((input_data_train[f'dom_1'].shape[0] + input_data_train[f'dom_2'].shape[0] + input_data_train[f'dom_3'].shape[0] + input_data_train[f'dom_4'].shape[0] + input_data_train[f'dom_5'].shape[0] + input_data_train[f'dom_6'].shape[0] + input_data_train[f'dom_7'].shape[0] + input_data_train[f'dom_8'].shape[0] + input_data_train[f'dom_9'].shape[0] + input_data_train[f'dom_10'].shape[0],) + input_data_train[f'dom_1'].shape[1:], dtype=input_data_train[f'dom_1'].dtype)
        concatenated_input_valid = np.empty((input_data_valid[f'dom_1'].shape[0] + input_data_valid[f'dom_2'].shape[0] + input_data_valid[f'dom_3'].shape[0] + input_data_valid[f'dom_4'].shape[0] + input_data_valid[f'dom_5'].shape[0] + input_data_valid[f'dom_6'].shape[0] + input_data_valid[f'dom_7'].shape[0] + input_data_valid[f'dom_8'].shape[0] + input_data_valid[f'dom_9'].shape[0] + input_data_valid[f'dom_10'].shape[0],) + input_data_valid[f'dom_1'].shape[1:], dtype=input_data_valid[f'dom_1'].dtype)
        concatenated_input_test = np.empty((input_data_test[f'dom_1'].shape[0] + input_data_test[f'dom_2'].shape[0] + input_data_test[f'dom_3'].shape[0] + input_data_test[f'dom_4'].shape[0] + input_data_test[f'dom_5'].shape[0] + input_data_test[f'dom_6'].shape[0] + input_data_test[f'dom_7'].shape[0] + input_data_test[f'dom_8'].shape[0] + input_data_test[f'dom_9'].shape[0] + input_data_test[f'dom_10'].shape[0],) + input_data_test[f'dom_1'].shape[1:], dtype=input_data_test[f'dom_1'].dtype)
        concatenated_target_train = np.empty((target_data_train[f'dom_1'].shape[0] + target_data_train[f'dom_2'].shape[0] + target_data_train[f'dom_3'].shape[0] + target_data_train[f'dom_4'].shape[0] + target_data_train[f'dom_5'].shape[0] + target_data_train[f'dom_6'].shape[0] + target_data_train[f'dom_7'].shape[0] + target_data_train[f'dom_8'].shape[0] + target_data_train[f'dom_9'].shape[0] + target_data_train[f'dom_10'].shape[0],) + target_data_train[f'dom_1'].shape[1:], dtype=target_data_train[f'dom_1'].dtype)
        concatenated_target_valid = np.empty((target_data_valid[f'dom_1'].shape[0] + target_data_valid[f'dom_2'].shape[0] + target_data_valid[f'dom_3'].shape[0] + target_data_valid[f'dom_4'].shape[0] + target_data_valid[f'dom_5'].shape[0] + target_data_valid[f'dom_6'].shape[0] + target_data_valid[f'dom_7'].shape[0] + target_data_valid[f'dom_8'].shape[0] + target_data_valid[f'dom_9'].shape[0] + target_data_valid[f'dom_10'].shape[0],) + target_data_valid[f'dom_1'].shape[1:], dtype=target_data_valid[f'dom_1'].dtype)
        concatenated_target_test = np.empty((target_data_test[f'dom_1'].shape[0] + target_data_test[f'dom_2'].shape[0] + target_data_test[f'dom_3'].shape[0] + target_data_test[f'dom_4'].shape[0] + target_data_test[f'dom_5'].shape[0] + target_data_test[f'dom_6'].shape[0] + target_data_test[f'dom_7'].shape[0] + target_data_test[f'dom_8'].shape[0] + target_data_test[f'dom_9'].shape[0] + target_data_test[f'dom_10'].shape[0],) + target_data_test[f'dom_1'].shape[1:], dtype=target_data_test[f'dom_1'].dtype)

        #Doing the operation for input train 
        concatenated_input_train[0::10] = input_data_train[f'dom_1']
        concatenated_input_train[1::10] = input_data_train[f'dom_2']
        concatenated_input_train[2::10] = input_data_train[f'dom_3']
        concatenated_input_train[3::10] = input_data_train[f'dom_4']
        concatenated_input_train[4::10] = input_data_train[f'dom_5']
        concatenated_input_train[5::10] = input_data_train[f'dom_6']
        concatenated_input_train[6::10] = input_data_train[f'dom_7']
        concatenated_input_train[7::10] = input_data_train[f'dom_8']
        concatenated_input_train[8::10] = input_data_train[f'dom_9']
        concatenated_input_train[9::10] = input_data_train[f'dom_10']
        #Deleting unused variables to save memory space.
        del input_data_train
        #Doing the operation for the input valid
        concatenated_input_valid[0::10] = input_data_valid[f'dom_1']
        concatenated_input_valid[1::10] = input_data_valid[f'dom_2']
        concatenated_input_valid[2::10] = input_data_valid[f'dom_3']
        concatenated_input_valid[3::10] = input_data_valid[f'dom_4']
        concatenated_input_valid[4::10] = input_data_valid[f'dom_5']
        concatenated_input_valid[5::10] = input_data_valid[f'dom_6']
        concatenated_input_valid[6::10] = input_data_valid[f'dom_7']
        concatenated_input_valid[7::10] = input_data_valid[f'dom_8']
        concatenated_input_valid[8::10] = input_data_valid[f'dom_9']
        concatenated_input_valid[9::10] = input_data_valid[f'dom_10']
        #Deleting unused variables to save memory space.
        del input_data_valid
        #Doing the operation for the input test
        concatenated_input_test[0::10] = input_data_test[f'dom_1']
        concatenated_input_test[1::10] = input_data_test[f'dom_2']
        concatenated_input_test[2::10] = input_data_test[f'dom_3']
        concatenated_input_test[3::10] = input_data_test[f'dom_4']
        concatenated_input_test[4::10] = input_data_test[f'dom_5']
        concatenated_input_test[5::10] = input_data_test[f'dom_6']
        concatenated_input_test[6::10] = input_data_test[f'dom_7']
        concatenated_input_test[7::10] = input_data_test[f'dom_8']
        concatenated_input_test[8::10] = input_data_test[f'dom_9']
        concatenated_input_test[9::10] = input_data_test[f'dom_10']
        #Deleting unused variables to save memory space.
        del input_data_test
        #Save the new numpy array
        np.save(path_dom_input_train, concatenated_input_train)
        np.save(path_dom_input_valid, concatenated_input_valid)
        np.save(path_dom_input_test, concatenated_input_test)
        #Deleting unused variables to save memory space.
        del concatenated_input_train, concatenated_input_valid, concatenated_input_test

        #Doing the operation for input train 
        concatenated_target_train[0::10] = target_data_train[f'dom_1']
        concatenated_target_train[1::10] = target_data_train[f'dom_2']
        concatenated_target_train[2::10] = target_data_train[f'dom_3']
        concatenated_target_train[3::10] = target_data_train[f'dom_4']
        concatenated_target_train[4::10] = target_data_train[f'dom_5']
        concatenated_target_train[5::10] = target_data_train[f'dom_6']
        concatenated_target_train[6::10] = target_data_train[f'dom_7']
        concatenated_target_train[7::10] = target_data_train[f'dom_8']
        concatenated_target_train[8::10] = target_data_train[f'dom_9']
        concatenated_target_train[9::10] = target_data_train[f'dom_10']
        #Deleting unused variables to save memory space.
        del target_data_train
        #Doing the operation for the input valid
        concatenated_target_valid[0::10] = target_data_valid[f'dom_1']
        concatenated_target_valid[1::10] = target_data_valid[f'dom_2']
        concatenated_target_valid[2::10] = target_data_valid[f'dom_3']
        concatenated_target_valid[3::10] = target_data_valid[f'dom_4']
        concatenated_target_valid[4::10] = target_data_valid[f'dom_5']
        concatenated_target_valid[5::10] = target_data_valid[f'dom_6']
        concatenated_target_valid[6::10] = target_data_valid[f'dom_7']
        concatenated_target_valid[7::10] = target_data_valid[f'dom_8']
        concatenated_target_valid[8::10] = target_data_valid[f'dom_9']
        concatenated_target_valid[9::10] = target_data_valid[f'dom_10']
        #Deleting unused variables to save memory space.
        del target_data_valid
        #Doing the operation for the input test
        concatenated_target_test[0::10] = target_data_test[f'dom_1']
        concatenated_target_test[1::10] = target_data_test[f'dom_2']
        concatenated_target_test[2::10] = target_data_test[f'dom_3']
        concatenated_target_test[3::10] = target_data_test[f'dom_4']
        concatenated_target_test[4::10] = target_data_test[f'dom_5']
        concatenated_target_test[5::10] = target_data_test[f'dom_6']
        concatenated_target_test[6::10] = target_data_test[f'dom_7']
        concatenated_target_test[7::10] = target_data_test[f'dom_8']
        concatenated_target_test[8::10] = target_data_test[f'dom_9']
        concatenated_target_test[9::10] = target_data_test[f'dom_10']
        #Deleting unused variables to save memory space.
        del target_data_test

        #Saving the numpy file to the choosen destination. 
        np.save(path_dom_target_train, concatenated_target_train)
        np.save(path_dom_target_valid, concatenated_target_valid)
        np.save(path_dom_target_test, concatenated_target_test)

        del concatenated_target_train, concatenated_target_valid, concatenated_target_test

        print(f'The concatenation of spatio-temporal value is done.')

    if do_topo:
        #Input variables 
        path_dom1_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine1}/topo/tg.npy'
        path_dom2_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine2}/topo/tg.npy'
        path_dom3_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine3}/topo/tg.npy'
        path_dom4_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine4}/topo/tg.npy'
        path_dom5_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine5}/topo/tg.npy'
        path_dom6_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine6}/topo/tg.npy'
        path_dom7_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine7}/topo/tg.npy'
        path_dom8_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine8}/topo/tg.npy'
        path_dom9_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine9}/topo/tg.npy'
        path_dom10_topo = f'{path_location_topo}/east_canada_squential_train_allpasses_domaine{domaine10}/topo/tg.npy'
        #Output path
        out_don_topo = f'{path_location_result_topo}/east_canada_squential_train_allpasses_domaine/topo/'

        #Create the path if it doesnt exist
        if not os.path.exists(out_don_topo):
            os.makedirs(out_don_topo)

        #Output path and numpy name
        out_don_topo = f'{path_location_result_topo}/east_canada_squential_train_allpasses_domaine/topo/all_dom'    

        #Load domain data
        topo_dom1 = np.load(path_dom1_topo)
        print(f'The shape of the topo data is: {topo_dom1.shape}')
        topo_dom2 = np.load(path_dom2_topo)
        topo_dom3 = np.load(path_dom3_topo)
        topo_dom4 = np.load(path_dom4_topo)
        topo_dom5 = np.load(path_dom5_topo)
        topo_dom6 = np.load(path_dom6_topo)
        topo_dom7 = np.load(path_dom7_topo)
        topo_dom8 = np.load(path_dom8_topo)
        topo_dom9 = np.load(path_dom9_topo)
        topo_dom10 = np.load(path_dom10_topo)
        print_memory_usage()

        #Combine domains together
        topo = np.concatenate([topo_dom1, topo_dom2, topo_dom3, topo_dom4, topo_dom5, topo_dom6, topo_dom7, topo_dom8, topo_dom9, topo_dom10], axis=0)

        #Save the topographie
        np.save(out_don_topo, topo)

    if do_skip:
        #Input domains
        path_dom1_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine1}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom1_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine1}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom1_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine1}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom2_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine2}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom2_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine2}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom2_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine2}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom3_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine3}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom3_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine3}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom3_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine3}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom4_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine4}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom4_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine4}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom4_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine4}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom5_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine5}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom5_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine5}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom5_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine5}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom6_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine6}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom6_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine6}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom6_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine6}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom7_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine7}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom7_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine7}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom7_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine7}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom8_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine8}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom8_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine8}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom8_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine8}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom9_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine9}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom9_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine9}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom9_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine9}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        path_dom10_input_train = f'{path_location_inputs_skip}/east_canada_squential_train_allpasses_domaine{domaine10}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom10_input_valid = f'{path_location_inputs_skip}/east_canada_squential_valid_allpasses_domaine{domaine10}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'
        path_dom10_input_test = f'{path_location_inputs_skip}/east_canada_squential_test_allpasses_domaine{domaine10}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/skip.npy'

        #Creating a list with the input data
        paths_inputs_train = [path_dom1_input_train, path_dom2_input_train, path_dom3_input_train, path_dom4_input_train, path_dom5_input_train, path_dom6_input_train, path_dom7_input_train, path_dom8_input_train, path_dom9_input_train, path_dom10_input_train]
        paths_inputs_valid = [path_dom1_input_valid, path_dom2_input_valid, path_dom3_input_valid, path_dom4_input_valid, path_dom5_input_valid, path_dom6_input_valid, path_dom7_input_valid, path_dom8_input_valid, path_dom9_input_valid, path_dom10_input_valid]
        paths_inputs_test = [path_dom1_input_test, path_dom2_input_test, path_dom3_input_test, path_dom4_input_test, path_dom5_input_test, path_dom6_input_test, path_dom7_input_test, path_dom8_input_test, path_dom9_input_test, path_dom10_input_test]

        #Result path: 
        path_dom_input_train = f'{path_location_result_inputs_skip}/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/'
        path_dom_input_valid = f'{path_location_result_inputs_skip}/east_canada_squential_valid_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/'
        path_dom_input_test = f'{path_location_result_inputs_skip}/east_canada_squential_test_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/'

        #Create the path if it doesnt exist
        if not os.path.exists(path_dom_input_train):
            os.makedirs(path_dom_input_train)
        if not os.path.exists(path_dom_input_valid):
            os.makedirs(path_dom_input_valid)
        if not os.path.exists(path_dom_input_test):
            os.makedirs(path_dom_input_test)

        #Create the name of the npy file that will be saved at the choosed path 
        path_dom_input_train = f'{path_location_result_inputs_skip}/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/all_dom'
        path_dom_input_valid = f'{path_location_result_inputs_skip}/east_canada_squential_valid_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/all_dom'
        path_dom_input_test = f'{path_location_result_inputs_skip}/east_canada_squential_test_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/all_dom'

        #Initialize empty dictionary
        input_data_train = {}
        input_data_valid = {}
        input_data_test = {}

        #Loading data
        for i, path in enumerate(paths_inputs_train):
            input_train = np.load(path)
            print(f'The shape of input_train is: {input_train.shape}')
            input_valid = np.load(paths_inputs_valid[i])
            input_test = np.load(paths_inputs_test[i])
            print_memory_usage()

            input_data_train[f'dom_{i+1}'] = input_train
            input_data_valid[f'dom_{i+1}'] = input_valid
            input_data_test[f'dom_{i+1}'] = input_test
        
        #Deleting unused variables to save memory space.
        del input_train, input_valid, input_test

        #Concatenate domaine tegether
        concatenated_input_train = np.empty((input_data_train[f'dom_1'].shape[0] + input_data_train[f'dom_2'].shape[0] + input_data_train[f'dom_3'].shape[0] + input_data_train[f'dom_4'].shape[0] + input_data_train[f'dom_5'].shape[0] + input_data_train[f'dom_6'].shape[0] + input_data_train[f'dom_7'].shape[0] + input_data_train[f'dom_8'].shape[0] + input_data_train[f'dom_9'].shape[0] + input_data_train[f'dom_10'].shape[0],) + input_data_train[f'dom_1'].shape[1:], dtype=input_data_train[f'dom_1'].dtype)
        concatenated_input_valid = np.empty((input_data_valid[f'dom_1'].shape[0] + input_data_valid[f'dom_2'].shape[0] + input_data_valid[f'dom_3'].shape[0] + input_data_valid[f'dom_4'].shape[0] + input_data_valid[f'dom_5'].shape[0] + input_data_valid[f'dom_6'].shape[0] + input_data_valid[f'dom_7'].shape[0] + input_data_valid[f'dom_8'].shape[0] + input_data_valid[f'dom_9'].shape[0] + input_data_valid[f'dom_10'].shape[0],) + input_data_valid[f'dom_1'].shape[1:], dtype=input_data_valid[f'dom_1'].dtype)
        concatenated_input_test = np.empty((input_data_test[f'dom_1'].shape[0] + input_data_test[f'dom_2'].shape[0] + input_data_test[f'dom_3'].shape[0] + input_data_test[f'dom_4'].shape[0] + input_data_test[f'dom_5'].shape[0] + input_data_test[f'dom_6'].shape[0] + input_data_test[f'dom_7'].shape[0] + input_data_test[f'dom_8'].shape[0] + input_data_test[f'dom_9'].shape[0] + input_data_test[f'dom_10'].shape[0],) + input_data_test[f'dom_1'].shape[1:], dtype=input_data_test[f'dom_1'].dtype)

        #Doing the operation for input train 
        concatenated_input_train[0::10] = input_data_train[f'dom_1']
        concatenated_input_train[1::10] = input_data_train[f'dom_2']
        concatenated_input_train[2::10] = input_data_train[f'dom_3']
        concatenated_input_train[3::10] = input_data_train[f'dom_4']
        concatenated_input_train[4::10] = input_data_train[f'dom_5']
        concatenated_input_train[5::10] = input_data_train[f'dom_6']
        concatenated_input_train[6::10] = input_data_train[f'dom_7']
        concatenated_input_train[7::10] = input_data_train[f'dom_8']
        concatenated_input_train[8::10] = input_data_train[f'dom_9']
        concatenated_input_train[9::10] = input_data_train[f'dom_10']
        #Deleting unused variables to save memory space.
        del input_data_train
        #Doing the operation for the input valid
        concatenated_input_valid[0::10] = input_data_valid[f'dom_1']
        concatenated_input_valid[1::10] = input_data_valid[f'dom_2']
        concatenated_input_valid[2::10] = input_data_valid[f'dom_3']
        concatenated_input_valid[3::10] = input_data_valid[f'dom_4']
        concatenated_input_valid[4::10] = input_data_valid[f'dom_5']
        concatenated_input_valid[5::10] = input_data_valid[f'dom_6']
        concatenated_input_valid[6::10] = input_data_valid[f'dom_7']
        concatenated_input_valid[7::10] = input_data_valid[f'dom_8']
        concatenated_input_valid[8::10] = input_data_valid[f'dom_9']
        concatenated_input_valid[9::10] = input_data_valid[f'dom_10']
        #Deleting unused variables to save memory space.
        del input_data_valid
        #Doing the operation for the input test
        concatenated_input_test[0::10] = input_data_test[f'dom_1']
        concatenated_input_test[1::10] = input_data_test[f'dom_2']
        concatenated_input_test[2::10] = input_data_test[f'dom_3']
        concatenated_input_test[3::10] = input_data_test[f'dom_4']
        concatenated_input_test[4::10] = input_data_test[f'dom_5']
        concatenated_input_test[5::10] = input_data_test[f'dom_6']
        concatenated_input_test[6::10] = input_data_test[f'dom_7']
        concatenated_input_test[7::10] = input_data_test[f'dom_8']
        concatenated_input_test[8::10] = input_data_test[f'dom_9']
        concatenated_input_test[9::10] = input_data_test[f'dom_10']
        #Deleting unused variables to save memory space.
        del input_data_test
        #Save the new numpy array
        np.save(path_dom_input_train, concatenated_input_train)
        np.save(path_dom_input_valid, concatenated_input_valid)
        np.save(path_dom_input_test, concatenated_input_test)
