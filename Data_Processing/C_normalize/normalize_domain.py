#!/usr/bin/env python

""" 
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Created by: Jean-Sébastien Giroux, Date: 2023-11-22
    Inspired by: Kevin Gauthier, Data: 2019
"""


#.**************************************************************************************************************************************.#
#                                                          Code description                                                              #
#.**************************************************************************************************************************************.#
"""
    TASK NAME - Standardize the input and label data into their given range based on their variable. 

     STATUS - experimental

     DESCRIPTION - This script will take the input and label crop result and convert these results into Numpy format to allow the Neural
                   Network to work efficently with the data. 
"""     


#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil

#.**************************************************************************************************************************************.#
#                                                             Functions                                                                  #
#.**************************************************************************************************************************************.#
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024 * 1024):.2f} GB")

class Normalize():
    def compute_pixel_statistics(self, data):
        mean_data = np.mean(data, axis=0, keepdims=True)  # Compute mean across samples
        std_data = np.std(data, axis=0, keepdims=True)    # Compute std across samples
        std_data[std_data == 0] = 1  # Avoid division by zero
        return mean_data, std_data
    
    def compute_channel_statistics(self, data):
        # Calculate mean and standard deviation for each channel
        means = np.mean(data, axis=(0, 2, 3), keepdims=True)
        stds = np.std(data, axis=(0, 2, 3), keepdims=True)
        stds[stds == 0] = 1  # Avoid division by zero

        
        return means, stds

    def standardize_data(self, data, mean, std):
        return (data - mean) / std

    def standardize_dataset(self, train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels,
                            mean_train_inputs, std_train_inputs, mean_train_labels, std_train_labels):
        print('Starting standardization')
        train_inputs_standardized = self.standardize_data(train_inputs, mean_train_inputs, std_train_inputs)
        train_labels_standardized = self.standardize_data(train_labels, mean_train_labels, std_train_labels)

        # Standardize validation data using training statistics for inputs and labels
        val_inputs_standardized = self.standardize_data(val_inputs, mean_train_inputs, std_train_inputs)
        val_labels_standardized = self.standardize_data(val_labels, mean_train_labels, std_train_labels)

        # Standardize test data using training statistics for inputs and labels
        test_inputs_standardized = self.standardize_data(test_inputs, mean_train_inputs, std_train_inputs)
        test_labels_standardized = self.standardize_data(test_labels, mean_train_labels, std_train_labels)

        return (train_inputs_standardized, train_labels_standardized,
                val_inputs_standardized, val_labels_standardized,
                test_inputs_standardized, test_labels_standardized)
    
    def unstandardize_data(self, standardized_data, mean, std):
        return standardized_data * std + mean

    def destandardize_data(self, standardized_inputs, standardized_labels, mean_inputs, std_inputs, labels_shape):
        # De-standardize inputs
        original_inputs = (standardized_inputs * std_inputs) + mean_inputs
        
        # Resize mean and std for labels to match the labels' dimensions
        mean_labels_resized = np.tile(mean_inputs[:, 0:1, :, :], (1, 1, labels_shape[2], labels_shape[3]))
        std_labels_resized = np.tile(std_inputs[:, 0:1, :, :], (1, 1, labels_shape[2], labels_shape[3]))

        # De-standardize labels
        original_labels = (standardized_labels * std_labels_resized) + mean_labels_resized

        return original_inputs, original_labels

    def get_normalization(self, NPY_INPUT_DATA, NPY_LABEL_DATA, NPY_SKIP_DATA, NPY_TOPO_DATA, NORMALIZE_INPUT_DATA, NORMALIZE_LABEL_DATA, NORMALIZE_SKIP_DATA, NORMALIZE_TOPO_DATA, VARIABLE_NAME_ARRAY_INPUT,
                                 VARIABLE_NAME_ARRAY_LABEL, TEST, PATH_SAVE_MEAN, PATH_SAVE_STD, TOPOGRAPHIE, PREDICTED_VALUE, INTERPOLATION_TYPE):
        if TOPOGRAPHIE:
            domaine = np.load(NPY_TOPO_DATA)
            print(f'domaine shape is: {domaine.shape}')
            domaine = np.swapaxes(domaine, -2, -1)

            # Do path for validation and test set 
            base_path = '/home/jfg000/ss5/data_superResolution'
            domain_path = f'topography/{INTERPOLATION_TYPE}/data_used_in_neural_network_cad_5' #data_used_in_neural_network_qcnb
            input_data_path = 'east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data'

            # Loading the mean and std deviation for the general model
            mean_topo_path = os.path.join(base_path, domain_path, input_data_path, 'mean.npy')
            std_topo_path = os.path.join(base_path, domain_path, input_data_path, 'std.npy')
            mean_inputs = np.load(mean_topo_path)
            std_inputs = np.load(std_topo_path)

            # Standardize the input data
            domaine_standardized = (domaine - mean_inputs) / std_inputs
            # Destandardize the input data
            original_inputs = (domaine_standardized * std_inputs) + mean_inputs
            are_inputs_close = np.allclose(original_inputs, domaine, atol=1e-3)
            print("Topo are close:", are_inputs_close)
            print(f'The path is : {NORMALIZE_TOPO_DATA}')
           
            np.save(NORMALIZE_TOPO_DATA, domaine_standardized)
        elif PREDICTED_VALUE:
            skip = np.load(NPY_SKIP_DATA)   

            # Do path for validation and test set
            base_path = '/home/jfg000/ss5/data_superResolution'
            domain_path = f'domaine/{INTERPOLATION_TYPE}/domaine_creation' #data_used_in_neural_network
            domain_path_2 = f'domaine/{INTERPOLATION_TYPE}/data_used_in_neural_network_cad_5' #data_used_in_neural_network
            input_data_path = 'skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
            domain_number = 1

            # Compute pixel-wise statistics based on training inputs
            train_skip_mean_path = os.path.join(base_path, 'input', domain_path_2, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'mean.npy')
            train_skip_std_path = os.path.join(base_path, 'input', domain_path_2, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'std.npy')
            mean_skip = np.load(train_skip_mean_path)
            std_skip = np.load(train_skip_std_path)


            val_skip_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_valid_allpasses_domaine{domain_number}', input_data_path, 'skip.npy')
            test_skip_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_test_allpasses_domaine{domain_number}', input_data_path, 'skip.npy')
            
            val_skip = np.load(val_skip_path) 
            test_skip = np.load(test_skip_path)

            skip = np.swapaxes(skip, -2, -1)
            val_skip = np.swapaxes(val_skip, -2, -1)
            test_skip = np.swapaxes(test_skip, -2, -1)
            print(f'The shape of the skip is : {skip.shape}')
            print(f'The shape of the skip is : {val_skip.shape}')
            print(f'The shape of the skip is : {test_skip.shape}')

            # Standardize the input data
            skip_standardized = (skip - mean_skip) / std_skip
            skip_standardized_valid = (val_skip - mean_skip) / std_skip
            skip_standardized_test = (test_skip - mean_skip) / std_skip

            # Destandardize the input data to verify
            original_inputs = (skip_standardized * std_skip) + mean_skip
            are_inputs_close = np.allclose(original_inputs, skip, atol=1e-3)
            print("Topo are close:", are_inputs_close)

            original_inputs_valid = (skip_standardized_valid * std_skip) + mean_skip
            are_inputs_close = np.allclose(original_inputs_valid, val_skip, atol=1e-3)
            print("Topo are close:", are_inputs_close)

            original_inputs_test = (skip_standardized_test * std_skip) + mean_skip
            are_inputs_close = np.allclose(original_inputs_test, test_skip, atol=1e-3)
            print("Topo are close:", are_inputs_close)

            print(f'Saving the standardized train skip')
            print(f'The saving path is : {NORMALIZE_SKIP_DATA}')
            np.save(NORMALIZE_SKIP_DATA, skip_standardized)

            print(f'Saving the standardized valid skip')
            val_skip_save_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_valid_allpasses_domaine{domain_number}', input_data_path, 'normalize.npy')   #normalize_10.npy
            np.save(val_skip_save_path, skip_standardized_valid)

            print(f'Saving the standardized test skip')
            test_skip_save_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_test_allpasses_domaine{domain_number}', input_data_path, 'normalize.npy')
            np.save(test_skip_save_path, skip_standardized_test)

        else:
            print('Loading the input and label data...')
            train_inputs = np.load(NPY_INPUT_DATA)
            print(f'The shape of the input is: {train_inputs.shape}')
            train_labels = np.load(NPY_LABEL_DATA)
            print_memory_usage()

            #Do path for validation and test set
            base_path = '/home/jfg000/ss5/data_superResolution'
            print(f'The interpolation type is: {INTERPOLATION_TYPE}')
            domain_path = f'domaine/{INTERPOLATION_TYPE}/domaine_creation' #data_used_in_neural_network
            domain_path_2 = f'domaine/{INTERPOLATION_TYPE}/data_used_in_neural_network_cad_5' #data_used_in_neural_network_qcnb
            input_data_path = 'UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
            domain_number = 1

            # Compute pixel-wise statistics based on training inputs
            mean_input_path = os.path.join(base_path, 'input', domain_path_2, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'mean.npy')
            std_input_path = os.path.join(base_path, 'input', domain_path_2, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'std.npy')
            mean_train_inputs = np.load(mean_input_path)
            std_train_inputs = np.load(std_input_path)
            print(f'The shape for mean_inputs is: {mean_train_inputs.shape}')
            print(f'The shape for std_inputs is: {std_train_inputs.shape}')

            # Compute pixel-wise statistics based on training labels
            mean_labels_path = os.path.join(base_path, 'label', domain_path_2, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'mean.npy')
            std_labels_path = os.path.join(base_path, 'label', domain_path_2, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'std.npy')
            mean_train_labels = np.load(mean_labels_path)
            std_train_labels = np.load(std_labels_path)


            val_inputs_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_valid_allpasses_domaine{domain_number}', input_data_path, 'input.npy')     # all_dom.npy
            val_labels_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_valid_allpasses_domaine{domain_number}', input_data_path, 'label.npy')
            test_inputs_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_test_allpasses_domaine{domain_number}', input_data_path, 'input.npy')
            test_labels_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_test_allpasses_domaine{domain_number}', input_data_path, 'label.npy')

            val_inputs = np.load(val_inputs_path)
            val_labels = np.load(val_labels_path)
            test_inputs = np.load(test_inputs_path)
            test_labels = np.load(test_labels_path)

            #Inversing the image to put in the original orientation (Maybee if I have more time understand why the image is inverted when loading it. Probably in readStdFile method)
            train_inputs = np.swapaxes(train_inputs, -2, -1)
            train_labels = np.swapaxes(train_labels, -2, -1)
            val_inputs = np.swapaxes(val_inputs, -2, -1)
            val_labels = np.swapaxes(val_labels, -2, -1)
            test_inputs = np.swapaxes(test_inputs, -2, -1)
            test_labels = np.swapaxes(test_labels, -2, -1)

            print(f'The shape of the input is : {train_inputs.shape}')
            print(f'The shape of the target is : {train_labels.shape}')
            print(f'The shape of the input is : {val_inputs.shape}')
            print(f'The shape of the target is : {val_labels.shape}')
            print(f'The shape of the input is : {test_inputs.shape}')
            print(f'The shape of the target is : {test_labels.shape}')

            #Standardize the input and label data
            (train_inputs_standardized, train_labels_standardized,
             val_inputs_standardized, val_labels_standardized,
             test_inputs_standardized, test_labels_standardized) = self.standardize_dataset(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels,
                                                                             mean_train_inputs, std_train_inputs, mean_train_labels, std_train_labels)
            print_memory_usage()

            print(f'Saving the standardize train input')
            np.save(NORMALIZE_INPUT_DATA, train_inputs_standardized)

            print(f'Saving the standardized train label')
            np.save(NORMALIZE_LABEL_DATA, train_labels_standardized)
            print(f'Saving the standardized train std_labels')

            print(f'Saving the standardized valid input')
            val_inputs_save_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_valid_allpasses_domaine{domain_number}', input_data_path, 'normalize.npy')  # normalize_10.npy
            np.save(val_inputs_save_path, val_inputs_standardized)
            print(f'Saving the standardized valid label')
            val_labels_save_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_valid_allpasses_domaine{domain_number}', input_data_path, 'normalize.npy')
            np.save(val_labels_save_path, val_labels_standardized)

            print(f'Saving the standardized test input')
            test_inputs_save_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_test_allpasses_domaine{domain_number}', input_data_path, 'normalize.npy')
            np.save(test_inputs_save_path, test_inputs_standardized)
            print(f'Saving the standardized test label')
            test_labels_save_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_test_allpasses_domaine{domain_number}', input_data_path, 'normalize.npy')
            np.save(test_labels_save_path, test_labels_standardized)
             
            # if TEST: 
            #     #De-standardize the input and label data
            #     print('De-standardize the input data...')
            #     de_standardized_inputs, de_standardized_labels = self.de_standardize(standardized_inputs, standardized_labels, VARIABLE_NAME_ARRAY_INPUT, VARIABLE_NAME_ARRAY_LABEL, mean_standardized_input, std_standardized_input, mean_standardized_label, std_standardized_label)
            #     #Verify that input, label data are the same than de-standardized data
            #     print('Verify if everything is working well...') 
            #     self.compare_input_de_standardize(n_inputs, de_standardized_inputs, new_labels, de_standardized_labels)

            # #Plot result
            # for i in range(10):
            #     #Reshaping the arrays for quiver plot
            #     U_inputs = inputs[i, 0, :, :]
            #     U_inputs_standardize = standardized_inputs[i, 0, :, :]
            #     U_labels_standardize = standardized_labels[i, 0, :, :]
            #     U_labels = labels[i, 0, :, :]

            #     fig, axes = plt.subplots(1,4, figsize=(36,16))
            #     fig.suptitle("Results", fontsize=20)

            #     axes[0].set_title("Inputs", fontsize=20)
            #     axes[1].set_title("standardize input", fontsize=20)
            #     axes[2].set_title("Target standardize", fontsize=20)
            #     axes[3].set_title("target", fontsize=20)

            #     im0 = axes[0].imshow(U_inputs, origin = "lower", norm=plt.Normalize(vmin=U_inputs.min(), vmax=U_inputs.max()), cmap='magma') 
            #     im1 = axes[1].imshow(U_inputs_standardize, origin = "lower", norm=plt.Normalize(vmin=U_inputs_standardize.min(), vmax=U_inputs_standardize.max()), cmap='magma')
            #     im2 = axes[2].imshow(U_labels_standardize, origin = "lower", norm=plt.Normalize(vmin=U_labels_standardize.min(), vmax=U_labels_standardize.max()), cmap='magma')
            #     im3 = axes[3].imshow(U_labels, origin = "lower", norm=plt.Normalize(vmin=U_labels.min(), vmax=U_labels.max()), cmap='magma')

            #     sm0 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=U_inputs.min(), vmax=U_inputs.max()))
            #     sm1 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=U_inputs_standardize.min(), vmax=U_inputs_standardize.max()))
            #     sm2 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=U_labels_standardize.min(), vmax=U_labels_standardize.max()))
            #     sm3 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=U_labels.min(), vmax=U_labels.max()))

            #     sm0.set_array([])
            #     sm1.set_array([])
            #     sm2.set_array([])
            #     sm3.set_array([])

            #     cbar0 = fig.colorbar(sm0, ax=axes[0], fraction=0.046, pad=0.04)
            #     cbar0.ax.tick_params(labelsize=14)
            #     cbar1 = fig.colorbar(sm1, ax=axes[1], fraction=0.046, pad=0.04)
            #     cbar1.ax.tick_params(labelsize=14)
            #     cbar2 = fig.colorbar(sm2, ax=axes[2], fraction=0.046, pad=0.04)
            #     cbar2.ax.tick_params(labelsize=14)
            #     cbar3 = fig.colorbar(sm3, ax=axes[3], fraction=0.046, pad=0.04) 
            #     cbar3.ax.tick_params(labelsize=14)

            #     plt.show()

            # for i in range(1):
            #     for j in range(8):
            #         #Reshaping the arrays for quiver plot
            #         U_inputs = n_inputs[i, j, :, :]
            #         U_labels = new_labels[i, 0, :, :]

            #         vmin = min(U_inputs.min(), U_labels.min())
            #         vmax = max(U_inputs.max(), U_labels.max())

            #         fig, axes = plt.subplots(1,2)
            #         fig.suptitle("Original data")
            #         axes[0].set_title(f"inputs{j}")
            #         axes[0].imshow(U_inputs, origin = "lower", vmin=vmin, vmax=vmax, cmap='magma')
            #         axes[1].set_title("labels")
            #         axes[1].imshow(U_labels, origin = "lower", vmin=vmin, vmax=vmax, cmap='magma')

            #         # Optionally, add a colorbar
            #         fig.colorbar(plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=vmin, vmax=vmax)), ax=axes.ravel().tolist())

            #         plt.show()

        print('The standardization is done and had work successfully.')