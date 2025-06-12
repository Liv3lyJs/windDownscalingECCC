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

    def standardize_dataset(self, train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels):
        print('Starting standardization')

        # Compute pixel-wise statistics based on training inputs
        mean_train_inputs, std_train_inputs = self.compute_channel_statistics(train_inputs)

        # Compute pixel-wise statistics based on training labels
        mean_train_labels, std_train_labels = self.compute_channel_statistics(train_labels)

        # Standardize training data
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
                test_inputs_standardized, test_labels_standardized,
                mean_train_inputs, std_train_inputs,
                mean_train_labels, std_train_labels)
    
    def unstandardize_data(self, standardized_data, mean, std):
        return standardized_data * std + mean
    
    # def standardize_data(self, inputs, labels):
    #     print(f'Starting the standardization')

    #     # Compute mean and std deviation for each channel in the inputs
    #     mean_inputs = np.mean(inputs, axis=(0, 2, 3), keepdims=True)  # Shape (num_channels, 1, 1, 1)
    #     std_inputs = np.std(inputs, axis=(0, 2, 3), keepdims=True)
    #     std_inputs[std_inputs == 0] = 1  # Avoid division by zero

    #     # Standardize the input data
    #     inputs_standardized = (inputs - mean_inputs) / std_inputs

    #     # Standardize the labels using the statistics of the first input channel
    #     mean_labels = mean_inputs[:, 0:1, :, :]  # Select the mean for the first channel only
    #     std_labels = std_inputs[:, 0:1, :, :]    # Select the std for the first channel only

    #     # Correctly resize mean and std to match label dimensions
    #     # Ensure dimensions (1, 1, 128, 128) to fit labels of shape (7350, 1, 128, 128)
    #     mean_labels_resized = np.tile(mean_labels, (1, 1, labels.shape[2], labels.shape[3]))
    #     std_labels_resized = np.tile(std_labels, (1, 1, labels.shape[2], labels.shape[3]))

    #     labels_standardized = (labels - mean_labels_resized) / std_labels_resized

    #     return inputs_standardized, labels_standardized, mean_inputs, std_inputs
    
        # Standardizing channel wise.
        # print(f'Starting the standardization')
        
        # # Initialize arrays to hold standardized data
        # inputs_standardized = np.empty_like(inputs)
        # labels_standardized = np.empty_like(labels)
        
        # # Compute mean and std deviation for each sample across all channels
        # for i in range(inputs.shape[0]):  # Loop over each sample
        #     mean_inputs = np.mean(inputs[i], axis=(1, 2), keepdims=True)  # Per-sample, per-channel
        #     std_inputs = np.std(inputs[i], axis=(1, 2), keepdims=True)
        #     std_inputs[std_inputs == 0] = 1  # Avoid division by zero

        #     # Standardize the input data for this sample
        #     inputs_standardized[i] = (inputs[i] - mean_inputs) / std_inputs
            
        #     # Standardize the labels using the statistics of the first input channel of this sample
        #     mean_labels = mean_inputs[0]  # First channel mean
        #     std_labels = std_inputs[0]    # First channel std
        #     labels_standardized[i] = (labels[i] - mean_labels) / std_labels
        
        # return inputs_standardized, labels_standardized

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

            # /data_used_in_neural_network/east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data
            # Do path for validation and test set 
            base_path = '/home/jfg000/ss5/data_superResolution'
            domain_path = f'topography/{INTERPOLATION_TYPE}/data_used_in_neural_network_cad_10' #data_used_in_neural_network
            input_data_path = 'east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data'

            # for dom in domaine:
            #     print(f'The domaine shape is: {dom.shape}')
            #     plt.imshow(dom[0, :, :])
            #     plt.show()

            domaine = np.swapaxes(domaine, -2, -1)

            # print(f'The domaine shape is: {domaine.shape}')
            # for dom in domaine:
            #     print(f'The domaine shape is: {dom.shape}')
            #     plt.imshow(dom[2, :, :])
            #     plt.show()

            #Standardize the input:
            print('Standardized the domain data: ')
            # Compute mean and std deviation for each channel in the inputs
            mean_inputs = np.mean(domaine, axis=(0, 2, 3), keepdims=True)  # Shape (num_channels, 1, 1, 1)
            std_inputs = np.std(domaine, axis=(0, 2, 3), keepdims=True)
            std_inputs[std_inputs == 0] = 1  # Avoid division by zero

            # Standardize the input data
            domaine_standardized = (domaine - mean_inputs) / std_inputs
            # Destandardize the input data
            original_inputs = (domaine_standardized * std_inputs) + mean_inputs
            are_inputs_close = np.allclose(original_inputs, domaine, atol=1e-3)
            print("Topo are close:", are_inputs_close)
            print(f'The path is : {NORMALIZE_TOPO_DATA}')

            mean_topo_path = os.path.join(base_path, domain_path, input_data_path, 'mean.npy')
            std_topo_path = os.path.join(base_path, domain_path, input_data_path, 'std.npy')

            np.save(NORMALIZE_TOPO_DATA, domaine_standardized)
            np.save(mean_topo_path, mean_inputs)
            np.save(std_topo_path, std_inputs)
        elif PREDICTED_VALUE:
            skip = np.load(NPY_SKIP_DATA)

            # Do path for validation and test set
            base_path = '/home/jfg000/ss5/data_superResolution'
            domain_path = f'domaine/{INTERPOLATION_TYPE}/data_used_in_neural_network_cad_10_UU' #data_used_in_neural_network
            input_data_path = 'skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'

            val_skip_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_valid_allpasses_domaine', input_data_path, 'all_dom.npy') #all_dom
            test_skip_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_test_allpasses_domaine', input_data_path, 'all_dom.npy')
            
            val_skip = np.load(val_skip_path)
            test_skip = np.load(test_skip_path)

            skip = np.swapaxes(skip, -2, -1)
            val_skip = np.swapaxes(val_skip, -2, -1)
            test_skip = np.swapaxes(test_skip, -2, -1)
            print(f'The shape of the skip is : {skip.shape}')
            print(f'The shape of the skip is : {val_skip.shape}')
            print(f'The shape of the skip is : {test_skip.shape}')

            # # Compute pixel-wise mean and std deviation for each channel in the inputs
            # mean_skip = np.mean(skip, axis=0, keepdims=True)  # Compute mean across samples
            # std_skip = np.std(skip, axis=0, keepdims=True)    # Compute std across samples
            # std_skip[std_skip == 0] = 1  # Avoid division by zero
            mean_skip = np.mean(skip, axis=(0, 2, 3), keepdims=True)  # Shape (num_channels, 1, 1, 1)
            std_skip = np.std(skip, axis=(0, 2, 3), keepdims=True)
            std_skip[std_skip == 0] = 1  # Avoid division by zero

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
            print(f'Saving the standardize train mean_inputs')
            train_skip_mean_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'mean.npy')
            np.save(train_skip_mean_path, mean_skip)
            print(f'Saving the standardize train std_inputs')
            train_skip_std_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'std.npy')
            np.save(train_skip_std_path, std_skip)

            print(f'Saving the standardized valid skip')
            val_skip_save_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_valid_allpasses_domaine', input_data_path, 'normalize.npy')
            np.save(val_skip_save_path, skip_standardized_valid)

            print(f'Saving the standardized test skip')
            test_skip_save_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_test_allpasses_domaine', input_data_path, 'normalize.npy')
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
            domain_path = f'domaine/{INTERPOLATION_TYPE}/data_used_in_neural_network_cad_10_UU'
            input_data_path = 'UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'

            val_inputs_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_valid_allpasses_domaine', input_data_path, 'all_dom.npy')     # all_dom.npy
            val_labels_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_valid_allpasses_domaine', input_data_path, 'all_dom.npy')
            test_inputs_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_test_allpasses_domaine', input_data_path, 'all_dom.npy')
            test_labels_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_test_allpasses_domaine', input_data_path, 'all_dom.npy')

            val_inputs = np.load(val_inputs_path)
            val_labels = np.load(val_labels_path)
            test_inputs = np.load(test_inputs_path)
            test_labels = np.load(test_labels_path)
           
            # for dom in inputs:
            #     print(f'The domaine shape is: {dom.shape}')
            #     plt.imshow(dom[0, :, :])
            #     plt.show()
            #     break
            # for dom in labels:
            #     print(f'The domaine shape is: {dom.shape}')
            #     plt.imshow(dom[0, :, :])
            #     plt.show()
            #     break

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
            # for dom in inputs:
            #     print(f'The domaine shape is: {dom.shape}')
            #     plt.imshow(dom[0, :, :])
            #     plt.show()
            #     break
            # for dom in labels:
            #     print(f'The domaine shape is: {dom.shape}')
            #     plt.imshow(dom[0, :, :])
            #     plt.show()
            #     break

            #Standardize the input and label data
            (train_inputs_standardized, train_labels_standardized,
             val_inputs_standardized, val_labels_standardized,
             test_inputs_standardized, test_labels_standardized,
             mean_train_inputs, std_train_inputs,
             mean_train_labels, std_train_labels) = self.standardize_dataset(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels)
            print(f'The shape for mean_inputs is: {mean_train_inputs.shape}')
            print(f'The shape for std_inputs is: {std_train_inputs.shape}')
            print_memory_usage()

            # unstandardized_train_inputs = self.unstandardize_data(train_inputs_standardized, mean_train_inputs, std_train_inputs)       # standardized_inputs[0:1]
            # unstandardized_train_labels = self.unstandardize_data(train_labels_standardized, mean_train_labels, std_train_labels)       # standardized_labels[0:1]
            # unstandardized_val_inputs = self.unstandardize_data(val_inputs_standardized, mean_train_inputs, std_train_inputs) 
            # unstandardized_val_labels = self.unstandardize_data(val_labels_standardized, mean_train_labels, std_train_labels) 
            # unstandardized_test_inputs = self.unstandardize_data(test_inputs_standardized, mean_train_inputs, std_train_inputs) 
            # unstandardized_test_labels = self.unstandardize_data(test_labels_standardized, mean_train_labels, std_train_labels)   

            # # Check if the de-standardized inputs are close to the original inputs
            # are_inputs_train_close = np.allclose(unstandardized_train_inputs, train_inputs, atol=1e-3)
            # # Check if the de-standardized labels are close to the original labels
            # are_labels_train_close = np.allclose(unstandardized_train_labels, train_labels, atol=1e-3)

            # are_inputs_valid_close = np.allclose(unstandardized_val_inputs, val_inputs, atol=1e-3)
            # are_labels_valid_close = np.allclose(unstandardized_val_labels, val_labels, atol=1e-3)
            # are_inputs_test_close = np.allclose(unstandardized_test_inputs, test_inputs, atol=1e-3)
            # are_labels_test_close = np.allclose(unstandardized_test_labels, test_labels, atol=1e-3)

            # print("Inputs train are close:", are_inputs_train_close)
            # print("Labels train are close:", are_labels_train_close)
            # print("Inputs valid are close:", are_inputs_valid_close)
            # print("Labels valid are close:", are_labels_valid_close)
            # print("Inputs test are close:", are_inputs_test_close)
            # print("Labels test are close:", are_labels_test_close)

            print(f'The final shape for the input is: {train_inputs_standardized.shape}')
            print(f'Saving the standardize train input')
            np.save(NORMALIZE_INPUT_DATA, train_inputs_standardized)
            print(f'Saving the standardize train mean_inputs')
            np.save(PATH_SAVE_MEAN, mean_train_inputs)
            print(f'Saving the standardize train std_inputs')
            np.save(PATH_SAVE_STD, std_train_inputs)

            print(f'The final shape for the input is: {train_labels_standardized.shape}')
            print(f'Saving the standardized train label')
            np.save(NORMALIZE_LABEL_DATA, train_labels_standardized)
            print(f'Saving the standardized train mean_labels')
            mean_labels_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'mean.npy')
            np.save(mean_labels_path, mean_train_labels)
            print(f'Saving the standardized train std_labels')
            std_labels_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_train_allpasses_domaine', input_data_path, 'std.npy')
            np.save(std_labels_path, std_train_labels)

            print(f'The final shape for the input is: {val_inputs_standardized.shape}')
            print(f'Saving the standardized valid input')
            val_inputs_save_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_valid_allpasses_domaine', input_data_path, 'normalize.npy')
            np.save(val_inputs_save_path, val_inputs_standardized)
            print(f'Saving the standardized valid label')
            val_labels_save_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_valid_allpasses_domaine', input_data_path, 'normalize.npy')
            np.save(val_labels_save_path, val_labels_standardized)

            print(f'The final shape for the input is: {test_inputs_standardized.shape}')
            print(f'Saving the standardized test input')
            test_inputs_save_path = os.path.join(base_path, 'input', domain_path, f'east_canada_squential_test_allpasses_domaine', input_data_path, 'normalize.npy')
            np.save(test_inputs_save_path, test_inputs_standardized)
            print(f'Saving the standardized test label')
            test_labels_save_path = os.path.join(base_path, 'label', domain_path, f'east_canada_squential_test_allpasses_domaine', input_data_path, 'normalize.npy')
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


#.**************************************************************************************************************************************.#
#                                                                 Main                                                                   #
#.**************************************************************************************************************************************.#
if __name__ == '__main__':
    normalize = Normalize()

    #List of variables used for debugging. 
    NPY_INPUT_DATA = '/home/jfg000/ss5/data_superResolution/input/domaine/nearestNeighbour_interpolation/domaine_creation/east_canada_squential_train_allpasses_domaine1/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/input.npy'
    NPY_LABEL_DATA = '/home/jfg000/ss5/data_superResolution/label/domaine/nearestNeighbour_interpolation/domaine_creation/east_canada_squential_train_allpasses_domaine1/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/label.npy'
    NPY_SKIP_DATA = None
    NPY_TOPO_DATA = None
    NORMALIZE_INPUT_DATA = '/home/jfg000/ss5/data_superResolution/input/domaine/nearestNeighbour_interpolation/domaine_creation/east_canada_squential_train_allpasses_domaine1/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    NORMALIZE_LABEL_DATA = '/home/jfg000/ss5/data_superResolution/label/domaine/nearestNeighbour_interpolation/domaine_creation/east_canada_squential_train_allpasses_domaine1/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    PATH_SAVE_MEAN = '/home/jfg000/ss5/data_superResolution/input/domaine/nearestNeighbour_interpolation/domaine_creation/east_canada_squential_train_allpasses_domaine1/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    PATH_SAVE_STD = '/home/jfg000/ss5/data_superResolution/input/domaine/nearestNeighbour_interpolation/domaine_creation/east_canada_squential_train_allpasses_domaine1/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    NORMALIZE_SKIP_DATA = None
    NORMALIZE_TOPO_DATA = None
    VARIABLE_NAME_ARRAY_INPUT = None
    VARIABLE_NAME_ARRAY_LABEL = None
    TEST = False
    COMBINE_UV = False
    PATH_CSV = None
    TOPOGRAPHIE = False
    PREDICTED_VALUE = False

    #Call the method to do normalization from crop class.
    normalize.get_normalization(NPY_INPUT_DATA, NPY_LABEL_DATA, NPY_SKIP_DATA, NPY_TOPO_DATA, NORMALIZE_INPUT_DATA, NORMALIZE_LABEL_DATA, NORMALIZE_SKIP_DATA, NORMALIZE_TOPO_DATA, VARIABLE_NAME_ARRAY_INPUT,
                                 VARIABLE_NAME_ARRAY_LABEL, TEST, PATH_SAVE_MEAN, PATH_SAVE_STD, TOPOGRAPHIE, PREDICTED_VALUE)
