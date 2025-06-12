#!/usr/bin/env python
""" 
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Developed by: Jean-Sébastien Giroux, Date: 2023-10-23
"""


#.**************************************************************************************************************************************.#
#                                                          Code description                                                              #
#.**************************************************************************************************************************************.#
"""
    TASK NAME - Training and testing of a a CNN architecture model using Keras to do supervised machine learning. 

     STATUS - experimental

     DESCRIPTION - When using the code, the user can choose if he want to do an hyper-parameter search of use a model with parameter the user define as optimal. 
                CASE1: Hyper-parameter search (Most commun case)
                The user will have to define the hyper-parameters he will want to test in the function hyper_parameters_search. Note that the 
                user might provide a batch size hyper-parameter in the function run_trial. It's also possible to add other hyper-parameters, just to copy 
                the structure used for the one that are currently used. 
                The code will work as follow: Keras_tuner will do a random search of the hyper-parameter the user gave and will train the choosen architecture.
                Once the model have been trained on the hyper-parameters, the script will then take the N best models and train them on the model with thir set
                of hyper-parameters. Once the training is done, a file nammed Best_model will be created in the folder outside of the gitlab repository. In this file, it will be 
                possible to see the test that the user created and the user will be able to see inside of it. Once inside the folder, there will be four other folders
                and documents. 
                    1- The document best_models.csv countain an array that rank the best set of hyper-parameters that were tested based on the choosen metric
                       (The loss function of the metric). The number 1 stand for the best model and the last stend for the less performing model. 
                    2- The folder nammed Image will have the generated images for the n best models. They will be ranked from 1 to n which 1 represent the best model
                        and n will be the less performing model of the selection. The generated images are created based on the Test_set which are image the 
                        Neural network haven't been trained on 
                    3- The folder nammed model will have the saved weights of the best epoch for the given hyper-parameters sets. The user can ignore this section 
                       when trying to find the best set of hyper-parameters. (This folder is used to restore the model weights).
                    4- The folder nammed Training will have the graph of the training and validation accuracy during the training of the model and the graph of the 
                       training and validation loss. This graph can give you a pretty good idea on how well the model performed during the training. 
                CASE2: Training with predefined hyper-parameters(No hyper-parameter search).
                TODO
                
"""           


#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import argparse
import csv
import copy
import keras_tuner # Because of this genius github user: jobsZhao99 we can use keras_tuner, run in your conda env: conda install -c conda-forge wrapt=1.14.1  He knows more than GPT 4.0...
from kerastuner.engine import multi_execution_tuner
from keras_tuner import BayesianOptimization, RandomSearch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
#keras.mixed_precision.set_global_policy('mixed_float16') # Use this line when you have access to GPU, will speed up the training by 3x. CANNOT BE USED WITH A LOSS OF mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import backend as K
import gc
from scipy.fft import fft2, fftshift

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


import callback as cb
import model as mod


#.**************************************************************************************************************************************.#
#                                                             Variables                                                                  #
#.**************************************************************************************************************************************.#
"""
    :Constant Param POST_HPS: Flag that indicate if the hyper-parameter search is done.
"""
global POST_HPS
POST_HPS = 0


#.**************************************************************************************************************************************.#
#                                                             Functions                                                                  #
#.**************************************************************************************************************************************.#
def path_creation(INPUT_FILE, LABEL_FILE, TOPO_FILE, PATH_RESULTS, HP_NAME, DATA_ALREADY_SPLIT):
    """
    Create the paths that will be used by the program.
    """
    #Create the directory name to store the best hyper-parameters(hp) in the hp search.
    if WIND:
        var_name = 'UUVV'
    elif WIND_UU:
        var_name = 'UU'
    elif WIND_VV:
        var_name = 'VV'
    else:
        var_name = 'UV'
    directory = f'{PATH_RESULTS}/hyper_parameters_search_{var_name}'
    HP_RESULT_PATH = f"{PATH_RESULTS}/non_interpolation_cad_10_transfer_learning{var_name}/{HP_NAME}_HP_Search" 
    HP_DIR = f"{HP_NAME}_{var_name}_hp_search"

    dom = 'domaine2'
    #Input file
    PATH_INPUT_TRAIN = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/domaine_creation/east_canada_squential_train_allpasses_{dom}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'    #/home/jfg000/ss5/Data/input/domaine/east_canada_test_sequential_train/UU_VV_TT_P0_PN_H_CX_SD_WGE_data
    PATH_INPUT_VALID = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/domaine_creation/east_canada_squential_valid_allpasses_{dom}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    PATH_INPUT_TEST = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/domaine_creation/east_canada_squential_test_allpasses_{dom}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    INPUT_FILE_TRAIN = os.path.join(PATH_INPUT_TRAIN, 'normalize_10.npy')
    INPUT_FILE_VALID = os.path.join(PATH_INPUT_VALID, 'normalize_10.npy')
    INPUT_FILE_TEST = os.path.join(PATH_INPUT_TEST, 'normalize_10.npy')

    #label file
    PATH_LABEL_TRAIN = f'/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/domaine_creation/east_canada_squential_train_allpasses_{dom}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    PATH_LABEL_VALID = f'/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/domaine_creation/east_canada_squential_valid_allpasses_{dom}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    PATH_LABEL_TEST = f'/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/domaine_creation/east_canada_squential_test_allpasses_{dom}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    LABEL_FILE_TRAIN = os.path.join(PATH_LABEL_TRAIN, 'normalize_10.npy')
    LABEL_FILE_VALID = os.path.join(PATH_LABEL_VALID, 'normalize_10.npy')
    LABEL_FILE_TEST = os.path.join(PATH_LABEL_TEST, 'normalize_10.npy')

    #Skip connection file 
    PATH_SKIP_TRAIN = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/domaine_creation/east_canada_squential_train_allpasses_{dom}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    PATH_SKIP_VALID = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/domaine_creation/east_canada_squential_valid_allpasses_{dom}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    PATH_SKIP_TEST =  f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/domaine_creation/east_canada_squential_test_allpasses_{dom}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data'
    SKIP_FILE_TRAIN = os.path.join(PATH_SKIP_TRAIN, 'normalize_10.npy')
    SKIP_FILE_VALID = os.path.join(PATH_SKIP_VALID, 'normalize_10.npy')
    SKIP_FILE_TEST = os.path.join(PATH_SKIP_TEST, 'normalize_10.npy')

    #Topo file
    PATH_TOPO = f'/home/jfg000/ss5/data_superResolution/topography/non_interpolation/domaine_creation/east_canada_squential_train_allpasses_{dom}/ME_MG_Z0_data'
    TOPO_FILE = os.path.join(PATH_TOPO, 'normalize_10.npy') #normalize.npy

    return INPUT_FILE_TRAIN, INPUT_FILE_VALID, INPUT_FILE_TEST, LABEL_FILE_TRAIN, LABEL_FILE_VALID, LABEL_FILE_TEST, SKIP_FILE_TRAIN, SKIP_FILE_VALID, SKIP_FILE_TEST, TOPO_FILE, directory, HP_RESULT_PATH, HP_DIR, var_name

def load_data_already_split(INPUT_FILE_TRAIN, INPUT_FILE_VALID, INPUT_FILE_TEST, LABEL_FILE_TRAIN, LABEL_FILE_VALID, LABEL_FILE_TEST, SKIP_FILE_TRAIN, SKIP_FILE_VALID, SKIP_FILE_TEST, TOPO_FILE):
    """
    Loading the input, label and topo data to train the model with. This is for a set that is already split between training and testing
    """
    print('load the input data')
    #Loading the input data
    inputs_train = np.load(INPUT_FILE_TRAIN)
    print(f'The input shape is: {inputs_train.shape}')
    inputs_valid = np.load(INPUT_FILE_VALID)
    inputs_test = np.load(INPUT_FILE_TEST)
    #Loading the label data
    labels_train = np.load(LABEL_FILE_TRAIN)
    print(f'The labels_train is: {labels_train.shape}')
    label_valid = np.load(LABEL_FILE_VALID)
    label_test = np.load(LABEL_FILE_TEST)
    #Loading skip connection data
    skip_train = np.load(SKIP_FILE_TRAIN)
    print(f'The skip_train is: {skip_train.shape}')
    skip_valid = np.load(SKIP_FILE_VALID)
    skip_test = np.load(SKIP_FILE_TEST)

    #Loading the topo data
    topo_train = np.load(TOPO_FILE)
    print(f'The topo_train is : {topo_train.shape}')

    return inputs_train, inputs_valid, inputs_test, labels_train, label_valid, label_test, skip_train, skip_valid, skip_test, topo_train

def dataProcessing_already_split(inputs_train, inputs_valid, inputs_test, labels_train, label_valid, label_test, skip_train, skip_valid, skip_test, topo_train):
    """
    Split the data into three set. Training, validation and testing. 
    """
    # Reshaping the data [sample, width, height, channel]
    inputs_train = inputs_train.transpose((0,2,3,1)) 
    inputs_valid = inputs_valid.transpose((0,2,3,1)) 
    inputs_test = inputs_test.transpose((0,2,3,1)) 

    labels_train = labels_train.transpose((0,2,3,1)) 
    label_valid = label_valid.transpose((0,2,3,1)) 
    label_test = label_test.transpose((0,2,3,1)) 

    skip_train = skip_train.transpose((0,2,3,1)) 
    skip_valid = skip_valid.transpose((0,2,3,1)) 
    skip_test = skip_test.transpose((0,2,3,1)) 

    # print(f'The shape of the topo is: {topo.shape}')
    topo = topo_train.transpose((0,2,3,1))
    # print(f'The shape of the topo is: {topo.shape}')
    # for top in topo:
    #     plt.imshow(top[:,:,2])
    #     plt.show() 

    topo_train_reshaped = np.tile(topo, (inputs_train.shape[0], 1, 1, 1))
    print(f'The shape of inputs_train is: {inputs_train.shape}')
    print(f'The shape of topo_train_reshaped is : {topo_train_reshaped.shape}')
    topo_val_reshaped = np.tile(topo, (inputs_valid.shape[0], 1, 1, 1))
    topo_test_reshaped = np.tile(topo, (inputs_test.shape[0], 1, 1, 1))

    # #Visualize the input associate with his target:
    # for i in range(15):
    #     #Reshaping the arrays for quiver plot
    #     U_inputs = inputs_train[i, :, :, 0]
    #     topo_input = topo_train_reshaped[i, :, :, 0]
    #     U_labels = labels_train[i, :, :, 0]

    #     fig, axes = plt.subplots(1,3, figsize=(36,16))
    #     fig.suptitle("Results", fontsize=20)

    #     axes[0].set_title("Inputs", fontsize=20)
    #     axes[1].set_title("Topo", fontsize=20)
    #     axes[2].set_title("Labels", fontsize=20)

    #     im0 = axes[0].imshow(U_inputs, origin = "lower", norm=plt.Normalize(vmin=U_inputs.min(), vmax=U_inputs.max()), cmap='magma') 
    #     im1 = axes[1].imshow(topo_input, origin = "lower", norm=plt.Normalize(vmin=topo_input.min(), vmax=topo_input.max()), cmap='magma')
    #     im2 = axes[2].imshow(U_labels, origin = "lower", norm=plt.Normalize(vmin=U_labels.min(), vmax=U_labels.max()), cmap='magma')

    #     sm0 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=U_inputs.min(), vmax=U_inputs.max()))
    #     sm1 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=topo_input.min(), vmax=topo_input.max()))
    #     sm2 = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=U_labels.min(), vmax=U_labels.max()))

    #     sm0.set_array([])
    #     sm1.set_array([])
    #     sm2.set_array([])

    #     cbar0 = fig.colorbar(sm0, ax=axes[0], fraction=0.046, pad=0.04)
    #     cbar0.ax.tick_params(labelsize=14)
    #     cbar1 = fig.colorbar(sm1, ax=axes[1], fraction=0.046, pad=0.04)
    #     cbar1.ax.tick_params(labelsize=14)
    #     cbar2 = fig.colorbar(sm2, ax=axes[2], fraction=0.046, pad=0.04)
    #     cbar2.ax.tick_params(labelsize=14)

    #     plt.show()

    #Add patche to the training to do a harder training
    #inputs_train, inputs_valid = get_random_blackout(inputs_train, inputs_valid)

    # Converting the numpy array to a tensor of float32 (See book p.449 yellow section).
    inputs_train = tf.convert_to_tensor(inputs_train, dtype='float32')
    inputs_valid = tf.convert_to_tensor(inputs_valid, dtype='float32')
    inputs_test = tf.convert_to_tensor(inputs_test, dtype='float32')

    labels_train = tf.convert_to_tensor(labels_train, dtype='float32')
    label_valid = tf.convert_to_tensor(label_valid, dtype='float32')
    label_test = tf.convert_to_tensor(label_test, dtype='float32')

    skip_train = tf.convert_to_tensor(skip_train, dtype='float32')
    skip_valid = tf.convert_to_tensor(skip_valid, dtype='float32')
    skip_test = tf.convert_to_tensor(skip_test, dtype='float32')


    topo_train_reshaped = tf.convert_to_tensor(topo_train_reshaped, dtype='float32')
    topo_val_reshaped = tf.convert_to_tensor(topo_val_reshaped, dtype='float32')
    topo_test_reshaped = tf.convert_to_tensor(topo_test_reshaped, dtype='float32')

    return inputs_train, inputs_valid, inputs_test, labels_train, label_valid, label_test, skip_train, skip_valid, skip_test, topo_train_reshaped, topo_val_reshaped, topo_test_reshaped

def tuner_initialization(hyper_parameters_search, NUM_TRIALS, EXECUTIONS_PER_TRIAL, directory, HP_DIR):
    """
    Initialize the hyper-parameter search. Keras doc here: https://keras.io/api/keras_tuner/tuners/random/ 
    Input
    :Param hyper_parameters_search: The function name of the function doing the hyper-parameter search. 
    :Param NUM_TRIALS: The epoch number done when doing an hyper-parameter search. 
    :Param EXECUTIONS_PER_TRIAL: The number of time a same set of hyper-parameter is tested (Minimize risk).
    :param directory: The directory where the hyper-parameter will be saved. 
    :Param HP_DIR: The name of the folder used to represent the hyper-parameter search. 
    Output
    :Param tuner: Instance of the hyper-parameter tuning class. 
    """
    tuner = BayesianOptimization(                            # RandomSearch
        hypermodel=hyper_parameters_search,
        objective=keras_tuner.Objective("val_mse", direction="min"), #TODO val_mean_absolute_error  val_mean_squared_error val_SSIMLoss
        max_trials=NUM_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        overwrite=True,
        directory='/fs/site5/eccc/cmd/x/jfg000/CNN_results_article/hyper_parameters_search_UU',  #directory
        project_name='NoInterpol_10dom_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_01_domaine2_transfer_learning_Final_HP_Search' #HP_DIR
    )
    
    return tuner

def hyper_parameters_search(hp):
    """
    Test a set of hyper-parameters for the architecture of the model being optimize. This will allow the program to test multiples output filters, 
    activation functions, learning rate and much more. 
    INPUT
    :Param hp: refer to trial.hyperparameters and can be use to select different approach to search for hyper-parameters
    OUTPUT
    :Param model: Object of the neural network with the selected hyper-parameters.
    """
    #Define the units (number of kernels) that can be used as an hyper-parameter to train the model with.
    units = hp.Choice("units", values=UNITS_HP)
    #Define the activation function that can be used as an hyper-parameter to train the model with.
    activation = hp.Choice("activation", ACTIVATION_FUNCTION_HP)
    #Define the learning rate that can be used as an hyper-parameter to train the model with. 
    lr = hp.Choice('lr', values=LEARNING_RATE_HP)
    #Define the kernel size that can be used as an hyper-parameter to train the model with.5
    kernel_size = hp.Choice("kernel_size", values=[3])
    kernel_size_e1 = hp.Choice("kernel_size_e1", values=[3])
    kernel_size_e2 = hp.Choice("kernel_size_e2", values=[3])
    kernel_size_e3 = hp.Choice("kernel_size_e3", values=[3])
    kernel_size_d1 = hp.Choice("kernel_size_d1", values=[3])
    kernel_size_d2 = hp.Choice("kernel_size_d2", values=[3])
    kernel_size_d3 = hp.Choice("kernel_size_d3", values=[3])
    #Define the number of hidden layers that can be used as an hyper-parameter to train the model with. 
    num_hidden_layer = hp.Choice("num_hidden_layer", values=NUM_HIDDEN_LAYERS_HP)
    #Define the batch_size that can be used as an hyper-parameter to train the model with. 
    batch_size = hp.Int("batch_size", min_value=BATCH_SIZE_MIN, max_value=BATCH_SIZE_MAX, step=BATCH_SIZE_STEP)
    #Define regularization L2 for the conv01 of DeepSD model
    regL1_conv01 = hp.Choice('regL1_conv01', values=[0.0]) #0.0, 0.01, 0.001, 0.0001
    #Define regularization L2 for the conv01 of DeepSD model
    regL1_conv02 = hp.Choice('regL1_conv02', values=[0.0])
    #Define regularization L2 for the conv01 of DeepSD model
    regL2_conv01 = hp.Choice('regL2_conv01', values=[0.0])
    #Define regularization L2 for the conv01 of DeepSD model
    regL2_conv02 = hp.Choice('regL2_conv02', values=[0.0])
    #Loss function weights
    loss_weights = hp.Choice('loss_weights', values=[1.0])
    #Define the dropout rate for the DeepSD model
    drop_rate = hp.Choice('drop_rate', values=[0.40]) #, 0.05, 0.10, 0.15, 0.2     0.10, 0.20, 0.30, 0.40, 0.50
    #Define the alpha value for the leakyRelu activation function
    alpha = hp.Choice('alpha', values=[0.20]) #0.01, 0.1, 0.2, 0.3
 
    #Choose the model from model.py that the user want to work with.
    #TODO, the model could also be an hyper-parameter. 
    model = mod.deepRU_article_non_interpolation_combine_loss_transferLearning(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                       num_hidden_layer=num_hidden_layer, WIND=WIND, regL1_conv01=regL1_conv01, regL1_conv02=regL1_conv02, regL2_conv01=regL2_conv01, regL2_conv02=regL2_conv02, drop_rate=drop_rate, alpha=alpha, loss_weights=loss_weights)
    
    #Load the weights of the base model (The model the transfer learning will be done on)
    load_model = True
    if load_model:
        model.load_weights(f'/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000001_Final_search_HP_Search/model/Searchmodel_1_cb')

    commun_name = 'trainable'
    #Freeze the top layers and just train the decoder and post U-Net part
    for layer in model.layers:
        if commun_name in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    #     print(f"Layer: {layer.name}, Trainable: {layer.trainable}")
    # print(f'Model summary for the main code.')
    # model.summary()

    return model, batch_size

def run_trial(self, trial, *args, **kwargs):
    """
    Custom run_trial method for the hyperparameter tuning process.

    This function overrides the default run_trial method of a Keras Tuner tuner.
    It is designed to integrate the custom hyperparameter tuning logic, including
    the dynamic setting of batch size and capturing the performance metric (val_SSIMLoss).

    The function first retrieves the hyperparameters for the current trial, then uses these
    hyperparameters to construct a model via the hyper_parameters_search function. It then
    sets the 'batch_size' for training, which is dynamically determined by the tuning process.
    
    After training the model, it captures the last 'val_SSIMLoss' value from the training history,
    and updates the trial with this metric for performance tracking.

    Parameters:
    - self: Reference to the tuner instance.
    - trial: The Trial object containing information about the current trial, including hyperparameters.
    - *args, **kwargs: Additional arguments and keyword arguments for model.fit(), 
      such as training and validation data.

    The method updates the trial with the computed 'val_SSIMLoss', which is used by the tuner 
    to evaluate and compare different hyperparameter configurations.
    """
    #Retrieve the hyperparameters for the current trial.
    hp = trial.hyperparameters
    #Call hyper_parameter function used to define the hp search values and call the CNN model the use is training. 
    model, batch_size = hyper_parameters_search(hp)
    #Update the 'batch_size' in the kwargs dictionary, which will be used in model.fit().
    kwargs['batch_size'] = batch_size
    #Train the model with the provided arguments (*args) and keyword arguments (**kwargs)
    history = model.fit(*args, **kwargs)
    # print(f'Model summary after the fit method is called.')
    # model.summary()
    print(history.history.keys())
    #Extract the best value of 'val_SSIMLoss' from the training history.
    best_metric = min(history.history['val_mse']) #TODO  val_mean_absolute_error val_mean_squared_error val_SSIMLoss
    #Prevent -inf to be the best metric
    if best_metric == float('-inf'):
        best_metric = 1
    #Update the trial with the obtained 'val_SSIMLoss' metric.
    self.oracle.update_trial(trial.trial_id, {'val_mse': best_metric}) #TODO val_mean_squared_error val_mean_absolute_error val_SSIMLoss
    #Clear memory
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
    gc.collect()

def tuner_search(input_train, label_train, skip_train, topo_train, topo_val, EPOCH_TRY, input_val, label_val, skip_valid, earlystop, reduceLROnPlateau, endWhenNaN, VERBOSE):
    """
    Do the hyper-parameter search. Indeed it will train the model with the given set of parameters.
    Input
    :Param input_train: The training input data used to train the Neural Network. 
    :Param label_train: The training label data used to train the Neural Network. 
    :Param EPOCH_TRY: The number of epochs the Neural Network will train on. 
    :Param input_val: The validation input data used to validate the Neural Network.
    :Param label_val: The validation label data used to validate the Neural Network. 
    :Param earlystop: Callback function used to stop the model when the training isnt showing sign of amelioration for a certain epoch number. 
    :Param reduceLROnPlateau: Callback function used to reduce the learning rate when the training isnt showing sign of amelioration for a certain epoch number. 
    :Param endWhenNaN: Callback function used to end the training when NaN appear in the loss function calculation. 
    :Param VERBOSE: To display user information. 
    Output
    :Param tuner: Instance of the hyper-parameter tuning class. 
    """
    # print(f'The shape of the input_train is {input_train.shape}')
    # print(f'The shape of the topo_train is {topo_train.shape}')
    # print(f'The shape of the label_train is {label_train.shape}')
    # print(f'The shape of the input_val is {input_val.shape}')
    # print(f'The shape of the topo_val is {topo_val.shape}')
    # print(f'The shape of the label_val is {label_val.shape}')

    tuner.search(
        [input_train, topo_train, skip_train], 
        label_train, 
        epochs=EPOCH_TRY, 
        validation_data=([input_val, topo_val, skip_valid], label_val),
        callbacks=[earlystop , reduceLROnPlateau, endWhenNaN], #earlystop , reduceLROnPlateau, endWhenNaN
        verbose=VERBOSE,
        shuffle=True
    )

    return tuner

def write_hp_csv(HP_RESULT_PATH, best_trials):
    """
    Write into a csv the information of the hyper-parameters(hp) used when doing the search. Also the csv keep in memory the 
    best value of the metric loss for every single set of hp used during the search. This csv file can be used to visialize 
    which set of hp perform better than other and present valuable information for futur hp search. 
    Input 
    :Param HP_RESULT_PATH: The path where will be stored the result of the hyper-parameter search. 
    :Param best_trials: The hyper-parameters trials results obtain in the hp search. 
    Output
    :none.
    """
    #Create the CSV file headers, which are hyper-parameters and the value of the validation loss metric. 
    fieldnames = ['model_rank', 'units', 'activation', 'lr', 'batch_size', 'kernel_size', 'kernel_size_e1', 'kernel_size_e2', 'kernel_size_e3', 'kernel_size_d1', 'kernel_size_d2', 'kernel_size_d3','num_hidden_layer', 'regL1_conv01', 'regL1_conv02', 'regL2_conv01', 'regL2_conv02', 'drop_rate', 'alpha', 'loss_weights', 'val_mse'] # , 'val_psnr'  TODO val_mean_squared_error val_mean_absolute_error val_SSIMLoss
    #Open the csv file and append to the file the hyper-parameters and the value of the validation loss metric of for each trials. 
    with open(f"{HP_RESULT_PATH}/best_models.csv", "a", newline="") as csvfile:
        #Initiate the writer of the csv file to write inside the file.  
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        #Loop throw all the hyper-parameters trials. 
        for i, trial in enumerate(best_trials):
            #Access the hyperparameters and the best value of val_SSIMLoss from the current trial. 
            hps = trial.hyperparameters
            #Access the validation loss metric value (Is not part of the hyper-parameter search so work differently than others). 
            val_mse = trial.metrics.get_best_value('val_mse') # TODO  val_mean_absolute_error val_mean_squared_error val_SSIMLoss
            #val_psnr = trial.metrics.get_best_value('val_psnr')
            #Create the row information for the ranking, hyper-parameter and validation loss metric for the current model with his actual hyper-parameters values
            row_dict = {
                #Write the rank of the model. 
                "model_rank": i + 1,
                #Write the hyper-parameters information.
                "units": hps.get("units"),
                "activation": hps.get("activation"),
                "lr": hps.get("lr"),
                "batch_size": hps.get("batch_size"),
                "kernel_size": hps.get("kernel_size"),
                "kernel_size_e1": hps.get("kernel_size_e1"), 
                "kernel_size_e2": hps.get("kernel_size_e2"), 
                "kernel_size_e3": hps.get("kernel_size_e3"), 
                "kernel_size_d1": hps.get("kernel_size_d1"), 
                "kernel_size_d2": hps.get("kernel_size_d2"), 
                "kernel_size_d3": hps.get("kernel_size_d3"),
                "num_hidden_layer": hps.get("num_hidden_layer"),
                'regL1_conv01': hps.get("regL1_conv01"),
                'regL1_conv02': hps.get("regL1_conv02"),
                'regL2_conv01': hps.get("regL2_conv01"),
                'regL2_conv02': hps.get("regL2_conv02"),
                'drop_rate': hps.get('drop_rate'),
                'alpha': hps.get('alpha'),
                'loss_weights': hps.get('loss_weights'),
                #Write the validation loss metric of the current hyper-parameters set tested for the model.
                'val_mse': val_mse #TODO  val_mean_absolute_error  val_mean_squared_error val_SSIMLoss
                #'val_psnr': val_psnr
            }
            #Write the row infomation.
            writer.writerow(row_dict)

def model_train(model, input_train, topo_train, label_train, skip_train, BATCH_SIZE, EPOCHS, VERBOSE, input_val, topo_val, label_val, skip_valid, best_model_weights, earlystop, reduceLROnPlateau, endWhenNaN):
    """
    This is the training loop used to train the Neural Network. Here are define the hyper-parameters that the user 
    is trying to find using this program. To have more information on how the model.fit function works, please refers 
    to this documentation page of Keras: https://keras.io/api/models/model_training_apis/#fit-method 
    INPUT
    :Param input_train: inputs input data of the NN used to train the model.
    :Param label_train: labels input data of the NN used to train the model. 
    :Param BATCH_SIZE: Number of inputs that are processed in a single forward and backward pass during the training of the neural network.
    :Param EPOCHS: Number of complete pass through the entire training dataset during the training of the neural network.
    :Param VERBOSE: If you want the code to comment to help understand what is going on (Recommended). 
    :Param input_val: inputs validation data of the NN used to validate the performances of the model at each epochs. 
    :Param label_val: labels validation data of the NN used to validate the performances of the model at each epochs. 
    OUTPUT
    :Param history: Object of the record of the training process. Contains various metrics that were computed after each epoch during training
                    and during the validation of the NN. 
    """
    history = model.fit(
        x=[input_train, topo_train, skip_train],
        y=label_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[best_model_weights, earlystop, reduceLROnPlateau, endWhenNaN],  #, earlystop, reduceLROnPlateau, endWhenNaN
        verbose=VERBOSE,
        validation_data=([input_val, topo_val, skip_valid], label_val),
        shuffle=True   
    )

    return history

def evaluate_model(model, input_test, topo_test, label_test):
    """
    Evaluating the model performances with the testing data abd analysing the accuracy of the model.
    Information about the method can be found here on Keras documentation: # https://keras.io/api/models/model_training_apis/#fit-method
    INPUT
    :Param input_test: inputs testing data of the NN used to test the performances of the model at the end of the training. 
    :Param label_test: labels testing data of the NN used to test the performances of the model at the end of the training. 
    OUTPUT
    :None
    """
    test_loss, test_acc = model.evaluate(
    x=[input_test, topo_test],
    y=label_test,
    batch_size=1,
    verbose=VERBOSE
    )
    print(f"Test loss: {test_loss:.3f}")

def plot_graph(history):
    """
    Save a graph of the training and validation loss and the training and validation accuracy. These graphs
    are going to be saved at this path: HP_RESULT_PATH. 
    INPUT
    :Param history: Access the history of the training. 
    OUTPUT
    :None
    """
    #Plot training and validation loss
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, "blue", label="Training loss")
    plt.plot(epochs, val_loss, "orange", label="validation loss")
    plt.title("Training and validation loss MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    directory = HP_RESULT_PATH + "/Training"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/Train_val_loss_{name}.png')
    plt.close()
    # Plot training and validation accuracy
    acc = history.history["mse"] #  TODO mean_absolute_error SSIMLoss mean_squared_error
    val_acc = history.history["val_mse"] #    TODO val_mean_absolute_error val_mean_squared_error
    plt.plot(epochs, acc, "blue", label="Training accuracy")
    plt.plot(epochs, val_acc, "orange", label="Validation accuracy")
    plt.title("Training and validation accuracy SSIMLoss")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'{HP_RESULT_PATH}/Training/Train_val_acc_{name}.png')
    plt.close()

def prediction(model, test_data_input, test_topo_data, skip_test_data, VERBOSE):
    """
    Take the tested model and predict the output value of the CNN with the test input data. 
    The Network haven't been trained with this data so it will be a great indicator on how
    well the model will be performing. 
    INPUT
    :Param model: The model structure of the curent Neural Network with his weights. 
    :Param test_data_input: The test data that are used to test the performance of the Neural Network model and weights
    OUTPUT
    :Param Prediction: The output prediction of the Neural Network with his optimal weights.
    """
    #Predict the result of the model with the testing data.
    predictions = model.predict(
        x=[test_data_input, test_topo_data, skip_test_data],
        batch_size=None,
        verbose=VERBOSE,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )

    return predictions

def de_standardize(test_data_input, predictions, test_data_label, skip_test_data):
    std_inputs = np.load('/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy')
    mean_inputs = np.load('/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy') 
    std_label = np.load('/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy')
    mean_label = np.load('/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy') 
    std_skip = np.load('/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy')
    mean_skip = np.load('/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy')
    
    # Ensure std and mean are broadcastable to the data shape
    assert std_inputs.shape == mean_inputs.shape, "Standard deviation and mean file shapes do not match."

    # De-standardize inputs, using only the first channel stats
    mean_inputs_adjusted = np.transpose(mean_inputs, (0, 2, 3, 1)) 
    std_inputs_adjusted = np.transpose(std_inputs, (0, 2, 3, 1)) 
    print(f'The original_input shape is: {test_data_input.shape}')
    print(f'The std_inputs shape is: {std_inputs.shape} and the mean_inputs shape is: {mean_inputs.shape}')
    print(f'The std_inputs_adjusted shape is: {std_inputs_adjusted.shape} and the mean_inputs_adjusted shape is: {mean_inputs_adjusted.shape}')
    original_inputs = (test_data_input * std_inputs_adjusted[..., 0:1]) + mean_inputs_adjusted[..., 0:1]

    # # Resize mean and std for predictions and labels to match their dimensions
    # labels_shape = test_data_label.shape
    # mean_labels_resized = np.tile(mean_inputs[:, 0:1, :, :], (1, 1, labels_shape[2], labels_shape[3]))
    # std_labels_resized = np.tile(std_inputs[:, 0:1, :, :], (1, 1, labels_shape[2], labels_shape[3]))

    # # De-standardize predictions and labels using the resized mean and std
    # original_prediction = (predictions * std_labels_resized) + mean_labels_resized
    # original_labels = (test_data_label * std_labels_resized) + mean_labels_resized

    # De-standardize predictions and labels using the resized mean and std
    mean_label_adjusted = np.transpose(mean_label, (0, 2, 3, 1)) 
    std_label_adjusted = np.transpose(std_label, (0, 2, 3, 1)) 
    print(f'The standardized prediction shape is : {predictions.shape}')
    print(f'The std_label shape is : {std_label.shape} and the mean_label shape is : {mean_label.shape}')
    print(f'The std_label_adjusted shape is: {std_label_adjusted.shape} and the mean_label_adjusted shape is: {mean_label_adjusted.shape}')

    original_prediction = (predictions * std_label_adjusted[..., 0:1]) + mean_label_adjusted[..., 0:1]
    original_labels = (test_data_label * std_label_adjusted[..., 0:1]) + mean_label_adjusted[..., 0:1]

    print(f'The shape before the de-standardization for test_data_skip_de_standardized is: {skip_test_data.shape}')
    print(f'The shape of mean_skip is : {mean_skip.shape} and the shapt of std_skip is: {std_skip.shape}')
    # De-standardize skip connection using the their standardization information
    mean_skip_adjusted = np.transpose(mean_skip, (0, 2, 3, 1))  # Shape: (1, 48, 48, 1)
    std_skip_adjusted = np.transpose(std_skip, (0, 2, 3, 1)) 
    original_skip = (skip_test_data * std_skip_adjusted) + mean_skip_adjusted
    print(f'The shape after the de-standardization for the test_data_skip_de_standardized is: {skip_test_data.shape}')

    return original_inputs, original_prediction, original_labels, original_skip

def crop_result(test_data_input, predictions, test_data_label, test_data_skip_de_standardized):
    print(f'The data shape is : {test_data_input.shape}')
    print(f'The data shape is : {predictions.shape}')
    print(f'The data shape is : {test_data_label.shape}')
    print(f'The data shape is : {test_data_skip_de_standardized.shape}')
    test_data_input = test_data_input[:, :, :, :]
    predictions = predictions[:, 4:-4, 4:-4, :]
    test_data_label = test_data_label[:, 4:-4, 4:-4, :]
    test_data_skip_de_standardized = test_data_skip_de_standardized[:, 4:-4, 4:-4, :]

    return test_data_input, predictions, test_data_label, test_data_skip_de_standardized

def metric_pred_imagecsv(directory, images, hyp, MAE, MSE, RMSE, PSNR, SSIM):
    """
    Write in a csv file metrics values between the difference of the prediction and label images. At the moment it will calculate the MAE, MSE, RMSE and PSNR between
    each choosen test images for every selected models. This metrics results can be used to tell the performances of the model by analysing their value. Be careful, some
    metrics are better the smaller they are and other are better the bigger they are. 
    Input 
    :Param directory: The directory where will be saved the csv file. 
    :Param images: An index to know which image to select in the numpy array for the test data. 
    :Param hyp: Object that allow to access the hyper-parameters information of the current analysed model. 
    :Param MAE: An initial numpy array to store the MAE results. 
    :Param MSE: An initial numpy array to store the MSE results. 
    :Param RMSE: An initial numpy array to store the RMSE results. 
    :Param PSNR: An initial numpy array to store the PSNR results. 
    Output
    :none
    """
    #Create csv file header, which are image number, learning_rate, batch_size, mae, mse, rmse, psnr, ssim.
    fieldnames = ['image_number', 'learning_rate', 'batch_size', 'mae', 'mse', 'rmse', 'psnr', 'ssim']
    #Open the csv file and append the to the file the information of the model and the metrics value between the original image and the predicted image
    with open(f'{directory}/metrics_predicted_images.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if images == 0:
            writer.writeheader()
        #Create the information in the csv file. #TODO optimize and take in consideration the numpy array. 
        row_dict = {
            #Write the image number that was regenerated
            'image_number': images+1,
            #Write the learning rate used
            'learning_rate': hyp.get("lr"),
            #Write the batch size used
            'batch_size': hyp.get("batch_size"),
            #Write the mae metric score
            'mae' : MAE[images],
            #Write the mse metric score
            'mse' : MSE[images],
            #Write the rmse metric score
            'rmse' : RMSE[images],
            #Write the psnr metric score
            'psnr' : PSNR[images],
            #Write the ssim metric score
            'ssim' : SSIM[images],
        }
        #Write the row information.
        writer.writerow(row_dict)

def rank_metric_pred_imagecsv(directory):
    """
    It will give a rank from 1 to N to every metrics to allow the user of the program to know which regenerated image
    have the best metrics over for a certain metric. The rank will be stored in a new column in the csv file and will 
    be sorted based on the mae ranking. Note that it is possible to change this metric for any other based on what the
    user think is the best metrics to analyse to generate the best images. It will then overwrite the previous csv file
    created with metric_pred_imagecsv function and add the rank metric and the sorted csv over a choosen metric. 
    Input 
    :Param directory: The directory where will be saved the csv file. 
    Output
    :none.
    """
    #Load the csv file
    df = pd.read_csv(f'{directory}/metrics_predicted_images.csv')
    #Rank the column based on a given metric. 
    df['mae Rank'] = df['mae'].rank(ascending=True)
    df['mse Rank'] = df['mse'].rank(ascending=True)
    df['rmse Rank'] = df['rmse'].rank(ascending=True)
    df['psnr Rank'] = df['psnr'].rank(ascending=False)
    df['ssim Rank'] = df['ssim'].rank(ascending=False)

    #Sorting the value based on the mae. 
    df_sorted = df.sort_values(by='ssim', ascending=False)
    #Save the file back to csv. 
    df_sorted.to_csv(f'{directory}/metrics_predicted_images.csv', index=False)

def mean_metric_pred_imagecsv(HP_RESULT_PATH, j, hyp, mean_MAE, mean_MSE, mean_RMSE, mean_PSNR, mean_SSIM):
    #Create csv file header, which are image number, learning_rate, batch_size, mae, mse, rmse, psnr, ssim.
    fieldnames = ['image_number', 'learning_rate', 'batch_size', 'mean_mae', 'mean_mse', 'mean_rmse', 'mean_psnr', 'mean_ssim'] #, 'mean_ssim'
    #Open the csv file and append the to the file the information of the model and the metrics value between the original image and the predicted image
    with open(f'{HP_RESULT_PATH}/average_metrics_predicted_images.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #If it is the first model, write the header. 
        if j == 1:
            writer.writeheader()
        #Create the information in the csv file. #TODO optimize and take in consideration the numpy array. 
        row_dict = {
            #Write the image number that was regenerated
            'image_number': j,
            #Write the learning rate used
            'learning_rate': hyp.get("lr"),
            #Write the batch size used
            'batch_size': hyp.get("batch_size"),
            #Write the mae metric score
            'mean_mae' : mean_MAE[j-1],
            #Write the mse metric score
            'mean_mse' : mean_MSE[j-1],
            #Write the rmse metric score
            'mean_rmse' : mean_RMSE[j-1],
            #Write the psnr metric score
            'mean_psnr' : mean_PSNR[j-1],
            #Write the ssim metric score
            'mean_ssim' : mean_SSIM[j-1],
        }
        #Write the row information.
        writer.writerow(row_dict)

def rank_mean_metric_pred_imagecsv(HP_RESULT_PATH):
    """
    It will give a rank from 1 to N to every metrics to allow the user of the program to know which regenerated image
    have the best metrics over for a certain metric. The rank will be stored in a new column in the csv file and will 
    be sorted based on the mean_mae ranking. Note that it is possible to change this metric for any other based on what the
    user think is the best metrics to analyse to generate the best images. It will then overwrite the previous csv file
    created with mean_metric_pred_imagecsv function and add the rank metric and the sorted csv over a choosen metric. 
    Input 
    :Param HP_RESULT_PATH: The directory where will be saved the csv file. 
    Output
    :none.
    """
    #Load the csv file
    df = pd.read_csv(f'{HP_RESULT_PATH}/average_metrics_predicted_images.csv')
    #Rank the column based on a given metric. 
    df['mean mae Rank'] = df['mean_mae'].rank(ascending=True)
    df['mean mse Rank'] = df['mean_mse'].rank(ascending=True)
    df['mean rmse Rank'] = df['mean_rmse'].rank(ascending=True)
    df['mean psnr Rank'] = df['mean_psnr'].rank(ascending=False)
    df['mean ssim Rank'] = df['mean_ssim'].rank(ascending=True)

    #Sorting the value based on the mean_mae. 
    df_sorted = df.sort_values(by='mean_mae', ascending=True)
    #Save the file back to csv. 
    df_sorted.to_csv(f'{HP_RESULT_PATH}/average_metrics_predicted_images.csv', index=False)

def create_prediction_image(images, test_data_input, predictions, test_data_label, index,
                            mae, mse, rmse, rmse_benchmark, psnr, ssim, SSIM_benchmark):
    input_float = test_data_input[images, :, :, index]
    predictions_float = predictions[images, :, :, 0]
    label_float = test_data_label[images, :, :, 0]
    difference = predictions_float - label_float

    # Determine the common color scale range for predictions and label
    vmin = np.min(label_float)
    vmax = np.max(label_float)

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle("Results", fontsize=20)

    data = [input_float, predictions_float, label_float, difference]
    titles = ["Inputs", "Result", "Target", "Difference", "Metrics"]
    cmaps = ['magma' for _ in range(4)] + [None]  # No colormap for the metrics

    for i, (ax, title) in enumerate(zip(axes, titles)):
        if i < 4:  # For the first four plots
            if i == 1 or i == 2:  # Apply shared color scale for predictions and label
                im = ax.imshow(data[i], origin='lower', cmap=cmaps[i], aspect='auto',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
            else:
                im = ax.imshow(data[i], origin='lower', cmap=cmaps[i], aspect='auto')
            fig.colorbar(im, ax=ax)
        else:  # For the metrics plot
            metrics_text = f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nRMSE_benchmark: {rmse_benchmark}\nPSNR: {psnr}\nSSIM: {ssim}\nSSIM_benchmark: {SSIM_benchmark}"
            ax.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=24, transform=ax.transAxes)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    return fig, axes

def compute_psd(image):
    # Perform the 2D FFT
    f_transform = fft2(image)
    # Shift the zero frequency component to the center
    f_transform_shifted = fftshift(f_transform)
    # Compute the power spectral density
    psd = np.abs(f_transform_shifted) ** 2
    return psd

def azimuthal_average(psd):
    y, x = np.indices(psd.shape)
    center = np.array([x.max() / 2.0, y.max() / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def power_spectral_density_graph_one_image(predicted_image, ground_truth_image, directory, images, test_data_skip_de_standardized):
    # print(f'The shape of the ground truth image is : {ground_truth_image[images, :, :, 0].shape}')
    # print(f'The shape of the predicted_image is : {predicted_image[images, :, :, 0].shape}')
    psd_ground_truth = compute_psd(ground_truth_image[images, :, :, 0])
    psd_predicted = compute_psd(predicted_image[images, :, :, 0])
    psd_skip = compute_psd(test_data_skip_de_standardized[images, :, :, 0])
    psd_gt_azimuthal = azimuthal_average(psd_ground_truth)
    psd_pred_azimuthal = azimuthal_average(psd_predicted)
    psd_skip_azimuthal = azimuthal_average(psd_skip)
    
    # Convert PSD to dB
    psd_gt_db = 10 * np.log10(psd_gt_azimuthal)
    psd_pred_db = 10 * np.log10(psd_pred_azimuthal)
    psd_benchmark = 10 * np.log10(psd_skip_azimuthal)
    
    size = psd_ground_truth.shape[0]
    freqs = np.fft.fftfreq(size)[:size // 2]  # Compute frequencies in cycles per pixel
    freqs = freqs[freqs > 0]  # Remove zero frequency for log-log plot

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_gt_db[:len(freqs)], label='Ground Truth')
    plt.plot(freqs, psd_pred_db[:len(freqs)], label='Predicted')
    plt.plot(freqs, psd_benchmark[:len(freqs)], label='benchmark')
    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Comparison of Power Spectral Densities in dB')
    plt.yscale('log')  # Set the y-axis to log scale
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{directory}/psd_{images+1}.png')
    plt.close()

# Compute and plot the average power spectral density
def power_spectral_density_graph_average(predicted_images, ground_truth_images, directory, test_data_skip_de_standardized):
    num_images = predicted_images.shape[0]
    
    psd_gt_azimuthals = []
    psd_pred_azimuthals = []
    psd_benchmark_azimuthals = []
    
    for i in range(num_images):
        psd_ground_truth = compute_psd(ground_truth_images[i, :, :, 0])
        psd_predicted = compute_psd(predicted_images[i, :, :, 0])
        psd_benchmark = compute_psd(test_data_skip_de_standardized[i, :, :, 0])
        
        psd_gt_azimuthals.append(azimuthal_average(psd_ground_truth))
        psd_pred_azimuthals.append(azimuthal_average(psd_predicted))
        psd_benchmark_azimuthals.append(azimuthal_average(psd_benchmark))
    
    # Compute the average PSD
    avg_psd_gt_azimuthal = np.mean(psd_gt_azimuthals, axis=0)
    avg_psd_pred_azimuthal = np.mean(psd_pred_azimuthals, axis=0)
    avg_psd_benchmark_azimuthal = np.mean(psd_benchmark_azimuthals, axis=0)
    
    # Convert PSD to dB
    avg_psd_gt_db = 10 * np.log10(avg_psd_gt_azimuthal)
    avg_psd_pred_db = 10 * np.log10(avg_psd_pred_azimuthal)
    avg_psd_benchmark_db = 10 * np.log10(avg_psd_benchmark_azimuthal)
    
    size = psd_ground_truth.shape[0]
    freqs = np.fft.fftfreq(size)[:size // 2]  # Compute frequencies in cycles per pixel
    freqs = freqs[freqs > 0]  # Remove zero frequency for log-log plot

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, avg_psd_gt_db[:len(freqs)], label='Ground Truth')
    plt.plot(freqs, avg_psd_pred_db[:len(freqs)], label='Predicted')
    plt.plot(freqs, avg_psd_benchmark_db[:len(freqs)], label='BenchMark')
    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Comparison of Average Power Spectral Densities in dB')
    plt.yscale('log')  # Set the y-axis to log scale
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{directory}/average_psd.png')
    plt.close()

def argparser():
    """
    Take the user input in the trainJs.sh script. This input will then be used in this code as variables and constants. 
    INPUT
    :none
    OUTPUT
    :Param parser.parse_args(): Used to parse the user input with a variable or a constant. 
    """
    # Argparse initialisation to allow external input to be used in the code
    parser = argparse.ArgumentParser(description="Arguments used in the code that are called in the bash file or the terminal")
    
    parser.add_argument("--INPUT_FILE", type=str, help="The path where the inputs file are located")
    parser.add_argument("--LABEL_FILE", type=str, help="The path where the labels data are located")
    parser.add_argument("--TOPO_FILE", type=str, help="The path where the topography data are located")
    parser.add_argument('--TRAINING_SIZE', type=int, help='Number of input images used in the training.')
    parser.add_argument('--VALID_SIZE', type=int, help='Number of input images used to evaluate the model during the training process.')
    parser.add_argument('--TEST_SIZE', type=int, help='Number of input images used to evaluate the model after the training.')
    parser.add_argument('--TEMPERATURE', action='store_true', help='Flag used when working with temperature model.')
    parser.add_argument('--WIND', action='store_true', help='Flag used when working with Wind model and predicting only UU.')
    parser.add_argument('--WIND_UU', action='store_true', help='Flag used when working with Wind model and predicting only UU.')
    parser.add_argument('--WIND_VV', action='store_true', help='Flag used when working with Wind model and predicting only VV.')
    parser.add_argument('--WIND_UV', action='store_true', help='Flag used when working with Wind model and predicting only UV.')
    parser.add_argument('--data_already_spllit', action='store_true', help='Flag used when the data is pre split into set (train, valid, test).')
    parser.add_argument('--PATH_DP', type=str, help='Path that refer to the folder nammed Data_Processing/C_normalize.')
    parser.add_argument('--PATH_RESULTS', type=str, help='The path where the results are going to be stored at.')
    parser.add_argument('--HYPER_PARAMETERS_SEARCHS', action='store_true', help='Flag if the user want do to a hyper-paremeters.')
    parser.add_argument("--HP_NAME", type=str, help="The name of the hyper-parameters the user is curently doing (To store the results at).")
    parser.add_argument("--NUM_TRIALS", type=int, help="The number of try to find the best hyper-parameters combinaison.")
    parser.add_argument('--EXECUTIONS_PER_TRIAL', type=int, help='The number of time the same hyper-parameters combinaison will be tested.')
    parser.add_argument("--N_BEST", type=int, help="The N best models obtain in the hp search that will be used to train and test the model.")
    parser.add_argument("--EPOCH_TRY", type=int, help="The epoch number for the hp search.")
    parser.add_argument('--UNITS_HP', nargs='+', help='List of units(number of kernels) that can be used as an hyper-parameter to train the model with.')
    parser.add_argument('--ACTIVATION_FUNCTION_HP', nargs='+', help='List of activation function that can be used as an hyper-parameter to train the model with.')
    parser.add_argument('--LEARNING_RATE_HP', nargs='+', type=float, help='List of Learning rate that can be used as an hyper-parameter to train the model with.')
    parser.add_argument('--KERNEL_SIZE_HP', nargs='+', help='List of kernel size that can be used as an hyper-parameter to train the model with.')
    parser.add_argument('--NUM_HIDDEN_LAYERS_HP', nargs='+', help='List of hidden layers that can be used as an hyper-parameter to train the model with.')
    parser.add_argument('--BATCH_SIZE_MIN', type=int, help='Min number of epochs used to train the Neural Network.')
    parser.add_argument('--BATCH_SIZE_MAX', type=int, help='Max number of epochs used to train the Neural Network.')
    parser.add_argument('--BATCH_SIZE_STEP', type=int, help='Step between number of epochs used to train the Neural Network.')
    parser.add_argument('--UNITS', type=int, help='Units(number of kernels) that can be used as an hyper-parameter to train the model with (Non hp search).')
    parser.add_argument('--ACTIVATION_FUNCTION', type=str, help='Activation function that can be used as an hyper-parameter to train the model with (Non hp search).')
    parser.add_argument('--LEARNING_RATE', type=float, help='Learning rate that can be used as an hyper-parameter to train the model with (Non hp search).')
    parser.add_argument('--KERNEL_SIZE', type=int, help='kernel size that can be used as an hyper-parameter to train the model with (Non hp search).')
    parser.add_argument('--NUM_HIDDEN_LAYERS', type=int, help='Hidden layers that can be used as an hyper-parameter to train the model with (Non hp search).')
    parser.add_argument('--BATCH_SIZE', type=int, help='Number of inputs that are processed in a single forward and backward pass during the training of the neural network (Non hp search).')
    parser.add_argument('--EPOCH', type=int, help='The epoch number for the hp search.')
    parser.add_argument('--NUM_TRAIN_REGENERATE', type=int, help='Number of training data that will be regenerated in the test set.')
    parser.add_argument('--VERBOSE', action='store_true', help='Flag if the user want the code to print to help debug (Non hp search).')

    return parser.parse_args()

#.**************************************************************************************************************************************.#
#                                                                 Main                                                                   #
#.**************************************************************************************************************************************.#
if __name__ == '__main__':
    #Parse the argument
    args = argparser()

    #Give the parse to a variable.
    INPUT_FILE = args.INPUT_FILE
    LABEL_FILE = args.LABEL_FILE
    TOPO_FILE = args.TOPO_FILE
    TRAINING_SIZE = args.TRAINING_SIZE
    VALID_SIZE = args.VALID_SIZE
    TEST_SIZE = args.TEST_SIZE
    TEMPERATURE = args.TEMPERATURE
    WIND = args.WIND
    WIND_UU = args.WIND_UU
    WIND_VV = args.WIND_VV
    WIND_UV = args.WIND_UV
    DATA_ALREADY_SPLIT = args.data_already_spllit
    PATH_DP = args.PATH_DP
    PATH_RESULTS = args.PATH_RESULTS
    HYPER_PARAMETERS_SEARCHS = args.HYPER_PARAMETERS_SEARCHS
    HP_NAME = args.HP_NAME
    NUM_TRIALS = args.NUM_TRIALS
    EXECUTIONS_PER_TRIAL =args.EXECUTIONS_PER_TRIAL
    N_BEST = args.N_BEST
    EPOCH_TRY = args.EPOCH_TRY
    UNITS_HP = args.UNITS_HP
    ACTIVATION_FUNCTION_HP = args.ACTIVATION_FUNCTION_HP
    LEARNING_RATE_HP = args.LEARNING_RATE_HP
    KERNEL_SIZE_HP = args.KERNEL_SIZE_HP
    NUM_HIDDEN_LAYERS_HP = args.NUM_HIDDEN_LAYERS_HP
    BATCH_SIZE_MIN = args.BATCH_SIZE_MIN
    BATCH_SIZE_MAX = args.BATCH_SIZE_MAX
    BATCH_SIZE_STEP = args.BATCH_SIZE_STEP
    UNITS = args.UNITS
    ACTIVATION_FUNCTION = args.ACTIVATION_FUNCTION
    LEARNING_RATE = args.LEARNING_RATE
    KERNEL_SIZE = args.KERNEL_SIZE
    NUM_HIDDEN_LAYERS = args.NUM_HIDDEN_LAYERS
    BATCH_SIZE = args.BATCH_SIZE
    EPOCH = args.EPOCH
    NUM_TRAIN_REGENERATE = args.NUM_TRAIN_REGENERATE
    VERBOSE = args.VERBOSE

    # #If the user want to use the python debugger and not use bash files.
    # INPUT_FILE = '/home/jfg000/ss5/Data/input/domaine/gaspe/UU_VV_UV_TT_P0_H_CX_SD_WGE_data'
    # LABEL_FILE = '/home/jfg000/ss5/Data/label/domaine/gaspe/UU_data'
    # TOPO_FILE = '/home/jfg000/ss5/Data/topography/east_canada_sequential_train_domaine1/ME_MG_Z0_data'
    # TRAINING_SIZE = 2194
    # VALID_SIZE = 1212
    # TEST_SIZE = 1212
    # TEMPERATURE = False
    # WIND = False
    # WIND_UU = True
    # WIND_VV = False
    # WIND_UV = False
    # DATA_ALREADY_SPLIT = True
    # PATH_DP = '/home/jfg000/ss5/SuperResolution/Data_Processing/C_normalize/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_de_standardized_information'
    # PATH_RESULTS='/home/jfg000/ss5/CNN_results'


    # HYPER_PARAMETERS_SEARCHS = True #If false, erase --HYPER_PARAMETERS_SEARCHS on line 101
    # HP_NAME = 'NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_sequential_testalldomains' #'Js_WindVV_DeepSD_l_ssim_m_mean_absolut_test1' #Always change this path.  #Js_Wind_FSRCNN_l_mean_absolut_m_ssim_test1
    # NUM_TRIALS = 1
    # EXECUTIONS_PER_TRIAL = 1
    # N_BEST = 1
    # EPOCH_TRY = 1
    # UNITS_HP = [64]
    # ACTIVATION_FUNCTION_HP = ['relu']
    # LEARNING_RATE_HP = [0.00001, 0.0001, 0.001]
    # KERNEL_SIZE_HP=[3]
    # NUM_HIDDEN_LAYERS_HP=[3]
    # BATCH_SIZE_MIN=16
    # BATCH_SIZE_MAX=240
    # BATCH_SIZE_STEP=16

    # #Constant for Non-Hyper-parameters search (If HYPER_PARAMETERS_SEARCHS=FALSE) Line 65 to 79 can be any value, they wont affect the program. 
    # UNITS = 64
    # ACTIVATION_FUNCTION = 'relu'
    # LEARNING_RATE = 0.0001
    # KERNEL_SIZE = 3
    # NUM_HIDDEN_LAYERS = 2
    # BATCH_SIZE = 8
    # EPOCH = 600

    # #Constant for model evaluation
    # NUM_TRAIN_REGENERATE = 40
    # VERBOSE = True

    #Find if there are GPUs on the device. 
    physical_device = tf.config.experimental.list_physical_devices('GPU')
    print(f'Device found : {physical_device}')

    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    #Create the path that will be used later in the program to load data and to save it. 
    if DATA_ALREADY_SPLIT:
        INPUT_FILE_TRAIN, INPUT_FILE_VALID, INPUT_FILE_TEST, LABEL_FILE_TRAIN, LABEL_FILE_VALID, LABEL_FILE_TEST, SKIP_FILE_TRAIN, SKIP_FILE_VALID, SKIP_FILE_TEST, TOPO_FILE, directory, HP_RESULT_PATH, HP_DIR, var_name = path_creation(INPUT_FILE, LABEL_FILE, TOPO_FILE, PATH_RESULTS, HP_NAME, DATA_ALREADY_SPLIT)
        inputs_train, inputs_valid, inputs_test, labels_train, label_valid, label_test, skip_train, skip_valid, skip_test, topo_train = load_data_already_split(INPUT_FILE_TRAIN, INPUT_FILE_VALID, INPUT_FILE_TEST, LABEL_FILE_TRAIN, LABEL_FILE_VALID, LABEL_FILE_TEST, SKIP_FILE_TRAIN, SKIP_FILE_VALID, SKIP_FILE_TEST, TOPO_FILE)
        #Calling dataProcessing function that will create a train, validation and test set for the input and label data.
        input_train, input_val, input_test, label_train, label_val, label_test, skip_train, skip_valid, skip_test, topo_train, topo_val, topo_test = dataProcessing_already_split(inputs_train, inputs_valid, inputs_test, labels_train, label_valid, label_test, skip_train, skip_valid, skip_test, topo_train)
    else:
        INPUT_FILE, LABEL_FILE, TOPO_FILE, directory, HP_RESULT_PATH, HP_DIR, var_name = path_creation(INPUT_FILE, LABEL_FILE, TOPO_FILE, PATH_RESULTS, HP_NAME, DATA_ALREADY_SPLIT)
        
    #User is doing an hyper-parameters search.
    if HYPER_PARAMETERS_SEARCHS:
        #Initialize the hyper-parameter search. Keras doc here: https://keras.io/api/keras_tuner/tuners/random/ 
        tuner = tuner_initialization(hyper_parameters_search, NUM_TRIALS, EXECUTIONS_PER_TRIAL, directory, HP_DIR)

        #Replace the defaut run_trial method of the MultiExecutionTuner class with the custom run_trial funciton.
        multi_execution_tuner.MultiExecutionTuner.run_trial = run_trial
        #Call the callback function without the best_model_weights, here we just want to search for the best hp fit. 
        earlystop, reduceLROnPlateau, endWhenNaN = cb.callbacks(name=None, HYPER_PARAMETERS_SEARCHS=HYPER_PARAMETERS_SEARCHS, POST_HPS=POST_HPS, VERBOSE=VERBOSE, HP_RESULT_PATH=None)

        #Start the hyper-parameter search and get the best model. Keras doc here: https://keras.io/keras_tuner/ 
        tuner = tuner_search(input_train, label_train, skip_train, topo_train, topo_val, EPOCH_TRY, input_val, label_val, skip_valid, earlystop, reduceLROnPlateau, endWhenNaN, VERBOSE)

        #Write the hyper-parameter information in a csv file ranking all of the best model from the best to the wrost:
        #Get all of the trials values. 
        best_trials = tuner.oracle.get_best_trials(num_trials=NUM_TRIALS)
        #Check and create the directory to store the result at. 
        if not os.path.exists(HP_RESULT_PATH):
            os.makedirs(HP_RESULT_PATH)
        write_hp_csv(HP_RESULT_PATH, best_trials)

    #If the user is not doing an hyper-parameter search, train the model with choosen values. 
    else:
        #Define the flag because the hyper-parameter search isn't being done.
        POST_HPS = 1
        #Call the choosen CNN model with the user value. 
        model = mod.deepRU_article_non_interpolation_combine_loss_transferLearning(units=UNITS, activation=ACTIVATION_FUNCTION, lr=LEARNING_RATE, kernel_size=KERNEL_SIZE, num_hidden_layer=NUM_HIDDEN_LAYERS, WIND=WIND, regL1_conv01=0.01, regL1_conv02=0.01, regL2_conv01=0.01, regL2_conv02=0.01, drop_rate=0.5, alpha=0.01, loss_weights=0.2)
        #Call the callback functions from scrips callback.py
        best_model_weights, earlystop, reduceLROnPlateau, endWhenNaN = cb.callbacks(name=None, HYPER_PARAMETERS_SEARCHS=HYPER_PARAMETERS_SEARCHS, POST_HPS=POST_HPS, VERBOSE=VERBOSE, HP_RESULT_PATH=None)
        #Train the model with this set of hyper-parameters and the callbacks functions. 
        history = model_train(model, input_train, label_train, BATCH_SIZE, EPOCH, VERBOSE, input_val, label_val, best_model_weights, earlystop, reduceLROnPlateau, endWhenNaN)
        #Call plot function to create training and validation loss, training and validation accuracy.
        plot_graph(history)

        #Exit the code after training the model
        sys.exit()

    #Get the n best model obtain in the training. Will be used later to train the model with data.  
    best_hp = tuner.get_best_hyperparameters(num_trials=N_BEST)

    #Initialize array for average prediction calculation. 
    mean_MAE = np.full(N_BEST,0.0)
    mean_MSE = np.full(N_BEST,0.0)
    mean_RMSE = np.full(N_BEST,0.0)
    mean_PSNR = np.full(N_BEST,0.0)
    mean_SSIM = np.full(N_BEST,0.0)

    #Enumerate over the best n models. 
    for j, hyp in enumerate(best_hp, start=1):
        print(f'The length for best_hp is {len(best_hp)}')
        print(f'The N_BEST value is {N_BEST}')
        print(f'Testing model {j}')
        #Recreate the best model with his set of hyper-parameters.
        model = mod.deepRU_article_non_interpolation_combine_loss_transferLearning(units=hyp.get("units"), activation=hyp.get("activation"), lr=hyp.get("lr"), kernel_size=hyp.get("kernel_size"), kernel_size_e1=hyp.get("kernel_size_e1"), kernel_size_e2=hyp.get("kernel_size_e2"),  kernel_size_e3=hyp.get("kernel_size_e3"),
                           kernel_size_d1=hyp.get("kernel_size_d1"), kernel_size_d2=hyp.get("kernel_size_d2"), kernel_size_d3=hyp.get("kernel_size_d3"),
                           num_hidden_layer=hyp.get("num_hidden_layer"), WIND=WIND, regL1_conv01=hyp.get('regL1_conv01'), regL1_conv02=hyp.get('regL1_conv02'),  regL2_conv01=hyp.get('regL2_conv01'), regL2_conv02=hyp.get('regL2_conv02'), drop_rate=hyp.get('drop_rate'), alpha=hyp.get('alpha'), loss_weights=hyp.get('loss_weights'))
            
        print(f'Loading model')
        load_model = True
        if load_model:
            model.load_weights(f'/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000001_Final_search_HP_Search/model/Searchmodel_1_cb')

        #Flag to tell the program you are done with the hp search and training the best models from that serach.
        POST_HPS = 1
        #Create the name for the current model. 
        name = f"model_{j}"
        name_cb = f"model_{j}_cb"

        print(f'Calling callbacks')
        #Calling the callback function to stop the training, save the best weights on the epoch, and change the learning rate if needed.
        best_model_weights, earlystop, reduceLROnPlateau, endWhenNaN = cb.callbacks(name=name_cb , HYPER_PARAMETERS_SEARCHS=HYPER_PARAMETERS_SEARCHS, POST_HPS=POST_HPS, VERBOSE=VERBOSE, HP_RESULT_PATH=HP_RESULT_PATH)
        
        print(f'Doing history of training')
        #When training the model form the hp search, need to put more epochs, like said in Francois Chollet book. 
        #Train the model with the current hyper-parameters set and the selected callbacks. 
        history = model_train(model, input_train, topo_train, label_train, skip_train, hyp.get("batch_size"), int(1.2*EPOCH_TRY), VERBOSE, input_val, topo_val, label_val, skip_valid, best_model_weights, earlystop, reduceLROnPlateau, endWhenNaN)

        #Call plot function to create training and validation loss, training and validation accuracy.
        print(f'Plotting history')
        plot_graph(history)

        #TESTING TODO create a new script for the testing to be separate from the training of the Neural Network. 
        #Evaluate model.
        #evaluate_model(model, input_test, topo_test, label_test)

        #Load the best model weights.
        print(f'Loading weights')
        model.load_weights(f"{HP_RESULT_PATH}/model/Search{name_cb}")

        #Select all or a specify size for the input test values. 
        test_data_input = input_test[0:NUM_TRAIN_REGENERATE] #TODO Change if you do not want to take all the test data each time you test the model. 
        test_topo_data = topo_test[0:NUM_TRAIN_REGENERATE]
        skip_test_data =  skip_test[0:NUM_TRAIN_REGENERATE]
        test_data_label = label_test[0:NUM_TRAIN_REGENERATE]

        #Predict the result of the model with the testing data.
        print(f'Doing the prediction')
        predictions = prediction(model, test_data_input, test_topo_data, skip_test_data, VERBOSE)
        
        #Convert the tensor back to numpy array
        #Go to the cpu to allow the convertion to works
        test_data_input = test_data_input.cpu().numpy()
        test_topo_data = test_topo_data.cpu().numpy()
        test_data_label = test_data_label.cpu().numpy()
        skip_test_data = skip_test_data.cpu().numpy()

        #De-standardize the selected test input and label data. 
        print(f'Destandardizing')
        test_data_input_de_standardized, predictions_de_standardized, test_data_label_de_standardized, test_data_skip_de_standardized = de_standardize(test_data_input, predictions, test_data_label, skip_test_data)
        print(f'The shape for the test_data_skip_de_standardized after de-standardization is: {test_data_skip_de_standardized.shape}')


        #Remove the padding pixels for the the input, prediction, label image and skip connection 
        print(f'Removing padding')
        test_data_input_de_standardized, predictions_de_standardized, test_data_label_de_standardized, test_data_skip_de_standardized = crop_result(test_data_input_de_standardized, predictions_de_standardized, test_data_label_de_standardized, test_data_skip_de_standardized)

        #Calculate metrics to note the performances of the predicted images and regenerate test images. 
        #Initialize Array for metric calculation 
        NUM_TRAIN_REGENERATE = len(input_test)
        MAE = np.full(NUM_TRAIN_REGENERATE, 0.0)
        MSE = np.full(NUM_TRAIN_REGENERATE, 0.0)
        RMSE = np.full(NUM_TRAIN_REGENERATE, 0.0)
        RMSE_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)
        PSNR = np.full(NUM_TRAIN_REGENERATE, 0.0)
        SSIM = np.full(NUM_TRAIN_REGENERATE, 0.0)
        SSIM_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)
        

        #Itterate over the range of the TEST_SIZE to calculate the metrics and regenerate all images. 
        for images in range(300):
            print(f'The length of test_data_input_de_standardized is {len(test_data_input_de_standardized)}')
            print(f' we are at image: {images}')
            #Calculate metrics between the prediction of the Neural Network and the label image
            MAE[images] = np.mean(np.abs(predictions_de_standardized[images] - test_data_label_de_standardized[images]))
            MSE[images] = np.mean(np.square(predictions_de_standardized[images] - test_data_label_de_standardized[images]))
            RMSE[images] = np.sqrt(np.mean(np.square(predictions_de_standardized[images] - test_data_label_de_standardized[images])))
            RMSE_benchmark[images] = np.sqrt(np.mean(np.square(test_data_skip_de_standardized[images] - test_data_label_de_standardized[images])))
            data_range = test_data_label_de_standardized[images].max() - test_data_label_de_standardized[images].min()
            PSNR[images] = psnr(test_data_label_de_standardized[images], predictions_de_standardized[images], data_range=data_range)
            SSIM[images] = ssim(test_data_label_de_standardized[images].squeeze(), predictions_de_standardized[images].squeeze(), data_range=data_range)
            SSIM_benchmark[images] = ssim(test_data_skip_de_standardized[images].squeeze(), predictions_de_standardized[images].squeeze(), data_range=data_range)


            #Create an image of the result of the prediction. [inputs, output, labels]
            if TEMPERATURE: 
                fig, axes = create_prediction_image(images, test_data_skip_de_standardized, predictions_de_standardized, test_data_label_de_standardized, 5,
                                                    MAE[images], MSE[images], RMSE[images], PSNR[images], SSIM[images])
            if (WIND_UU or WIND_VV or WIND_UV) and images <= 10:
                fig, axes = create_prediction_image(images, test_data_skip_de_standardized, predictions_de_standardized, test_data_label_de_standardized, 0,
                                                    MAE[images], MSE[images], RMSE[images], RMSE_benchmark[images], PSNR[images], SSIM[images], SSIM_benchmark[images])
                
                directory = HP_RESULT_PATH + '/image' + f'/CNN_Results_{name}'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                axes[2].get_figure().savefig(f'{directory}/Ranking_{images+1}.png')
                plt.close()

            print(f'doing metric_pred')
            #Write the metrics in a csv file for every testing images generated by a model 
            metric_pred_imagecsv(directory, images, hyp, MAE, MSE, RMSE, PSNR, SSIM)

        #Rank the results from the best to the wrost.
        print(f'Ranking metric pred')
        rank_metric_pred_imagecsv(directory)

        #Write the average metrics in a csv file for a model on the testing data. 
        #Calculate the average for every metrics.
        mean_MAE[j-1] = np.mean(MAE)
        mean_MSE[j-1] = np.mean(MSE)
        mean_RMSE[j-1] = np.mean(RMSE)
        mean_PSNR[j-1] = np.mean(PSNR)
        mean_SSIM[j-1] = np.mean(SSIM)

        #Write the metrics in a csv file for every testing images generated by a model
        print(f'doing mean metric pred..')
        mean_metric_pred_imagecsv(HP_RESULT_PATH, j, hyp, mean_MAE, mean_MSE, mean_RMSE, mean_PSNR, mean_SSIM)

    #Rank the results from the best to the wrost.
    rank_mean_metric_pred_imagecsv(HP_RESULT_PATH)
