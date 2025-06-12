#!/bin/bash

#Definition of reed only variables (Constants)
"""
:Constant PATH_NN: Path that refer to the folder nammed Neural_Network. 
:Constant PATH_MAMBA: Path that refer to the mamba installation done localy. 
:Constant ENVNAME: Name of the conda environment that the user created for the project. 
:Constant SCRIPTNAME: Name of the script that will be run. 
:Constant INPUT_FILE: Path where the input data used for the training of the Neural Network are located. 
:Constant LABEL_FILE: Path where the label data used for the training of the Neural Network are located. 
:Constant TRAINING_SIZE: Number of input images used in the training.
:Constant VALID_SIZE: Number of input images used to evaluate the model during the training process.
:Constant TEST_SIZE: Number of input images used to evaluate the model after the training.
:Constant TEMPERATURE: Flag used when working with temperature model.
:Constant WIND: Flag used when working with Wind model.
:Constant WIND_UU: Flag used when working with Wind model and predicting only UU.
:Constant WIND_VV: Flag used when working with Wind model and predicting only VV.
:Constant WIND_UV: Flag used when working with Wind model and predicting only UV.

         ############### Constants for hyper-parameters shearch ###############

:Constant HYPER_PARAMETERS_SEARCHS: Flag = True if the user want do to a hyper-paremeters.
:Constant NUM_TRIALS: Number of trials that will be performed using different set of hyper-parameters.
:Constant EXECUTIONS_PER_TRIAL: The number of time the same hyper-parameters combinaison will be tested.
:Constant N_BEST: The N best models obtain in the hp search that will be used to train and test the model. 
:Constant EPOCH_TRY: Number of epochs used to train the Neural Network. 
:Constant UNITS_HP: Array of units(number of kernels) that can be used as an hyper-parameter to train the model with.
:Constant ACTIVATION_FUNCTION_HP: Array of activation function that can be used as an hyper-parameter to train the model with.
:Constant LEARNING_RATE_HP: Array of Learning rate that can be used as an hyper-parameter to train the model with.
:Constant KERNEL_SIZE_HP: Array of kernel size that can be used as an hyper-parameter to train the model with.
:Constant NUM_HIDDEN_LAYERS_HP: Array of hidden layers that can be used as an hyper-parameter to train the model with. 
:Constant BATCH_SIZE_MIN: Min number of epochs used to train the Neural Network.
:Constant BATCH_SIZE_MAX: Max number of epochs used to train the Neural Network.
:Constant BATCH_SIZE_STEP: Step between number of epochs used to train the Neural Network.
:Constant PATH_DP: Path that refer to the folder nammed Data_Processing/C_normalize. 
:Constant PATH_RESULTS: The path where the results are going to be stored at. 

         ############### Constants for non hyper-parameters shearch ###############

:Constant UNITS: Units(number of kernels) that can be used as an hyper-parameter to train the model with.
:Constant ACTIVATION_FUNCTION: Activation function that can be used as an hyper-parameter to train the model with.
:Constant LEARNING_RATE: Learning rate that can be used as an hyper-parameter to train the model with.
:Constant KERNEL_SIZE: kernel size that can be used as an hyper-parameter to train the model with.
:Constant NUM_HIDDEN_LAYERS: Hidden layers that can be used as an hyper-parameter to train the model with. 
:Constant BATCH_SIZE: Number of inputs that are processed in a single forward and backward pass during the training of the neural network (NOT USED WITH THE HYPER-PARAMETER SERACH).
:Constant EPOCH: Number of epochs used to train the Neural Network. 

         ############### Constants for Model evaluation ###############

:Constant NUM_TRAIN_REGENERATE: Number of training data that will be regenerated in the test set. 

:Constant VERBOSE: If the user want the code to print to help debug.
"""
readonly PATH_NN="/home/jfg000/ss5/SuperResolution/Neural_Network"
readonly PATH_MAMBA="/fs/ssm/eccc/cmd/cmds/apps/mamba/master/mamba_2023.11.23_all"
readonly ENVNAME="MyEnv"
readonly SCRIPTNAME="train_nearset_neighbour.py"
readonly INPUT_FILE='/home/jfg000/ss5/Data/input/domaine/gaspe/UU_VV_UV_TT_P0_H_CX_SD_WGE_data'  #/home/jfg000/ss5/JS_Kevin_CNN/Kevin_Entire_tar_files/site5_Unzip/aiData/qc/normalize/00Z
readonly LABEL_FILE='/home/jfg000/ss5/Data/label/domaine/gaspe/UU_data'  #/home/jfg000/ss5/JS_Kevin_CNN/Kevin_Entire_tar_files/site5_Unzip/aiData/qc/normalize/00Z
readonly TOPO_FILE='/home/jfg000/ss5/Data/topography/east_canada_sequential_train_domaine1/ME_MG_Z0_data' #/home/jfg000/ss5/Data/topography/gaspe/ME_MG_Z0_data
readonly TRAINING_SIZE=2194 #3230   #510
readonly VALID_SIZE=1212 #924 #925   #110
readonly TEST_SIZE=1212    #110
readonly TEMPERATURE=False #If false, erase --HYPER_PARAMETERS_SEARCHS on line 101
readonly WIND=False #If false, erase --WIND on line 101
readonly WIND_UU=True #If false, erase --WIND_UU on line 101
readonly WIND_VV=False #If false, erase --WIND_VV on line 101
readonly WIND_UV=False #If false, erase --WIND_UV on line 101
readonly data_already_spllit=True #If the training, validation and test set are already done
readonly PATH_DP='/home/jfg000/ss5/SuperResolution/Data_Processing/C_normalize/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_de_standardized_information' #'/home/jfg000/ss5/SuperResolution/Data_Processing/C_normalize/gaspe/UU_VV_TT_P0_H_CX_SD_ME_MG_Z0_de_standardized_information'
readonly PATH_RESULTS='/home/jfg000/ss5/CNN_results_article'

#Constants for Hyper-parameters search (If HYPER_PARAMETERS_SEARCHS=TRUE) Line 81 to 88 can be any value, they wont affect the program. 
readonly HYPER_PARAMETERS_SEARCHS=TRUE #If false, erase --HYPER_PARAMETERS_SEARCHS on line 101
readonly HP_NAME='nearest_neighboor_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000001_Final_search' #'Js_WindVV_DeepSD_l_ssim_m_mean_absolut_test1' #Always change this path.  #Js_Wind_FSRCNN_l_mean_absolut_m_ssim_test1
readonly NUM_TRIALS=1
readonly EXECUTIONS_PER_TRIAL=1
readonly N_BEST=1
readonly EPOCH_TRY=70
readonly UNITS_HP=(64)
readonly ACTIVATION_FUNCTION_HP=('relu')
readonly LEARNING_RATE_HP=(0.001)
readonly KERNEL_SIZE_HP=(3)
readonly NUM_HIDDEN_LAYERS_HP=(3)
readonly BATCH_SIZE_MIN=32
readonly BATCH_SIZE_MAX=32
readonly BATCH_SIZE_STEP=32

#Constant for Non-Hyper-parameters search (If HYPER_PARAMETERS_SEARCHS=FALSE) Line 65 to 79 can be any value, they wont affect the program. 
readonly UNITS=64
readonly ACTIVATION_FUNCTION='relu'
readonly LEARNING_RATE=0.0001
readonly KERNEL_SIZE=3
readonly NUM_HIDDEN_LAYERS=2
readonly BATCH_SIZE=8
readonly EPOCH=600

#Constant for model evaluation
readonly NUM_TRAIN_REGENERATE=300

readonly VERBOSE=TRUE #If false, erase --VERBOSE on line 101

#Initialization of the shell to use cuda
. ssmuse-sh -x $PATH_MAMBA
conda config --set changeps1 false

#Activate virtual environment 
. activate $ENVNAME

#Run script train_GAN.py
cd $PATH_NN
./$SCRIPTNAME --INPUT_FILE $INPUT_FILE --LABEL_FILE $LABEL_FILE --TOPO_FILE $TOPO_FILE --TRAINING_SIZE $TRAINING_SIZE --VALID_SIZE $VALID_SIZE --TEST_SIZE $TEST_SIZE --WIND_UU --PATH_DP $PATH_DP --PATH_RESULTS $PATH_RESULTS --HYPER_PARAMETERS_SEARCHS --HP_NAME $HP_NAME --NUM_TRIALS $NUM_TRIALS --EXECUTIONS_PER_TRIAL $EXECUTIONS_PER_TRIAL --N_BEST $N_BEST --EPOCH_TRY $EPOCH_TRY --UNITS_HP "${UNITS_HP[@]}" --ACTIVATION_FUNCTION_HP "${ACTIVATION_FUNCTION_HP[@]}" --LEARNING_RATE_HP "${LEARNING_RATE_HP[@]}" --KERNEL_SIZE_HP "${KERNEL_SIZE_HP[@]}" --NUM_HIDDEN_LAYERS_HP "${NUM_HIDDEN_LAYERS_HP[@]}" --BATCH_SIZE_MIN $BATCH_SIZE_MIN --BATCH_SIZE_MAX $BATCH_SIZE_MAX --BATCH_SIZE_STEP $BATCH_SIZE_STEP --UNITS $UNITS --ACTIVATION_FUNCTION $ACTIVATION_FUNCTION --LEARNING_RATE $LEARNING_RATE --KERNEL_SIZE $KERNEL_SIZE --NUM_HIDDEN_LAYERS $NUM_HIDDEN_LAYERS --BATCH_SIZE $BATCH_SIZE --EPOCH $EPOCH --NUM_TRAIN_REGENERATE $NUM_TRAIN_REGENERATE --VERBOSE --data_already_spllit
