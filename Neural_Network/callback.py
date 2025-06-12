#!/usr/bin/env python
""" 
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Developed by: Jean-Sébastien Giroux, Date: 2023-11-07
"""


#.**************************************************************************************************************************************.#
#                                                          Code description                                                              #
#.**************************************************************************************************************************************.#
"""
    TASK NAME - Training of a neural network (NN).

     STATUS - experimental

     DESCRIPTION - The script named callbacks incorporates four different callback functions utilized during the training of a neural network model with 
                   TensorFlow and Keras. Each callback serves a specific purpose in enhancing the training process, ensuring efficiency, and preventing overfitting. 
                   Below is a description of each callback in the script:
                    
                   A- Model Checkpoint Callback (best_model_weights):
                   Purpose: To save the model weights at each epoch where there is an improvement in validation SSIM Loss (or mean absolute error).
                   Details: The ModelCheckpoint callback monitors the val_SSIMLoss metric and saves the model weights to the specified path whenever an improvement is detected. It is configured to save only the best model weights (save_best_only=True) and to operate in 'min' mode, meaning improvements are defined as reductions in the monitored metric.
                   Usage: Ideal for scenarios where you need to retrieve the best-performing model after the training process.

                   B- Early Stopping Callback (earlystop):
                   Purpose: To stop training when the model stops improving on the validation SSIM Loss.
                   Details: The EarlyStopping callback monitors val_SSIMLoss and stops training if no improvement is seen for a specified number of epochs (patience). The patience value is adjusted based on whether post-hyperparameter search tuning is occurring (POST_HPS).
                   Usage: Useful to prevent overfitting and to save computational resources by stopping training early if the model isnt improving.

                   Reduce Learning Rate on Plateau Callback (reduceLROnPlateau):
                   Purpose: To reduce the learning rate when the models improvement on validation SSIM Loss plateaus.
                   Details: This callback reduces the learning rate by a factor of 0.10 if no improvement in val_SSIMLoss is observed for a certain number of epochs (patience). The patience and other parameters are also adjusted based on the training phase.
                   Usage: Helps in fine-tuning the model by making smaller changes to the weights when approaching a minimum in the loss function.

                   Terminate on NaN Callback (endWhenNaN):
                   Purpose: To terminate training if the loss becomes NaN (Not a Number).
                   Details: The TerminateOnNaN callback stops training immediately if the loss at any epoch is NaN, preventing wasted training cycles on a diverging model.
                   Usage: Acts as a fail-safe to prevent training from continuing when an unrecoverable numerical error occurs.
"""                     
#TODO create a class insted of a function for the callback list. 


#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import tensorflow as tf
import keras
keras.mixed_precision.set_global_policy('mixed_float16') # Use this line when you have access to GPU, will speed up the training by 3x.


#.**************************************************************************************************************************************.#
#                                                             Functions                                                                  #
#.**************************************************************************************************************************************.#
def callbacks(name, HYPER_PARAMETERS_SEARCHS, POST_HPS, VERBOSE, HP_RESULT_PATH):
    """
    List of callbacks that will be used in the training of the NN. The callbacks are used during the training of the 
    NN when certain conditions are meet to help the training get the best possible results. In this section ModelCheckpoint
    is used to save the model that gave the best performances in the training based on the validation loss. EarlyStopping is 
    used to stop the training if the validation_loss haven't improve during a certain epoch number, ReduceLROnPlateau is used
    to reduce the learning rate when there is a plateau of consecutive Validation loss that stay at the almost the same value
    during a certain number of epochs.
    INPUT 
    :Param name: Name for the current model being tested. Can be None value  
    :Param HYPER_PARAMETERS_SEARCHS: Flag = True if the user want do to a hyper-paremeters.
    :Param POST_HPS: Flag that indicate if the hyper-parameter search is done. 
    :Param VERBOSE: If the user want the code to print to help debug.
    :Param HP_RESULT_PATH: Path where will be located the results of the hyper-parameters tuning.
    OUTPUT
    :Param best_model_weights: The best model weight obtain during the validation loss of the NN for a specific epoch. 
    :Param earlystop: Stopping the training when the validation loss isn't improving for a certain number of epochs. 
    :Param reduceLROnPlateau: Change the value of the learning rate if the validation loss isn't improving for a certain number of epochs. 
    :Param endWhenNaN: End the training of the NN is the loss is Not a number (NaN).
    """
    if not HYPER_PARAMETERS_SEARCHS or POST_HPS:
        # Save best model weights https://keras.io/api/callbacks/model_checkpoint/ 
        best_model_weights = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{HP_RESULT_PATH}/model/Search{name}",
            monitor="val_mse", #   TODO   val_mean_squared_error val_SSIMLoss mse
            save_best_only=True,
            mode="min",
            save_weights_only=True, 
            save_freq="epoch",
            initial_value_threshold=None
        )

    # Early stop when the model isn't improving anymore https://keras.io/api/callbacks/early_stopping/
    if POST_HPS:
        # Based on Francois Chollet book, the patience need to be higher when training the models with the best hp sets
        patience = 15
    else:
        patience = 10
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_mse", #TODO val_mean_squared_error    val_mean_absolute_error    val_SSIMLoss
        min_delta=0.001, # Min_delta could be considered as an hyper-parameter...
        patience=patience,
        verbose=VERBOSE,
        mode="min"
    )

    # Reduce the learning rate on a plateau https://keras.io/api/callbacks/reduce_lr_on_plateau/ 
    if POST_HPS:
        # Based on Francois Chollet book, the patience need to be higher when training the models with the best hp sets
        patience = 10
    else:
        patience = 7
    reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mse", #TODO      val_mean_absolute_error val_mean_squared_error
        factor=0.30,
        patience=patience,
        verbose=VERBOSE,
        mode="min",
        min_delta=0.001,
        cooldown=0,
        min_lr=0
    )

    # End training if the lost became NAN https://keras.io/api/callbacks/terminate_on_nan/ 
    endWhenNaN = tf.keras.callbacks.TerminateOnNaN()

    if not HYPER_PARAMETERS_SEARCHS or POST_HPS:
        return best_model_weights, earlystop, reduceLROnPlateau, endWhenNaN
    else:
        return earlystop, reduceLROnPlateau, endWhenNaN
