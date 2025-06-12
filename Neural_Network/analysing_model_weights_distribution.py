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
    TASK NAME - #TODO

     STATUS - experimental

     DESCRIPTION - #TODO
"""    


#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

import model as mod


#.**************************************************************************************************************************************.#
#                                                             Variables                                                                  #
#.**************************************************************************************************************************************.#
"""
    #TODO
"""
NN_PATH = '/local/drive4/cmdd/afsyjsg/SuperResolution/Neural_Network'
HP_NAME = 'Js_WindUU_DeepSD_l_ssim_m_mean_absolut_test1'
HP_RESULT_PATH = f'{NN_PATH}/Best_model_UU/{HP_NAME}_HP_Search'
j = 4
name = name = f"model_{j}"

#.**************************************************************************************************************************************.#
#                                                                 Main                                                                   #
#.**************************************************************************************************************************************.#
if __name__ == '__main__':
    #Load the best model. 
    model = mod.DeepSD(units=2, activation='relu', lr=0.01, kernel_size=3, num_hidden_layer=3, WIND=False, regL2_conv01=0.01, regL2_conv02=0.01, drop_rate=0.2)
    #Load the best model weights.
    model.load_weights(f'{HP_RESULT_PATH}/model/Search{name}')

    #Create directory to save the images at.
    directory = HP_RESULT_PATH + '/analyse_model_weights_distribution' + f'/CNN_Results_{name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    #Visualize the weights distribution of the model. 
    for i, layer in enumerate(model.layers):
        print(f'Analysing layer {i} of the Neural Network')
        #Load the weights of the current layer. 
        weights = layer.get_weights()
        #To check if the layer has any trainable weights.
        if len(weights) > 0:
            plt.hist(weights[0].flatten(), bins=50)
            plt.title(f'Weight Distribution in Layer: {layer.name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.savefig(f'{directory}/layer_{layer.name}.png')
            plt.close()
