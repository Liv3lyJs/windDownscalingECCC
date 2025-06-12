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
    TASK NAME - Neural network model for weather and wind prediction. There are CNNs models, GANs models and many more. Most of them have 
                been found within the litterature. 

     STATUS - experimental

     DESCRIPTION - This script have all the architectures that are being used with the train_CNN and train_GAN script. This architectures
                   are based on the available litterature and in their title, there are a link to the original paper and for some, a link
                   to a video explication of the architecture and a review of the code and even a github link to the original code. You 
                   can add as many new architecture as you want. It's easier to take architecture from litterature than create one. 
"""                     


#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import tensorflow as tf
import keras
from keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, initializers
from tensorflow.keras.layers import Lambda, ZeroPadding2D, Resizing, Conv2D, Conv3D, Conv3DTranspose, UpSampling3D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add, PReLU, Input, LeakyReLU, UpSampling2D, Concatenate
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.losses import MeanSquaredError
keras.mixed_precision.set_global_policy('mixed_float16') # Use this line when you have access to GPU, will speed up the training by 3x.


#.**************************************************************************************************************************************.#
#                                                             Functions                                                                  #
#.**************************************************************************************************************************************.#
#TT DOWNSCALING CNN MODELS
def model_convolution(units, activation, lr, kernel_size, num_hidden_layer):
    """
    Architecture of a CNN. This architecture is based on the work of Kevin Gauthier and has ben improve with hyper-parameters tunning. This was
    a bench matk architecture to based futur result on. 
    This is the hyper-parameters that gave the best result based on the SSIM custom loss metric:
    Units= activation= lr= Kernel_size= num_hidden_layer= TODO complete the best hp section and the list of parameters used in the function. 
    INPUT
    :Param units: Units(number of kernels) that can be used as an hyper-parameter to train the model with.
    :Param activation: Activation function that can be used as an hyper-parameter to train the model with.
    :Param lr: Learning rate that can be used as an hyper-parameter to train the model with.
    :Param kernel_size: kernel size that can be used as an hyper-parameter to train the model with.
    :Param num_hidden_layer: Hidden layers that can be used as an hyper-parameter to train the model with. 
    OUTPUT
    :Param model: Object of the neural network model architecture.
    """
    # https://keras.io/api/layers/convolution_layers/convolution2d/ and Francois Chollet's p.202
    # Define the input shape that the model will receive. In this case, it's channel last [width, height, channel(s)]
    inputs = Input(shape=(129,129,15))

    # Define each layers of the NN.
    # Input layer
    x = Conv2D(filters=units, kernel_size=(kernel_size,kernel_size), strides=(1,1), padding="same", activation=activation, bias_initializer='zeros', use_bias=True, kernel_initializer="he_normal", name="Conv00")(inputs) # Kernel_initializer glorot_normal when working with tanh-sigmoid. when working with selu use lecun_normal
    for layer in range(num_hidden_layer): 
        # Hidden layer
        x = Conv2D(filters=units, kernel_size=(kernel_size,kernel_size), strides=(1,1), padding="same", bias_initializer='zeros', use_bias=True, kernel_initializer="he_normal", name=f"Conv0{layer+1}")(x) # Kernel_initializer glorot_normal when working with tanh-sigmoid lecun_normal
        x = BatchNormalization()(x) # Normalize the data between 0 and 1 to prevent the loss to be NaN or inf. 
        x = Activation(activation)(x) # Put the activation function after the batch normalization for better results. 
    # Output layer
    outputs = Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding="same", activation=activation, use_bias=True, bias_initializer='zeros', kernel_initializer="he_normal", name="ConvOut")(x) # Kernel_initializer glorot_normal when working with tanh-sigmoid lecun_normal

    # Define the input and the output of the model.
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Define the optimizer, the loss function and the metrics that will be analyse while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=["mean_absolute_error"], #"mean_absolute_error"
        metrics=[SSIMLoss] 
    )

    return model

def David_model(units, activation, lr, kernel_size, num_hidden_layer):
    """
    Architecture of a CNN. This architecture is based on the work of David Landry and has ben improve with hyper-parameters tunning. This was
    a second version of a bench matk architecture to based futur result on. 
    This is the hyper-parameters that gave the best result based on the SSIM custom loss metric:
    Units= activation= lr= Kernel_size= num_hidden_layer= TODO complete the best hp section and the list of parameters used in the function. 
    INPUT
    :Param units: Units(number of kernels) that can be used as an hyper-parameter to train the model with.
    :Param activation: Activation function that can be used as an hyper-parameter to train the model with.
    :Param lr: Learning rate that can be used as an hyper-parameter to train the model with.
    :Param kernel_size: kernel size that can be used as an hyper-parameter to train the model with.
    :Param num_hidden_layer: Hidden layers that can be used as an hyper-parameter to train the model with.
    OUTPUT
    :Param model: Object of the neural network model architecture.
    """
    # Define the input shape
    inputs = Input(shape=(129,129,15))

    # Define each layers of the Neural Network
    #Input layer
    input_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", bias_initializer='zeros', use_bias=True, kernel_initializer="he_normal", name="Conv00")(inputs)
    x = input_tensor
    # Hidden layer
    for i in range(3):
        x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x,input_tensor])
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
    # Output layer
    outputs = Conv2D(filters=1, kernel_size=(9,9), padding='same', activation="sigmoid")(x)

    # Define the input and the output of the model.
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Define the optimizer, the loss function and the metrics that will be analyse while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=["mean_absolute_error"], #"mean_absolute_error"
        metrics=[SSIMLoss] 
    )

    return model 

#SRGAN using from this article p. 5 https://arxiv.org/pdf/1609.04802.pdf 
#I will suggest the user of the program to watch this video, it will explain all the code : https://www.youtube.com/watch?v=nbRkLE2fiVI

#This architecture is used to calculate the loss function
#VGG-19:
def vgg_19(hr_shape):
    vgg = VGG19(weights=None, include_top=False, input_shape=hr_shape)
    weights_paths = '/home/jfg000/SuperResolution/Neural_Network/VGG/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg.load_weights(weights_paths)

    return keras.Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

#Generator: 
def res_block(ip):
    res_model = Conv2D(64, (3,3), padding='same')(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1,2])(res_model)

    res_model = Conv2D(64, (3,3), padding='same')(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = Add()([ip, res_model])

    return res_model

def upscale_block(ip):
    up_model = Conv2D(256, (3,3), padding='same')(ip)
    # up_model UpSampling2D(size=2)(up_model)         # Use this line of code when the input is 4x less then the label (If no normalization is done for the input data)
    up_model = PReLU(shared_axes=[1,2])(up_model)

    return up_model

def generator(lr_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding='same')(lr_ip)
    layers = PReLU(shared_axes=[1,2])(layers)
    temp = layers

    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3,3), padding='same')(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = Add()([layers, temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    outputs = Conv2D(3, (9,9), padding='same')(layers)

    return keras.Model(inputs=lr_ip, outputs=outputs)

# Discriminator:
def discriminator_block(ip, filters, strides=1, bn=True):
    disc_model = Conv2D(filters, (3,3), strides=strides, padding='same')(ip)
    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)
    
    disc_model = LeakyReLU(alpha=0.2)(disc_model)

    return disc_model

def discriminator(hr_ip):
    df = 64

    d1 = discriminator_block(hr_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)

    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)

    outputs = Dense(1, activation='sigmoid')(d10)

    return keras.Model(inputs=hr_ip, outputs=outputs) 

# Combine the generator, VGG-19 and the discriminator 
def DiscGen_comb(generator, discriminator, vgg, lr_ip, hr_ip):
    genrator_ima = generator(lr_ip)

    generator_features = vgg(genrator_ima)

    # Not possible to train the discriminator model while the generator model is being trained
    discriminator.trainable = False
    validity = discriminator(genrator_ima)
                                                      # Adversarial Loss, Content loss(Using VGG19)
    return keras.Model(inputs=[lr_ip, hr_ip], outputs=[validity, generator_features])




#UUVV DOWNSCALING CNN MODELS
def David_model_wind(units, activation, lr, kernel_size, num_hidden_layer, WIND):
    """
    Architecture of a CNN. This architecture is based on the work of David Landry and has ben improve with hyper-parameters tunning. This was
    a second version of a bench matk architecture to based futur result on. This was the best performing model that the team found when trying 
    to reproduce the work of Kevin and David. It will be the bench mark for the wind model prediction. 
    This is the hyper-parameters that gave the best result based on the SSIM custom loss metric:
    Units= activation= lr= Kernel_size= num_hidden_layer= TODO complete the best hp section and the list of parameters used in the function. 
    INPUT
    :Param units: Units(number of kernels) that can be used as an hyper-parameter to train the model with.
    :Param activation: Activation function that can be used as an hyper-parameter to train the model with.
    :Param lr: Learning rate that can be used as an hyper-parameter to train the model with.
    :Param kernel_size: kernel size that can be used as an hyper-parameter to train the model with.
    :Param num_hidden_layer: Hidden layers that can be used as an hyper-parameter to train the model with.
    OUTPUT
    :Param model: Object of the neural network model architecture.
    """
    # Define the input shape
    if WIND:
        inputs = Input(shape=(129,129,2))
    else:
        inputs = Input(shape=(129,129,5))

    # Define each layers of the Neural Network
    #Input layer
    input_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", bias_initializer='zeros', use_bias=True, kernel_initializer="he_normal", name="Conv00")(inputs)
    x = input_tensor
    # Hidden layer
    for i in range(3):
        x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x,input_tensor])
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
    # Output layer
    if WIND:
        outputs = Conv2D(filters=2, kernel_size=(9,9), padding='same', activation="sigmoid")(x)
    else:
        outputs = Conv2D(filters=1, kernel_size=(9,9), padding='same', activation="sigmoid")(x)

    # Define the input and the output of the model.
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Define the optimizer, the loss function and the metrics that will be analyse while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=["mean_absolute_error"], #SSIMLoss
        metrics=[SSIMLoss]  
    )

    return model 

def DeepSD(units, activation, lr, kernel_size, num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate):
    """
    This architecture has been taken from: A comparative study of convolutional neural network models for wind field downscaling 
    There are many pieces missing like the activation function that need to be taken, but it will be found by testing multiples 
    options. 
    """
    #Define the input shape 
    inputs = Input(shape=(128,128,10))

    #Define Neural Network layers
    x = Conv2D(64, (9,9), padding='same', use_bias=True, bias_initializer='zeros', name="Conv00")(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    #Add the dropout layer
    x = Dropout(drop_rate)(x)

    x = Conv2D(32, (1,1), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv01), name="Conv01")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(1, (5,5), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name="Conv02")(x)
    outputs = PReLU()(x)

    #Define the input-output of the model 
    model = keras.Model(inputs=inputs, outputs=outputs)

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=["mean_absolute_error"], #SSIMLoss
        metrics=[SSIMLoss]  
    )

    return model 

#Let the model learn the interpollation
def DeepSDv2(units, activation, lr, kernel_size, num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate):
    """
    This architecture has been taken from: A comparative study of convolutional neural network models for wind field downscaling 
    There are many pieces missing like the activation function that need to be taken, but it will be found by testing multiples 
    options. 
    """
    #Define the input shape 
    inputs = Input(shape=(32,32,7))

    #Define Neural Network layers
    x = Conv2D(64, (9,9), padding='same', use_bias=True, bias_initializer='zeros', name="Conv00")(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    #Add the dropout layer
    x = Dropout(drop_rate)(x)

    x = Conv2D(32, (1,1), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv01), name="Conv01")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    #Doing the topo input 
    inputs_topo = Input(shape=(64, 64, 3))
    #Define Neural Network layers
    topo = Conv2D(64, (9,9), padding='same', use_bias=True, bias_initializer='zeros', name="Conv03")(inputs_topo)
    topo = BatchNormalization()(topo)
    topo = PReLU()(topo)
    #Add the dropout layer
    topo = Dropout(drop_rate)(topo)
    topo = Conv2D(32, (1,1), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv01), name="Conv04")(topo)
    topo = BatchNormalization()(topo)
    topo = PReLU()(topo)

    #Put both together. 
    #Couche de up-sampling
    x = Concatenate()([UpSampling2D(size=(2,2))(x), topo])

    x = Conv2D(1, (5,5), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name="Conv02")(x)
    outputs = PReLU()(x)

    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo] , outputs=outputs)

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=["mean_absolute_error"], #SSIMLoss
        metrics=["mean_squared_error"] #SSIMLoss
    )

    return model 

def FSRCNN(units, activation, lr, kernel_size, num_hidden_layer, WIND):
    """
    This architecture has been taken from: A comparative study of convolutional neural network models for wind field downscaling 
    There are many pieces missing like the activation function that need to be taken, but it will be found by testing multiples 
    options. 
    """
    #Define the input shape
    if WIND:
        inputs = Input(shape=(129,129,2))
    else:
        inputs = Input(shape=(129,129,6))

    #Define Neural Network layers
    x = Conv2D(12, (1,1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(12, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(12, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(12, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(12, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    if WIND: 
        x = Conv2D(2, (1,1), padding='same')(x)
    else:
        x = Conv2D(1, (1,1), padding='same')(x)
    x = BatchNormalization()(x)
    outputs = x = PReLU()(x)

    #Define the input-output of the model 
    model = keras.Model(inputs=inputs, outputs=outputs)

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=["mean_absolute_error"], # SSIMLoss
        metrics=["root_mean_squared_error"]  
    )

    return model 

def deepRU_residual_block(x, filters, kernel_size=3, alpha=0.1, regL1=0.01, regL2=0.02, name='layer'):
    # Save the input tensor for the skip connection
    residual = x
    
    # First convolution layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l1_l2(l1=regL1, l2=regL2), name=f'{name}_00')(x)
    x = BatchNormalization(name=f'{name}_01')(x)
    x = LeakyReLU(alpha=alpha)(x)
    
    # Second convolution layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l1_l2(l1=regL1, l2=regL2), name=f'{name}_02')(x)
    x = BatchNormalization(name=f'{name}_03')(x)
    x = LeakyReLU(alpha=alpha)(x)
    
    # Third convolution layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l1_l2(l1=regL1, l2=regL2), name=f'{name}_04')(x)
    x = BatchNormalization(name=f'{name}_05')(x)
    
    # Add the input tensor (residual) to the output of the third convolution layer
    x = Add()([x, residual])
    x = LeakyReLU(alpha=alpha)(x)
    
    return x

def deepRU_v0(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
           num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha):
    """
    Input and topo are NOT concatenated together and go throw the U-Net architecture simultaniously: 
    - Pass the original low resolution input throw multiples convolution layers and do an upsample x2
    - Pass the topographie high resolution input throw multiples convolution layers. 

    In the U-Net:
    - The input is only the low resolution 
    - The depth of the network is 3 and is based on the paper that we found this architecture in, but do not have the same depth as the article.

    End of model:
    - Concatenate the input low resolution with the topographie high resolution. 
    """
    inputs = Input(shape=(32,32,10))
    inputs_topo = Input(shape=(125, 125, 3))
    inputs_topo = ZeroPadding2D(padding=((1, 2), (1, 2)))(inputs_topo)
    #Skip connection with only wind variable (Variable that we are trying to predict)
    wind = Lambda(lambda x: x[...,0:1])(inputs)
    wind = UpSampling2D(size=(2, 2))(wind)
    # print(f'The wind shape is: {wind.shape}')
    #Take all other spatio-temporal variables except wind variables
    spatio_temp = Lambda(lambda x: x[...,1:])(inputs)
    # print(f'Spatio-temporal variable shape is: {spatio_temp.shape}')

    #Pre-encoder predictors
    x = Conv2D(64, (5, 5), padding='same')(spatio_temp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    # print(f'Spatio-temporal variable shape is: {x.shape}')

    #Pre-encoder predictors
    topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros')(inputs_topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(32, (3 ,5), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros')(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros')(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)
    # print(f'The topo shape is {topo.shape}')

    #Concatenate the input 
    inp = Concatenate()([x, topo])
    inp = UpSampling2D(size=(2, 2))(inp)
    #print(f'The inp shape is {inp.shape}')

    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(2,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(inp)
    e1 = Dropout(drop_rate)(e1)
    e1 = BatchNormalization()(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0)
    #print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(160, (3, 3), strides=(2,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e1)
    e2 = Dropout(drop_rate)(e2)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    e2 = deepRU_residual_block(e2, 160, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    #print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(192, (3, 3), strides=(2,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e2)
    e3 = Dropout(drop_rate)(e3)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    e3 = deepRU_residual_block(e3, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    #print(f'The e3 shape is: {e3.shape}')

    #Bottleneck layer
    x = Conv2D(224, (3,3), strides=(2,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e3)
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 224, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    #print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d3 = UpSampling2D()(x)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(192, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d3)
    d3 = Dropout(drop_rate)(d3)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    d3 = deepRU_residual_block(d3, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    #print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D()(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(160, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d2)
    d2 = Dropout(drop_rate)(d2)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    d2 = deepRU_residual_block(d2, 160, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    #print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D()(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d1)
    d1 = Dropout(drop_rate)(d1)
    d1 = BatchNormalization()(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    #print(f'The d1 shape is: {d1.shape}')
    d1 = UpSampling2D()(d1)
    #print(f'The d1 before skip connection layer is: {d1.shape}')

    x = Concatenate()([d1, wind])
    x = Conv2D(64, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02))(x)
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 64, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    #print(f'Shape before output is: {x.shape}')

    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02))(x)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo] , outputs=outputs)

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=["mean_squared_error"], #SSIMLoss mean_squared_error  
        metrics=[SSIMLoss, psnr] #SSIMLoss mean_absolute_error
    )

    #Print the model complexity
    #model.summary()

    return model

def deepRU_article_non_interpolation(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                                     num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha):

    inputs = Input(shape=(32,32,34))
    skip_train = Input(shape=(64, 64, 1))
    inputs_topo = Input(shape=(125, 125, 3))
    padded_inputs_topo = ZeroPadding2D(padding=((1, 2), (1, 2)))(inputs_topo)
    print(f'the inputs_topo shape is {padded_inputs_topo}')

    ###### STEP1 ######
    #Multiples levels UV
    wind_uv = Lambda(lambda x: x[...,0:4])(inputs)
    print(f'The shape of wind_u is: {wind_uv.shape}')
    gradient_uv = Lambda(lambda x: x[...,24:25])(inputs)
    print(f'The shape of gradient_uv is: {gradient_uv.shape}')
    #Convolution sequence
    wind_uv = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_uv00")(wind_uv)
    wind_uv = BatchNormalization()(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
    wind_uv = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_uv01")(wind_uv)
    wind_uv = BatchNormalization()(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
    #Concatenation with gradient
    wind_uv = Concatenate()([wind_uv, gradient_uv])
    #Convolution sequence
    wind_uv = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_uv02")(wind_uv)
    wind_uv = BatchNormalization()(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
    print(f'The shape of wind_uv at the end of the first step is: {wind_uv.shape}')

    #Multiples levels UU
    wind_u = Lambda(lambda x: x[...,6:10])(inputs)
    print(f'The shape of wind_v is: {wind_u.shape}')
    gradient_u = Lambda(lambda x: x[...,25:26])(inputs)
    print(f'The shape of gradient_u is: {gradient_u.shape}')
    #Convolution sequence
    wind_u = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_u00")(wind_u)
    wind_u = BatchNormalization()(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)
    wind_u = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_u01")(wind_u)
    wind_u = BatchNormalization()(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)
    #Concatenation with gradient
    wind_u = Concatenate()([wind_u, gradient_u])
    #Convolution sequence
    wind_u = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_u02")(wind_u)
    wind_u = BatchNormalization()(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)
    print(f'The shape of wind_u at the end of the first step is: {wind_u.shape}')

    #Multiples levels VV
    wind_v = Lambda(lambda x: x[...,12:16])(inputs)
    print(f'The shape of wind_t is: {wind_v.shape}')
    gradient_v = Lambda(lambda x: x[...,26:27])(inputs)
    print(f'The shape of gradient_v is: {gradient_v.shape}')
    #Convolution sequence
    wind_v = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_v00")(wind_v)
    wind_v = BatchNormalization()(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)
    wind_v = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_v01")(wind_v)
    wind_v = BatchNormalization()(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)
    #Concatenation with gradient
    wind_v = Concatenate()([wind_v, gradient_v])
    #Convolution sequence
    wind_v = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_v02")(wind_v)
    wind_v = BatchNormalization()(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)
    print(f'The shape of wind_v at the end of the first step is: {wind_v.shape}')

    #Multiples levels TT
    wind_t = Lambda(lambda x: x[...,18:22])(inputs)
    print(f'The shape of wind_u is: {wind_t.shape}')
    gradient_t = Lambda(lambda x: x[...,27:28])(inputs)
    print(f'The shape of gradient_t is: {gradient_t.shape}')
    #Convolution sequence
    wind_t = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_t00")(wind_t)
    wind_t = BatchNormalization()(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)
    wind_t = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_t01")(wind_t)
    wind_t = BatchNormalization()(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)
    #Concatenation with gradient
    wind_t = Concatenate()([wind_t, gradient_t])
    #Convolution sequence
    wind_t = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_t02")(wind_t)
    wind_t = BatchNormalization()(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)
    print(f'The shape of wind_t at the end of the first step is: {wind_t.shape}')

    ###### STEP2 ######
    #Concatenate uv, u, v, t and gradient together
    wind = Concatenate()([wind_uv, wind_u, wind_v, wind_t])
    wind = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_combine_levels")(wind)
    print(f'At the end of step 2 the shape of wind is: {wind.shape}')

    ###### STEP3 ######
    #Putting topographie the same resolution of spatio-temporal data
    topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_geo00")(padded_inputs_topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_geo01")(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', name="pre_UNet_geo02")(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    ###### STEP4 ######
    #Concatenate all input data together
    other_input = Lambda(lambda x: x[...,28:])(inputs)
    print(f'The shape of the other input is : {other_input.shape}')
    input_concatenate = Concatenate()([wind, other_input, topo])
    print(f'The shape of the input before the u-net is: {input_concatenate.shape}')

    ###### STEP4 ######
    #U-net
    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name="UNet_e1")(input_concatenate)
    e1 = Dropout(drop_rate)(e1)
    e1 = BatchNormalization()(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    #e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0)
    print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e1)
    e2 = Dropout(drop_rate)(e2)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    #e2 = deepRU_residual_block(e2, 160, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e2)
    e3 = Dropout(drop_rate)(e3)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    #e3 = deepRU_residual_block(e3, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The e3 shape is: {e3.shape}')

    e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e3)
    e4 = Dropout(drop_rate)(e4)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    #e4 = deepRU_residual_block(e4, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The e4 shape is: {e4.shape}')

    e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e4)
    e5 = Dropout(drop_rate)(e5)
    e5 = BatchNormalization()(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    #e5 = deepRU_residual_block(e5, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The e5 shape is: {e5.shape}')

    #Bottleneck layer
    x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e5)
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    #x = deepRU_residual_block(x, 224, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d5 = UpSampling2D(size=(2, 1))(x)
    d5 = Concatenate()([d5, e5])
    d5 = Conv2D(384, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d5)
    d5 = Dropout(drop_rate)(d5)
    d5 = BatchNormalization()(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    #d5 = deepRU_residual_block(d5, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The d5 shape is: {d5.shape}')

    d4 = UpSampling2D(size=(1, 2))(d5)
    d4 = Concatenate()([d4, e4])
    d4 = Conv2D(320, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d4)
    d4 = Dropout(drop_rate)(d4)
    d4 = BatchNormalization()(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    #d4 = deepRU_residual_block(d4, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The d4 shape is: {d4.shape}')

    d3 = UpSampling2D(size=(2, 1))(d4)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d3)
    d3 = Dropout(drop_rate)(d3)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    #d3 = deepRU_residual_block(d3, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D(size=(1, 2))(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(192, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d2)
    d2 = Dropout(drop_rate)(d2)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    #d2 = deepRU_residual_block(d2, 160, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D(size=(2, 1))(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d1)
    d1 = Dropout(drop_rate)(d1)
    d1 = BatchNormalization()(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    #d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    print(f'The d1 shape is: {d1.shape}')

    #Gradually putting the network the same shape of the skip connection. 
    d0 = UpSampling2D(size=(1, 2))(d1)
    d0 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d0)
    d0 = Dropout(drop_rate)(d0)
    d0 = BatchNormalization()(d0)
    d0 = LeakyReLU(alpha=alpha)(d0)
    d0 = UpSampling2D(size=(2, 2))(d0)
    d0 = Conv2D(1, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d0)

    #Combining with an addition the skip connection with the network
    result = Add()([skip_train, d0])
    result = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(result)
    result = Dropout(drop_rate)(result)
    result = BatchNormalization()(result)
    result = LeakyReLU(alpha=alpha)(result)


    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02))(result)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=["mean_squared_error"], #SSIMLoss mean_squared_error  
        metrics=[SSIMLoss, psnr] #SSIMLoss mean_absolute_error
    )

    #Print the model complexity
    #model.summary()

    return model

# def deepRU_article_non_interpolation_combine_loss(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
#                                      num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):

#     # inputs = Input(shape=(32,32,34))
#     # skip_train = Input(shape=(64, 64, 1))
#     # inputs_topo = Input(shape=(125, 125, 3))
#     # padded_inputs_topo = ZeroPadding2D(padding=((1, 2), (1, 2)))(inputs_topo)
#     # print(f'the inputs_topo shape is {padded_inputs_topo}')

#     inputs = Input(shape=(16,16,26))
#     skip_train = Input(shape=(48, 48, 1))
#     inputs_topo = Input(shape=(64, 64, 3))
#     # padded_inputs_topo = ZeroPadding2D(padding=((1, 2), (1, 2)))(inputs_topo)
#     # print(f'the inputs_topo shape is {padded_inputs_topo}')

#     ###### STEP1 ######
#     #Multiples levels UV
#     wind_uv = Lambda(lambda x: x[...,0:4])(inputs)
#     # print(f'The shape of wind_u is: {wind_uv.shape}')
#     gradient_uv = Lambda(lambda x: x[...,16:17])(inputs)
#     # print(f'The shape of gradient_uv is: {gradient_uv.shape}')
#     #Convolution sequence
#     wind_uv = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv00")(wind_uv)
#     wind_uv = BatchNormalization(name="pre_UNet_uv01")(wind_uv)
#     wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
#     wind_uv = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv02")(wind_uv)
#     wind_uv = BatchNormalization(name="pre_UNet_uv03")(wind_uv)
#     wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
#     #Concatenation with gradient
#     wind_uv = Concatenate()([wind_uv, gradient_uv])
#     #Convolution sequence
#     wind_uv = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv04")(wind_uv)
#     wind_uv = BatchNormalization(name="pre_UNet_uv05")(wind_uv)
#     wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
#     # print(f'The shape of wind_uv at the end of the first step is: {wind_uv.shape}')

#     #Multiples levels UU
#     wind_u = Lambda(lambda x: x[...,4:8])(inputs)
#     # print(f'The shape of wind_v is: {wind_u.shape}')
#     gradient_u = Lambda(lambda x: x[...,17:18])(inputs)
#     # print(f'The shape of gradient_u is: {gradient_u.shape}')
#     #Convolution sequence
#     wind_u = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u00")(wind_u)
#     wind_u = BatchNormalization(name="pre_UNet_u01")(wind_u)
#     wind_u = LeakyReLU(alpha=alpha)(wind_u)
#     wind_u = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u02")(wind_u)
#     wind_u = BatchNormalization(name="pre_UNet_u03")(wind_u)
#     wind_u = LeakyReLU(alpha=alpha)(wind_u)
#     #Concatenation with gradient
#     wind_u = Concatenate()([wind_u, gradient_u])
#     #Convolution sequence
#     wind_u = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u04")(wind_u)
#     wind_u = BatchNormalization(name="pre_UNet_u05")(wind_u)
#     wind_u = LeakyReLU(alpha=alpha)(wind_u)
#     # print(f'The shape of wind_u at the end of the first step is: {wind_u.shape}')

#     #Multiples levels VV
#     wind_v = Lambda(lambda x: x[...,8:12])(inputs)
#     # print(f'The shape of wind_t is: {wind_v.shape}')
#     gradient_v = Lambda(lambda x: x[...,18:19])(inputs)
#     # print(f'The shape of gradient_v is: {gradient_v.shape}')
#     #Convolution sequence
#     wind_v = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v00")(wind_v)
#     wind_v = BatchNormalization(name="pre_UNet_v01")(wind_v)
#     wind_v = LeakyReLU(alpha=alpha)(wind_v)
#     wind_v = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v02")(wind_v)
#     wind_v = BatchNormalization(name="pre_UNet_v03")(wind_v)
#     wind_v = LeakyReLU(alpha=alpha)(wind_v)
#     #Concatenation with gradient
#     wind_v = Concatenate()([wind_v, gradient_v])
#     #Convolution sequence
#     wind_v = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v04")(wind_v)
#     wind_v = BatchNormalization(name="pre_UNet_v05")(wind_v)
#     wind_v = LeakyReLU(alpha=alpha)(wind_v)
#     # print(f'The shape of wind_v at the end of the first step is: {wind_v.shape}')

#     #Multiples levels TT
#     wind_t = Lambda(lambda x: x[...,12:16])(inputs)
#     # print(f'The shape of wind_u is: {wind_t.shape}')
#     gradient_t = Lambda(lambda x: x[...,19:20])(inputs)
#     # print(f'The shape of gradient_t is: {gradient_t.shape}')
#     #Convolution sequence
#     wind_t = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t00")(wind_t)
#     wind_t = BatchNormalization(name="pre_UNet_t01")(wind_t)
#     wind_t = LeakyReLU(alpha=alpha)(wind_t)
#     wind_t = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t02")(wind_t)
#     wind_t = BatchNormalization(name="pre_UNet_t03")(wind_t)
#     wind_t = LeakyReLU(alpha=alpha)(wind_t)
#     #Concatenation with gradient
#     wind_t = Concatenate()([wind_t, gradient_t])
#     #Convolution sequence
#     wind_t = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t04")(wind_t)
#     wind_t = BatchNormalization(name="pre_UNet_t05")(wind_t)
#     wind_t = LeakyReLU(alpha=alpha)(wind_t)
#     # print(f'The shape of wind_t at the end of the first step is: {wind_t.shape}')

#     ###### STEP2 ######
#     #Concatenate uv, u, v, t and gradient together
#     wind = Concatenate()([wind_uv, wind_u, wind_v, wind_t])
#     wind = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_combine_levels")(wind)
#     # print(f'At the end of step 2 the shape of wind is: {wind.shape}')

#     ###### STEP3 ######
#     #Putting topographie the same resolution of spatio-temporal data
#     topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo00")(inputs_topo)
#     topo = BatchNormalization(name="pre_UNet_geo01")(topo)
#     topo = LeakyReLU(alpha=alpha)(topo)

#     topo = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo02")(topo)
#     topo = BatchNormalization(name="pre_UNet_geo03")(topo)
#     topo = LeakyReLU(alpha=alpha)(topo)

#     topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo04")(topo)
#     topo = BatchNormalization(name="pre_UNet_geo05")(topo)
#     topo = LeakyReLU(alpha=alpha)(topo)

#     ###### STEP4 ######
#     #Concatenate all input data together
#     other_input = Lambda(lambda x: x[...,20:])(inputs)
#     # print(f'The shape of the other input is : {other_input.shape}')
#     input_concatenate = Concatenate()([wind, other_input, topo])
#     # print(f'The shape of the input before the u-net is: {input_concatenate.shape}')

#     ###### STEP4 ######
#     #U-net
#     #Encoder
#     e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e1_00')(input_concatenate)
#     e1 = BatchNormalization(name='U_Net_encoder_e1_01')(e1)
#     e1 = LeakyReLU(alpha=alpha)(e1)
#     e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0, name='U_Net_encoder_e1_02')
#     e1 = Dropout(drop_rate)(e1)
#     #print(f'The e1 shape is: {e1.shape}')

#     e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e2_00')(e1)
#     e2 = BatchNormalization(name='U_Net_encoder_e2_01')(e2)
#     e2 = LeakyReLU(alpha=alpha)(e2)
#     e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e2_02')
#     e2 = Dropout(drop_rate)(e2)
#     #print(f'The e2 shape is: {e2.shape}')

#     e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e3_00')(e2)
#     e3 = BatchNormalization(name='U_Net_encoder_e3_01')(e3)
#     e3 = LeakyReLU(alpha=alpha)(e3)
#     e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e3_02')
#     e3 = Dropout(drop_rate)(e3)
#     #print(f'The e3 shape is: {e3.shape}')

#     e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e4_00')(e3)
#     e4 = BatchNormalization(name='U_Net_encoder_e4_01')(e4)
#     e4 = LeakyReLU(alpha=alpha)(e4)
#     e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e4_02')
#     e4 = Dropout(drop_rate)(e4)
#     #print(f'The e4 shape is: {e4.shape}')

#     e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e5_00')(e4)
#     e5 = BatchNormalization(name='U_Net_encoder_e5_01')(e5)
#     e5 = LeakyReLU(alpha=alpha)(e5)
#     e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e5_02')
#     e5 = Dropout(drop_rate)(e5)
#     #print(f'The e5 shape is: {e5.shape}')

#     #Bottleneck layer
#     x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_bottleneck_00')(e5)
#     x = BatchNormalization(name='U_Net_bottleneck_01')(x)
#     x = LeakyReLU(alpha=alpha)(x)
#     x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_bottleneck_02')
#     x = Dropout(drop_rate)(x)
#     #print(f'The bottleneck shape is: {x.shape}')

#     #Decoder
#     d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
#     d5 = Concatenate()([d5, e5])
#     d5 = Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e5_trainable_00')(d5)
#     d5 = BatchNormalization(name='U_Net_decoder_e5_trainable_01')(d5)
#     d5 = LeakyReLU(alpha=alpha)(d5)
#     d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e5_trainable_02')
#     d5 = Dropout(drop_rate)(d5)
#     #print(f'The d5 shape is: {d5.shape}')

#     d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
#     d4 = Concatenate()([d4, e4])
#     d4 = Conv2D(320, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e4_trainable_00')(d4)
#     d4 = BatchNormalization(name='U_Net_decoder_e4_trainable_01')(d4)
#     d4 = LeakyReLU(alpha=alpha)(d4)
#     d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e4_trainable_02')
#     d4 = Dropout(drop_rate)(d4)
#     #print(f'The d4 shape is: {d4.shape}')

#     d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
#     d3 = Concatenate()([d3, e3])
#     d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e3_trainable_00')(d3)
#     d3 = BatchNormalization(name='U_Net_decoder_e3_trainable_01')(d3)
#     d3 = LeakyReLU(alpha=alpha)(d3)
#     d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e3_trainable_02')
#     d3 = Dropout(drop_rate)(d3)
#     #print(f'The d3 shape is: {d3.shape}')
    
#     d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
#     d2 = Concatenate()([d2, e2])
#     d2 = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e2_trainable_00')(d2)
#     d2 = BatchNormalization(name='U_Net_decoder_e2_trainable_01')(d2)
#     d2 = LeakyReLU(alpha=alpha)(d2)
#     d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e2_trainable_02')
#     d2 = Dropout(drop_rate)(d2)
#     #print(f'The d2 shape is: {d2.shape}')
    
#     d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
#     d1 = Concatenate()([d1, e1])
#     d1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e1_trainable_00')(d1)
#     d1 = BatchNormalization(name='U_Net_decoder_e1_trainable_01')(d1)
#     d1 = LeakyReLU(alpha=alpha)(d1)
#     d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e1_trainable_02')
#     d1 = Dropout(drop_rate)(d1)
#     #print(f'The d1 shape is: {d1.shape}')

#     #Gradually putting the network the same shape of the skip connection. 
#     d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
#     d0 = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_00')(d0)
#     d0 = BatchNormalization(name='post_UNet_d0_trainable_01')(d0)
#     d0 = LeakyReLU(alpha=alpha)(d0)
#     d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
#     d0 = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_02')(d0)

#     #Combining with an addition the skip connection with the network
#     result = Add()([skip_train, d0])
#     result = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_results_trainable_00')(result)
#     result = BatchNormalization(name='post_UNet_results_trainable_01')(result)
#     result = LeakyReLU(alpha=alpha)(result)


#     outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name='post_UNet_output_trainable')(result)
    
#     #Define the input-output of the model 
#     model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
#     margin = 4

#     #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
#     opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=5.0)
#     model.compile(
#         optimizer=opt,
#         loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function central_region_combined_loss(margin, loss_weights)
#         metrics=[central_region_ssim(margin),
#                  central_region_mse(margin) 
#         ]
#     )

#     # #Print the model complexity
#     # print(f'This is the model summary for the model code.')
#     # model.summary()

#     return model

def deepRU_article_non_interpolation_combine_loss(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                                     num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):
    inputs = Input(shape=(16,16,26))
    skip_train = Input(shape=(48, 48, 1))
    inputs_topo = Input(shape=(64, 64, 3))

    wind_uv = Lambda(lambda x: x[...,0:4])(inputs)
    gradient_uv = Lambda(lambda x: x[...,16:17])(inputs)
    wind_u = Lambda(lambda x: x[...,4:8])(inputs)
    gradient_u = Lambda(lambda x: x[...,17:18])(inputs)
    wind_v = Lambda(lambda x: x[...,8:12])(inputs)
    gradient_v = Lambda(lambda x: x[...,18:19])(inputs)
    wind_t = Lambda(lambda x: x[...,12:16])(inputs)
    gradient_t = Lambda(lambda x: x[...,19:20])(inputs)
    other_input = Lambda(lambda x: x[...,20:])(inputs)

    ###### STEP1 ######
    #Other variables
    #Convolution sequence
    other_input = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o00")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o01")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)
    other_input = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o02")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o03")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)

    #Multiples levels UU
    #Convolution sequence
    wind_u = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u00")(wind_u)
    wind_u = BatchNormalization(name="pre_UNet_u01")(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)
    wind_u = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u02")(wind_u)
    wind_u = BatchNormalization(name="pre_UNet_u03")(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)

    #Multiples levels VV
    #Convolution sequence
    wind_v = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v00")(wind_v)
    wind_v = BatchNormalization(name="pre_UNet_v01")(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)
    wind_v = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v02")(wind_v)
    wind_v = BatchNormalization(name="pre_UNet_v03")(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)

    #Multiples levels TT
    #Convolution sequence
    wind_t = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t00")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t01")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)
    wind_t = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t02")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t03")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)

    #Multiples levels UV
    #Convolution sequence
    wind_uv = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv00")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv01")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
    wind_uv = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv02")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv03")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)

    #Robinson's number
    robinson = Concatenate()([gradient_u, gradient_v, gradient_t])
    robinson = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_rob00")(robinson)
    robinson = BatchNormalization(name="pre_UNet_rob01")(robinson)
    robinson = LeakyReLU(alpha=alpha)(robinson)
    robinson = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_rob02")(robinson)
    robinson = BatchNormalization(name="pre_UNet_rob03")(robinson)
    robinson = LeakyReLU(alpha=alpha)(robinson)

    ###### STEP2 ######
    #Combine spatio temporal variables with multiples levels variables and robinson 
    spatio_temp = Concatenate()([other_input, wind_u, wind_v, wind_t, wind_uv, robinson])
    spatio_temp = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio00")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio01")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)
    spatio_temp = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio02")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio03")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)

    ###### STEP3 ######
    #Putting topographie the same resolution of spatio-temporal data
    topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo00")(inputs_topo)
    topo = BatchNormalization(name="pre_UNet_geo01")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo02")(topo)
    topo = BatchNormalization(name="pre_UNet_geo03")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo04")(topo)
    topo = BatchNormalization(name="pre_UNet_geo05")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    ###### STEP4 ######
    #Concatenate the spatio temporal value with topo 
    input_concatenate = Concatenate()([spatio_temp, topo])

    ###### STEP5 ######
    #U-net
    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e1_00')(input_concatenate)
    e1 = BatchNormalization(name='U_Net_encoder_e1_01')(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0, name='U_Net_encoder_e1_02')
    e1 = Dropout(drop_rate)(e1)
    #print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e2_00')(e1)
    e2 = BatchNormalization(name='U_Net_encoder_e2_01')(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e2_02')
    e2 = Dropout(drop_rate)(e2)
    #print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e3_00')(e2)
    e3 = BatchNormalization(name='U_Net_encoder_e3_01')(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e3_02')
    e3 = Dropout(drop_rate)(e3)
    #print(f'The e3 shape is: {e3.shape}')

    e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e4_00')(e3)
    e4 = BatchNormalization(name='U_Net_encoder_e4_01')(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e4_02')
    e4 = Dropout(drop_rate)(e4)
    #print(f'The e4 shape is: {e4.shape}')

    e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e5_00')(e4)
    e5 = BatchNormalization(name='U_Net_encoder_e5_01')(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e5_02')
    e5 = Dropout(drop_rate)(e5)
    #print(f'The e5 shape is: {e5.shape}')

    #Bottleneck layer
    x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_bottleneck_00')(e5)
    x = BatchNormalization(name='U_Net_bottleneck_01')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_bottleneck_02')
    x = Dropout(drop_rate)(x)
    #print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
    d5 = Concatenate()([d5, e5])
    d5 = Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e5_trainable_00')(d5)
    d5 = BatchNormalization(name='U_Net_decoder_e5_trainable_01')(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e5_trainable_02')
    d5 = Dropout(drop_rate)(d5)
    #print(f'The d5 shape is: {d5.shape}')

    d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
    d4 = Concatenate()([d4, e4])
    d4 = Conv2D(320, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e4_trainable_00')(d4)
    d4 = BatchNormalization(name='U_Net_decoder_e4_trainable_01')(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e4_trainable_02')
    d4 = Dropout(drop_rate)(d4)
    #print(f'The d4 shape is: {d4.shape}')

    d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e3_trainable_00')(d3)
    d3 = BatchNormalization(name='U_Net_decoder_e3_trainable_01')(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e3_trainable_02')
    d3 = Dropout(drop_rate)(d3)
    #print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e2_trainable_00')(d2)
    d2 = BatchNormalization(name='U_Net_decoder_e2_trainable_01')(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e2_trainable_02')
    d2 = Dropout(drop_rate)(d2)
    #print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e1_trainable_00')(d1)
    d1 = BatchNormalization(name='U_Net_decoder_e1_trainable_01')(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e1_trainable_02')
    d1 = Dropout(drop_rate)(d1)
    #print(f'The d1 shape is: {d1.shape}')

    #Gradually putting the network the same shape of the skip connection. 
    d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
    d0 = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_00')(d0)
    d0 = BatchNormalization(name='post_UNet_d0_trainable_01')(d0)
    d0 = LeakyReLU(alpha=alpha)(d0)
    d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
    d0 = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_02')(d0)

    #Combining with an addition the skip connection with the network
    result = Add()([skip_train, d0])
    result = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_results_trainable_00')(result)
    result = BatchNormalization(name='post_UNet_results_trainable_01')(result)
    result = LeakyReLU(alpha=alpha)(result)


    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name='post_UNet_output_trainable')(result)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
    margin = 4

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=3.0)
    model.compile(
        optimizer=opt,
        loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function central_region_combined_loss(margin, loss_weights)
        metrics=[central_region_ssim(margin),
                 central_region_mse(margin) 
        ]
    )

    # #Print the model complexity
    # print(f'This is the model summary for the model code.')
    # model.summary()

    return model

def deepRU_article_non_interpolation_combine_loss_onlyUV(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                                     num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):
    inputs = Input(shape=(16,16,26))
    skip_train = Input(shape=(48, 48, 1))
    inputs_topo = Input(shape=(64, 64, 3))

    wind_uv = Lambda(lambda x: x[...,0:4])(inputs)

    #Multiples levels UV
    #Convolution sequence
    wind_uv = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv00")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv01")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
    wind_uv = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv02")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv03")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)

    ###### STEP5 ######
    #U-net
    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e1_00')(wind_uv)
    e1 = BatchNormalization(name='U_Net_encoder_e1_01')(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0, name='U_Net_encoder_e1_02')
    e1 = Dropout(drop_rate)(e1)
    #print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e2_00')(e1)
    e2 = BatchNormalization(name='U_Net_encoder_e2_01')(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e2_02')
    e2 = Dropout(drop_rate)(e2)
    #print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e3_00')(e2)
    e3 = BatchNormalization(name='U_Net_encoder_e3_01')(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e3_02')
    e3 = Dropout(drop_rate)(e3)
    #print(f'The e3 shape is: {e3.shape}')

    e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e4_00')(e3)
    e4 = BatchNormalization(name='U_Net_encoder_e4_01')(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e4_02')
    e4 = Dropout(drop_rate)(e4)
    #print(f'The e4 shape is: {e4.shape}')

    e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e5_00')(e4)
    e5 = BatchNormalization(name='U_Net_encoder_e5_01')(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e5_02')
    e5 = Dropout(drop_rate)(e5)
    #print(f'The e5 shape is: {e5.shape}')

    #Bottleneck layer
    x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_bottleneck_00')(e5)
    x = BatchNormalization(name='U_Net_bottleneck_01')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_bottleneck_02')
    x = Dropout(drop_rate)(x)
    #print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
    d5 = Concatenate()([d5, e5])
    d5 = Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e5_trainable_00')(d5)
    d5 = BatchNormalization(name='U_Net_decoder_e5_trainable_01')(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e5_trainable_02')
    d5 = Dropout(drop_rate)(d5)
    #print(f'The d5 shape is: {d5.shape}')

    d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
    d4 = Concatenate()([d4, e4])
    d4 = Conv2D(320, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e4_trainable_00')(d4)
    d4 = BatchNormalization(name='U_Net_decoder_e4_trainable_01')(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e4_trainable_02')
    d4 = Dropout(drop_rate)(d4)
    #print(f'The d4 shape is: {d4.shape}')

    d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e3_trainable_00')(d3)
    d3 = BatchNormalization(name='U_Net_decoder_e3_trainable_01')(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e3_trainable_02')
    d3 = Dropout(drop_rate)(d3)
    #print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e2_trainable_00')(d2)
    d2 = BatchNormalization(name='U_Net_decoder_e2_trainable_01')(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e2_trainable_02')
    d2 = Dropout(drop_rate)(d2)
    #print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e1_trainable_00')(d1)
    d1 = BatchNormalization(name='U_Net_decoder_e1_trainable_01')(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e1_trainable_02')
    d1 = Dropout(drop_rate)(d1)
    #print(f'The d1 shape is: {d1.shape}')

    #Gradually putting the network the same shape of the skip connection. 
    d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
    d0 = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_00')(d0)
    d0 = BatchNormalization(name='post_UNet_d0_trainable_01')(d0)
    d0 = LeakyReLU(alpha=alpha)(d0)
    d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
    d0 = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_02')(d0)

    #Combining with an addition the skip connection with the network
    result = Add()([skip_train, d0])
    result = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_results_trainable_00')(result)
    result = BatchNormalization(name='post_UNet_results_trainable_01')(result)
    result = LeakyReLU(alpha=alpha)(result)


    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name='post_UNet_output_trainable')(result)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
    margin = 4

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
    model.compile(
        optimizer=opt,
        loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function central_region_combined_loss(margin, loss_weights)
        metrics=[central_region_ssim(margin),
                 central_region_mse(margin) 
        ]
    )

    # #Print the model complexity
    # print(f'This is the model summary for the model code.')
    # model.summary()

    return model

def deepRU_article_non_interpolation_combine_loss_notUV(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                                     num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):
    inputs = Input(shape=(16,16,26))
    skip_train = Input(shape=(48, 48, 1))
    inputs_topo = Input(shape=(64, 64, 3))

    wind_t = Lambda(lambda x: x[...,12:16])(inputs)
    gradient_t = Lambda(lambda x: x[...,19:20])(inputs)
    other_input = Lambda(lambda x: x[...,20:])(inputs)

    ###### STEP1 ######
    #Other variables
    #Convolution sequence
    other_input = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o00")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o01")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)
    other_input = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o02")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o03")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)

    #Multiples levels TT
    #Convolution sequence
    wind_t = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t00")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t01")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)
    wind_t = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t02")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t03")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)

    ###### STEP2 ######
    #Combine spatio temporal variables with multiples levels variables and robinson 
    spatio_temp = Concatenate()([other_input, wind_t, gradient_t])
    spatio_temp = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio00")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio01")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)
    spatio_temp = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio02")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio03")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)

    ###### STEP3 ######
    #Putting topographie the same resolution of spatio-temporal data
    topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo00")(inputs_topo)
    topo = BatchNormalization(name="pre_UNet_geo01")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo02")(topo)
    topo = BatchNormalization(name="pre_UNet_geo03")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo04")(topo)
    topo = BatchNormalization(name="pre_UNet_geo05")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    ###### STEP4 ######
    #Concatenate the spatio temporal value with topo 
    input_concatenate = Concatenate()([spatio_temp, topo])

    ###### STEP5 ######
    #U-net
    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e1_00')(input_concatenate)
    e1 = BatchNormalization(name='U_Net_encoder_e1_01')(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0, name='U_Net_encoder_e1_02')
    e1 = Dropout(drop_rate)(e1)
    #print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e2_00')(e1)
    e2 = BatchNormalization(name='U_Net_encoder_e2_01')(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e2_02')
    e2 = Dropout(drop_rate)(e2)
    #print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e3_00')(e2)
    e3 = BatchNormalization(name='U_Net_encoder_e3_01')(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e3_02')
    e3 = Dropout(drop_rate)(e3)
    #print(f'The e3 shape is: {e3.shape}')

    e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e4_00')(e3)
    e4 = BatchNormalization(name='U_Net_encoder_e4_01')(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e4_02')
    e4 = Dropout(drop_rate)(e4)
    #print(f'The e4 shape is: {e4.shape}')

    e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e5_00')(e4)
    e5 = BatchNormalization(name='U_Net_encoder_e5_01')(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e5_02')
    e5 = Dropout(drop_rate)(e5)
    #print(f'The e5 shape is: {e5.shape}')

    #Bottleneck layer
    x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_bottleneck_00')(e5)
    x = BatchNormalization(name='U_Net_bottleneck_01')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_bottleneck_02')
    x = Dropout(drop_rate)(x)
    #print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
    d5 = Concatenate()([d5, e5])
    d5 = Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e5_trainable_00')(d5)
    d5 = BatchNormalization(name='U_Net_decoder_e5_trainable_01')(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e5_trainable_02')
    d5 = Dropout(drop_rate)(d5)
    #print(f'The d5 shape is: {d5.shape}')

    d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
    d4 = Concatenate()([d4, e4])
    d4 = Conv2D(320, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e4_trainable_00')(d4)
    d4 = BatchNormalization(name='U_Net_decoder_e4_trainable_01')(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e4_trainable_02')
    d4 = Dropout(drop_rate)(d4)
    #print(f'The d4 shape is: {d4.shape}')

    d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e3_trainable_00')(d3)
    d3 = BatchNormalization(name='U_Net_decoder_e3_trainable_01')(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e3_trainable_02')
    d3 = Dropout(drop_rate)(d3)
    #print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e2_trainable_00')(d2)
    d2 = BatchNormalization(name='U_Net_decoder_e2_trainable_01')(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e2_trainable_02')
    d2 = Dropout(drop_rate)(d2)
    #print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e1_trainable_00')(d1)
    d1 = BatchNormalization(name='U_Net_decoder_e1_trainable_01')(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e1_trainable_02')
    d1 = Dropout(drop_rate)(d1)
    #print(f'The d1 shape is: {d1.shape}')

    #Gradually putting the network the same shape of the skip connection. 
    d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
    d0 = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_00')(d0)
    d0 = BatchNormalization(name='post_UNet_d0_trainable_01')(d0)
    d0 = LeakyReLU(alpha=alpha)(d0)
    d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
    d0 = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_02')(d0)

    #Combining with an addition the skip connection with the network
    result = Add()([skip_train, d0])
    result = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_results_trainable_00')(result)
    result = BatchNormalization(name='post_UNet_results_trainable_01')(result)
    result = LeakyReLU(alpha=alpha)(result)


    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name='post_UNet_output_trainable')(result)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
    margin = 4

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
    model.compile(
        optimizer=opt,
        loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function central_region_combined_loss(margin, loss_weights)
        metrics=[central_region_ssim(margin),
                 central_region_mse(margin) 
        ]
    )

    # #Print the model complexity
    # print(f'This is the model summary for the model code.')
    # model.summary()

    return model

def deepRU_article_non_interpolation_combine_loss_uu(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                                     num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):
    inputs = Input(shape=(16,16,26))
    skip_train = Input(shape=(48, 48, 1))
    inputs_topo = Input(shape=(64, 64, 3))

    wind_uv = Lambda(lambda x: x[...,0:4])(inputs)
    gradient_uv = Lambda(lambda x: x[...,16:17])(inputs)
    wind_u = Lambda(lambda x: x[...,4:8])(inputs)
    gradient_u = Lambda(lambda x: x[...,17:18])(inputs)
    wind_v = Lambda(lambda x: x[...,8:12])(inputs)
    gradient_v = Lambda(lambda x: x[...,18:19])(inputs)
    wind_t = Lambda(lambda x: x[...,12:16])(inputs)
    gradient_t = Lambda(lambda x: x[...,19:20])(inputs)
    other_input = Lambda(lambda x: x[...,20:])(inputs)

    ###### STEP1 ######
    #Other variables
    #Convolution sequence
    other_input = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o00")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o01")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)
    other_input = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o02")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o03")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)

    #Multiples levels UU
    #Convolution sequence
    wind_u = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u00")(wind_u)
    wind_u = BatchNormalization(name="pre_UNet_u01")(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)
    wind_u = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u02")(wind_u)
    wind_u = BatchNormalization(name="pre_UNet_u03")(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)

    #Multiples levels VV
    #Convolution sequence
    wind_v = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v00")(wind_v)
    wind_v = BatchNormalization(name="pre_UNet_v01")(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)
    wind_v = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v02")(wind_v)
    wind_v = BatchNormalization(name="pre_UNet_v03")(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)

    #Multiples levels TT
    #Convolution sequence
    wind_t = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t00")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t01")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)
    wind_t = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t02")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t03")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)

    #Multiples levels UV
    #Convolution sequence
    wind_uv = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv00")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv01")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
    wind_uv = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv02")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv03")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)

    #Robinson's number
    robinson = Concatenate()([gradient_u, gradient_v, gradient_t])
    robinson = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_rob00")(robinson)
    robinson = BatchNormalization(name="pre_UNet_rob01")(robinson)
    robinson = LeakyReLU(alpha=alpha)(robinson)
    robinson = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_rob02")(robinson)
    robinson = BatchNormalization(name="pre_UNet_rob03")(robinson)
    robinson = LeakyReLU(alpha=alpha)(robinson)

    ###### STEP2 ######
    #Combine spatio temporal variables with multiples levels variables and robinson 
    spatio_temp = Concatenate()([other_input, wind_u, wind_v, wind_t, wind_uv, robinson])
    spatio_temp = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio00")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio01")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)
    spatio_temp = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio02")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio03")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)

    ###### STEP3 ######
    #Putting topographie the same resolution of spatio-temporal data
    topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo00")(inputs_topo)
    topo = BatchNormalization(name="pre_UNet_geo01")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo02")(topo)
    topo = BatchNormalization(name="pre_UNet_geo03")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo04")(topo)
    topo = BatchNormalization(name="pre_UNet_geo05")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    ###### STEP4 ######
    #Concatenate the spatio temporal value with topo 
    input_concatenate = Concatenate()([spatio_temp, topo])

    ###### STEP5 ######
    #U-net
    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e1_00')(input_concatenate)
    e1 = BatchNormalization(name='U_Net_encoder_e1_01')(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0, name='U_Net_encoder_e1_02')
    e1 = Dropout(drop_rate)(e1)
    #print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e2_00')(e1)
    e2 = BatchNormalization(name='U_Net_encoder_e2_01')(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e2_02')
    e2 = Dropout(drop_rate)(e2)
    #print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e3_00')(e2)
    e3 = BatchNormalization(name='U_Net_encoder_e3_01')(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e3_02')
    e3 = Dropout(drop_rate)(e3)
    #print(f'The e3 shape is: {e3.shape}')

    e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e4_00')(e3)
    e4 = BatchNormalization(name='U_Net_encoder_e4_01')(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e4_02')
    e4 = Dropout(drop_rate)(e4)
    #print(f'The e4 shape is: {e4.shape}')

    e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e5_00')(e4)
    e5 = BatchNormalization(name='U_Net_encoder_e5_01')(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e5_02')
    e5 = Dropout(drop_rate)(e5)
    #print(f'The e5 shape is: {e5.shape}')

    #Bottleneck layer
    x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_bottleneck_00')(e5)
    x = BatchNormalization(name='U_Net_bottleneck_01')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_bottleneck_02')
    x = Dropout(drop_rate)(x)
    #print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
    d5 = Concatenate()([d5, e5])
    d5 = Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e5_trainable_00')(d5)
    d5 = BatchNormalization(name='U_Net_decoder_e5_trainable_01')(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e5_trainable_02')
    d5 = Dropout(drop_rate)(d5)
    #print(f'The d5 shape is: {d5.shape}')

    d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
    d4 = Concatenate()([d4, e4])
    d4 = Conv2D(320, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e4_trainable_00')(d4)
    d4 = BatchNormalization(name='U_Net_decoder_e4_trainable_01')(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e4_trainable_02')
    d4 = Dropout(drop_rate)(d4)
    #print(f'The d4 shape is: {d4.shape}')

    d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e3_trainable_00')(d3)
    d3 = BatchNormalization(name='U_Net_decoder_e3_trainable_01')(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e3_trainable_02')
    d3 = Dropout(drop_rate)(d3)
    #print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e2_trainable_00')(d2)
    d2 = BatchNormalization(name='U_Net_decoder_e2_trainable_01')(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e2_trainable_02')
    d2 = Dropout(drop_rate)(d2)
    #print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e1_trainable_00')(d1)
    d1 = BatchNormalization(name='U_Net_decoder_e1_trainable_01')(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e1_trainable_02')
    d1 = Dropout(drop_rate)(d1)
    #print(f'The d1 shape is: {d1.shape}')

    #Gradually putting the network the same shape of the skip connection. 
    d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
    d0 = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_00')(d0)
    d0 = BatchNormalization(name='post_UNet_d0_trainable_01')(d0)
    d0 = LeakyReLU(alpha=alpha)(d0)
    d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
    d0 = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_02')(d0)

    #Combining with an addition the skip connection with the network
    result = Add()([skip_train, d0])
    result = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_results_trainable_00')(result)
    result = BatchNormalization(name='post_UNet_results_trainable_01')(result)
    result = LeakyReLU(alpha=alpha)(result)


    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name='post_UNet_output_trainable')(result)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
    margin = 4

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=3.0)
    model.compile(
        optimizer=opt,
        loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function central_region_combined_loss(margin, loss_weights)
        metrics=[central_region_ssim(margin),
                 central_region_mse(margin) 
        ]
    )

    # #Print the model complexity
    # print(f'This is the model summary for the model code.')
    # model.summary()

    return model

def deepRU_article_non_interpolation_combine_loss_transferLearning(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                                     num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):
    inputs = Input(shape=(16,16,26))
    skip_train = Input(shape=(48, 48, 1))
    inputs_topo = Input(shape=(64, 64, 3))

    wind_uv = Lambda(lambda x: x[...,0:4])(inputs)
    gradient_uv = Lambda(lambda x: x[...,16:17])(inputs)
    wind_u = Lambda(lambda x: x[...,4:8])(inputs)
    gradient_u = Lambda(lambda x: x[...,17:18])(inputs)
    wind_v = Lambda(lambda x: x[...,8:12])(inputs)
    gradient_v = Lambda(lambda x: x[...,18:19])(inputs)
    wind_t = Lambda(lambda x: x[...,12:16])(inputs)
    gradient_t = Lambda(lambda x: x[...,19:20])(inputs)
    other_input = Lambda(lambda x: x[...,20:])(inputs)

    ###### STEP1 ######
    #Other variables
    #Convolution sequence
    other_input = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o00")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o01")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)
    other_input = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o02")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o03")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)

    #Multiples levels UU
    #Convolution sequence
    wind_u = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u00")(wind_u)
    wind_u = BatchNormalization(name="pre_UNet_u01")(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)
    wind_u = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u02")(wind_u)
    wind_u = BatchNormalization(name="pre_UNet_u03")(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)

    #Multiples levels VV
    #Convolution sequence
    wind_v = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v00")(wind_v)
    wind_v = BatchNormalization(name="pre_UNet_v01")(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)
    wind_v = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v02")(wind_v)
    wind_v = BatchNormalization(name="pre_UNet_v03")(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)

    #Multiples levels TT
    #Convolution sequence
    wind_t = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t00")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t01")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)
    wind_t = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t02")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t03")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)

    #Multiples levels UV
    #Convolution sequence
    wind_uv = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv00")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv01")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
    wind_uv = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv02")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv03")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)

    #Robinson's number
    robinson = Concatenate()([gradient_u, gradient_v, gradient_t])
    robinson = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_rob00")(robinson)
    robinson = BatchNormalization(name="pre_UNet_rob01")(robinson)
    robinson = LeakyReLU(alpha=alpha)(robinson)
    robinson = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_rob02")(robinson)
    robinson = BatchNormalization(name="pre_UNet_rob03")(robinson)
    robinson = LeakyReLU(alpha=alpha)(robinson)

    ###### STEP2 ######
    #Combine spatio temporal variables with multiples levels variables and robinson 
    spatio_temp = Concatenate()([other_input, wind_u, wind_v, wind_t, wind_uv, robinson])
    spatio_temp = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio00")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio01")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)
    spatio_temp = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio02")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio03")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)

    ###### STEP3 ######
    #Putting topographie the same resolution of spatio-temporal data
    topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo00")(inputs_topo)
    topo = BatchNormalization(name="pre_UNet_geo01")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo02")(topo)
    topo = BatchNormalization(name="pre_UNet_geo03")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo04")(topo)
    topo = BatchNormalization(name="pre_UNet_geo05")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    ###### STEP4 ######
    #Concatenate the spatio temporal value with topo 
    input_concatenate = Concatenate()([spatio_temp, topo])

    ###### STEP5 ######
    #U-net
    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e1_00')(input_concatenate)
    e1 = BatchNormalization(name='U_Net_encoder_e1_01')(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0, name='U_Net_encoder_e1_02')
    e1 = Dropout(drop_rate)(e1)
    #print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e2_00')(e1)
    e2 = BatchNormalization(name='U_Net_encoder_e2_01')(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e2_02')
    e2 = Dropout(drop_rate)(e2)
    #print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e3_00')(e2)
    e3 = BatchNormalization(name='U_Net_encoder_e3_01')(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e3_02')
    e3 = Dropout(drop_rate)(e3)
    #print(f'The e3 shape is: {e3.shape}')

    e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e4_00')(e3)
    e4 = BatchNormalization(name='U_Net_encoder_e4_01')(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e4_02')
    e4 = Dropout(drop_rate)(e4)
    #print(f'The e4 shape is: {e4.shape}')

    e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e5_00')(e4)
    e5 = BatchNormalization(name='U_Net_encoder_e5_01')(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e5_02')
    e5 = Dropout(drop_rate)(e5)
    #print(f'The e5 shape is: {e5.shape}')

    #Bottleneck layer
    x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_bottleneck_00')(e5)
    x = BatchNormalization(name='U_Net_bottleneck_01')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_bottleneck_02')
    x = Dropout(drop_rate)(x)
    #print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
    d5 = Concatenate()([d5, e5])
    d5 = Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e5_trainable_00')(d5)
    d5 = BatchNormalization(name='U_Net_decoder_e5_trainable_01')(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e5_trainable_02')
    d5 = Dropout(drop_rate)(d5)
    #print(f'The d5 shape is: {d5.shape}')

    d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
    d4 = Concatenate()([d4, e4])
    d4 = Conv2D(320, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e4_trainable_00')(d4)
    d4 = BatchNormalization(name='U_Net_decoder_e4_trainable_01')(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e4_trainable_02')
    d4 = Dropout(drop_rate)(d4)
    #print(f'The d4 shape is: {d4.shape}')

    d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e3_trainable_00')(d3)
    d3 = BatchNormalization(name='U_Net_decoder_e3_trainable_01')(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e3_trainable_02')
    d3 = Dropout(drop_rate)(d3)
    #print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e2_trainable_00')(d2)
    d2 = BatchNormalization(name='U_Net_decoder_e2_trainable_01')(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e2_trainable_02')
    d2 = Dropout(drop_rate)(d2)
    #print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e1_trainable_00')(d1)
    d1 = BatchNormalization(name='U_Net_decoder_e1_trainable_01')(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e1_trainable_02')
    d1 = Dropout(drop_rate)(d1)
    #print(f'The d1 shape is: {d1.shape}')

    #Gradually putting the network the same shape of the skip connection. 
    d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
    d0 = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_00')(d0)
    d0 = BatchNormalization(name='post_UNet_d0_trainable_01')(d0)
    d0 = LeakyReLU(alpha=alpha)(d0)
    d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
    d0 = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_02')(d0)

    #Combining with an addition the skip connection with the network
    result = Add()([skip_train, d0])
    result = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_results_trainable_00')(result)
    result = BatchNormalization(name='post_UNet_results_trainable_01')(result)
    result = LeakyReLU(alpha=alpha)(result)


    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name='post_UNet_output_trainable')(result)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
    commun_name = 'trainable'
    #Freeze the top layers and just train the decoder and post U-Net part
    for layer in model.layers:
        if commun_name in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    margin = 4

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=3.0)
    model.compile(
        optimizer=opt,
        loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function central_region_combined_loss(margin, loss_weights)
        metrics=[central_region_ssim(margin),
                 central_region_mse(margin) 
        ]
    )

    # #Print the model complexity
    # print(f'This is the model summary for the model code.')
    # model.summary()

    return model






# def deepRU_article_interpolation_combine_loss(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
#                                               num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):

#     inputs = Input(shape=(64,64,26))
#     skip_train = Input(shape=(48, 48, 1))
#     inputs_topo = Input(shape=(64, 64, 3))

#     ###### STEP1 ######
#     #Multiples levels UV
#     wind_uv = Lambda(lambda x: x[...,0:4])(inputs)
#     # print(f'The shape of wind_u is: {wind_uv.shape}')
#     gradient_uv = Lambda(lambda x: x[...,16:17])(inputs)
#     # print(f'The shape of gradient_uv is: {gradient_uv.shape}')
#     #Convolution sequence
#     wind_uv = Conv2D(16, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_uv00")(wind_uv)
#     wind_uv = BatchNormalization(name="pre_UNet_uv01")(wind_uv)
#     wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
#     wind_uv = Conv2D(32, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_uv02")(wind_uv)
#     wind_uv = BatchNormalization(name="pre_UNet_uv03")(wind_uv)
#     wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
#     #Concatenation with gradient
#     wind_uv = Concatenate()([wind_uv, gradient_uv])
#     #Convolution sequence
#     wind_uv = Conv2D(64, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_uv04")(wind_uv)
#     wind_uv = BatchNormalization(name="pre_UNet_uv05")(wind_uv)
#     wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
#     print(f'The shape of wind_uv at the end of the first step is: {wind_uv.shape}')

#     #Multiples levels UU
#     wind_u = Lambda(lambda x: x[...,4:8])(inputs)
#     # print(f'The shape of wind_v is: {wind_u.shape}')
#     gradient_u = Lambda(lambda x: x[...,17:18])(inputs)
#     # print(f'The shape of gradient_u is: {gradient_u.shape}')
#     #Convolution sequence
#     wind_u = Conv2D(16, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_u00")(wind_u)
#     wind_u = BatchNormalization(name="pre_UNet_u01")(wind_u)
#     wind_u = LeakyReLU(alpha=alpha)(wind_u)
#     wind_u = Conv2D(32, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_u02")(wind_u)
#     wind_u = BatchNormalization(name="pre_UNet_u03")(wind_u)
#     wind_u = LeakyReLU(alpha=alpha)(wind_u)
#     #Concatenation with gradient
#     wind_u = Concatenate()([wind_u, gradient_u])
#     #Convolution sequence
#     wind_u = Conv2D(64, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_u04")(wind_u)
#     wind_u = BatchNormalization(name="pre_UNet_u05")(wind_u)
#     wind_u = LeakyReLU(alpha=alpha)(wind_u)
#     print(f'The shape of wind_u at the end of the first step is: {wind_u.shape}')

#     #Multiples levels VV
#     wind_v = Lambda(lambda x: x[...,8:12])(inputs)
#     # print(f'The shape of wind_t is: {wind_v.shape}')
#     gradient_v = Lambda(lambda x: x[...,18:19])(inputs)
#     # print(f'The shape of gradient_v is: {gradient_v.shape}')
#     #Convolution sequence
#     wind_v = Conv2D(16, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_v00")(wind_v)
#     wind_v = BatchNormalization(name="pre_UNet_v01")(wind_v)
#     wind_v = LeakyReLU(alpha=alpha)(wind_v)
#     wind_v = Conv2D(32, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_v02")(wind_v)
#     wind_v = BatchNormalization(name="pre_UNet_v03")(wind_v)
#     wind_v = LeakyReLU(alpha=alpha)(wind_v)
#     #Concatenation with gradient
#     wind_v = Concatenate()([wind_v, gradient_v])
#     #Convolution sequence
#     wind_v = Conv2D(64, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_v04")(wind_v)
#     wind_v = BatchNormalization(name="pre_UNet_v05")(wind_v)
#     wind_v = LeakyReLU(alpha=alpha)(wind_v)
#     print(f'The shape of wind_v at the end of the first step is: {wind_v.shape}')

#     #Multiples levels TT
#     wind_t = Lambda(lambda x: x[...,12:16])(inputs)
#     # print(f'The shape of wind_u is: {wind_t.shape}')
#     gradient_t = Lambda(lambda x: x[...,19:20])(inputs)
#     # print(f'The shape of gradient_t is: {gradient_t.shape}')
#     #Convolution sequence
#     wind_t = Conv2D(16, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_t00")(wind_t)
#     wind_t = BatchNormalization(name="pre_UNet_t01")(wind_t)
#     wind_t = LeakyReLU(alpha=alpha)(wind_t)
#     wind_t = Conv2D(32, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_t02")(wind_t)
#     wind_t = BatchNormalization(name="pre_UNet_t03")(wind_t)
#     wind_t = LeakyReLU(alpha=alpha)(wind_t)
#     #Concatenation with gradient
#     wind_t = Concatenate()([wind_t, gradient_t])
#     #Convolution sequence
#     wind_t = Conv2D(64, (3 ,3), padding='same', use_bias=True, kernel_initializer=initializers.HeNormal(), bias_initializer='zeros', name="pre_UNet_t04")(wind_t)
#     wind_t = BatchNormalization(name="pre_UNet_t05")(wind_t)
#     wind_t = LeakyReLU(alpha=alpha)(wind_t)
#     print(f'The shape of wind_t at the end of the first step is: {wind_t.shape}')

#     ###### STEP2 ######
#     #Concatenate uv, u, v, t and gradient together
#     wind = Concatenate()([wind_uv, wind_u, wind_v, wind_t]) 
#     wind = Conv2D(32, (3 ,3), strides=(2,2), padding='same', kernel_initializer=initializers.HeNormal(), use_bias=True, bias_initializer='zeros', name="pre_UNet_wind00")(wind)
#     wind = Conv2D(64, (3 ,3), strides=(2,2), padding='same', kernel_initializer=initializers.HeNormal(), use_bias=True, bias_initializer='zeros', name="pre_UNet_wind01")(wind)
#     print(f'At the end of step 2 the shape of wind is: {wind.shape}')

#     ###### STEP3 ######
#     #Putting topographie the same resolution of spatio-temporal data
#     topo = Conv2D(16, (3 ,3), strides=(2,2), padding='same', kernel_initializer=initializers.HeNormal(), use_bias=True, bias_initializer='zeros', name="pre_UNet_topo00")(inputs_topo)
#     topo = BatchNormalization(name="pre_UNet_topo01")(topo)
#     topo = LeakyReLU(alpha=alpha)(topo)

#     topo = Conv2D(32, (3 ,3), strides=(2,2), padding='same', kernel_initializer=initializers.HeNormal(), use_bias=True, bias_initializer='zeros', name="pre_UNet_topo02")(topo)
#     topo = BatchNormalization(name="pre_UNet_topo03")(topo)
#     topo = LeakyReLU(alpha=alpha)(topo)

#     topo = Conv2D(64, (3 ,3), padding='same', kernel_initializer=initializers.HeNormal(), use_bias=True, bias_initializer='zeros', name="pre_UNet_topo04")(topo)
#     topo = BatchNormalization(name="pre_UNet_topo05")(topo)
#     topo = LeakyReLU(alpha=alpha)(topo)

#     ###### STEP4 ######
#     #Concatenate all input data together
#     other_input = Lambda(lambda x: x[...,20:])(inputs)
#     other_input = Conv2D(32, (3 ,3), strides=(2,2), padding='same', kernel_initializer=initializers.HeNormal(), use_bias=True, bias_initializer='zeros', name="pre_UNet_con00")(other_input)
#     other_input = Conv2D(64, (3 ,3), strides=(2,2), padding='same', kernel_initializer=initializers.HeNormal(), use_bias=True, bias_initializer='zeros', name="pre_UNet_con01")(other_input)
#     print(f'The shape of the other input is : {other_input.shape}')
#     input_concatenate = Concatenate()([wind, other_input, topo])
#     print(f'The shape of the input before the u-net is: {input_concatenate.shape}')

#     ###### STEP4 ######
#     #U-net
#     #Encoder
#     e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e1_00')(input_concatenate)
#     e1 = BatchNormalization(name='U_Net_encoder_e1_01')(e1)
#     e1 = LeakyReLU(alpha=alpha)(e1)
#     e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0, name='U_Net_encoder_e1_02')
#     e1 = Dropout(drop_rate)(e1)
#     print(f'The e1 shape is: {e1.shape}')

#     e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e2_00')(e1)
#     e2 = BatchNormalization(name='U_Net_encoder_e2_01')(e2)
#     e2 = LeakyReLU(alpha=alpha)(e2)
#     e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e2_02')
#     e2 = Dropout(drop_rate)(e2)
#     print(f'The e2 shape is: {e2.shape}')

#     e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e3_00')(e2)
#     e3 = BatchNormalization(name='U_Net_encoder_e3_01')(e3)
#     e3 = LeakyReLU(alpha=alpha)(e3)
#     e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e3_02')
#     e3 = Dropout(drop_rate)(e3)
#     print(f'The e3 shape is: {e3.shape}')

#     e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e4_00')(e3)
#     e4 = BatchNormalization(name='U_Net_encoder_e4_01')(e4)
#     e4 = LeakyReLU(alpha=alpha)(e4)
#     e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e4_02')
#     e4 = Dropout(drop_rate)(e4)
#     print(f'The e4 shape is: {e4.shape}')

#     e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e5_00')(e4)
#     e5 = BatchNormalization(name='U_Net_encoder_e5_01')(e5)
#     e5 = LeakyReLU(alpha=alpha)(e5)
#     e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e5_02')
#     e5 = Dropout(drop_rate)(e5)
#     print(f'The e5 shape is: {e5.shape}')

#     #Bottleneck layer
#     x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_bottleneck_00')(e5)
#     x = BatchNormalization(name='U_Net_bottleneck_01')(x)
#     x = LeakyReLU(alpha=alpha)(x)
#     x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_bottleneck_02')
#     x = Dropout(drop_rate)(x)
#     print(f'The bottleneck shape is: {x.shape}')

#     #Decoder
#     d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
#     d5 = Concatenate()([d5, e5])
#     d5 = Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_d5_trainable_00')(d5)
#     d5 = BatchNormalization(name='U_Net_decoder_d5_trainable_01')(d5)
#     d5 = LeakyReLU(alpha=alpha)(d5)
#     d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_d5_trainable_02')
#     d5 = Dropout(drop_rate)(d5)
#     print(f'The d5 shape is: {d5.shape}')

#     d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
#     d4 = Concatenate()([d4, e4])
#     d4 = Conv2D(320, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_d4_trainable_00')(d4)
#     d4 = BatchNormalization(name='U_Net_decoder_d4_trainable_01')(d4)
#     d4 = LeakyReLU(alpha=alpha)(d4)
#     d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_d4_trainable_02')
#     d4 = Dropout(drop_rate)(d4)
#     print(f'The d4 shape is: {d4.shape}')

#     d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
#     d3 = Concatenate()([d3, e3])
#     d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_d3_trainable_00')(d3)
#     d3 = BatchNormalization(name='U_Net_decoder_d3_trainable_01')(d3)
#     d3 = LeakyReLU(alpha=alpha)(d3)
#     d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_d3_trainable_02')
#     d3 = Dropout(drop_rate)(d3)
#     print(f'The d3 shape is: {d3.shape}')
    
#     d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
#     d2 = Concatenate()([d2, e2])
#     d2 = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_d2_trainable_00')(d2)
#     d2 = BatchNormalization(name='U_Net_decoder_d2_trainable_01')(d2)
#     d2 = LeakyReLU(alpha=alpha)(d2)
#     d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_d2_trainable_02')
#     d2 = Dropout(drop_rate)(d2)
#     print(f'The d2 shape is: {d2.shape}')
    
#     d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
#     d1 = Concatenate()([d1, e1])
#     d1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_d1_trainable_00')(d1)
#     d1 = BatchNormalization(name='U_Net_decoder_d1_trainable_01')(d1)
#     d1 = LeakyReLU(alpha=alpha)(d1)
#     d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_d1_trainable_02')
#     d1 = Dropout(drop_rate)(d1)
#     print(f'The d1 shape is: {d1.shape}')

#     #Gradually putting the network the same shape of the skip connection. 
#     d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
#     d0 = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_00')(d0)
#     d0 = BatchNormalization(name='post_UNet_d0_trainable_01')(d0)
#     d0 = LeakyReLU(alpha=alpha)(d0)
#     d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
#     d0 = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_02')(d0)

#     #Combining with an addition the skip connection with the network
#     result = Add()([skip_train, d0])
#     result = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_results_trainable_00')(result)
#     result = BatchNormalization(name='post_UNet_results_trainable_01')(result)
#     result = LeakyReLU(alpha=alpha)(result)

#     outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name='post_UNet_output_trainable')(result)
    
#     #Define the input-output of the model 
#     model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
#     margin = 4

#     #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
#     opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
#     model.compile(
#         optimizer=opt,
#         loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function
#         metrics=[central_region_ssim(margin),
#                  central_region_mse(margin) 
#         ]
#     )

#     #Print the model complexity
#     #model.summary()

#     return model

def deepRU_article_interpolation_combine_loss(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                                              num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):
    inputs = Input(shape=(64,64,26))
    skip_train = Input(shape=(48, 48, 1))
    inputs_topo = Input(shape=(64, 64, 3))

    wind_uv = Lambda(lambda x: x[...,0:4])(inputs)
    gradient_uv = Lambda(lambda x: x[...,16:17])(inputs)
    wind_u = Lambda(lambda x: x[...,4:8])(inputs)
    gradient_u = Lambda(lambda x: x[...,17:18])(inputs)
    wind_v = Lambda(lambda x: x[...,8:12])(inputs)
    gradient_v = Lambda(lambda x: x[...,18:19])(inputs)
    wind_t = Lambda(lambda x: x[...,12:16])(inputs)
    gradient_t = Lambda(lambda x: x[...,19:20])(inputs)
    other_input = Lambda(lambda x: x[...,20:])(inputs)

    ###### STEP1 ######
    #Other variables
    #Convolution sequence
    other_input = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o00")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o01")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)
    other_input = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_o02")(other_input)
    other_input = BatchNormalization(name="pre_UNet_o03")(other_input)
    other_input = LeakyReLU(alpha=alpha)(other_input)

    #Multiples levels UU
    #Convolution sequence
    wind_u = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u00")(wind_u)
    wind_u = BatchNormalization(name="pre_UNet_u01")(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)
    wind_u = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_u02")(wind_u)
    wind_u = BatchNormalization(name="pre_UNet_u03")(wind_u)
    wind_u = LeakyReLU(alpha=alpha)(wind_u)

    #Multiples levels VV
    #Convolution sequence
    wind_v = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v00")(wind_v)
    wind_v = BatchNormalization(name="pre_UNet_v01")(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)
    wind_v = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_v02")(wind_v)
    wind_v = BatchNormalization(name="pre_UNet_v03")(wind_v)
    wind_v = LeakyReLU(alpha=alpha)(wind_v)

    #Multiples levels TT
    #Convolution sequence
    wind_t = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t00")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t01")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)
    wind_t = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_t02")(wind_t)
    wind_t = BatchNormalization(name="pre_UNet_t03")(wind_t)
    wind_t = LeakyReLU(alpha=alpha)(wind_t)

    #Multiples levels UV
    #Convolution sequence
    wind_uv = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv00")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv01")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)
    wind_uv = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_uv02")(wind_uv)
    wind_uv = BatchNormalization(name="pre_UNet_uv03")(wind_uv)
    wind_uv = LeakyReLU(alpha=alpha)(wind_uv)

    #Robinson's number
    robinson = Concatenate()([gradient_u, gradient_v, gradient_t])
    robinson = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_rob00")(robinson)
    robinson = BatchNormalization(name="pre_UNet_rob01")(robinson)
    robinson = LeakyReLU(alpha=alpha)(robinson)
    robinson = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_rob02")(robinson)
    robinson = BatchNormalization(name="pre_UNet_rob03")(robinson)
    robinson = LeakyReLU(alpha=alpha)(robinson)

    ###### STEP2 ######
    #Combine spatio temporal variables with multiples levels variables and robinson 
    spatio_temp = Concatenate()([other_input, wind_u, wind_v, wind_t, wind_uv, robinson])
    spatio_temp = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio00")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio01")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)
    spatio_temp = Conv2D(64, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_spatio02")(spatio_temp)
    spatio_temp = BatchNormalization(name="pre_UNet_spatio03")(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)

    ###### STEP3 ######
    #Putting topographie the same resolution of spatio-temporal data
    topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo00")(inputs_topo)
    topo = BatchNormalization(name="pre_UNet_geo01")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo02")(topo)
    topo = BatchNormalization(name="pre_UNet_geo03")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), name="pre_UNet_geo04")(topo)
    topo = BatchNormalization(name="pre_UNet_geo05")(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    ###### STEP4 ######
    #Concatenate the spatio temporal value with topo 
    input_concatenate = Concatenate()([spatio_temp, topo])

    ###### STEP5 ######
    #U-net
    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e1_00')(input_concatenate)
    e1 = BatchNormalization(name='U_Net_encoder_e1_01')(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0, name='U_Net_encoder_e1_02')
    e1 = Dropout(drop_rate)(e1)
    #print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e2_00')(e1)
    e2 = BatchNormalization(name='U_Net_encoder_e2_01')(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e2_02')
    e2 = Dropout(drop_rate)(e2)
    #print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e3_00')(e2)
    e3 = BatchNormalization(name='U_Net_encoder_e3_01')(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e3_02')
    e3 = Dropout(drop_rate)(e3)
    #print(f'The e3 shape is: {e3.shape}')

    e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e4_00')(e3)
    e4 = BatchNormalization(name='U_Net_encoder_e4_01')(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e4_02')
    e4 = Dropout(drop_rate)(e4)
    #print(f'The e4 shape is: {e4.shape}')

    e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_encoder_e5_00')(e4)
    e5 = BatchNormalization(name='U_Net_encoder_e5_01')(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_encoder_e5_02')
    e5 = Dropout(drop_rate)(e5)
    #print(f'The e5 shape is: {e5.shape}')

    #Bottleneck layer
    x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_bottleneck_00')(e5)
    x = BatchNormalization(name='U_Net_bottleneck_01')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_bottleneck_02')
    x = Dropout(drop_rate)(x)
    #print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
    d5 = Concatenate()([d5, e5])
    d5 = Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e5_trainable_00')(d5)
    d5 = BatchNormalization(name='U_Net_decoder_e5_trainable_01')(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e5_trainable_02')
    d5 = Dropout(drop_rate)(d5)
    #print(f'The d5 shape is: {d5.shape}')

    d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
    d4 = Concatenate()([d4, e4])
    d4 = Conv2D(320, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e4_trainable_00')(d4)
    d4 = BatchNormalization(name='U_Net_decoder_e4_trainable_01')(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e4_trainable_02')
    d4 = Dropout(drop_rate)(d4)
    #print(f'The d4 shape is: {d4.shape}')

    d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e3_trainable_00')(d3)
    d3 = BatchNormalization(name='U_Net_decoder_e3_trainable_01')(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e3_trainable_02')
    d3 = Dropout(drop_rate)(d3)
    #print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e2_trainable_00')(d2)
    d2 = BatchNormalization(name='U_Net_decoder_e2_trainable_01')(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e2_trainable_02')
    d2 = Dropout(drop_rate)(d2)
    #print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='U_Net_decoder_e1_trainable_00')(d1)
    d1 = BatchNormalization(name='U_Net_decoder_e1_trainable_01')(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0, name='U_Net_decoder_e1_trainable_02')
    d1 = Dropout(drop_rate)(d1)
    #print(f'The d1 shape is: {d1.shape}')

    #Gradually putting the network the same shape of the skip connection. 
    d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
    d0 = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_00')(d0)
    d0 = BatchNormalization(name='post_UNet_d0_trainable_01')(d0)
    d0 = LeakyReLU(alpha=alpha)(d0)
    d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
    d0 = Conv2D(1, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_d0_trainable_02')(d0)

    #Combining with an addition the skip connection with the network
    result = Add()([skip_train, d0])
    result = Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02), name='post_UNet_results_trainable_00')(result)
    result = BatchNormalization(name='post_UNet_results_trainable_01')(result)
    result = LeakyReLU(alpha=alpha)(result)


    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=initializers.HeNormal(), kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02), name='post_UNet_output_trainable')(result)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
    margin = 4

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
    model.compile(
        optimizer=opt,
        loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function central_region_combined_loss(margin, loss_weights)
        metrics=[central_region_ssim(margin),
                 central_region_mse(margin) 
        ]
    )

    # #Print the model complexity
    # print(f'This is the model summary for the model code.')
    # model.summary()

    return model

def deepRU_article_testing_similar_architecture(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
                                              num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha, loss_weights):

    inputs = Input(shape=(16,16,26))
    skip_train = Input(shape=(48, 48, 1))
    inputs_topo = Input(shape=(64, 64, 3))

    ###### STEP1 ######
    spatio_temp = Conv2D(16, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros')(inputs)
    spatio_temp = BatchNormalization()(spatio_temp)
    spatio_temp = LeakyReLU(alpha=alpha)(spatio_temp)

    ###### STEP2 ######
    #Putting topographie the same resolution of spatio-temporal data
    topo = Conv2D(16, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros')(inputs_topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(32, (3 ,3), strides=(2, 2), padding='same', use_bias=True, bias_initializer='zeros')(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(64, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros')(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    ###### STEP4 ######
    #Concatenate all input data together
    input_concatenate = Concatenate()([spatio_temp, topo])
    print(f'The shape of the input before the u-net is: {input_concatenate.shape}')

    ###### STEP4 ######
    #U-net
    #Encoder
    e1 = Conv2D(128, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(input_concatenate)
    e1 = BatchNormalization()(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    e1 = deepRU_residual_block(e1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0.0)
    e1 = Dropout(drop_rate)(e1)
    print(f'The e1 shape is: {e1.shape}')

    e2 = Conv2D(192, (3, 3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    e2 = deepRU_residual_block(e2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    e2 = Dropout(drop_rate)(e2)
    print(f'The e2 shape is: {e2.shape}')

    e3 = Conv2D(256, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    e3 = deepRU_residual_block(e3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    e3 = Dropout(drop_rate)(e3)
    print(f'The e3 shape is: {e3.shape}')

    e4 = Conv2D(320, (3, 3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e3)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    e4 = deepRU_residual_block(e4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    e4 = Dropout(drop_rate)(e4)
    print(f'The e4 shape is: {e4.shape}')

    e5 = Conv2D(384, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e4)
    e5 = BatchNormalization()(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    e5 = deepRU_residual_block(e5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    e5 = Dropout(drop_rate)(e5)
    print(f'The e5 shape is: {e5.shape}')

    #Bottleneck layer
    x = Conv2D(448, (3,3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e5)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = deepRU_residual_block(x, 448, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    x = Dropout(drop_rate)(x)
    print(f'The bottleneck shape is: {x.shape}')

    #Decoder
    d5 = UpSampling2D(size=(2, 1), interpolation='bilinear')(x)
    d5 = Concatenate()([d5, e5])
    d5 = Conv2D(384, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d5)
    d5 = BatchNormalization()(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    d5 = deepRU_residual_block(d5, 384, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    d5 = Dropout(drop_rate)(d5)
    print(f'The d5 shape is: {d5.shape}')

    d4 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d5)
    d4 = Concatenate()([d4, e4])
    d4 = Conv2D(320, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d4)
    d4 = BatchNormalization()(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    d4 = deepRU_residual_block(d4, 320, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    d4 = Dropout(drop_rate)(d4)
    print(f'The d4 shape is: {d4.shape}')

    d3 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
    d3 = Concatenate()([d3, e3])
    d3 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d3)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    d3 = deepRU_residual_block(d3, 256, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    d3 = Dropout(drop_rate)(d3)
    print(f'The d3 shape is: {d3.shape}')
    
    d2 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d3)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(192, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d2)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    d2 = deepRU_residual_block(d2, 192, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    d2 = Dropout(drop_rate)(d2)
    print(f'The d2 shape is: {d2.shape}')
    
    d1 = UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
    d1 = Concatenate()([d1, e1])
    d1 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d1)
    d1 = BatchNormalization()(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    d1 = deepRU_residual_block(d1, 128, kernel_size=3, alpha=alpha, regL1=0, regL2=0)
    d1 = Dropout(drop_rate)(d1)
    print(f'The d1 shape is: {d1.shape}')

    #Gradually putting the network the same shape of the skip connection. 
    d0 = UpSampling2D(size=(1, 2), interpolation='bilinear')(d1)
    d0 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d0)
    d0 = BatchNormalization()(d0)
    d0 = LeakyReLU(alpha=alpha)(d0)
    d0 = UpSampling2D(size=(3, 3), interpolation='bilinear')(d0)
    d0 = Conv2D(1, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d0)

    #Combining with an addition the skip connection with the network
    result = Add()([skip_train, d0])
    result = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(result)
    result = BatchNormalization()(result)
    result = LeakyReLU(alpha=alpha)(result)


    outputs = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02))(result)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo, skip_train] , outputs=outputs)
    margin = 4

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=5.0)
    model.compile(
        optimizer=opt,
        loss=central_region_combined_loss(margin, loss_weights),  # Use the correct custom loss function
        metrics=[central_region_ssim(margin),
                 central_region_mse(margin) 
        ]
    )

    #Print the model complexity
    #model.summary()

    return model

def deepRU_v01(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
           num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha):
    """
    Modele semple interessant!
    The same as v0, but inversed, only the wind input do not pass throw the network. 
    Input and topo are NOT concatenated together and go throw the U-Net architecture simultaniously: 
    - Pass the original low resolution input throw multiples convolution layers and do an upsample x2
    - Pass the topographie high resolution input throw multiples convolution layers. 

    In the U-Net:
    - The input is only the low resolution 
    - The depth of the network is 3 and is based on the paper that we found this architecture in, but do not have the same depth as the article.

    End of model:
    - Concatenate the input low resolution with the topographie high resolution. 
    """
    inputs = Input(shape=(32,32,10))
    inputs_topo = Input(shape=(64, 64, 3))
    #Pre-encoder predictors
    topo = Conv2D(18, (3 ,3), strides=(1, 2), padding='same', use_bias=True, bias_initializer='zeros')(inputs_topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(36, (3 ,5), strides=(2, 1), padding='same', use_bias=True, bias_initializer='zeros')(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)
    print(f'The topo shape is {topo.shape}')

    #Concatenate the input 
    inp = Concatenate()([inputs, topo])
    #inp = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear'))(inp)
    inp = UpSampling2D(size=(2, 2))(inp)
    print(f'The inp shape is {inp.shape}')
    #Skip connection with only wind data
    wind = Lambda(lambda x: x[...,0:1])(inp)

    #Encoder
    e1 = Conv2D(32, (3, 3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(inp)
    e1 = BatchNormalization()(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    print(f'The e1 layer shape is {e1.shape}')

    e2 = Conv2D(64, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    print(f'The e2 layer shape is {e2.shape}')

    e3 = Conv2D(128, (3, 3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    print(f'The e3 layer shape is {e3.shape}')

    e4 = Conv2D(192, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e3)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    print(f'The e4 layer shape is {e4.shape}')

    e5 = Conv2D(256, (3, 3), strides=(2,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e4)
    e5 = BatchNormalization()(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    print(f'The e5 layer shape is {e5.shape}')

    #Bottleneck layer
    x = Conv2D(320, (3,3), strides=(2,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e5)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    x = LeakyReLU(alpha=alpha)(x)
    print(f'The x layer shape is {x.shape}')

    #Decoder
    #d3 = Conv2DTranspose(512, (kernel_size_e1, kernel_size_e1), strides=(2,2), padding='same')(x)
    d5 = UpSampling2D(size=(2, 2))(x)
    #d5 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear'))(x)
    d5 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d5)
    d5 = BatchNormalization()(d5)
    d5 = Concatenate()([d5, e5])
    d5 = Dropout(drop_rate)(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    print(f'The d5 layer shape is {d5.shape}')
    
    #d2 = Conv2DTranspose(256, (kernel_size_e2, kernel_size_e2), strides=(2,2), padding='same')(d3)
    d4 = UpSampling2D(size=(2, 2))(d5)
    #d4 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear'))(d5)
    d4 = Conv2D(192, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d4)
    d4 = BatchNormalization()(d4)
    d4 = Concatenate()([d4, e4])
    d4 = Dropout(drop_rate)(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    print(f'The d4 layer shape is {d4.shape}')

    #d3 = Conv2DTranspose(512, (kernel_size_e1, kernel_size_e1), strides=(2,2), padding='same')(x)
    d3 = UpSampling2D(size=(1, 2))(d4)
    #d3 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 1, x.shape[2] * 2), method='bilinear'))(d4)
    d3 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d3)
    d3 = BatchNormalization()(d3)
    d3 = Concatenate()([d3, e3])
    d3 = Dropout(drop_rate)(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    print(f'The d3 layer shape is {d3.shape}')
    
    #d2 = Conv2DTranspose(256, (kernel_size_e2, kernel_size_e2), strides=(2,2), padding='same')(d3)
    d2 = UpSampling2D(size=(2, 1))(d3)
    #d2 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 1), method='bilinear'))(d3)
    d2 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d2)
    d2 = BatchNormalization()(d2)
    d2 = Concatenate()([d2, e2])
    d2 = Dropout(drop_rate)(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    print(f'The d2 layer shape is {d2.shape}')
    
    #d1 = Conv2DTranspose(128, (kernel_size_e3, kernel_size_e3), strides=(2,2), padding='same')(d2)
    d1 = UpSampling2D(size=(1, 2))(d2)
    #d1 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 1, x.shape[2] * 2), method='bilinear'))(d2)
    d1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d1)
    d1 = BatchNormalization()(d1)
    d1 = Concatenate()([d1, e1])
    d1 = Dropout(drop_rate)(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    print(f'The d1 layer shape is {d1.shape}')
    
    d0 = Conv2DTranspose(32, (3,3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d1)
    print(f'The d0 layer shape is {d0.shape}')

    #Adding the predictions to the topographie
    x = Concatenate()([d0, wind])
    x = Dropout(drop_rate)(x)
    x = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02))(x)
    outputs = LeakyReLU(alpha=alpha)(x)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo] , outputs=outputs)

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=[SSIMLoss], #SSIMLoss mean_squared_error 
        metrics=["mean_squared_error", psnr] #SSIMLoss mean_absolute_error
    )

    #Print the model complexity
    model.summary()

    return model

def deepRU_v1(units, activation, lr, kernel_size, kernel_size_e1, kernel_size_e2, kernel_size_e3, kernel_size_d1, kernel_size_d2, kernel_size_d3,
           num_hidden_layer, WIND, regL1_conv01, regL1_conv02, regL2_conv01, regL2_conv02, drop_rate, alpha):
    #Input spatiotemp and topo variables
    inputs = Input(shape=(32,32,19))
    inputs_topo = Input(shape=(64, 64, 3))

    predict_UV = Lambda(lambda x: x[...,0:1])(inputs)
    wind_UV = Lambda(lambda x: x[...,0:3])(inputs)
    wind_U = Lambda(lambda x: x[...,3:6])(inputs)
    wind_V = Lambda(lambda x: x[...,6:9])(inputs)
    temperature = Lambda(lambda x: x[...,9:12])(inputs)
    others = Lambda(lambda x: x[...,12:])(inputs)
    #Pre-encoder multiples variables values
    #Wind UV multiples levels
    wind_UV = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros')(wind_UV)
    wind_UV = BatchNormalization()(wind_UV)
    wind_UV = LeakyReLU(alpha=alpha)(wind_UV)
    #Wind U multiples levels
    wind_U = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros')(wind_U)
    wind_U = BatchNormalization()(wind_U)
    wind_U = LeakyReLU(alpha=alpha)(wind_U)
    #Wind V multiples levels
    wind_V = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros')(wind_V)
    wind_V = BatchNormalization()(wind_V)
    wind_V = LeakyReLU(alpha=alpha)(wind_V)
    #Temperature multiples levels
    temperature = Conv2D(32, (3 ,3), padding='same', use_bias=True, bias_initializer='zeros')(temperature)
    temperature = BatchNormalization()(temperature)
    temperature = LeakyReLU(alpha=alpha)(temperature)
    #Combine predicted variables
    spatio_temp = Concatenate()([predict_UV, wind_UV, wind_U, wind_V, temperature, others])
    print(f'The spatio_temp shape is {spatio_temp.shape}')

    #Pre-encoder topographie predictor
    topo = Conv2D(18, (3 ,3), strides=(1, 2), padding='same', use_bias=True, bias_initializer='zeros')(inputs_topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(36, (3 ,5), strides=(2, 1), padding='same', use_bias=True, bias_initializer='zeros')(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)

    topo = Conv2D(72, (3 ,5), padding='same', use_bias=True, bias_initializer='zeros')(topo)
    topo = BatchNormalization()(topo)
    topo = LeakyReLU(alpha=alpha)(topo)
    print(f'The topo shape is {topo.shape}')

    #Concatenate the spatio-temporal and topographie variables
    inp = Concatenate()([spatio_temp, topo])
    # inp = UpSampling2D(size=(2, 2))(inp)
    # print(f'The inp shape is {inp.shape}')
    # #Skip connection with only wind data
    # wind = Lambda(lambda x: x[...,0:1])(inp)

    #Encoder
    e1 = Conv2D(32, (3, 3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(inp)
    e1 = BatchNormalization()(e1)
    e1 = LeakyReLU(alpha=alpha)(e1)
    print(f'The e1 layer shape is {e1.shape}')

    e2 = Conv2D(64, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(alpha=alpha)(e2)
    print(f'The e2 layer shape is {e2.shape}')

    e3 = Conv2D(128, (3, 3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(alpha=alpha)(e3)
    print(f'The e3 layer shape is {e3.shape}')

    e4 = Conv2D(192, (3, 3), strides=(1,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e3)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(alpha=alpha)(e4)
    print(f'The e4 layer shape is {e4.shape}')

    e5 = Conv2D(256, (3, 3), strides=(2,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e4)
    e5 = BatchNormalization()(e5)
    e5 = LeakyReLU(alpha=alpha)(e5)
    print(f'The e5 layer shape is {e5.shape}')

    #Bottleneck layer
    x = Conv2D(320, (3,3), strides=(2,2), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(e5)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    x = LeakyReLU(alpha=alpha)(x)
    print(f'The x layer shape is {x.shape}')

    #Decoder
    #d3 = Conv2DTranspose(512, (kernel_size_e1, kernel_size_e1), strides=(2,2), padding='same')(x)
    d5 = UpSampling2D(size=(2, 2))(x)
    #d5 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear'))(x)
    d5 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d5)
    d5 = BatchNormalization()(d5)
    d5 = Concatenate()([d5, e5])
    d5 = Dropout(drop_rate)(d5)
    d5 = LeakyReLU(alpha=alpha)(d5)
    print(f'The d5 layer shape is {d5.shape}')
    
    #d2 = Conv2DTranspose(256, (kernel_size_e2, kernel_size_e2), strides=(2,2), padding='same')(d3)
    d4 = UpSampling2D(size=(2, 2))(d5)
    #d4 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear'))(d5)
    d4 = Conv2D(192, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d4)
    d4 = BatchNormalization()(d4)
    d4 = Concatenate()([d4, e4])
    d4 = Dropout(drop_rate)(d4)
    d4 = LeakyReLU(alpha=alpha)(d4)
    print(f'The d4 layer shape is {d4.shape}')

    #d3 = Conv2DTranspose(512, (kernel_size_e1, kernel_size_e1), strides=(2,2), padding='same')(x)
    d3 = UpSampling2D(size=(1, 2))(d4)
    #d3 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 1, x.shape[2] * 2), method='bilinear'))(d4)
    d3 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d3)
    d3 = BatchNormalization()(d3)
    d3 = Concatenate()([d3, e3])
    d3 = Dropout(drop_rate)(d3)
    d3 = LeakyReLU(alpha=alpha)(d3)
    print(f'The d3 layer shape is {d3.shape}')
    
    #d2 = Conv2DTranspose(256, (kernel_size_e2, kernel_size_e2), strides=(2,2), padding='same')(d3)
    d2 = UpSampling2D(size=(2, 1))(d3)
    #d2 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 1), method='bilinear'))(d3)
    d2 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d2)
    d2 = BatchNormalization()(d2)
    d2 = Concatenate()([d2, e2])
    d2 = Dropout(drop_rate)(d2)
    d2 = LeakyReLU(alpha=alpha)(d2)
    print(f'The d2 layer shape is {d2.shape}')
    
    #d1 = Conv2DTranspose(128, (kernel_size_e3, kernel_size_e3), strides=(2,2), padding='same')(d2)
    d1 = UpSampling2D(size=(1, 2))(d2)
    #d1 = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 1, x.shape[2] * 2), method='bilinear'))(d2)
    d1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d1)
    d1 = BatchNormalization()(d1)
    d1 = Concatenate()([d1, e1])
    d1 = Dropout(drop_rate)(d1)
    d1 = LeakyReLU(alpha=alpha)(d1)
    print(f'The d1 layer shape is {d1.shape}')
    
    d0 = Conv2DTranspose(32, (3,3), strides=(2,1), padding='same', kernel_regularizer=l1_l2(l1=regL1_conv01, l2=regL2_conv02))(d1)
    print(f'The d0 layer shape is {d0.shape}')

    #Adding the predictions to the topographie
    x = Concatenate()([d0, predict_UV])
    x = UpSampling2D(size=(2, 2))(x)
    x = Dropout(drop_rate)(x)
    x = Conv2D(1, (3,3), padding='same', use_bias=True, bias_initializer='zeros', kernel_regularizer=l1_l2(l1=regL1_conv02, l2=regL2_conv02))(x)
    outputs = LeakyReLU(alpha=alpha)(x)
    
    #Define the input-output of the model 
    model = keras.Model(inputs=[inputs, inputs_topo] , outputs=outputs)

    #Define the optimizer, the loss function and the metrics that will be used while training the NN. 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=[SSIMLoss], #SSIMLoss mean_squared_error 
        metrics=["mean_squared_error", psnr] #SSIMLoss mean_absolute_error
    )

    #Print the model complexity
    model.summary()

    return model

# # Custom loss metrics
# def SSIMLoss(y_true, y_pred):
#     # Calculate the maximum absolute value in each image
#     max_pixel_true = tf.reduce_max(tf.abs(y_true))
#     max_pixel_pred = tf.reduce_max(tf.abs(y_pred))
    
#     max_pixel = tf.maximum(max_pixel_true, max_pixel_pred)
    
#     return tf.image.ssim(y_true, y_pred, max_val=max_pixel)

# def psnr(y_true, y_pred):
#     # Calculate the maximum absolute value in each image
#     max_pixel_true = tf.reduce_max(tf.abs(y_true))
#     max_pixel_pred = tf.reduce_max(tf.abs(y_pred))

#     max_val = tf.maximum(max_pixel_true, max_pixel_pred)
    
#     return tf.image.psnr(y_true, y_pred, max_val=max_val)

# def ssim_loss(y_true, y_pred):
#     ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
#     return (1 - ssim) / 2

# # Combined SSIM and MSE loss function
# def combined_loss(y_true, y_pred):
#     mse = MeanSquaredError()(y_true, y_pred)
#     ssim = ssim_loss(y_true, y_pred)
#     return mse + ssim








# Define PSNR metric
def psnr(y_true, y_pred):
    max_pixel_true = tf.reduce_max(tf.abs(y_true))
    max_pixel_pred = tf.reduce_max(tf.abs(y_pred))
    max_val = tf.maximum(max_pixel_true, max_pixel_pred)
    return tf.image.psnr(y_true, y_pred, max_val=1)

# Define SSIM loss function
def ssim_loss(y_true, y_pred):
    max_pixel_true = tf.reduce_max(tf.abs(y_true))
    max_pixel_pred = tf.reduce_max(tf.abs(y_pred))

    max_pixel = tf.maximum(max_pixel_true, max_pixel_pred)

    ssim = tf.image.ssim(y_true, y_pred, max_val=1)
    return (1 - ssim) / 2

# Define SSIM metric
def ssim_metric(y_true, y_pred):
    # Calculate the maximum absolute value in each image
    max_pixel_true = tf.reduce_max(tf.abs(y_true))
    max_pixel_pred = tf.reduce_max(tf.abs(y_pred))
    
    max_pixel = tf.maximum(max_pixel_true, max_pixel_pred)
    
    return tf.image.ssim(y_true, y_pred, max_val=1)

# Combined SSIM and MSE loss function
def combined_loss(y_true, y_pred, loss_weights):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mse = MeanSquaredError()(y_true, y_pred)
    y_true_normalized = normalize_data(y_true)
    y_pred_normalized = normalize_data(y_pred)
    ssim = 1 - tf.image.ssim(y_true_normalized, y_pred_normalized, max_val=1)
    return mse + (loss_weights * ssim)  #mse +     # 0.7 * mse + ((1-0.7) * ssim)

# Custom loss function to calculate combined loss only in the central region
def central_region_combined_loss(margin, loss_weights):
    def loss(y_true, y_pred):
        # Assuming y_true and y_pred have shape (batch_size, height, width, channels)
        y_true_central = y_true #[:, margin:-margin, margin:-margin, :]
        y_pred_central = y_pred #[:, margin:-margin, margin:-margin, :]
        return combined_loss(y_true_central, y_pred_central, loss_weights)
    return loss

# Custom metric function for PSNR in the central region
def central_region_psnr(margin):
    def PSNR(y_true, y_pred):
        y_true_central = y_true #[:, margin:-margin, margin:-margin, :]
        y_pred_central = y_pred #[:, margin:-margin, margin:-margin, :]
        y_true_central_normalized = normalize_data(y_true_central)
        y_pred_central_normalized = normalize_data(y_pred_central)
        return psnr(y_true_central_normalized, y_pred_central_normalized)
    return PSNR

# Custom metric function for SSIM in the central region
def central_region_ssim(margin):
    def SSIMLoss(y_true, y_pred):
        y_true_central = y_true #[:, margin:-margin, margin:-margin, :]
        y_pred_central = y_pred #[:, margin:-margin, margin:-margin, :]
        y_true_central_normalized = normalize_data(y_true_central)
        y_pred_central_normalized = normalize_data(y_pred_central)
        return tf.image.ssim(y_true_central_normalized, y_pred_central_normalized, max_val=1)
    return SSIMLoss

def central_region_mse(margin):
    def mse(y_true, y_pred):
        y_true_central = y_true #[:, margin:-margin, margin:-margin, :]
        y_pred_central = y_pred #[:, margin:-margin, margin:-margin, :]
        mse_centered = MeanSquaredError()(y_true_central, y_pred_central)
        return mse_centered
    return mse

# Normalize data for ssim and psnr calculation
def normalize_data(data):
    min_val = tf.reduce_min(data, axis=(1, 2, 3), keepdims=True)
    max_val = tf.reduce_max(data, axis=(1, 2, 3), keepdims=True)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

#Put a fully connected layer here to ajust the image size. 
# # Augmentation des canaux pour la convolution de sous-pixels
# # Pour un facteur d'upsampling de 2, augmentez les canaux par un facteur de 4 (2^2)
# x = Conv2D(4, (3, 3), padding='same', use_bias=True, name="SubPixelConv")(x)

# # Réarrangement des pixels (Pixel Shuffle)
# x = Lambda(pixel_shuffle(scale=2), name="PixelShuffle")(x)

# #Et pour gerer les images qui ne sont pas des multiples:
# # Option 1 : Interpolation légère
# x = tf.image.resize(x, [128, 128], method='bilinear')  # Interpolation bilinéaire

# # OU

# # Option 2 : Padding
# x = ZeroPadding2D(padding=((2, 2), (2, 2)))(x)