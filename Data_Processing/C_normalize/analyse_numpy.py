#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


inputs = np.load('/home/jfg000/ss5/Data/input/UU_VV_TT_P0_H_CX_SD_data/normalize.npy')
label = np.load('/home/jfg000/ss5/Data/label/UU_data/normalize.npy')
inputs = inputs[0:30, :, :, :]

#Display the input and label
for i in range(9):
    #Reshaping the arrays for quiver plot
    U_inputs = inputs[0, i, :, :]

    fig, axes = plt.subplots(1,2)
    fig.suptitle("Original data")
    axes[0].set_title(f"inputs{i}")
    axes[0].imshow(U_inputs, origin = "lower", cmap='magma')

    # Optionally, add a colorbar
    fig.colorbar(plt.cm.ScalarMappable(cmap='magma'), ax=axes.ravel().tolist())

    plt.show()

print('Finish')