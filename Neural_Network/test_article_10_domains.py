#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.fft import fft2, fftshift
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import model as mod

#Functions
def de_standardize(test_data_input, predictions, test_data_label, skip_test_data, std_inputs_de_standardize_path, mean_inputs_de_standardize_path, 
                   std_label_de_standardize_path, mean_label_de_standardize_path, std_skip_de_standardize_path, mean_skip_de_standardize_path):
    std_inputs = np.load(std_inputs_de_standardize_path)       
    mean_inputs = np.load(mean_inputs_de_standardize_path)       
    std_label = np.load(std_label_de_standardize_path)       
    mean_label = np.load(mean_label_de_standardize_path)      
    std_skip = np.load(std_skip_de_standardize_path)   
    mean_skip = np.load(mean_skip_de_standardize_path)   
    
    # Ensure std and mean are broadcastable to the data shape
    assert std_inputs.shape == mean_inputs.shape, "Standard deviation and mean file shapes do not match."

    # De-standardize inputs, using only the first channel stats
    mean_inputs_adjusted = np.transpose(mean_inputs, (0, 2, 3, 1)) 
    std_inputs_adjusted = np.transpose(std_inputs, (0, 2, 3, 1)) 
    print(f'The original_input shape is: {test_data_input.shape}')
    print(f'The std_inputs shape is: {std_inputs.shape} and the mean_inputs shape is: {mean_inputs.shape}')
    print(f'The std_inputs_adjusted shape is: {std_inputs_adjusted.shape} and the mean_inputs_adjusted shape is: {mean_inputs_adjusted.shape}')
    original_inputs = (test_data_input * std_inputs_adjusted[..., 0:1]) + mean_inputs_adjusted[..., 0:1]

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

def crop_result(test_data_input, predictions, test_data_label, test_data_skip_de_standardized, verif_flag=1):
    print(f'The data shape of test_data_input is : {test_data_input.shape}')
    print(f'The data shape of predictions is : {predictions.shape}')
    print(f'The data shape of test_data_label is : {test_data_label.shape}')
    print(f'The data shape of test_data_skip_de_standardized is : {test_data_skip_de_standardized.shape}')
    test_data_input = test_data_input[:, :, :, :]
    predictions = predictions[:, 4:-4, 4:-4, :]
    if not verif_flag:
        test_data_label = test_data_label[:, 4:-4, 4:-4, :]
        test_data_skip_de_standardized = test_data_skip_de_standardized[:, 4:-4, 4:-4, :]

    if verif_flag:
        return test_data_input, predictions
    else:
        return test_data_input, predictions, test_data_label, test_data_skip_de_standardized
    
def create_prediction_image(images, test_data_input, predictions, test_data_label, index,
                            mae, mse, rmse, rmse_benchmark, psnr, ssim, SSIM_benchmark, HP_RESULT_PATH, search_type):
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

    # Save the figure
    directory = HP_RESULT_PATH + '/image' + '/CNN_Results' + search_type
    if not os.path.exists(directory):
        os.makedirs(directory)

    axes[2].get_figure().savefig(f'{directory}/Ranking_{images+1}.png')
    plt.close()

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

def power_spectral_density_graph_one_image(predicted_image1, ground_truth_image, directory, images, test_data_skip_de_standardized):
    # print(f'The shape of the ground truth image is : {ground_truth_image[images, :, :, 0].shape}')
    # print(f'The shape of the predicted_image is : {predicted_image[images, :, :, 0].shape}')
    psd_ground_truth = compute_psd(ground_truth_image[images, :, :, 0])
    psd_predicted = compute_psd(predicted_image1[images, :, :, 0])
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
    plt.plot(freqs, psd_pred_db[:len(freqs)], label='Predicted no interpolation')
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
    print(f'The power spectral density number of images to regenerate is: {predicted_images.shape}')
    
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
    plt.plot(freqs, avg_psd_pred_db[:len(freqs)], label='Predicted No interpolation')
    plt.plot(freqs, avg_psd_benchmark_db[:len(freqs)], label='BenchMark')
    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Comparison of Average Power Spectral Densities in dB')
    plt.yscale('log')  # Set the y-axis to log scale
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/average_psd.png')
    plt.close()

# Main 
if __name__ == '__main__':
    #VARIABLES
    # Hyper-parameters used for each model type
    # General path for all type of strategies:
    # Path to the test files
    label_file = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    skip_file = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_test_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    # Path to the std deviation and mean to de-standardize the input and label. 
    std_label_de_standardize_path = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_label_de_standardize_path = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    std_skip_de_standardize_path = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_skip_de_standardize_path = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    # Path to save the output prediction 
    HP_RESULT_PATH = f"/home/jfg000/ss5/CNN_results/articles_testing/test_new_architecture_cad_10domaines" 

    # Parameter for no interpolation: 
    lr_no_interp = 0.0006
    drop_rate_no_interp = 0.2
    alpha_no_interp = 0.2
    loss_weights_no_interp = 0.8
    # Path for model weights
    path_weights_no_interp = '/home/jfg000/ss5/CNN_results/no_interpolation_article_new_model_cad_10dom_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_00000000_HP_Search/model/Searchmodel_1'
    # Input data to test the model with 
    input_file_test_no_inter = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    # Path for topo data
    topo_no_inter = '/home/jfg000/ss5/data_superResolution/topography/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data/normalize.npy'
    # Std deviation and mean to de-standardize the prediction
    std_inputs_de_standardize_path_no_inter = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_inputs_de_standardize_path_no_inter = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_10dom_cad/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    
    # Create the models that will be used to do prediction on 
    model_no_interpolation = mod.deepRU_article_non_interpolation_combine_loss(units=0, activation=None, lr=lr_no_interp, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, 
                                                                            kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0, num_hidden_layer=0, WIND=None, 
                                                                            regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, drop_rate=drop_rate_no_interp, alpha=alpha_no_interp, loss_weights=loss_weights_no_interp)

    # Load the model to do prediction on: 
    # Load model weights for no interpolation, bilinear and nearest neighbour
    model_no_interpolation.load_weights(path_weights_no_interp)

    # Load the test data
    #Loading the label data
    label_test = np.load(label_file)
    print(f'The labels_train is: {label_test.shape}')
    #Loading skip connection data
    skip_test = np.load(skip_file)
    print(f'The skip_train is: {skip_test.shape}')

    # Load test data for specific strategie
    print('load the input data for non interpolation')
    inputs_test_no_inter = np.load(input_file_test_no_inter)
    print(f'The input shape is: {inputs_test_no_inter.shape}')
    # Loading the topo data
    topo_no_inter = np.load(topo_no_inter)
    print(f'The topo_test is : {topo_no_inter.shape}')

    # Reshape the test data [sample, width, height, channel] 
    inputs_test_no_inter = inputs_test_no_inter.transpose((0,2,3,1)) 
    label_test = label_test.transpose((0,2,3,1)) 
    skip_test = skip_test.transpose((0,2,3,1)) 
    topo_no_inter = topo_no_inter.transpose((0,2,3,1))

    # # Adjust the topographie if the user is only testing on a single domain
    # topo_no_inter = np.tile(topo_no_inter, (inputs_test_no_inter.shape[0], 1, 1, 1))
    # print(f'The shape of inputs_test_no_inter is: {inputs_test_no_inter.shape}')
    # print(f'The shape of topo_train_reshaped is : {topo.shape}')
    # Adjust the topographie if the user is testing multiples domains
    topo_no_inter = np.tile(topo_no_inter, (int(inputs_test_no_inter.shape[0] / 10), 1, 1, 1))
    print(f'The shape of inputs_test_no_inter is: {inputs_test_no_inter.shape}')
    print(f'The shape of topo_train_reshaped is : {topo_no_inter.shape}')

    # Do the prediction 
    # Prediction for non interpolation 
    predictions_no_inter = model_no_interpolation.predict(
        x=[inputs_test_no_inter, topo_no_inter, skip_test],
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )

    # De-standardize the selected test input and label data. 
    # Do it for Non interpolation, bilinear interpolation and nearest neighbour interpolation
    inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, skip_test_de_standardized = de_standardize(inputs_test_no_inter, predictions_no_inter, label_test, skip_test, 
                                                                                                                                                        std_inputs_de_standardize_path_no_inter, mean_inputs_de_standardize_path_no_inter,
                                                                                                                                                        std_label_de_standardize_path, mean_label_de_standardize_path, 
                                                                                                                                                        std_skip_de_standardize_path, mean_skip_de_standardize_path)

    # Remove the padding pixels for the the input, prediction, label image and skip connection 
    # Do it for Non interpolation, bilinear interpolation and nearest neighbour interpolation
    verif_flag = 0
    inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, skip_test_de_standardized = crop_result(inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, skip_test_de_standardized, verif_flag)

    # Initialize numpy array to store the metric values
    NUM_TRAIN_REGENERATE = len(inputs_test_no_inter_de_standardized)
    RMSE_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    # Non interpolation 
    MAE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    MSE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    RMSE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    PSNR_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  

    # Itterate over all the images predicted, analyse them, plot them and calculate metrics
    for images_prediction in range(NUM_TRAIN_REGENERATE):
        print(f'Currently analysing image: {images_prediction+1}')
        # Calculate metrics between the prediction of the Neural Network and the label image 
        data_range = label_test_de_standardized[images_prediction].max() - label_test_de_standardized[images_prediction].min()
        RMSE_benchmark[images_prediction] = np.sqrt(np.mean(np.square(skip_test_de_standardized[images_prediction] - label_test_de_standardized[images_prediction])))
        SSIM_benchmark[images_prediction] = ssim(label_test_de_standardized[images_prediction].squeeze(), skip_test_de_standardized[images_prediction].squeeze(), data_range=data_range)
        # No interpolation 
        MAE_ni[images_prediction] = np.mean(np.abs(predictions_no_inter_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        MSE_ni[images_prediction] = np.mean(np.square(predictions_no_inter_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        RMSE_ni[images_prediction] = np.sqrt(np.mean(np.square(predictions_no_inter_de_standardized[images_prediction] - label_test_de_standardized[images_prediction])))
        data_range = label_test_de_standardized[images_prediction].max() - label_test_de_standardized[images_prediction].min()
        PSNR_ni[images_prediction] = psnr(label_test_de_standardized[images_prediction], predictions_no_inter_de_standardized[images_prediction], data_range=data_range)
        SSIM_ni[images_prediction] = ssim(label_test_de_standardized[images_prediction].squeeze(), predictions_no_inter_de_standardized[images_prediction].squeeze(), data_range=data_range)

        if images_prediction < 30:
            # Create prediction image for non interpolation 
            create_prediction_image(images_prediction, inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, 0,
                                    MAE_ni[images_prediction], MSE_ni[images_prediction], RMSE_ni[images_prediction], RMSE_benchmark[images_prediction], 
                                    PSNR_ni[images_prediction], SSIM_ni[images_prediction], SSIM_benchmark[images_prediction], HP_RESULT_PATH, 'no_interpolation')
            
            # Compute the power spectral graphic for the three interpolation strategies
            power_spectral_density_graph_one_image(predictions_no_inter_de_standardized, label_test_de_standardized, HP_RESULT_PATH, images_prediction, skip_test_de_standardized)

    # Compute the average Power spectal on the whole test set
    power_spectral_density_graph_average(predictions_no_inter_de_standardized, label_test_de_standardized, HP_RESULT_PATH, skip_test_de_standardized)

    # Calculate the average value for every metric
    #For no interpolation
    mean_MAE_ni = np.mean(MAE_ni)
    mean_MSE_ni = np.mean(MSE_ni)
    mean_RMSE_ni = np.mean(RMSE_ni)
    mean_PSNR_ni = np.mean(PSNR_ni)
    mean_SSIM_ni = np.mean(SSIM_ni)
    #For benchMark
    mean_RMSE_benchmark = np.mean(RMSE_benchmark)
    mean_SSIM_benchmark = np.mean(SSIM_benchmark)
    #Plot the graph for the mean metric values
    fig, axes = plt.subplots(1, 4, figsize=(30, 6))
    fig.suptitle("Average Metrics Results", fontsize=20)

    titles = ["Non Interpolation", "Bilinear interpolation", "Nearest Neighbour interpolation", "benchmark"]
    cmaps = [None] 

    metrics_text_ni = f"Mean MAE: {mean_MAE_ni}\nMean MSE: {mean_MSE_ni}\nMean RMSE: {mean_RMSE_ni}\nMean PSNR: {mean_PSNR_ni}\nMean SSIM: {mean_SSIM_ni}"
    metrics_text_bm = f"Mean RMSE: {mean_RMSE_benchmark}\nMean SSIM: {mean_SSIM_benchmark}"


    axes[0].text(0.5, 0.5, metrics_text_ni, ha='center', va='center', fontsize=24, transform=axes[0].transAxes)
    axes[0].set_title(titles[0], fontsize=14)
    axes[0].axis('off')

    axes[1].text(0.5, 0.5, metrics_text_bm, ha='center', va='center', fontsize=24, transform=axes[1].transAxes)
    axes[1].set_title(titles[3], fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()

    # Save the figure
    directory = HP_RESULT_PATH + '/image'
    if not os.path.exists(directory):
        os.makedirs(directory)

    axes[1].get_figure().savefig(f'{directory}/average_metrics_images.png')
    plt.close()
