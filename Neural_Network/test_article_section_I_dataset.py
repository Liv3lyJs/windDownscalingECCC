#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.fft import fft2, fftshift
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import gaussian_kde

import model as mod

#Functions
def de_standardize(test_data_input, predictions, test_data_label, std_inputs_de_standardize_path, mean_inputs_de_standardize_path, 
                   std_label_de_standardize_path, mean_label_de_standardize_path, bm_bilinear, bm_flag=0):
    std_inputs = np.load(std_inputs_de_standardize_path)       
    mean_inputs = np.load(mean_inputs_de_standardize_path)       
    std_label = np.load(std_label_de_standardize_path)       
    mean_label = np.load(mean_label_de_standardize_path)       
    
    # Ensure std and mean are broadcastable to the data shape
    assert std_inputs.shape == mean_inputs.shape, "Standard deviation and mean file shapes do not match."

    # De-standardize inputs, using only the first channel stats
    mean_inputs_adjusted = np.transpose(mean_inputs, (0, 2, 3, 1)) 
    std_inputs_adjusted = np.transpose(std_inputs, (0, 2, 3, 1)) 
    print(f'The original_input shape is: {test_data_input.shape}')
    print(f'The std_inputs shape is: {std_inputs.shape} and the mean_inputs shape is: {mean_inputs.shape}')
    print(f'The std_inputs_adjusted shape is: {std_inputs_adjusted.shape} and the mean_inputs_adjusted shape is: {mean_inputs_adjusted.shape}')
    original_inputs = (test_data_input * std_inputs_adjusted[..., 0:1]) + mean_inputs_adjusted[..., 0:1]
    if bm_flag: 
        bm_bilinear = (bm_bilinear * std_inputs_adjusted[..., 0:1]) + mean_inputs_adjusted[..., 0:1]

    # De-standardize predictions and labels using the resized mean and std
    mean_label_adjusted = np.transpose(mean_label, (0, 2, 3, 1)) 
    std_label_adjusted = np.transpose(std_label, (0, 2, 3, 1)) 
    print(f'The standardized prediction shape is : {predictions.shape}')
    print(f'The std_label shape is : {std_label.shape} and the mean_label shape is : {mean_label.shape}')
    print(f'The std_label_adjusted shape is: {std_label_adjusted.shape} and the mean_label_adjusted shape is: {mean_label_adjusted.shape}')

    original_prediction = (predictions * std_label_adjusted[..., 0:1]) + mean_label_adjusted[..., 0:1]
    original_labels = (test_data_label * std_label_adjusted[..., 0:1]) + mean_label_adjusted[..., 0:1]

    if bm_flag: 
        return original_inputs, original_prediction, original_labels, bm_bilinear
    else:
        return original_inputs, original_prediction, original_labels

def crop_result(test_data_input, predictions, test_data_label, bm_bilinear_de_standardized, verif_flag=0):
    print(f'The data shape of test_data_input is : {test_data_input.shape}')
    print(f'The data shape of predictions is : {predictions.shape}')
    print(f'The data shape of test_data_label is : {test_data_label.shape}')
    print(f'The data shape of bm_bilinear_de_standardized is : {bm_bilinear_de_standardized.shape}')

    test_data_input = test_data_input[:, :, :, :]
    predictions = predictions[:, 4:-4, 4:-4, :]
    if verif_flag:
        test_data_label = test_data_label[:, 4:-4, 4:-4, :]
        bm_bilinear_de_standardized = bm_bilinear_de_standardized[:, 4:-4, 4:-4, :]

    if verif_flag:
        return test_data_input, predictions, test_data_label, bm_bilinear_de_standardized
    else:
        return test_data_input, predictions
        
def convert_node_msec(inputs_test, predictions, label, bm_bilinear_de_standardized, flag=0):
    factor = 0.514444

    inputs_test = inputs_test * factor
    predictions = predictions * factor
    if flag:
        label = label * factor
        bm_bilinear_de_standardized = bm_bilinear_de_standardized * factor

    if flag:
        return inputs_test, predictions, label, bm_bilinear_de_standardized
    else:
        return inputs_test, predictions
    
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
                im = ax.imshow(data[i], origin='lower', cmap=cmaps[i], aspect='equal',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
            else:
                im = ax.imshow(data[i], origin='lower', cmap=cmaps[i], aspect='equal')
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

def power_spectral_density_graph_one_image(predicted_image1, predicted_image2, predicted_image3, ground_truth_image, directory, images, test_data_skip_de_standardized):
    # print(f'The shape of the ground truth image is : {ground_truth_image[images, :, :, 0].shape}')
    # print(f'The shape of the predicted_image is : {predicted_image[images, :, :, 0].shape}')
    psd_ground_truth = compute_psd(ground_truth_image[images, :, :, 0])
    psd_predicted = compute_psd(predicted_image1[images, :, :, 0])
    psd_predicted2 = compute_psd(predicted_image2[images, :, :, 0])
    psd_predicted3 = compute_psd(predicted_image3[images, :, :, 0])
    psd_skip = compute_psd(test_data_skip_de_standardized[images, :, :, 0])
    psd_gt_azimuthal = azimuthal_average(psd_ground_truth)
    psd_pred_azimuthal = azimuthal_average(psd_predicted)
    psd_pred_azimuthal2 = azimuthal_average(psd_predicted2)
    psd_pred_azimuthal3 = azimuthal_average(psd_predicted3)
    psd_skip_azimuthal = azimuthal_average(psd_skip)
    
    # Convert PSD to dB
    psd_gt_db = 10 * np.log10(psd_gt_azimuthal)
    psd_pred_db = 10 * np.log10(psd_pred_azimuthal)
    psd_pred_db2 = 10 * np.log10(psd_pred_azimuthal2)
    psd_pred_db3 = 10 * np.log10(psd_pred_azimuthal3)
    psd_benchmark = 10 * np.log10(psd_skip_azimuthal)
    
    size = psd_ground_truth.shape[0]
    freqs = np.fft.fftfreq(size)[:size // 2]  # Compute frequencies in cycles per pixel
    freqs = freqs[freqs > 0]  # Remove zero frequency for log-log plot

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_gt_db[:len(freqs)], label='Ground Truth')
    plt.plot(freqs, psd_pred_db[:len(freqs)], label='Predicted no interpolation')
    plt.plot(freqs, psd_pred_db2[:len(freqs)], label='Predicted bilinear interpolation')
    plt.plot(freqs, psd_pred_db3[:len(freqs)], label='Predicted nearest neighbour interpolation')
    plt.plot(freqs, psd_benchmark[:len(freqs)], label='Baseline')
    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Comparison of Power Spectral Densities in dB')
    plt.yscale('log')  # Set the y-axis to log scale
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{directory}/psd_{images+1}.png')
    plt.close()

# Compute and plot the average power spectral density
def power_spectral_density_graph_average(predicted_images, predicted_images2, predicted_images3, ground_truth_images, directory, test_data_skip_de_standardized):
    num_images = predicted_images.shape[0]
    print(f'The power spectral density number of images to regenerate is: {predicted_images.shape}')
    
    psd_gt_azimuthals = []
    psd_pred_azimuthals = []
    psd_pred_azimuthals2 = []
    psd_pred_azimuthals3 = []
    psd_benchmark_azimuthals = []
    
    for i in range(num_images):
        psd_ground_truth = compute_psd(ground_truth_images[i, :, :, 0])
        psd_predicted = compute_psd(predicted_images[i, :, :, 0])
        psd_predicted2 = compute_psd(predicted_images2[i, :, :, 0])
        psd_predicted3 = compute_psd(predicted_images3[i, :, :, 0])
        psd_benchmark = compute_psd(test_data_skip_de_standardized[i, :, :, 0])
        
        psd_gt_azimuthals.append(azimuthal_average(psd_ground_truth))
        psd_pred_azimuthals.append(azimuthal_average(psd_predicted))
        psd_pred_azimuthals2.append(azimuthal_average(psd_predicted2))
        psd_pred_azimuthals3.append(azimuthal_average(psd_predicted3))
        psd_benchmark_azimuthals.append(azimuthal_average(psd_benchmark))
    
    # Compute the average PSD
    avg_psd_gt_azimuthal = np.mean(psd_gt_azimuthals, axis=0)
    avg_psd_pred_azimuthal = np.mean(psd_pred_azimuthals, axis=0)
    avg_psd_pred_azimuthal2 = np.mean(psd_pred_azimuthals2, axis=0)
    avg_psd_pred_azimuthal3 = np.mean(psd_pred_azimuthals3, axis=0)
    avg_psd_benchmark_azimuthal = np.mean(psd_benchmark_azimuthals, axis=0)
    
    # Convert PSD to dB
    avg_psd_gt_db = 10 * np.log10(avg_psd_gt_azimuthal)
    avg_psd_pred_db = 10 * np.log10(avg_psd_pred_azimuthal)
    avg_psd_pred_db2 = 10 * np.log10(avg_psd_pred_azimuthal2)
    avg_psd_pred_db3 = 10 * np.log10(avg_psd_pred_azimuthal3)
    avg_psd_benchmark_db = 10 * np.log10(avg_psd_benchmark_azimuthal)
    
    size = psd_ground_truth.shape[0]
    freqs = np.fft.fftfreq(size)[:size // 2]  # Compute frequencies in cycles per pixel
    freqs = freqs[freqs > 0]  # Remove zero frequency for log-log plot

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, avg_psd_gt_db[:len(freqs)], label='Ground Truth')
    plt.plot(freqs, avg_psd_pred_db[:len(freqs)], label='Predicted No interpolation')
    plt.plot(freqs, avg_psd_pred_db2[:len(freqs)], label='Predicted Bilinear interpolation')
    plt.plot(freqs, avg_psd_pred_db3[:len(freqs)], label='Predicted Nearest Neighbour interpolation')
    plt.plot(freqs, avg_psd_benchmark_db[:len(freqs)], label='Baseline')
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

def pdf(predicted_image1, predicted_image2, predicted_image3, ground_truth_image, directory, images, test_data_skip_de_standardized):
    # Flatten the data
    predict1_flatten = predicted_image1[images, :, :, 0].flatten()
    predict2_flatten = predicted_image2[images, :, :, 0].flatten()
    predict3_flatten = predicted_image3[images, :, :, 0].flatten()
    bench_mark = test_data_skip_de_standardized[images, :, :, 0].flatten() 
    ground_truth_flatten = ground_truth_image[images, :, :, 0].flatten()

    # Find max and min wind speed
    min_wind_speed = min(np.min(predicted_image1), np.min(predicted_image2), np.min(predicted_image3), np.min(ground_truth_image), np.min(test_data_skip_de_standardized))
    max_wind_speed = max(np.max(predicted_image1), np.max(predicted_image2), np.max(predicted_image3), np.max(ground_truth_image), np.max(test_data_skip_de_standardized))
    # Set the wind speed range:
    wind_speed_range = np.linspace(min_wind_speed, max_wind_speed, 1000)

    # Calculate the kernel density estimation
    kde1 = gaussian_kde(predict1_flatten)
    kde2 = gaussian_kde(predict2_flatten)
    kde3 = gaussian_kde(predict3_flatten)
    kdebm = gaussian_kde(bench_mark)
    kdegt = gaussian_kde(ground_truth_flatten)

    # Calculate the PDF
    pdf1 = kde1(wind_speed_range)
    pdf2 = kde2(wind_speed_range)
    pdf3 = kde3(wind_speed_range)
    pdfbm = kdebm(wind_speed_range)
    pdfgt = kdegt(wind_speed_range)

    plt.figure(figsize=(10, 6))

    plt.plot(wind_speed_range, pdf1, label='No interpolation')
    plt.plot(wind_speed_range, pdf2, label='Bilinear interpolation')
    plt.plot(wind_speed_range, pdf3, label='Nearest-Neighbour interpolation')
    plt.plot(wind_speed_range, pdfbm, label='Bench mark')
    plt.plot(wind_speed_range, pdfgt, label='Ground truth')

    plt.xlabel('Wind Speed (m/sec)')
    plt.ylabel('PDF')
    plt.title('PDF of Wind Speed Predictions')
    plt.legend(loc='upper right')

    plt.savefig(f'{directory}/wind_speed_pdf{images+1}.png')
    plt.close()

def average_pdf(predicted_image1, predicted_image2, predicted_image3, ground_truth_image, directory, test_data_skip_de_standardized):
    # Flatten the data for all samples
    predict1_flatten = predicted_image1[:, :, :, 0].flatten()
    predict2_flatten = predicted_image2[:, :, :, 0].flatten()
    predict3_flatten = predicted_image3[:, :, :, 0].flatten()
    bench_mark = test_data_skip_de_standardized[:, :, :, 0].flatten() 
    ground_truth_flatten = ground_truth_image[:, :, :, 0].flatten()

    # Find min and max wind speeds
    min_wind_speed = min(np.min(predicted_image1), np.min(predicted_image2), np.min(predicted_image3), np.min(test_data_skip_de_standardized), np.min(ground_truth_image))
    max_wind_speed = max(np.max(predicted_image1), np.max(predicted_image2), np.max(predicted_image3), np.max(test_data_skip_de_standardized), np.max(ground_truth_image))

    # Set the wind speed range
    wind_speed_range = np.linspace(min_wind_speed, max_wind_speed, 400)

    # Calculate the kernel density estimation
    kde1 = gaussian_kde(predict1_flatten, bw_method=0.95)
    kde2 = gaussian_kde(predict2_flatten, bw_method=0.95)
    kde3 = gaussian_kde(predict3_flatten, bw_method=0.95)
    kdebm = gaussian_kde(bench_mark, bw_method=0.95)
    kdegt = gaussian_kde(ground_truth_flatten, bw_method=0.95)

    # Calculate the PDF for the specified range
    print(f'pdf 1')
    pdf1 = kde1(wind_speed_range)
    print(f'pdf 2')
    pdf2 = kde2(wind_speed_range)
    print(f'pdf 3')
    pdf3 = kde3(wind_speed_range)
    print(f'pdf bm')
    pdfbm = kdebm(wind_speed_range)
    print(f'pdf gt')
    pdfgt = kdegt(wind_speed_range)

    plt.figure(figsize=(10, 6))

    # Plot the PDFs with appropriate labels
    plt.plot(wind_speed_range, pdf1, label='Prediction 1')
    plt.plot(wind_speed_range, pdf2, label='Prediction 2')
    plt.plot(wind_speed_range, pdf3, label='Prediction 3')
    plt.plot(wind_speed_range, pdfbm, label='Baseline')
    plt.plot(wind_speed_range, pdfgt, label='Ground Truth')

    # Add labels, title, and legend
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('PDF')
    plt.title('Average PDF of Wind Speed Predictions')
    plt.legend(loc='upper right')

    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}//wind_speed_pdf.png')
    plt.close()

# Main 
if __name__ == '__main__':
    plt.rcParams['font.size'] = 14
    #VARIABLES
    # Hyper-parameters used for each model type
    # General path for all type of strategies:
    # Path to the test files
    label_file = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    skip_file = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    # Path to the std deviation and mean to de-standardize the input and label. 
    std_label_de_standardize_path = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_label_de_standardize_path = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    std_skip_de_standardize_path = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_skip_de_standardize_path = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    # Path to save the output prediction 
    HP_RESULT_PATH = f"/home/jfg000/ss5/CNN_results_article/articles_testing/section_I_alldomains_FINAL_01" 

    # Parameter for no interpolation: 
    lr_no_interp = 0.001
    drop_rate_no_interp = 0.30
    alpha_no_interp = 0.30
    loss_weights_no_interp = 0.50
    # Path for model weights
    path_weights_no_interp = '/home/jfg000/ss5/CNN_results_article/non_interpolation_qcnb_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000000_Final_search_HP_Search/model/Searchmodel_9_cb'
    # Input data to test the model with 
    input_file_test_no_inter = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    # Path for topo data
    topo_no_inter = '/home/jfg000/ss5/data_superResolution/topography/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data/normalize.npy'
    # Std deviation and mean to de-standardize the prediction
    std_inputs_de_standardize_path_no_inter = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_inputs_de_standardize_path_no_inter = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'

    # Parameter for bilinear interpolation:
    lr_bilinear = 0.001
    drop_bilinear = 0.25
    alpha_bilinear = 0.25
    loss_bilinear = 1.0
    # Path for model weights
    path_weights_bilinear = path_weights_bilinear = '/home/jfg000/ss5/CNN_results_article/bilinear_interpolation_qcnb_UU/bilinear_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000000_Final_search_HP_Search/model/Searchmodel_7_cb'
    # Input data to test the model with 
    input_file_test_bilinear = '/home/jfg000/ss5/data_superResolution/input/domaine/bilinear_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    # Path for topo data
    topo_inter = '/home/jfg000/ss5/data_superResolution/topography/bilinear_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data/normalize.npy'
    # Std deviation and mean to de-standardize the prediction
    std_inputs_de_standardize_path_bilinear = '/home/jfg000/ss5/data_superResolution/input/domaine/bilinear_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_inputs_de_standardize_path_bilinear = '/home/jfg000/ss5/data_superResolution/input/domaine/bilinear_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    
    # Parameter for nearest neighbour interpolation:
    lr_nearest_neighbour = 0.001
    drop_nearest_neighbour = 0.25
    alpha_nearest_neighbour = 0.25
    loss_nearest_neighbour = 1.0
    # Path for model weights
    path_weights_nearest_neighbour = '/home/jfg000/ss5/CNN_results_article/nearest_neighbour_interpolation_qcnb_UU/nearest_neighboor_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000000_Final_search_HP_Search/model/Searchmodel_2_cb'
    # Input data to test the model with 
    input_file_test_nearest_neighbour = '/home/jfg000/ss5/data_superResolution/input/domaine/nearestNeighbour_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    # Std deviation and mean to de-standardize the prediction
    std_inputs_de_standardize_path_nearest_neighbour = '/home/jfg000/ss5/data_superResolution/input/domaine/nearestNeighbour_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_inputs_de_standardize_path_nearest_neighbour = '/home/jfg000/ss5/data_superResolution/input/domaine/nearestNeighbour_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    
    # Create the models that will be used to do prediction on 
    model_no_interpolation = mod.deepRU_article_non_interpolation_combine_loss(units=0, activation=None, lr=lr_no_interp, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, 
                                                                            kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0, num_hidden_layer=0, WIND=None, 
                                                                            regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, drop_rate=drop_rate_no_interp, alpha=alpha_no_interp, loss_weights=loss_weights_no_interp)
    model_bilinear = mod.deepRU_article_interpolation_combine_loss(units=0, activation=None, lr=lr_bilinear, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0,
                                                                num_hidden_layer=0, WIND=None, regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, 
                                                                drop_rate=drop_bilinear, alpha=alpha_bilinear, loss_weights=loss_bilinear)
    model_nearest_neighbour = mod.deepRU_article_interpolation_combine_loss(units=0, activation=None, lr=lr_nearest_neighbour, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, 
                                                                            kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0, num_hidden_layer=0, WIND=None, 
                                                                            regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, 
                                                                            drop_rate=drop_nearest_neighbour, alpha=alpha_nearest_neighbour, loss_weights=loss_nearest_neighbour)
    # Load the model to do prediction on: 
    # Load model weights for no interpolation, bilinear and nearest neighbour
    model_no_interpolation.load_weights(path_weights_no_interp)
    model_bilinear.load_weights(path_weights_bilinear)
    model_nearest_neighbour.load_weights(path_weights_nearest_neighbour)

    # Load the test data
    #Loading the label data
    label_test = np.load(label_file)
    print(f'The labels_train is: {label_test.shape}')
    #Loading skip connection data
    skip_test = np.load(skip_file)
    print(f'The skip_train is: {skip_test.shape}')
    bm_bilinear = np.load(input_file_test_bilinear)
    print(f'The bm_bilinear is: {bm_bilinear.shape}')


    # Load test data for specific strategie
    print('load the input data for non interpolation')
    inputs_test_no_inter = np.load(input_file_test_no_inter)
    print(f'The input shape is: {inputs_test_no_inter.shape}')
    # Loading the topo data
    topo_no_inter = np.load(topo_no_inter)
    print(f'The topo_test is : {topo_no_inter.shape}')

    print('load the input data for bilinear')
    inputs_test_bilinear = np.load(input_file_test_bilinear)
    print(f'The input shape is: {inputs_test_bilinear.shape}')
    # Loading the topo data
    topo_inter = np.load(topo_inter)
    print(f'The topo_test is : {topo_inter.shape}')

    print('load the input data for nearest neighbour')
    inputs_test_nearest_neighbour = np.load(input_file_test_nearest_neighbour)
    print(f'The input shape is: {inputs_test_nearest_neighbour.shape}')

    # Reshape the test data [sample, width, height, channel] 
    inputs_test_no_inter = inputs_test_no_inter.transpose((0,2,3,1)) 
    inputs_test_bilinear = inputs_test_bilinear.transpose((0,2,3,1)) 
    inputs_test_nearest_neighbour = inputs_test_nearest_neighbour.transpose((0,2,3,1)) 
    label_test = label_test.transpose((0,2,3,1)) 
    skip_test = skip_test.transpose((0,2,3,1)) 
    topo_no_inter = topo_no_inter.transpose((0,2,3,1))
    topo_inter = topo_inter.transpose((0,2,3,1))

    bm_bilinear = bm_bilinear.transpose((0,2,3,1))[:, 8:56, 8:56, 0]
    bm_bilinear = bm_bilinear[:, :, :, np.newaxis]
    print(f'New bm_bilinear shape is: {bm_bilinear.shape}')

    assert inputs_test_no_inter.shape[0] == inputs_test_bilinear.shape[0] == inputs_test_nearest_neighbour.shape[0], 'There is not the same number of input data for both files'

    # # Adjust the topographie if the user is only testing on a single domain
    # topo_no_inter = np.tile(topo_no_inter, (inputs_test_no_inter.shape[0], 1, 1, 1))
    # print(f'The shape of inputs_test_no_inter is: {inputs_test_no_inter.shape}')
    # print(f'The shape of topo_train_reshaped is : {topo.shape}')
    # Adjust the topographie if the user is testing multiples domains
    topo_no_inter = np.tile(topo_no_inter, (int(inputs_test_no_inter.shape[0] / 5), 1, 1, 1))
    print(f'The shape of inputs_test_no_inter is: {inputs_test_no_inter.shape}')
    print(f'The shape of topo_train_reshaped is : {topo_no_inter.shape}')

    topo_inter = np.tile(topo_inter, (int(inputs_test_no_inter.shape[0] / 5), 1, 1, 1))
    print(f'The shape of inputs_test_no_inter is: {inputs_test_no_inter.shape}')
    print(f'The shape of topo_train_reshaped is : {topo_inter.shape}')

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
    # Prediction for bilinear interpolation 
    predictions_bilinear = model_bilinear.predict(
        x=[inputs_test_bilinear, topo_inter, skip_test],
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    # Prediction for nearest neighbour interpolation 
    predictions_nearest_neighbour = model_nearest_neighbour.predict(
        x=[inputs_test_nearest_neighbour, topo_inter, skip_test],
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    print(f'The shape of the prediction for no inter: {predictions_no_inter.shape}')
    print(f'The shape of the prediction for no bi: {predictions_bilinear.shape}')
    print(f'The shape of the prediction for no NN: {predictions_nearest_neighbour.shape}')

    # De-standardize the selected test input and label data. 
    # Do it for Non interpolation, bilinear interpolation and nearest neighbour interpolation
    inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized = de_standardize(inputs_test_no_inter, predictions_no_inter, label_test, 
                                                                                                                                                         std_inputs_de_standardize_path_no_inter, mean_inputs_de_standardize_path_no_inter,
                                                                                                                                                         std_label_de_standardize_path, mean_label_de_standardize_path, bm_bilinear, 1)
    inputs_test_bilinear_de_standardized, predictions_bilinear_de_standardized, label_test_de_standardized_1 =                            de_standardize(inputs_test_bilinear, predictions_bilinear, label_test, 
                                                                                                                                                         std_inputs_de_standardize_path_bilinear, mean_inputs_de_standardize_path_bilinear,
                                                                                                                                                         std_label_de_standardize_path, mean_label_de_standardize_path, bm_bilinear)
    inputs_test_nearest_neighbour_de_standardized, predictions_nearest_neighbour_de_standardized, label_test_de_standardized_2 =          de_standardize(inputs_test_nearest_neighbour, predictions_nearest_neighbour, label_test, 
                                                                                                                                                         std_inputs_de_standardize_path_nearest_neighbour, mean_inputs_de_standardize_path_nearest_neighbour,
                                                                                                                                                         std_label_de_standardize_path, mean_label_de_standardize_path, bm_bilinear)

    # Remove the padding pixels for the the input, prediction, label image and skip connection 
    inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized = crop_result(inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized, 1)
    inputs_test_bilinear_de_standardized, predictions_bilinear_de_standardized = crop_result(inputs_test_bilinear_de_standardized, predictions_bilinear_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized)
    inputs_test_nearest_neighbour_de_standardized, predictions_nearest_neighbour_de_standardized = crop_result(inputs_test_nearest_neighbour_de_standardized, predictions_nearest_neighbour_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized)

    # Convert the units from knots to meters per seconds
    inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized = convert_node_msec(inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized, 1)
    inputs_test_bilinear_de_standardized, predictions_bilinear_de_standardized = convert_node_msec(inputs_test_bilinear_de_standardized, predictions_bilinear_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized)
    inputs_test_nearest_neighbour_de_standardized, predictions_nearest_neighbour_de_standardized = convert_node_msec(inputs_test_nearest_neighbour_de_standardized, predictions_nearest_neighbour_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized)

    # Initialize numpy array to store the metric values
    NUM_TRAIN_REGENERATE = len(inputs_test_no_inter_de_standardized)
    MAE_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)
    RMSE_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    # Non interpolation 
    MAE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    MSE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    RMSE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    PSNR_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    # Bilinear interpolation 
    MAE_b = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    MSE_b = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    RMSE_b = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    PSNR_b = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_b = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    # Nearest Neighbour interpolation 
    MAE_nn = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    MSE_nn = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    RMSE_nn = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    PSNR_nn = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_nn = np.full(NUM_TRAIN_REGENERATE, 0.0)  

    # Itterate over all the images predicted, analyse them, plot them and calculate metrics
    for images_prediction in range(NUM_TRAIN_REGENERATE):
        print(f'Currently analysing image: {images_prediction+1}')
        # _____Benchmark Metrics: 
        MAE_benchmark[images_prediction] = np.mean(np.abs(bm_bilinear_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        RMSE_benchmark[images_prediction] = np.sqrt(np.mean(np.square(bm_bilinear_de_standardized[images_prediction] - label_test_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted = bm_bilinear_de_standardized[images_prediction].squeeze()
        ground_truth = label_test_de_standardized[images_prediction].squeeze()
        max_value = max(predicted.max(), ground_truth.max())
        min_value = min(predicted.min(), ground_truth.min())
        data_rg = max_value - min_value
        predicted_std = (predicted - min_value) / data_rg
        ground_truth_std = (ground_truth - min_value) / data_rg
        # Calculate SSIM
        SSIM_benchmark[images_prediction] = ssim(predicted_std, ground_truth_std, data_range=1.0)

        # ______No interpolation  Metrics: 
        MAE_ni[images_prediction] = np.mean(np.abs(predictions_no_inter_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        MSE_ni[images_prediction] = np.mean(np.square(predictions_no_inter_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        RMSE_ni[images_prediction] = np.sqrt(np.mean(np.square(predictions_no_inter_de_standardized[images_prediction] - label_test_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted_ni = predictions_no_inter_de_standardized[images_prediction].squeeze()
        ground_truth_ni = label_test_de_standardized[images_prediction].squeeze()
        max_value_ni = max(predicted_ni.max(), ground_truth_ni.max())
        min_value_ni = min(predicted_ni.min(), ground_truth_ni.min())
        data_rg_ni = max_value_ni - min_value_ni
        predicted_std_ni = (predicted_ni - min_value_ni) / data_rg_ni
        ground_truth_std_ni = (ground_truth_ni - min_value_ni) / data_rg_ni
        # Calculate SSIM
        SSIM_ni[images_prediction] = ssim(predicted_std_ni, ground_truth_std_ni, data_range=1.0)
        # Standardize values to calculate psnr
        predicted_ni_p = predictions_no_inter_de_standardized[images_prediction]
        ground_truth_ni_p = label_test_de_standardized[images_prediction]
        max_value_ni_p = max(predicted_ni_p.max(), ground_truth_ni_p.max())
        min_value_ni_p = min(predicted_ni_p.min(), ground_truth_ni_p.min())
        data_rg_ni_p = max_value_ni_p - min_value_ni_p
        predicted_std_ni_p = (predicted_ni_p - min_value_ni_p) / data_rg_ni_p
        ground_truth_std_ni_p = (ground_truth_ni_p - min_value_ni_p) / data_rg_ni_p
        # Calculate PSNR
        PSNR_ni[images_prediction] = psnr(predicted_std_ni_p, ground_truth_std_ni_p, data_range=1.0)

        # _______bilinear interpolation Metrics: 
        MAE_b[images_prediction] = np.mean(np.abs(predictions_bilinear_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        MSE_b[images_prediction] = np.mean(np.square(predictions_bilinear_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        RMSE_b[images_prediction] = np.sqrt(np.mean(np.square(predictions_bilinear_de_standardized[images_prediction] - label_test_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted_b = predictions_bilinear_de_standardized[images_prediction].squeeze()
        ground_truth_b = label_test_de_standardized[images_prediction].squeeze()
        max_value_b = max(predicted_b.max(), ground_truth_b.max())
        min_value_b = min(predicted_b.min(), ground_truth_b.min())
        data_rg_b = max_value_b - min_value_b
        predicted_std_b = (predicted_b - min_value_b) / data_rg_b
        ground_truth_std_b = (ground_truth_b - min_value_b) / data_rg_b
        # Calculate SSIM
        SSIM_b[images_prediction] = ssim(predicted_std_b, ground_truth_std_b, data_range=1.0)
        # Standardize values to calculate psnr
        predicted_b_p = predictions_bilinear_de_standardized[images_prediction]
        ground_truth_b_p = label_test_de_standardized[images_prediction]
        max_value_b_p = max(predicted_b_p.max(), ground_truth_b_p.max())
        min_value_b_p = min(predicted_b_p.min(), ground_truth_b_p.min())
        data_rg_b_p = max_value_b_p - min_value_b_p
        predicted_std_b_p = (predicted_b_p - min_value_b_p) / data_rg_b_p
        ground_truth_std_b_p = (ground_truth_b_p - min_value_b_p) / data_rg_b_p
        # Calculate PSNR
        PSNR_b[images_prediction] = psnr(predicted_std_b_p, ground_truth_std_b_p, data_range=1.0)

        # ______Nearest neighbour Metrics: 
        MAE_nn[images_prediction] = np.mean(np.abs(predictions_nearest_neighbour_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        MSE_nn[images_prediction] = np.mean(np.square(predictions_nearest_neighbour_de_standardized[images_prediction] - label_test_de_standardized[images_prediction]))
        RMSE_nn[images_prediction] = np.sqrt(np.mean(np.square(predictions_nearest_neighbour_de_standardized[images_prediction] - label_test_de_standardized[images_prediction])))
        data_range = label_test_de_standardized[images_prediction].max() - label_test_de_standardized[images_prediction].min()
        # Standardize the values to calculate ssim
        predicted_nn = predictions_nearest_neighbour_de_standardized[images_prediction].squeeze()
        ground_truth_nn = label_test_de_standardized[images_prediction].squeeze()
        max_value_nn = max(predicted_nn.max(), ground_truth_nn.max())
        min_value_nn = min(predicted_nn.min(), ground_truth_nn.min())
        data_rg_nn = max_value_nn - min_value_nn
        predicted_std_nn = (predicted_nn - min_value_nn) / data_rg_nn
        ground_truth_std_nn = (ground_truth_nn - min_value_nn) / data_rg_nn
        # Calculate SSIM
        SSIM_nn[images_prediction] = ssim(predicted_std_nn, ground_truth_std_nn, data_range=1.0)
        # Standardize values to calculate psnr
        predicted_nn_p = predictions_nearest_neighbour_de_standardized[images_prediction]
        ground_truth_nn_p = label_test_de_standardized[images_prediction]
        max_value_nn_p = max(predicted_nn_p.max(), ground_truth_nn_p.max())
        min_value_nn_p = min(predicted_nn_p.min(), ground_truth_nn_p.min())
        data_rg_nn_p = max_value_nn_p - min_value_nn_p
        predicted_std_nn_p = (predicted_nn_p - min_value_nn_p) / data_rg_nn_p
        ground_truth_std_nn_p = (ground_truth_nn_p - min_value_nn_p) / data_rg_nn_p
        # Calculate PSNR
        PSNR_nn[images_prediction] = psnr(predicted_std_nn_p, ground_truth_std_nn_p, data_range=1.0)
        
        if images_prediction < 10:
            # Create prediction image for non interpolation 
            create_prediction_image(images_prediction, inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, 0,
                                    MAE_ni[images_prediction], MSE_ni[images_prediction], RMSE_ni[images_prediction], RMSE_benchmark[images_prediction], 
                                    PSNR_ni[images_prediction], SSIM_ni[images_prediction], SSIM_benchmark[images_prediction], HP_RESULT_PATH, 'no_interpolation')
            # Create prediction image for bilinear interpolation 
            create_prediction_image(images_prediction, inputs_test_bilinear_de_standardized, predictions_bilinear_de_standardized, label_test_de_standardized, 0,
                                    MAE_b[images_prediction], MSE_b[images_prediction], RMSE_b[images_prediction], RMSE_benchmark[images_prediction], 
                                    PSNR_b[images_prediction], SSIM_b[images_prediction], SSIM_benchmark[images_prediction], HP_RESULT_PATH, 'bilinear_interpolation')
            # Create prediction image for nearest neighbour interpolation 
            create_prediction_image(images_prediction, inputs_test_nearest_neighbour_de_standardized, predictions_nearest_neighbour_de_standardized, label_test_de_standardized, 0,
                                    MAE_nn[images_prediction], MSE_nn[images_prediction], RMSE_nn[images_prediction], RMSE_benchmark[images_prediction], 
                                    PSNR_nn[images_prediction], SSIM_nn[images_prediction], SSIM_benchmark[images_prediction], HP_RESULT_PATH, 'nearest_neighbour')
            
            # # Compute the power spectral graphic for the three interpolation strategies
            # power_spectral_density_graph_one_image(predictions_no_inter_de_standardized, predictions_bilinear_de_standardized, predictions_nearest_neighbour_de_standardized, label_test_de_standardized, HP_RESULT_PATH, images_prediction, bm_bilinear_de_standardized)
            # pdf(predictions_no_inter_de_standardized, predictions_bilinear_de_standardized, predictions_nearest_neighbour_de_standardized, label_test_de_standardized, HP_RESULT_PATH, images_prediction, bm_bilinear_de_standardized)

    # # Compute the average Power spectal on the whole test set
    # power_spectral_density_graph_average(predictions_no_inter_de_standardized, predictions_bilinear_de_standardized, predictions_nearest_neighbour_de_standardized, label_test_de_standardized, HP_RESULT_PATH, bm_bilinear_de_standardized)
    # average_pdf(predictions_no_inter_de_standardized, predictions_bilinear_de_standardized, predictions_nearest_neighbour_de_standardized, label_test_de_standardized, HP_RESULT_PATH, bm_bilinear_de_standardized)

    # Calculate the average value for every metric
    #For no interpolation
    mean_MAE_ni = np.mean(MAE_ni)
    median_MAE_ni = np.median(MAE_ni)
    q1_MAE_ni = np.percentile(MAE_ni, 25)
    q3_MAE_ni = np.percentile(MAE_ni, 75)
    std_MAE_ni = np.std(MAE_ni)

    mean_MSE_ni = np.mean(MSE_ni)
    median_MSE_ni = np.median(MSE_ni)
    q1_MSE_ni = np.percentile(MSE_ni, 25)
    q3_MSE_ni = np.percentile(MSE_ni, 75)
    std_mse_ni = np.std(MSE_ni)

    mean_RMSE_ni = np.mean(RMSE_ni)
    median_RMSE_ni = np.median(RMSE_ni)
    q1_RMSE_ni = np.percentile(RMSE_ni, 25)
    q3_RMSE_ni = np.percentile(RMSE_ni, 75)
    std_RMSE_ni = np.std(RMSE_ni)

    mean_PSNR_ni = np.mean(PSNR_ni)
    median_PSNR_ni = np.median(PSNR_ni)
    q1_PSNR_ni = np.percentile(PSNR_ni, 25)
    q3_PSNR_ni = np.percentile(PSNR_ni, 75)
    std_PSNR_ni = np.std(PSNR_ni)

    mean_SSIM_ni = np.mean(SSIM_ni)
    median_SSIM_ni = np.median(SSIM_ni)
    q1_SSIM_ni = np.percentile(SSIM_ni, 25)
    q3_SSIM_ni = np.percentile(SSIM_ni, 75)
    std_SSIM_ni = np.std(SSIM_ni)

    #For bilinear interpolation 
    mean_MAE_b = np.mean(MAE_b)
    median_MAE_b = np.median(MAE_b)
    q1_MAE_b = np.percentile(MAE_b, 25)
    q3_MAE_b = np.percentile(MAE_b, 75)
    std_MAE_b = np.std(MAE_b)

    mean_MSE_b = np.mean(MSE_b)
    median_MSE_b = np.median(MSE_b)
    q1_MSE_b = np.percentile(MSE_b, 25)
    q3_MSE_b = np.percentile(MSE_b, 75)
    std_MSE_b = np.std(MSE_b)

    mean_RMSE_b = np.mean(RMSE_b)
    median_RMSE_b = np.median(RMSE_b)
    q1_RMSE_b = np.percentile(RMSE_b, 25)
    q3_RMSE_b = np.percentile(RMSE_b, 75)
    std_RMSE_b = np.std(RMSE_b)

    mean_PSNR_b = np.mean(PSNR_b)
    median_PSNR_b = np.median(PSNR_b)
    q1_PSNR_b = np.percentile(PSNR_b, 25)
    q3_PSNR_b = np.percentile(PSNR_b, 75)
    std_PSNR_b = np.std(PSNR_b)

    mean_SSIM_b = np.mean(SSIM_b)
    median_SSIM_b = np.median(SSIM_b)
    q1_SSIM_b = np.percentile(SSIM_b, 25)
    q3_SSIM_b = np.percentile(SSIM_b, 75)
    std_SSIM_b = np.std(SSIM_b)

    #For nearest neighbour interpolation
    mean_MAE_nn = np.mean(MAE_nn)
    median_MAE_nn = np.median(MAE_nn)
    q1_MAE_nn = np.percentile(MAE_nn, 25)
    q3_MAE_nn = np.percentile(MAE_nn, 75)
    std_MAE_nn  = np.std(MAE_nn)

    mean_MSE_nn = np.mean(MSE_nn)
    median_MSE_nn = np.median(MSE_nn)
    q1_MSE_nn = np.percentile(MSE_nn, 25)
    q3_MSE_nn = np.percentile(MSE_nn, 75)
    std_MSE_nn  = np.std(MSE_nn)

    mean_RMSE_nn = np.mean(RMSE_nn)
    median_RMSE_nn = np.median(RMSE_nn)
    q1_RMSE_nn = np.percentile(RMSE_nn, 25)
    q3_RMSE_nn = np.percentile(RMSE_nn, 75)
    std_RMSE_nn  = np.std(RMSE_nn)

    mean_PSNR_nn = np.mean(PSNR_nn)
    median_PSNR_nn = np.median(PSNR_nn)
    q1_PSNR_nn = np.percentile(PSNR_nn, 25)
    q3_PSNR_nn = np.percentile(PSNR_nn, 75)
    std_PSNR_nn  = np.std(PSNR_nn)

    mean_SSIM_nn = np.mean(SSIM_nn)
    median_SSIM_nn = np.median(SSIM_nn)
    q1_SSIM_nn = np.percentile(SSIM_nn, 25)
    q3_SSIM_nn = np.percentile(SSIM_nn, 75)
    std_SSIM_nn  = np.std(SSIM_nn)

    #For Baseline
    mean_MAE_benchmark = np.mean(MAE_benchmark)
    median_MAE_benchmark = np.median(MAE_benchmark)
    q1_MAE_benchmark = np.percentile(MAE_benchmark, 25)
    q3_MAE_benchmark = np.percentile(MAE_benchmark, 75)
    std_MAE_benchmark  = np.std(MAE_benchmark)

    mean_RMSE_benchmark = np.mean(RMSE_benchmark)
    median_RMSE_benchmark = np.median(RMSE_benchmark)
    q1_RMSE_benchmark = np.percentile(RMSE_benchmark, 25)
    q3_RMSE_benchmark = np.percentile(RMSE_benchmark, 75)
    std_RMSE_benchmark  = np.std(RMSE_benchmark)

    mean_SSIM_benchmark = np.mean(SSIM_benchmark)
    median_SSIM_benchmark = np.median(SSIM_benchmark)
    q1_SSIM_benchmark = np.percentile(SSIM_benchmark, 25)
    q3_SSIM_benchmark = np.percentile(SSIM_benchmark, 75)
    std_SSIM_benchmark  = np.std(SSIM_benchmark)

    #Plot the graph for the mean metric values
    fig, axes = plt.subplots(1, 4, figsize=(30, 20))
    fig.suptitle("Average Metrics Results", fontsize=20)

    titles = ["Non Interpolation", "Bilinear interpolation", "Nearest Neighbour interpolation", "Baseline"]
    cmaps = [None] 

    metrics_text_ni = f"Mean MAE: {mean_MAE_ni:.2f}\nMedian MAE: {median_MAE_ni:.2f}\nQ1 MAE: {q1_MAE_ni:.2f}\nQ3 MAE: {q3_MAE_ni:.2f}\nMean RMSE: {mean_RMSE_ni:.2f}\n Std MAE: {std_MAE_ni:.2f}\nMedian RMSE: {median_RMSE_ni:.2f}\nQ1 RMSE: {q1_RMSE_ni:.2f}\nQ3 RMSE: {q3_RMSE_ni:.2f}\nStd RMSE: {std_RMSE_ni:.2f}\nMean SSIM: {mean_SSIM_ni:.2f}\nMedian SSIM: {median_SSIM_ni:.2f}\nQ1 SSIM: {q1_SSIM_ni:.2f}\nQ3 SSIM: {q3_SSIM_ni:.2f}\nStd SSIM: {std_SSIM_ni:.2f}"
    metrics_text_b = f"Mean MAE: {mean_MAE_b:.2f}\nMedian MAE: {median_MAE_b:.2f}\nQ1 MAE: {q1_MAE_b:.2f}\nQ3 MAE: {q3_MAE_b:.2f}\nMean RMSE: {mean_RMSE_b:.2f}\n Std MAE: {std_MAE_b:.2f}\nMean RMSE: {mean_RMSE_b:.2f}\nMedian RMSE: {median_RMSE_b:.2f}\nQ1 RMSE: {q1_RMSE_b:.2f}\nQ3 RMSE: {q3_RMSE_b:.2f}\nStd RMSE: {std_RMSE_b:.2f}\nMean SSIM: {mean_SSIM_b:.2f}\nMedian SSIM: {median_SSIM_b:.2f}\nQ1 SSIM: {q1_SSIM_b:.2f}\nQ3 SSIM:{q3_SSIM_b:.2f}\nStd SSIM: {std_SSIM_b:.2f}"
    metrics_text_nn = f"Mean MAE: {mean_MAE_nn:.2f}\nMedian MAE: {median_MAE_nn:.2f}\nQ1 MAE: {q1_MAE_nn:.2f}\nQ3 MAE: {q3_MAE_nn:.2f}\nMean RMSE: {mean_RMSE_nn:.2f}\n Std MAE: {std_MAE_nn:.2f}\nMean RMSE: {mean_RMSE_nn:.2f}\nMedian RMSE: {median_RMSE_nn:.2f}\nQ1 RMSE: {q1_RMSE_nn:.2f}\nQ3 RMSE: {q3_RMSE_nn:.2f}\nStd RMSE: {std_RMSE_nn:.2f}\nMean SSIM: {mean_SSIM_nn:.2f}\nMedian SSIM: {median_SSIM_nn:.2f}\nQ1 SSIM: {q1_SSIM_nn:.2f}\nQ3 SSIM:{q3_SSIM_nn:.2f}\nStd SSIM: {std_SSIM_nn:.2f}"
    metrics_text_bm = f"Mean MAE: {mean_MAE_benchmark:.2f}\nMedian MAE: {median_MAE_benchmark:.2f}\nQ1 MAE: {q1_MAE_benchmark:.2f}\nQ3 MAE: {q3_MAE_benchmark:.2f}\nMean RMSE: {mean_RMSE_benchmark:.2f}\n Std MAE: {std_MAE_benchmark:.2f}\nMean RMSE: {mean_RMSE_benchmark:.2f}\nMedian RMSE: {median_RMSE_benchmark:.2f}\nQ1 RMSE: {q1_RMSE_benchmark:.2f}\nQ3 RMSE: {q3_RMSE_benchmark:.2f}\nStd RMSE: {std_RMSE_benchmark:.2f}\nMean SSIM: {mean_SSIM_benchmark:.2f}\nMedian SSIM: {median_SSIM_benchmark:.2f}\nQ1 SSIM: {q1_SSIM_benchmark:.2f}\nQ3 SSIM:{q3_SSIM_benchmark:.2f}\nStd SSIM: {std_SSIM_benchmark:.2f}"


    axes[0].text(0.5, 0.5, metrics_text_ni, ha='center', va='center', fontsize=24, transform=axes[0].transAxes)
    axes[0].set_title(titles[0], fontsize=14)
    axes[0].axis('off')

    axes[1].text(0.5, 0.5, metrics_text_b, ha='center', va='center', fontsize=24, transform=axes[1].transAxes)
    axes[1].set_title(titles[1], fontsize=14)
    axes[1].axis('off')

    axes[2].text(0.5, 0.5, metrics_text_nn, ha='center', va='center', fontsize=24, transform=axes[2].transAxes)
    axes[2].set_title(titles[2], fontsize=14)
    axes[2].axis('off')

    axes[3].text(0.5, 0.5, metrics_text_bm, ha='center', va='center', fontsize=24, transform=axes[3].transAxes)
    axes[3].set_title(titles[3], fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()

    # Save the figure
    directory = HP_RESULT_PATH + '/image'
    if not os.path.exists(directory):
        os.makedirs(directory)

    axes[3].get_figure().savefig(f'{directory}/average_metrics_images.png')
    plt.close()

    # Create a box plot to show the metrics instead of a table. 
    # MAE_________________________________________
    data = [MAE_b, MAE_nn, MAE_ni, MAE_benchmark]
    labels = ['Bi-linear', 'Nearest Neighbour', 'No Interpolation', 'Baseline']
    color_pairs = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

    # Creating the box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True, notch=True, showfliers=False)

    # Assigning the color pairs manually to match the labels
    for patch, color in zip(box['boxes'], color_pairs):
        patch.set_facecolor(color)

    # Customizing the median line color
    median_color = 'black'
    for median in box['medians']:
        median.set_color(median_color)

    # Adding mean values to the plot
    for i, d in enumerate(data):
        mean_value = np.mean(d)
        median_value = np.median(d)
        q1_value = np.percentile(d, 25)
        q3_value = np.percentile(d, 75)

        plt.plot(i + 1, mean_value, 'D', color='black', markersize=5, label="Mean" if i == 0 else "")

    # Adding other plot elements
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel('m/s', fontsize=20)
    plt.grid(True)

    # Automatically determine y-axis limits
    plt.autoscale()

    # Get current y-axis limits
    y_min, y_max = plt.gca().get_ylim()

    # Set y-axis ticks to steps of 0.1 within the determined range
    y_ticks = np.arange(0, y_max, 0.1)
    plt.yticks(y_ticks)

    # Save the plot as an image file
    plt.savefig(f'{HP_RESULT_PATH}/mean_MAE_box_plot.png')


    # RMSE_________________________________________
    data = [RMSE_b, RMSE_nn, RMSE_ni, RMSE_benchmark]
    labels = ['Bi-linear', 'Nearest Neighbour', 'No Interpolation', 'Baseline']
    color_pairs = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

    # Creating the box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True, notch=True, showfliers=False)

    # Assigning the color pairs manually to match the labels
    for patch, color in zip(box['boxes'], color_pairs):
        patch.set_facecolor(color)

    # Customizing the median line color
    median_color = 'black'
    for median in box['medians']:
        median.set_color(median_color)

    # Adding mean values to the plot
    for i, d in enumerate(data):
        mean_value = np.mean(d)
        median_value = np.median(d)
        q1_value = np.percentile(d, 25)
        q3_value = np.percentile(d, 75)

        plt.plot(i + 1, mean_value, 'D', color='black', markersize=5, label="Mean" if i == 0 else "")

    # Adding other plot elements
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel('m/s', fontsize=14)
    plt.grid(True)

    # Automatically determine y-axis limits
    plt.autoscale()

    # Get current y-axis limits
    y_min, y_max = plt.gca().get_ylim()

    # Set y-axis ticks to steps of 0.1 within the determined range
    y_ticks = np.arange(0, y_max, 0.1)
    plt.yticks(y_ticks)

    # Save the plot as an image file
    plt.savefig(f'{HP_RESULT_PATH}/mean_RMSE_box_plot.png')


    # SSIM_________________________________________
    data = [SSIM_b, SSIM_nn, SSIM_ni, SSIM_benchmark]
    labels = ['Bi-linear', 'Nearest Neighbour', 'No Interpolation', 'Baseline']
    color_pairs = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

    # Creating the box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True, notch=True, showfliers=False)

    # Assigning the color pairs manually to match the labels
    for patch, color in zip(box['boxes'], color_pairs):
        patch.set_facecolor(color)

    # Customizing the median line color
    median_color = 'black'
    for median in box['medians']:
        median.set_color(median_color)

    # Adding mean values to the plot
    for i, d in enumerate(data):
        mean_value = np.mean(d)
        median_value = np.median(d)
        q1_value = np.percentile(d, 25)
        q3_value = np.percentile(d, 75)

        plt.plot(i + 1, mean_value, 'D', color='black', markersize=5, label="Mean" if i == 0 else "")

    # Adding other plot elements
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.grid(True)

    # Automatically determine y-axis limits
    plt.autoscale()

    # Get current y-axis limits
    y_min, y_max = plt.gca().get_ylim()

    # Set y-axis ticks to steps of 0.1 within the determined range
    y_ticks = np.arange(-0.10, y_max, 0.1)
    plt.yticks(y_ticks)

    # Save the plot as an image file
    plt.savefig(f'{HP_RESULT_PATH}/mean_SSIM_box_plot.png')


#_____________________________________________________________________________
#                           Plot for the extremes:                           #
#_____________________________________________________________________________
# A) Define the threshold and associate each images to a given category 
# Mean wind speed value per images for target
mean_windspeed_per_images = np.mean(label_test_de_standardized[..., 0], axis=(1,2))

# Define the threshold based on the target
bottom_1_threshold = np.percentile(mean_windspeed_per_images, 1)
bottom_5_threshold = np.percentile(mean_windspeed_per_images, 5)
top_5_threshold = np.percentile(mean_windspeed_per_images, 95)
top_1_threshold = np.percentile(mean_windspeed_per_images, 99)

print(f'Here are in m/sec bottom 1% : {bottom_1_threshold:.3f}, 5% : {bottom_5_threshold:.3f}, top 5% : {top_5_threshold:.3f}, top 1% : {top_1_threshold:.3f}')
print()

# Create the mask that will be used to access the value associate with their appropriate threshold. 
mask_bottom1 = mean_windspeed_per_images < bottom_1_threshold
mask_bottom5 = mean_windspeed_per_images < bottom_5_threshold
mask_top5 = mean_windspeed_per_images > top_5_threshold
mask_top1 = mean_windspeed_per_images > top_1_threshold
mask_middle_90 = ~(mask_top5 | mask_bottom5)
mask_overall = np.full_like(mean_windspeed_per_images, True, dtype=bool)

# B) Data processing before metric calculation
# Remove last channel for computation
gt_images_squeezed = label_test_de_standardized[..., 0]        # shape (14495, 48, 48)
pred_images_squeezed_no_inter = predictions_no_inter_de_standardized[..., 0]    # same shape
pred_images_squeezed_bilinear = predictions_bilinear_de_standardized[..., 0]    # same shape
pred_images_squeezed_nearest = predictions_nearest_neighbour_de_standardized[..., 0]    # same shape
pred_images_squeezed_baseline = bm_bilinear_de_standardized[..., 0]    # same shape

# C) Metric calculation
# No interpolation
# Calculate MSE
rmse_per_image_no_inter = np.sqrt(np.mean((pred_images_squeezed_no_inter - gt_images_squeezed) ** 2, axis=(1, 2)))
# Calculate ssim
ssim_per_image_no_inter = np.array([
    ssim(gt, pred, data_range=gt.max() - gt.min())
    for gt, pred in zip(gt_images_squeezed, pred_images_squeezed_no_inter)
])
# Bilinear
# Calculate MSE
rmse_per_image_bilinear = np.sqrt(np.mean((pred_images_squeezed_bilinear - gt_images_squeezed) ** 2, axis=(1, 2)))
# Calculate ssim
ssim_per_image_bilinear = np.array([
    ssim(gt, pred, data_range=gt.max() - gt.min())
    for gt, pred in zip(gt_images_squeezed, pred_images_squeezed_bilinear)
])
# Nearest neighbour
# Calculate MSE
rmse_per_image_nearest = np.sqrt(np.mean((pred_images_squeezed_nearest - gt_images_squeezed) ** 2, axis=(1, 2)))
# Calculate ssim
ssim_per_image_nearest = np.array([
    ssim(gt, pred, data_range=gt.max() - gt.min())
    for gt, pred in zip(gt_images_squeezed, pred_images_squeezed_nearest)
])
# Baseline
# Calculate MSE
rmse_per_image_baseline = np.sqrt(np.mean((pred_images_squeezed_baseline - gt_images_squeezed) ** 2, axis=(1, 2)))
# Calculate ssim
ssim_per_image_baseline = np.array([
    ssim(gt, pred, data_range=gt.max() - gt.min())
    for gt, pred in zip(gt_images_squeezed, pred_images_squeezed_baseline)
])

# D) Print the metrics results
# No-inter
print(f'No interpolation')
rows = []

def report_metrics_no_inter(mask, name):
    rmse = np.mean(rmse_per_image_no_inter[mask])
    ssim_ = np.mean(ssim_per_image_no_inter[mask])
    print(f"{name}: RMSE = {rmse:.4f}, SSIM = {ssim_:.4f}")
    return [name, f"{rmse:.4f}", f"{ssim_:.4f}"]

rows.append(report_metrics_no_inter(mask_overall, "Overall"))
rows.append(report_metrics_no_inter(mask_bottom1, "Bottom 1%"))
rows.append(report_metrics_no_inter(mask_bottom5, "Bottom 5%"))
rows.append(report_metrics_no_inter(mask_middle_90, "Middle 90%"))
rows.append(report_metrics_no_inter(mask_top5, "Top 5%"))
rows.append(report_metrics_no_inter(mask_top1, "Top 1%"))
print()

# Bilinear
print(f'Bilinear interpolation')
rows_bi = []

def report_metrics_bilinear(mask, name):
    rmse_bilinear = np.mean(rmse_per_image_bilinear[mask])
    ssim_bilinear = np.mean(ssim_per_image_bilinear[mask])
    print(f"{name}: RMSE = {rmse_bilinear:.4f}, SSIM = {ssim_bilinear:.4f}")
    return [name, f"{rmse_bilinear:.4f}", f"{ssim_bilinear:.4f}"]

rows_bi.append(report_metrics_bilinear(mask_overall, "Overall"))
rows_bi.append(report_metrics_bilinear(mask_bottom1, "Bottom 1%"))
rows_bi.append(report_metrics_bilinear(mask_bottom5, "Bottom 5%"))
rows_bi.append(report_metrics_bilinear(mask_middle_90, "Middle 90%"))
rows_bi.append(report_metrics_bilinear(mask_top5, "Top 5%"))
rows_bi.append(report_metrics_bilinear(mask_top1, "Top 1%"))
print()

# Neirest Neighbour
print(f'Nearest Neighbour interpolation')
rows_n = []

def report_metrics_nearest_neighbour(mask, name):
    rmse_nearest = np.mean(rmse_per_image_nearest[mask])
    ssim_nearest = np.mean(ssim_per_image_nearest[mask])
    print(f"{name}: RMSE = {rmse_nearest:.4f}, SSIM = {ssim_nearest:.4f}")
    return [name, f"{rmse_nearest:.4f}", f"{ssim_nearest:.4f}"]

rows_n.append(report_metrics_nearest_neighbour(mask_overall, "Overall"))
rows_n.append(report_metrics_nearest_neighbour(mask_bottom1, "Bottom 1%"))
rows_n.append(report_metrics_nearest_neighbour(mask_bottom5, "Bottom 5%"))
rows_n.append(report_metrics_nearest_neighbour(mask_middle_90, "Middle 90%"))
rows_n.append(report_metrics_nearest_neighbour(mask_top5, "Top 5%"))
rows_n.append(report_metrics_nearest_neighbour(mask_top1, "Top 1%"))
print()

# Baseline
print(f'Baseline')
rows_b = []

def report_metrics_baseline(mask, name):
    rmse_baseline = np.mean(rmse_per_image_baseline[mask])
    ssim_baseline = np.mean(ssim_per_image_baseline[mask])
    print(f"{name}: RMSE = {rmse_baseline:.4f}, SSIM = {ssim_baseline:.4f}")
    return [name, f"{rmse_baseline:.4f}", f"{ssim_baseline:.4f}"]

rows_b.append(report_metrics_baseline(mask_overall, "Overall"))
rows_b.append(report_metrics_baseline(mask_bottom1, "Bottom 1%"))
rows_b.append(report_metrics_baseline(mask_bottom5, "Bottom 5%"))
rows_b.append(report_metrics_baseline(mask_middle_90, "Middle 90%"))
rows_b.append(report_metrics_baseline(mask_top5, "Top 5%"))
rows_b.append(report_metrics_baseline(mask_top1, "Top 1%"))
print()


# Calculate the average metrics over each images of the test set for interpolation bi-linear [sample, width, height, channel]:
# # A - RMSE
# predictions_no_inter_de_standardized
# predictions_bilinear_de_standardized
# predictions_nearest_neighbour_de_standardized
# label_test_de_standardized
# # B - SSIM