#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.fft import fft2, fftshift
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

import model as mod

#Functions
def de_standardize(test_data_input, predictions, test_data_label, std_inputs_de_standardize_path, mean_inputs_de_standardize_path, 
                   std_label_de_standardize_path, mean_label_de_standardize_path, bm_bilinear):
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
    bm_bilinear = (bm_bilinear * std_inputs_adjusted[..., 0:1]) + mean_inputs_adjusted[..., 0:1]

    # De-standardize predictions and labels using the resized mean and std
    mean_label_adjusted = np.transpose(mean_label, (0, 2, 3, 1)) 
    std_label_adjusted = np.transpose(std_label, (0, 2, 3, 1)) 
    print(f'The standardized prediction shape is : {predictions.shape}')
    print(f'The std_label shape is : {std_label.shape} and the mean_label shape is : {mean_label.shape}')
    print(f'The std_label_adjusted shape is: {std_label_adjusted.shape} and the mean_label_adjusted shape is: {mean_label_adjusted.shape}')

    original_prediction = (predictions * std_label_adjusted[..., 0:1]) + mean_label_adjusted[..., 0:1]
    original_labels = (test_data_label * std_label_adjusted[..., 0:1]) + mean_label_adjusted[..., 0:1]
 
    return original_inputs, original_prediction, original_labels, bm_bilinear

def crop_result(test_data_input, predictions, test_data_label, bm_bilinear_de_standardized):
    print(f'The data shape of test_data_input is : {test_data_input.shape}')
    print(f'The data shape of predictions is : {predictions.shape}')
    print(f'The data shape of test_data_label is : {test_data_label.shape}')
    print(f'The data shape of bm_bilinear_de_standardized is : {bm_bilinear_de_standardized.shape}')

    test_data_input = test_data_input[:, :, :, :]
    predictions = predictions[:, 4:-4, 4:-4, :]
    test_data_label = test_data_label[:, 4:-4, 4:-4, :]
    bm_bilinear_de_standardized = bm_bilinear_de_standardized[:, 4:-4, 4:-4, :]

    return test_data_input, predictions, test_data_label, bm_bilinear_de_standardized
        
    
def convert_node_msec(inputs_test, predictions, label, bm_bilinear_de_standardized):
    factor = 0.514444

    inputs_test = inputs_test * factor
    predictions = predictions * factor
    label = label * factor
    bm_bilinear_de_standardized = bm_bilinear_de_standardized * factor

    return inputs_test, predictions, label, bm_bilinear_de_standardized
    
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
    plt.plot(freqs, psd_pred_db[:len(freqs)], label='Predicted no interpolation A')
    plt.plot(freqs, psd_pred_db2[:len(freqs)], label='Predicted no interpolation Configuration B')
    plt.plot(freqs, psd_pred_db3[:len(freqs)], label='Predicted no interpolation Configuration C')
    plt.plot(freqs, psd_benchmark[:len(freqs)], label='Baseline')
    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Comparison of Power Spectral Densities in dB')
    plt.yscale('log')  # Set the y-axis to log scale
    plt.legend()
    plt.grid(True)
    
    directory = directory + '/image_psd'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f'{directory}/psd_{images+1}.png')
    plt.close()

# def power_spectral_density_graph_average(predicted_images, predicted_images2, predicted_images3, ground_truth_images, directory, test_data_skip_de_standardized):
#     num_images = predicted_images.shape[0]
#     num_images_cad_05 = predicted_images2.shape[0]
#     num_images_cad_10 = predicted_images3.shape[0]
#     print(f'The power spectral density number of images to regenerate is: {predicted_images.shape}')
    
#     psd_gt_azimuthals = []
#     psd_pred_azimuthals = []
#     psd_pred_azimuthals2 = []
#     psd_pred_azimuthals3 = []
#     psd_benchmark_azimuthals = []
    
#     for i in range(num_images):
#         psd_ground_truth = compute_psd(ground_truth_images[i, :, :, 0])
#         psd_predicted = compute_psd(predicted_images[i, :, :, 0])
#         psd_benchmark = compute_psd(test_data_skip_de_standardized[i, :, :, 0])   
#         psd_gt_azimuthals.append(azimuthal_average(psd_ground_truth))
#         psd_pred_azimuthals.append(azimuthal_average(psd_predicted))
#         psd_benchmark_azimuthals.append(azimuthal_average(psd_benchmark))

#     for i in range(num_images_cad_05):
#         psd_predicted2 = compute_psd(predicted_images2[i, :, :, 0])
#         psd_pred_azimuthals2.append(azimuthal_average(psd_predicted2))

#     for i in range(num_images_cad_10):
#         psd_predicted3 = compute_psd(predicted_images3[i, :, :, 0])
#         psd_pred_azimuthals3.append(azimuthal_average(psd_predicted3))
    
#     # Compute the average PSD
#     avg_psd_gt_azimuthal = np.mean(psd_gt_azimuthals, axis=0)
#     avg_psd_pred_azimuthal = np.mean(psd_pred_azimuthals, axis=0)
#     avg_psd_pred_azimuthal2 = np.mean(psd_pred_azimuthals2, axis=0)
#     avg_psd_pred_azimuthal3 = np.mean(psd_pred_azimuthals3, axis=0)
#     avg_psd_benchmark_azimuthal = np.mean(psd_benchmark_azimuthals, axis=0)
    
#     # Convert PSD to dB
#     avg_psd_gt_db = 10 * np.log10(avg_psd_gt_azimuthal)
#     avg_psd_pred_db = 10 * np.log10(avg_psd_pred_azimuthal)
#     avg_psd_pred_db2 = 10 * np.log10(avg_psd_pred_azimuthal2)
#     avg_psd_pred_db3 = 10 * np.log10(avg_psd_pred_azimuthal3)
#     avg_psd_benchmark_db = 10 * np.log10(avg_psd_benchmark_azimuthal)
    
#     size = psd_ground_truth.shape[0]
#     freqs = np.fft.fftfreq(size)[:size // 2]  # Compute frequencies in cycles per pixel
#     freqs = freqs[freqs > 0]  # Remove zero frequency for log-log plot

#     plt.figure(figsize=(10, 6))
#     plt.plot(freqs, avg_psd_gt_db[:len(freqs)], label='Ground Truth')
#     plt.plot(freqs, avg_psd_pred_db[:len(freqs)], label='No interpolation Configuration A')
#     plt.plot(freqs, avg_psd_pred_db2[:len(freqs)], label='No interpolation Configuration B')
#     plt.plot(freqs, avg_psd_pred_db3[:len(freqs)], label='No interpolation Configuration C')
#     plt.plot(freqs, avg_psd_benchmark_db[:len(freqs)], label='Baseline')
#     plt.xlabel('Spatial Frequency (cycles per pixel)')
#     plt.ylabel('Power Spectral Density (dB)')
#     plt.title('Comparison of Average Power Spectral Densities in dB')
#     plt.yscale('log')  # Set the y-axis to log scale
#     plt.legend()
#     plt.grid(True)
    
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(f'{directory}/average_psd.png')
#     plt.close()

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

    plt.plot(wind_speed_range, pdf1, label='No interpolation Configuration A')
    plt.plot(wind_speed_range, pdf2, label='No interpolation Configuration B')
    plt.plot(wind_speed_range, pdf3, label='No interpolation Configuration C')
    plt.plot(wind_speed_range, pdfbm, label='Bench mark')
    plt.plot(wind_speed_range, pdfgt, label='Ground truth')

    plt.xlabel('Wind Speed (m/sec)')
    plt.ylabel('PDF')
    plt.title('PDF of Wind Speed Predictions')
    plt.legend(loc='upper right')

    directory = directory + '/image_pdf'
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    plt.plot(wind_speed_range, pdf1, label='U-Net A')
    plt.plot(wind_speed_range, pdf2, label='U-Net B')
    plt.plot(wind_speed_range, pdf3, label='U-Net C')
    plt.plot(wind_speed_range, pdfbm, label='Baseline')
    plt.plot(wind_speed_range, pdfgt, label='Ground Truth')

    # Add labels, title, and legend
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('PDF')
    plt.title('Average PDF of Wind Speed Predictions')
    plt.legend(loc='upper right')

    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/wind_speed_pdf.png')
    plt.close()

def average_error_location(predictions, prediction_bilinear, prediction_nearest, label, Baseline, directory):
    """
    Calculate the average error location on the whole test set for a specific domain. 
    """
    # Calculate the average error on the whole domain data
    # print(f'Shape of predictions is : {predictions.shape}')
    # print(f'Shape of Baseline is : {Baseline.shape}')
    # print(f'Shape of label is : {label.shape}')
    error_pred = np.abs(predictions - label)
    error_pred_bi = np.abs(prediction_bilinear - label)
    error_pred_near = np.abs(prediction_nearest - label)
    error_bm = np.abs(Baseline - label)
    # print(f'shape for error_pred and error_bm are : {error_pred.shape} and {error_bm.shape}')
    # Compute the average to know the location of the errors
    average_error_pred = np.mean(error_pred, axis=0)
    average_error_pred_bi = np.mean(error_pred_bi, axis=0)
    average_error_pred_near = np.mean(error_pred_near, axis=0)
    average_error_bm = np.mean(error_bm, axis=0)
    # print(f'Shape for average_error_pred and average_error_bm are: {average_error_pred.shape} and {average_error_bm.shape}')
    # Calculate the average error for the entire Configuration as one final value
    final_average_error_pred = round(np.mean(average_error_pred), 2)
    final_average_error_pred_bi = round(np.mean(average_error_pred_bi), 2)
    final_average_error_pred_near = round(np.mean(average_error_pred_near), 2)
    final_average_error_bm = round(np.mean(average_error_bm), 2)

    # Show the result on a graph
    fig, axes = plt.subplots(1, 4, figsize=(30, 8))


    vmin = min(np.min(average_error_pred), np.min(average_error_pred_bi), np.min(average_error_pred_near))
    vmax = max(np.max(average_error_pred), np.max(average_error_pred_bi), np.max(average_error_pred_near))

    im1 = axes[0].imshow(average_error_pred, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'No interpolation Configuration A:\n {final_average_error_pred:.2f} m/sec', fontsize=20)
    fig.colorbar(im1, ax=axes[0], orientation='vertical')

    im2 = axes[1].imshow(average_error_pred_bi, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'No interpolation Configuration B:\n {final_average_error_pred_bi:.2f} m/sec', fontsize=20)
    fig.colorbar(im2, ax=axes[1], orientation='vertical')

    im3 = axes[2].imshow(average_error_pred_near, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title(f'No interpolation Configuration C:\n {final_average_error_pred_near:.2f} m/sec', fontsize=20)
    fig.colorbar(im3, ax=axes[2], orientation='vertical')

    im4 = axes[3].imshow(average_error_bm, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[3].set_title(f'Baseline:\n {final_average_error_bm:.2f} m/sec', fontsize=20)
    fig.colorbar(im4, ax=axes[3], orientation='vertical')
    
    plt.suptitle('Difference in error for the prediction and Baseline')
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[3].axis('off')
    
    plt.savefig(f'{directory}/average_error_location.png')
    plt.close()

# Main 
if __name__ == '__main__':
    plt.rcParams['font.size'] = 14
    #VARIABLES
    ANALYSED_DOMAIN = f"domaine1"
    # Hyper-parameters used for each model type
    # General path for all type of strategies:
    # Path to the test files
    std_skip_de_standardize_path = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_skip_de_standardize_path = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    # Path to save the output prediction 
    HP_RESULT_PATH = f"/home/jfg000/ss5/CNN_results_article/articles_testing/article_section_II_allDomains" 
    

    # Parameter for no interpolation Qc: 
    lr_no_interp = 0.001
    drop_rate_no_interp = 0.30
    alpha_no_interp = 0.30
    loss_weights_no_interp = 0.5
    # Path for model weights
    path_weights_no_interp = '/home/jfg000/ss5/CNN_results_article/non_interpolation_qcnb_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000000_Final_search_HP_Search/model/Searchmodel_9_cb' 
    # Input data to test the model with 
    input_file_test_no_inter = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy' 
    skip_no_inter = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    topo_no_inter = f'/home/jfg000/ss5/data_superResolution/topography/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data/normalize.npy'
    # Label data
    label_file = f'/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy' 
    std_label_de_standardize_path = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_label_de_standardize_path = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    # Std deviation and mean to de-standardize the prediction
    std_inputs_de_standardize_path_no_inter = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_inputs_de_standardize_path_no_inter = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'

    # Parameter for no interpolation Cad 5 domains: 
    lr_no_inter_cad_05 = 0.001
    drop_no_inter_cad_05 = 0.25
    alpha_no_inter_cad_05 = 0.30
    loss_no_inter_cad_05 = 1.00
    # Path for model weights
    path_weights_no_inter_cad_05 = '/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_05_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000000_Final_search_HP_Search/model/Searchmodel_1_cb'
    # Input data to test the model with 
    input_file_test_no_inter_cad_05 = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy' 
    skip_no_inter_cad_05 = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_test_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    topo_no_inter_cad_05 = f'/home/jfg000/ss5/data_superResolution/topography/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data/normalize.npy'
    # Label data 
    label_cad_05 = f'/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    std_label_de_standardize_path_cad_05 = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_label_de_standardize_path_cad_05 = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    # Std deviation and mean to de-standardize the prediction
    std_inputs_de_standardize_path_no_inter_cad_05 = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_inputs_de_standardize_path_no_inter_cad_05 = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    
    # Parameter for no interpolation Cad 10 domains:
    lr_no_inter_cad_10 = 0.001
    drop_no_inter_cad_10 = 0.35
    alpha_no_inter_cad_10 = 0.25
    loss_no_inter_cad_10 = 1.00
    # Path for model weights
    path_weights_no_inter_cad_10 = '/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000001_Final_search_HP_Search/model/Searchmodel_1_cb' 
    # Input data to test the model with 
    input_file_test_no_inter_cad_10 = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    skip_no_inter_cad_10 = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_test_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    topo_no_inter_cad_10 = f'/home/jfg000/ss5/data_superResolution/topography/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data/normalize.npy'
    # Label data 
    label_cad_10 = f'/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    std_label_de_standardize_path_cad_10 = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_label_de_standardize_path_cad_10 = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    # Std deviation and mean to de-standardize the prediction
    std_inputs_de_standardize_path_no_inter_cad_10 = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_inputs_de_standardize_path_no_inter_cad_10 = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    
    # Loading the Baseline
    input_file_benchmark = f'/home/jfg000/ss5/data_superResolution/input/domaine/bilinear_interpolation/data_used_in_neural_network_qcnb/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    input_file_benchmark_cad_05 = f'/home/jfg000/ss5/data_superResolution/input/domaine/bilinear_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    input_file_benchmark_cad_10 = f'/home/jfg000/ss5/data_superResolution/input/domaine/bilinear_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    
    # Create the models that will be used to do prediction on 
    model_no_interpolation = mod.deepRU_article_non_interpolation_combine_loss(units=0, activation=None, lr=lr_no_interp, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, 
                                                                               kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0, num_hidden_layer=0, WIND=None, 
                                                                               regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, drop_rate=drop_rate_no_interp, alpha=alpha_no_interp, loss_weights=loss_weights_no_interp)
    model_no_inter_cad_05 = mod.deepRU_article_non_interpolation_combine_loss(units=0, activation=None, lr=lr_no_inter_cad_05, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0,
                                                                                                                            num_hidden_layer=None, WIND=None, regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, drop_rate=drop_no_inter_cad_05, alpha=alpha_no_inter_cad_05, loss_weights=loss_no_inter_cad_05)
    model_no_inter_cad_10 = mod.deepRU_article_non_interpolation_combine_loss(units=0, activation=None, lr=lr_no_inter_cad_10, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0,
                                                                                                                            num_hidden_layer=None, WIND=None, regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, drop_rate=drop_no_inter_cad_10, alpha=alpha_no_inter_cad_10, loss_weights=loss_no_inter_cad_10)
    
    # Load the model to do prediction on: 
    # Load model weights for no interpolation, bilinear and nearest neighbour
    model_no_interpolation.load_weights(path_weights_no_interp)
    model_no_inter_cad_05.load_weights(path_weights_no_inter_cad_05)
    model_no_inter_cad_10.load_weights(path_weights_no_inter_cad_10)

    # Load the test data
    #Loading the label data
    label_test = np.load(label_file)
    label_test_cad_05 = np.load(label_cad_05)
    label_test_cad_10 = np.load(label_cad_10)
    bm_bilinear = np.load(input_file_benchmark)
    bm_bilinear_cad_05 = np.load(input_file_benchmark_cad_05)
    bm_bilinear_cad_10 = np.load(input_file_benchmark_cad_10)

    # Load test data for specific strategie
    print('load the input data for non interpolation')
    inputs_test_no_inter = np.load(input_file_test_no_inter)
    skip_no_inter = np.load(skip_no_inter)
    topo_no_inter = np.load(topo_no_inter)

    print('load the input data for non interpolation Cad 05')
    inputs_test_no_inter_cad_05 = np.load(input_file_test_no_inter_cad_05)
    skip_no_inter_cad_05 = np.load(skip_no_inter_cad_05)
    topo_no_inter_cad_05 = np.load(topo_no_inter_cad_05)

    #TODO delete for test only
    PATH_INPUT_TEST = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    PATH_SKIP_TEST =  '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_test_allpasses_domaine/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'
    PATH_TOPO = '/home/jfg000/ss5/data_superResolution/topography/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/ME_MG_Z0_data/normalize.npy'
    PATH_LABEL_TEST = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_test_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy'

    std_inputs_05 = np.load('/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy')
    mean_inputs_05 = np.load('/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy') 
    std_label_05 = np.load('/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy')
    mean_label_05 = np.load('/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_5/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy') 
    
    print('load the input data for non interpolation Cad 10')
    inputs_test_no_inter_cad_10 = np.load(input_file_test_no_inter_cad_10)
    skip_no_inter_cad_10 = np.load(skip_no_inter_cad_10)
    topo_no_inter_cad_10 = np.load(topo_no_inter_cad_10)

    # Reshape the test data [sample, width, height, channel] 
    inputs_test_no_inter = inputs_test_no_inter.transpose((0,2,3,1)) 
    inputs_test_no_inter_cad_05 = inputs_test_no_inter_cad_05.transpose((0,2,3,1)) 
    inputs_test_no_inter_cad_10 = inputs_test_no_inter_cad_10.transpose((0,2,3,1)) 

    label_test = label_test.transpose((0,2,3,1))
    label_test_cad_05 = label_test_cad_05.transpose((0,2,3,1))
    label_test_cad_10 = label_test_cad_10.transpose((0,2,3,1))

    skip_no_inter = skip_no_inter.transpose((0,2,3,1)) 
    skip_no_inter_cad_05 = skip_no_inter_cad_05.transpose((0,2,3,1))
    skip_no_inter_cad_10 = skip_no_inter_cad_10.transpose((0,2,3,1))

    bm_bilinear = bm_bilinear.transpose((0,2,3,1))[:, 8:56, 8:56, 0]
    bm_bilinear = bm_bilinear[:, :, :, np.newaxis]
    bm_bilinear_cad_05 = bm_bilinear_cad_05.transpose((0,2,3,1))[:, 8:56, 8:56, 0]
    bm_bilinear_cad_05 = bm_bilinear_cad_05[:, :, :, np.newaxis]
    bm_bilinear_cad_10 = bm_bilinear_cad_10.transpose((0,2,3,1))[:, 8:56, 8:56, 0]
    bm_bilinear_cad_10 = bm_bilinear_cad_10[:, :, :, np.newaxis]

    topo_no_inter = topo_no_inter.transpose((0,2,3,1))
    topo_no_inter_cad_05 = topo_no_inter_cad_05.transpose((0,2,3,1))
    topo_no_inter_cad_10 = topo_no_inter_cad_10.transpose((0,2,3,1))

    # Adjust the topographie if the user is only testing on a single domain
    topo_no_inter = np.tile(topo_no_inter, (int(inputs_test_no_inter.shape[0] / 5), 1, 1, 1))
    topo_no_inter_cad_05 = np.tile(topo_no_inter_cad_05, (int(inputs_test_no_inter_cad_05.shape[0] / 5), 1, 1, 1))
    topo_no_inter_cad_10 = np.tile(topo_no_inter_cad_10, (int(inputs_test_no_inter_cad_10.shape[0] / 10), 1, 1, 1))

    #Display the shape for each network inputs
    print(f'Domaine Quebec and New-Brunswick')
    print(f'inputs_test_no_inter shape: {inputs_test_no_inter.shape}')
    print(f'topo_no_inter shape: {topo_no_inter.shape}')
    print(f'skip_no_inter shape: {skip_no_inter.shape}')
    print(f'bm_bilinear shape: {bm_bilinear.shape}')

    print(f'Domaine Cad05')
    print(f'inputs_test_no_inter_cad_05 shape: {inputs_test_no_inter_cad_05.shape}')
    print(f'topo_no_inter_cad_05 shape: {topo_no_inter_cad_05.shape}')
    print(f'skip_no_inter_cad_05 shape: {skip_no_inter_cad_05.shape}')
    print(f'bm_bilinear_cad_05 shape: {bm_bilinear_cad_05.shape}')

    print(f'Domaine Cad10')
    print(f'inputs_test_no_inter_cad_05 shape: {inputs_test_no_inter_cad_10.shape}')
    print(f'topo_no_inter_cad_05 shape: {topo_no_inter_cad_10.shape}')
    print(f'skip_no_inter_cad_05 shape: {skip_no_inter_cad_10.shape}')
    print(f'bm_bilinear_cad_10 shape: {bm_bilinear_cad_10.shape}')

    # Do the prediction 
    # Prediction for non interpolation 
    predictions_no_inter = model_no_interpolation.predict(
        x=[inputs_test_no_inter, topo_no_inter, skip_no_inter],
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    # Prediction for bi-linear
    predictions_no_inter_cad_05 = model_no_inter_cad_05.predict(
        x=[inputs_test_no_inter_cad_05, topo_no_inter_cad_05, skip_no_inter_cad_05],
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    # Prediction for nearest neighbour 
    predictions_no_inter_cad_10 = model_no_inter_cad_10.predict(
        x=[inputs_test_no_inter_cad_10, topo_no_inter_cad_10, skip_no_inter_cad_10],
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
    inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized = de_standardize(inputs_test_no_inter, predictions_no_inter, label_test, 
                                                                                                                                                         std_inputs_de_standardize_path_no_inter, mean_inputs_de_standardize_path_no_inter,
                                                                                                                                                         std_label_de_standardize_path, mean_label_de_standardize_path, bm_bilinear)
    inputs_test_no_inter_cad_05_de_standardized, predictions_no_inter_cad_05_de_standardized, label_test_no_inter_cad_05_de_standardized, bm_bilinear_cad_05_de_standardized = de_standardize(inputs_test_no_inter_cad_05, predictions_no_inter_cad_05, label_test_cad_05, 
                                                                                                                                                         std_inputs_de_standardize_path_no_inter_cad_05, mean_inputs_de_standardize_path_no_inter_cad_05,
                                                                                                                                                         std_label_de_standardize_path_cad_05, mean_label_de_standardize_path_cad_05, bm_bilinear_cad_05)
    inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized, label_test_no_inter_cad_10_de_standardized, bm_bilinear_cad_10_de_standardized = de_standardize(inputs_test_no_inter_cad_10, predictions_no_inter_cad_10, label_test_cad_10, 
                                                                                                                                                         std_inputs_de_standardize_path_no_inter_cad_10, mean_inputs_de_standardize_path_no_inter_cad_10,
                                                                                                                                                         std_label_de_standardize_path_cad_10, mean_label_de_standardize_path_cad_10, bm_bilinear_cad_10)

    # Remove the padding pixels for the the input, prediction, label image and skip connection 
    inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized = crop_result(inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized)
    inputs_test_no_inter_cad_05_de_standardized, predictions_no_inter_cad_05_de_standardized, label_test_no_inter_cad_05_de_standardized, bm_bilinear_cad_05_de_standardized = crop_result(inputs_test_no_inter_cad_05_de_standardized, predictions_no_inter_cad_05_de_standardized, label_test_no_inter_cad_05_de_standardized, bm_bilinear_cad_05_de_standardized)
    inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized, label_test_no_inter_cad_10_de_standardized, bm_bilinear_cad_10_de_standardized = crop_result(inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized, label_test_no_inter_cad_10_de_standardized, bm_bilinear_cad_10_de_standardized)

    # Convert the units from knots to meters per seconds
    inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized = convert_node_msec(inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, bm_bilinear_de_standardized)
    inputs_test_no_inter_cad_05_de_standardized, predictions_no_inter_cad_05_de_standardized, label_test_no_inter_cad_05_de_standardized, bm_bilinear_cad_05_de_standardized = convert_node_msec(inputs_test_no_inter_cad_05_de_standardized, predictions_no_inter_cad_05_de_standardized, label_test_no_inter_cad_05_de_standardized, bm_bilinear_cad_05_de_standardized)
    inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized, label_test_no_inter_cad_10_de_standardized, bm_bilinear_cad_10_de_standardized = convert_node_msec(inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized, label_test_no_inter_cad_10_de_standardized, bm_bilinear_cad_10_de_standardized)

    # Initialize numpy array to store the metric values
    NUM_TRAIN_REGENERATE = len(inputs_test_no_inter_de_standardized)
    NUM_TRAIN_REGENERATE_CAD05 = len(inputs_test_no_inter_cad_05_de_standardized)
    NUM_TRAIN_REGENERATE_CAD10 = len(inputs_test_no_inter_cad_10_de_standardized)
 
    # Non interpolation ___________________
    MAE_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)
    RMSE_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0) 

    MAE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    MSE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    RMSE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    PSNR_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    # Bilinear interpolation ___________________
    MAE_benchmark_cad_05 = np.full(NUM_TRAIN_REGENERATE_CAD05, 0.0)
    RMSE_benchmark_cad_05 = np.full(NUM_TRAIN_REGENERATE_CAD05, 0.0)  
    SSIM_benchmark_cad_05 = np.full(NUM_TRAIN_REGENERATE_CAD05, 0.0) 

    MAE_b = np.full(NUM_TRAIN_REGENERATE_CAD05, 0.0)  
    MSE_b = np.full(NUM_TRAIN_REGENERATE_CAD05, 0.0)  
    RMSE_b = np.full(NUM_TRAIN_REGENERATE_CAD05, 0.0)  
    PSNR_b = np.full(NUM_TRAIN_REGENERATE_CAD05, 0.0)  
    SSIM_b = np.full(NUM_TRAIN_REGENERATE_CAD05, 0.0)  
    # Nearest Neighbour interpolation ___________________
    MAE_benchmark_cad_10 = np.full(NUM_TRAIN_REGENERATE_CAD10, 0.0)
    RMSE_benchmark_cad_10 = np.full(NUM_TRAIN_REGENERATE_CAD10, 0.0)  
    SSIM_benchmark_cad_10 = np.full(NUM_TRAIN_REGENERATE_CAD10, 0.0) 

    MAE_nn = np.full(NUM_TRAIN_REGENERATE_CAD10, 0.0)  
    MSE_nn = np.full(NUM_TRAIN_REGENERATE_CAD10, 0.0)  
    RMSE_nn = np.full(NUM_TRAIN_REGENERATE_CAD10, 0.0)  
    PSNR_nn = np.full(NUM_TRAIN_REGENERATE_CAD10, 0.0)  
    SSIM_nn = np.full(NUM_TRAIN_REGENERATE_CAD10, 0.0)  

    # Itterate over all the images predicted, analyse them, plot them and calculate metrics
    for images_prediction in range(len(inputs_test_no_inter_de_standardized)):
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

        # ______No interpolation Qc metrics: 
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

        if images_prediction < 10:
            # Create prediction image for non interpolation 
            create_prediction_image(images_prediction, inputs_test_no_inter_de_standardized, predictions_no_inter_de_standardized, label_test_de_standardized, 0,
                                    MAE_ni[images_prediction], MSE_ni[images_prediction], RMSE_ni[images_prediction], RMSE_benchmark[images_prediction], 
                                    PSNR_ni[images_prediction], SSIM_ni[images_prediction], SSIM_benchmark[images_prediction], HP_RESULT_PATH, 'no_interpolation')

    # Itterate over all the images predicted, analyse them, plot them and calculate metrics
    for images_prediction in range(len(inputs_test_no_inter_cad_05_de_standardized)):
        print(f'Currently analysing image: {images_prediction+1}')
        # _____Benchmark Metrics: 
        MAE_benchmark_cad_05[images_prediction] = np.mean(np.abs(bm_bilinear_cad_05_de_standardized[images_prediction] - label_test_no_inter_cad_05_de_standardized[images_prediction]))
        RMSE_benchmark_cad_05[images_prediction] = np.sqrt(np.mean(np.square(bm_bilinear_cad_05_de_standardized[images_prediction] - label_test_no_inter_cad_05_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted = bm_bilinear_cad_05_de_standardized[images_prediction].squeeze()
        ground_truth = label_test_no_inter_cad_05_de_standardized[images_prediction].squeeze()
        max_value = max(predicted.max(), ground_truth.max())
        min_value = min(predicted.min(), ground_truth.min())
        data_rg = max_value - min_value
        predicted_std = (predicted - min_value) / data_rg
        ground_truth_std = (ground_truth - min_value) / data_rg
        # Calculate SSIM
        SSIM_benchmark_cad_05[images_prediction] = ssim(predicted_std, ground_truth_std, data_range=1.0)

        # ______No interpolation Cad 05 metrics: 
        MAE_b[images_prediction] = np.mean(np.abs(predictions_no_inter_cad_05_de_standardized[images_prediction] - label_test_no_inter_cad_05_de_standardized[images_prediction]))
        MSE_b[images_prediction] = np.mean(np.square(predictions_no_inter_cad_05_de_standardized[images_prediction] - label_test_no_inter_cad_05_de_standardized[images_prediction]))
        RMSE_b[images_prediction] = np.sqrt(np.mean(np.square(predictions_no_inter_cad_05_de_standardized[images_prediction] - label_test_no_inter_cad_05_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted_b = predictions_no_inter_cad_05_de_standardized[images_prediction].squeeze()
        ground_truth_b = label_test_no_inter_cad_05_de_standardized[images_prediction].squeeze()
        max_value_b = max(predicted_b.max(), ground_truth_b.max())
        min_value_b = min(predicted_b.min(), ground_truth_b.min())
        data_rg_b = max_value_b - min_value_b
        predicted_std_b = (predicted_b - min_value_b) / data_rg_b
        ground_truth_std_b = (ground_truth_b - min_value_b) / data_rg_b
        # Calculate SSIM
        SSIM_b[images_prediction] = ssim(predicted_std_b, ground_truth_std_b, data_range=1.0)
        # Standardize values to calculate psnr
        predicted_b_p = predictions_no_inter_cad_05_de_standardized[images_prediction]
        ground_truth_b_p = label_test_no_inter_cad_05_de_standardized[images_prediction]
        max_value_b_p = max(predicted_b_p.max(), ground_truth_b_p.max())
        min_value_b_p = min(predicted_b_p.min(), ground_truth_b_p.min())
        data_rg_b_p = max_value_b_p - min_value_b_p
        predicted_std_b_p = (predicted_b_p - min_value_b_p) / data_rg_b_p
        ground_truth_std_b_p = (ground_truth_b_p - min_value_b_p) / data_rg_b_p
        # Calculate PSNR
        PSNR_b[images_prediction] = psnr(predicted_std_b_p, ground_truth_std_b_p, data_range=1.0)

        if images_prediction < 10:
            # Create prediction image for bilinear interpolation 
            create_prediction_image(images_prediction, inputs_test_no_inter_cad_05_de_standardized, predictions_no_inter_cad_05_de_standardized, label_test_no_inter_cad_05_de_standardized, 0,
                                    MAE_b[images_prediction], MSE_b[images_prediction], RMSE_b[images_prediction], RMSE_benchmark_cad_05[images_prediction], 
                                    PSNR_b[images_prediction], SSIM_b[images_prediction], SSIM_benchmark_cad_05[images_prediction], HP_RESULT_PATH, 'no_interpolation_Cad_05')

    # Itterate over all the images predicted, analyse them, plot them and calculate metrics
    for images_prediction in range(len(inputs_test_no_inter_cad_10_de_standardized)):
        print(f'Currently analysing image: {images_prediction+1}')
        # _____Benchmark Metrics: 
        MAE_benchmark_cad_10[images_prediction] = np.mean(np.abs(bm_bilinear_cad_10_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        RMSE_benchmark_cad_10[images_prediction] = np.sqrt(np.mean(np.square(bm_bilinear_cad_10_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted = bm_bilinear_cad_10_de_standardized[images_prediction].squeeze()
        ground_truth = label_test_no_inter_cad_10_de_standardized[images_prediction].squeeze()
        max_value = max(predicted.max(), ground_truth.max())
        min_value = min(predicted.min(), ground_truth.min())
        data_rg = max_value - min_value
        predicted_std = (predicted - min_value) / data_rg
        ground_truth_std = (ground_truth - min_value) / data_rg
        # Calculate SSIM
        SSIM_benchmark_cad_10[images_prediction] = ssim(predicted_std, ground_truth_std, data_range=1.0)

        # ______No interpolation Cad 10 metrics:  
        MAE_nn[images_prediction] = np.mean(np.abs(predictions_no_inter_cad_10_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        MSE_nn[images_prediction] = np.mean(np.square(predictions_no_inter_cad_10_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        RMSE_nn[images_prediction] = np.sqrt(np.mean(np.square(predictions_no_inter_cad_10_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted_nn = predictions_no_inter_cad_10_de_standardized[images_prediction].squeeze()
        ground_truth_nn = label_test_no_inter_cad_10_de_standardized[images_prediction].squeeze()
        max_value_nn = max(predicted_nn.max(), ground_truth_nn.max())
        min_value_nn = min(predicted_nn.min(), ground_truth_nn.min())
        data_rg_nn = max_value_nn - min_value_nn
        predicted_std_nn = (predicted_nn - min_value_nn) / data_rg_nn
        ground_truth_std_nn = (ground_truth_nn - min_value_nn) / data_rg_nn
        # Calculate SSIM
        SSIM_nn[images_prediction] = ssim(predicted_std_nn, ground_truth_std_nn, data_range=1.0)
        # Standardize values to calculate psnr
        predicted_nn_p = predictions_no_inter_cad_10_de_standardized[images_prediction]
        ground_truth_nn_p = label_test_no_inter_cad_10_de_standardized[images_prediction]
        max_value_nn_p = max(predicted_nn_p.max(), ground_truth_nn_p.max())
        min_value_nn_p = min(predicted_nn_p.min(), ground_truth_nn_p.min())
        data_rg_nn_p = max_value_nn_p - min_value_nn_p
        predicted_std_nn_p = (predicted_nn_p - min_value_nn_p) / data_rg_nn_p
        ground_truth_std_nn_p = (ground_truth_nn_p - min_value_nn_p) / data_rg_nn_p
        # Calculate PSNR
        PSNR_nn[images_prediction] = psnr(predicted_std_nn_p, ground_truth_std_nn_p, data_range=1.0)

        if images_prediction < 10:
            # Create prediction image for nearest neighbour interpolation 
            create_prediction_image(images_prediction, inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized, label_test_no_inter_cad_10_de_standardized, 0,
                                    MAE_nn[images_prediction], MSE_nn[images_prediction], RMSE_nn[images_prediction], RMSE_benchmark_cad_10[images_prediction], 
                                    PSNR_nn[images_prediction], SSIM_nn[images_prediction], SSIM_benchmark_cad_10[images_prediction], HP_RESULT_PATH, 'no_interpolation_Cad_10')

    # Create a box plot to show the metrics instead of a table. 
    # MAE_________________________________________
    data = [MAE_ni, MAE_benchmark, MAE_b, MAE_benchmark_cad_05, MAE_nn, MAE_benchmark_cad_10]
    labels = ['U-Net', 'Baseline', 'U-Net', 'Baseline', 'U-Net', 'Baseline']

    # Creating the box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True, notch=True, showfliers=False)

    # Defining custom color pairs for config and baseline
    color_pairs = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']

    # Assigning colors to the boxes
    for patch, color in zip(box['boxes'], color_pairs):
        patch.set_facecolor(color)

    # Customizing the median line color
    median_color = 'black'  
    for median in box['medians']:
        median.set_color(median_color)

    # Adding mean values to the plot
    for i, d in enumerate(data):
        mean_value = np.mean(d)
        plt.plot(i + 1, mean_value, 'D', color='black', markersize=5, label="Mean" if i == 0 else "")

    # Adding other plot elements
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel('m/s')
    plt.grid(True)

    # Automatically determine y-axis limits
    plt.autoscale()

    # Get current y-axis limits
    y_min, y_max = plt.gca().get_ylim()

    # Set y-axis ticks to steps of 0.1 within the determined range
    y_ticks = np.arange(0, y_max, 0.20)
    plt.yticks(y_ticks)

    # Save the plot as an image file
    plt.savefig(f'{HP_RESULT_PATH}/mean_MAE_box_plot.png')

    # RMSE_________________________________________
    data = [RMSE_ni, RMSE_benchmark, RMSE_b, RMSE_benchmark_cad_05, RMSE_nn, RMSE_benchmark_cad_10]
    labels = ['U-Net', 'Baseline', 'U-Net', 'Baseline', 'U-Net', 'Baseline']

    # Creating the box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True, notch=True, showfliers=False)

    # Defining custom color pairs for config and baseline
    color_pairs = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']

    # Assigning the color pairs manually
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
    y_ticks = np.arange(0, y_max, 0.20)
    plt.yticks(y_ticks)

    # Create custom legend
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Configuration A'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Configuration B'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Configuration C')
    ]

    # Add the legend
    plt.legend(handles=legend_elements, loc='best')

    # Save the plot as an image file
    plt.savefig(f'{HP_RESULT_PATH}/mean_RMSE_box_plot.png')

    # SSIM_________________________________________
    data = [SSIM_ni, SSIM_benchmark, SSIM_b, SSIM_benchmark_cad_05, SSIM_nn, SSIM_benchmark_cad_10]
    labels = ['U-Net', 'Baseline', 'U-Net', 'Baseline', 'U-Net', 'Baseline']

    # Creating the box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True, notch=True, showfliers=False)

    # Defining custom color pairs for config and baseline
    color_pairs = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']

    # Assigning the color pairs manually
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
