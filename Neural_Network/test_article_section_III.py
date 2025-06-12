#!/usr/bin/env python

"""
Load this package if the user want to recreate standard file. 
. ~spst900/spooki/use_nb_master_python.dot
"""

import fstpy
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
    test_data_label = test_data_label[:, 4:-4, 4:-4, :]
    if verif_flag:
        bm_bilinear_de_standardized = bm_bilinear_de_standardized[:, 4:-4, 4:-4, :]

    if verif_flag:
        return test_data_input, predictions, test_data_label, bm_bilinear_de_standardized
    else:
        return test_data_input, predictions, test_data_label
        
    
def convert_node_msec(inputs_test, predictions, label, bm_bilinear_de_standardized, flag=0):
    factor = 0.514444

    inputs_test = inputs_test * factor
    predictions = predictions * factor
    label = label * factor
    if flag:
        bm_bilinear_de_standardized = bm_bilinear_de_standardized * factor

    if flag:
        return inputs_test, predictions, label, bm_bilinear_de_standardized
    else:
        return inputs_test, predictions, label
    
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

    axes[2].get_figure().savefig(f'{directory}/Ranking_{images+1}.png', bbox_inches='tight')
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

    plt.plot(freqs, psd_pred_db[:len(freqs)], label='General')
    plt.plot(freqs, psd_pred_db2[:len(freqs)], label='Specific')
    plt.plot(freqs, psd_pred_db3[:len(freqs)], label='Zero')
    plt.plot(freqs, psd_benchmark[:len(freqs)], label='Baseline')
    plt.plot(freqs, psd_gt_db[:len(freqs)], label='Ground Truth')

    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Comparison of Power Spectral Densities in dB')
    plt.yscale('log')  # Set the y-axis to log scale
    plt.legend()
    plt.grid(True)
    
    directory = directory + '/image_psd'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f'{directory}/psd_{images+1}.png', bbox_inches='tight')
    plt.close()

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
    
    plt.plot(freqs, avg_psd_pred_db[:len(freqs)], label='General')
    plt.plot(freqs, avg_psd_pred_db2[:len(freqs)], label='Specific')
    plt.plot(freqs, avg_psd_pred_db3[:len(freqs)], label='Zero')
    plt.plot(freqs, avg_psd_benchmark_db[:len(freqs)], label='Baseline')
    plt.plot(freqs, avg_psd_gt_db[:len(freqs)], label='Ground Truth')

    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.yscale('log')  # Set the y-axis to log scale
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/average_psd.png', bbox_inches='tight')
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

    plt.plot(wind_speed_range, pdf1, label='General')
    plt.plot(wind_speed_range, pdf2, label='Specific')
    plt.plot(wind_speed_range, pdf3, label='Zero')
    plt.plot(wind_speed_range, pdfbm, label='Baseline')
    plt.plot(wind_speed_range, pdfgt, label='Ground truth')

    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('PDF')
    plt.title('PDF of Wind Speed Predictions')
    plt.legend(loc='upper right')

    directory = directory + '/image_pdf'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f'{directory}/wind_speed_pdf{images+1}.png', bbox_inches='tight')
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
    plt.plot(wind_speed_range, pdf1, label='General')
    plt.plot(wind_speed_range, pdf2, label='Specific')
    plt.plot(wind_speed_range, pdf3, label='Zero')
    plt.plot(wind_speed_range, pdfbm, label='Baseline')
    plt.plot(wind_speed_range, pdfgt, label='Ground Truth')

    # Add labels, title, and legend
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('PDF')
    plt.title('Average PDF of Wind Speed Predictions')
    plt.legend(loc='upper right')

    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/wind_speed_pdf.png', bbox_inches='tight')
    plt.close()

def combine_graphs(predicted_image1, predicted_image2, predicted_image3, ground_truth_image, directory, test_data_skip_de_standardized):
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
    wind_speed_range = np.linspace(min_wind_speed, max_wind_speed, 55)

    # Calculate the kernel density estimation for non-logarithmic
    kde1 = gaussian_kde(predict1_flatten, bw_method=0.7)
    kde2 = gaussian_kde(predict2_flatten, bw_method=0.7)
    kde3 = gaussian_kde(predict3_flatten, bw_method=0.7)
    kdebm = gaussian_kde(bench_mark, bw_method=0.7)
    kdegt = gaussian_kde(ground_truth_flatten, bw_method=0.7)

    # Calculate the PDFs for the specified range (non-logarithmic)
    pdf1 = kde1(wind_speed_range)
    pdf2 = kde2(wind_speed_range)
    pdf3 = kde3(wind_speed_range)
    pdfbm = kdebm(wind_speed_range)
    pdfgt = kdegt(wind_speed_range)

    # Now calculate the PDFs for the logarithmic graph
    pdf1_log = np.clip(kde1(wind_speed_range), 1e-10, None)
    pdf2_log = np.clip(kde2(wind_speed_range), 1e-10, None)
    pdf3_log = np.clip(kde3(wind_speed_range), 1e-10, None)
    pdfbm_log = np.clip(kdebm(wind_speed_range), 1e-10, None)
    pdfgt_log = np.clip(kdegt(wind_speed_range), 1e-10, None)

    # Create a base figure for non-logarithmic plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot non-logarithmic PDFs on the primary axis
    ax1.plot(wind_speed_range, pdf1, label='General')
    ax1.plot(wind_speed_range, pdf2, label='Specific')
    ax1.plot(wind_speed_range, pdf3, label='Zero')
    ax1.plot(wind_speed_range, pdfbm, label='Baseline')
    ax1.plot(wind_speed_range, pdfgt, label='Ground Truth')

    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('PDF')

    # Move the legend inside the plot in the upper right corner
    ax1.legend(loc='upper right')

    # Create a secondary inset axis for the logarithmic plot (for wind speed > 15 m/s)
    # Position it fully inside the non-logarithmic graph
    ax2 = fig.add_axes([0.47, 0.35, 0.40, 0.35])  # Adjust these values for the inset position and size
    high_speed_filter = wind_speed_range > 6  # Filter wind speeds > 15 m/s
    ax2.plot(wind_speed_range[high_speed_filter], pdf1_log[high_speed_filter], label='Bilinear interpolation (log)')
    ax2.plot(wind_speed_range[high_speed_filter], pdf2_log[high_speed_filter], label='Nearest-Neighbour interpolation (log)')
    ax2.plot(wind_speed_range[high_speed_filter], pdf3_log[high_speed_filter], label='No interpolation (log)')
    ax2.plot(wind_speed_range[high_speed_filter], pdfbm_log[high_speed_filter], label='Baseline (log)')
    ax2.plot(wind_speed_range[high_speed_filter], pdfgt_log[high_speed_filter], label='Ground Truth (log)')

    ax2.set_yscale('log')
    ax2.set_xlabel('Wind Speed (m/s)', fontsize=10)
    ax2.set_ylabel('PDF (log scale)', fontsize=10)

    # No legend for the inset
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)

    # Save the combined figure
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/combined_wind_speed_pdf.png', bbox_inches='tight')
    plt.close()

def average_error_location(predictions, prediction_bilinear, prediction_nearest, label, benchMark, directory):
    """
    Calculate the average error location on the whole test set for a specific domain. 
    """
    # Calculate the average error on the whole domain data
    # print(f'Shape of predictions is : {predictions.shape}')
    # print(f'Shape of benchmark is : {benchMark.shape}')
    # print(f'Shape of label is : {label.shape}')
    error_pred = np.abs(predictions - label)
    error_pred_bi = np.abs(prediction_bilinear - label)
    error_pred_near = np.abs(prediction_nearest - label)
    error_bm = np.abs(benchMark - label)
    # print(f'shape for error_pred and error_bm are : {error_pred.shape} and {error_bm.shape}')
    # Compute the average to know the location of the errors
    average_error_pred = np.mean(error_pred, axis=0)
    average_error_pred_bi = np.mean(error_pred_bi, axis=0)
    average_error_pred_near = np.mean(error_pred_near, axis=0)
    average_error_bm = np.mean(error_bm, axis=0)
    # print(f'Shape for average_error_pred and average_error_bm are: {average_error_pred.shape} and {average_error_bm.shape}')
    # Calculate the average error for the entire dataset as one final value
    final_average_error_pred = round(np.mean(average_error_pred), 2)
    final_average_error_pred_bi = round(np.mean(average_error_pred_bi), 2)
    final_average_error_pred_near = round(np.mean(average_error_pred_near), 2)
    final_average_error_bm = round(np.mean(average_error_bm), 2)

    # Show the result on a graph
    fig, axes = plt.subplots(1, 4, figsize=(30, 8))

    vmin = min(np.min(average_error_pred_bi), np.min(average_error_pred_near))  #min(np.min(average_error_pred), np.min(average_error_pred_bi), np.min(average_error_pred_near))
    vmax = max(np.max(average_error_pred_bi), np.max(average_error_pred_near))   #max(np.max(average_error_pred), np.max(average_error_pred_bi), np.max(average_error_pred_near))

    im1 = axes[0].imshow(average_error_pred, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'General model:\n Average: {final_average_error_pred:.2f} m/s', fontsize=20)
    fig.colorbar(im1, ax=axes[0], orientation='vertical')

    im2 = axes[1].imshow(average_error_pred_bi, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'Specific model:\n Average: {final_average_error_pred_bi:.2f} m/s', fontsize=20)
    fig.colorbar(im2, ax=axes[1], orientation='vertical')

    im3 = axes[2].imshow(average_error_pred_near, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title(f'Zero model:\n Average: {final_average_error_pred_near:.2f} m/s', fontsize=20)
    fig.colorbar(im3, ax=axes[2], orientation='vertical')

    im4 = axes[3].imshow(average_error_bm, cmap='magma', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[3].set_title(f'Baseline:\n Average: {final_average_error_bm:.2f} m/s (capped colorscale)', fontsize=20)
    fig.colorbar(im4, ax=axes[3], orientation='vertical')
    
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[3].axis('off')
    
    plt.savefig(f'{directory}/average_error_location.png', bbox_inches='tight')
    plt.close()

# Main 
if __name__ == '__main__':
    plt.rcParams['font.size'] = 14
    #VARIABLES
    ANALYSED_DOMAIN = f"domaine2"
    # Path to save the output prediction 
    HP_RESULT_PATH = f"/home/jfg000/ss5/CNN_results_article/articles_testing/article_section_III_{ANALYSED_DOMAIN}" 
    
    # Parameter for no interpolation Cad 10 domains:
    lr_no_inter_cad_10 = 0.001
    drop_no_inter_cad_10 = 0.35
    alpha_no_inter_cad_10 = 0.25
    loss_no_inter_cad_10 = 1.00
    # Path for model weights
    path_weights_no_inter_cad_10 = '/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000001_Final_search_HP_Search/model/Searchmodel_1_cb' 
    # Input data to test the model with 
    input_file_test_no_inter_cad_10 = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/domaine_creation/east_canada_squential_test_allpasses_{ANALYSED_DOMAIN}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize_10.npy'
    skip_no_inter_cad_10 = f'/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/domaine_creation/east_canada_squential_test_allpasses_{ANALYSED_DOMAIN}/skip_connection_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize_10.npy'
    topo_no_inter_cad_10 = f'/home/jfg000/ss5/data_superResolution/topography/non_interpolation/domaine_creation/east_canada_squential_train_allpasses_{ANALYSED_DOMAIN}/ME_MG_Z0_data/normalize_10.npy'
    # Label data 
    label_cad_10 = f'/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/domaine_creation/east_canada_squential_test_allpasses_{ANALYSED_DOMAIN}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize_10.npy'
    std_label_de_standardize_path_cad_10 = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_label_de_standardize_path_cad_10 = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    # Std deviation and mean to de-standardize the prediction
    std_inputs_de_standardize_path_no_inter_cad_10 = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/std.npy'
    mean_inputs_de_standardize_path_no_inter_cad_10 = '/home/jfg000/ss5/data_superResolution/input/domaine/non_interpolation/data_used_in_neural_network_cad_10/east_canada_squential_train_allpasses_domaine/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/mean.npy'
    
    # Parameter for no interpolation transfer learning Cad 10 domains:
    lr_no_inter_cad_10_tl = 0.00001
    drop_no_inter_cad_10_tl = 0.30  
    alpha_no_inter_cad_10_tl = 0.20
    loss_no_inter_cad_10_tl = 1.00
    # Path for model weights 
    path_weights_no_inter_cad_10_tl = f'/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_transfer_learningUU/tl_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_01_{ANALYSED_DOMAIN}_transfer_learning_Final_HP_Search_HP_Search/model/Searchmodel_5_cb'   #f'/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_transfer_learningUU/transferLearning_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_00000000000_Preliminary_search_{ANALYSED_DOMAIN}_transfer_learning_HP_Search_HP_Search/model/Searchmodel_1_cb'

    # Parameter for no interpolation domaine trained from scratch:
    lr_no_inter_cad_10_ts = 0.0001
    drop_no_inter_cad_10_ts = 0.35
    alpha_no_inter_cad_10_ts = 0.25
    loss_no_inter_cad_10_ts = 1.00
    # Path for model weights
    path_weights_no_inter_cad_10_ts = f'/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_transfer_learningUU/tl_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_01_{ANALYSED_DOMAIN}_no_transfer_learning_Final_HP_Search_HP_Search/model/Searchmodel_1_cb'   #f'/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_transfer_learningUU/transferLearning_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_00000000000_Preliminary_search_{ANALYSED_DOMAIN}_no_transfer_learning_HP_Search_HP_Search/model/Searchmodel_1_cb' 

    # Loading the Baseline
    input_file_benchmark = f'/home/jfg000/ss5/data_superResolution/input/domaine/bilinear_interpolation/domaine_creation/east_canada_squential_test_allpasses_{ANALYSED_DOMAIN}/UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_data/normalize.npy' #normalize_10
    

    # Create the models that will be used to do prediction on 
    model_no_inter_cad_10 = mod.deepRU_article_non_interpolation_combine_loss(units=0, activation=None, lr=lr_no_inter_cad_10, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0,
                                                                                                                            num_hidden_layer=None, WIND=None, regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, drop_rate=drop_no_inter_cad_10, alpha=alpha_no_inter_cad_10, loss_weights=loss_no_inter_cad_10)
    model_no_inter_cad_10_tl = mod.deepRU_article_non_interpolation_combine_loss_transferLearning(units=0, activation=None, lr=lr_no_inter_cad_10_tl, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0,
                                                                                                                            num_hidden_layer=None, WIND=None, regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, drop_rate=drop_no_inter_cad_10_tl, alpha=alpha_no_inter_cad_10_tl, loss_weights=loss_no_inter_cad_10_tl)
    model_no_inter_cad_10_ts = mod.deepRU_article_non_interpolation_combine_loss_transferLearning(units=0, activation=None, lr=lr_no_inter_cad_10_ts, kernel_size=0, kernel_size_e1=0, kernel_size_e2=0, kernel_size_e3=0, kernel_size_d1=0, kernel_size_d2=0, kernel_size_d3=0,
                                                                                                                            num_hidden_layer=None, WIND=None, regL1_conv01=0, regL1_conv02=0, regL2_conv01=0, regL2_conv02=0, drop_rate=drop_no_inter_cad_10_ts, alpha=alpha_no_inter_cad_10_ts, loss_weights=loss_no_inter_cad_10_ts)
    # Load the model to do prediction on: 
    # Load model weights for no interpolation, bilinear and nearest neighbour
    model_no_inter_cad_10.load_weights(path_weights_no_inter_cad_10)
    model_no_inter_cad_10_tl.load_weights(path_weights_no_inter_cad_10_tl)
    model_no_inter_cad_10_ts.load_weights(path_weights_no_inter_cad_10_ts)

    # Load the test data
    #Loading the label data
    label_test_cad_10 = np.load(label_cad_10)
    bm_bilinear = np.load(input_file_benchmark)

    # Load test data for specific strategie
    print('load the input data for non interpolation Cad 10')
    inputs_test_no_inter_cad_10 = np.load(input_file_test_no_inter_cad_10)
    skip_no_inter_cad_10 = np.load(skip_no_inter_cad_10)
    topo_no_inter_cad_10 = np.load(topo_no_inter_cad_10)

    # Reshape the test data [sample, width, height, channel] 
    inputs_test_no_inter_cad_10 = inputs_test_no_inter_cad_10.transpose((0,2,3,1)) 
    label_test_cad_10 = label_test_cad_10.transpose((0,2,3,1))
    skip_no_inter_cad_10 = skip_no_inter_cad_10.transpose((0,2,3,1))

    bm_bilinear = bm_bilinear.transpose((0,2,3,1))[:, 8:56, 8:56, 0]
    bm_bilinear = bm_bilinear[:, :, :, np.newaxis]

    topo_no_inter_cad_10 = topo_no_inter_cad_10.transpose((0,2,3,1))

    # Adjust the topographie if the user is only testing on a single domain
    topo_no_inter_cad_10 = np.tile(topo_no_inter_cad_10, (inputs_test_no_inter_cad_10.shape[0], 1, 1, 1))

    # Do the prediction 
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
    # Prediction for nearest neighbour 
    predictions_no_inter_cad_10_tl = model_no_inter_cad_10_tl.predict(
        x=[inputs_test_no_inter_cad_10, topo_no_inter_cad_10, skip_no_inter_cad_10],
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    # Prediction for nearest neighbour 
    predictions_no_inter_cad_10_ts = model_no_inter_cad_10_ts.predict(
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
    inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized, label_test_no_inter_cad_10_de_standardized, bm_bilinear_de_standardized = de_standardize(inputs_test_no_inter_cad_10, predictions_no_inter_cad_10, label_test_cad_10, 
                                                                                                                                                         std_inputs_de_standardize_path_no_inter_cad_10, mean_inputs_de_standardize_path_no_inter_cad_10,
                                                                                                                                                         std_label_de_standardize_path_cad_10, mean_label_de_standardize_path_cad_10, bm_bilinear, 1)
    inputs_test_no_inter_cad_10_de_standardized_1, predictions_no_inter_cad_10_tl_de_standardized, label_test_no_inter_cad_10_de_standardized_1 = de_standardize(inputs_test_no_inter_cad_10, predictions_no_inter_cad_10_tl, label_test_cad_10, 
                                                                                                                                                         std_inputs_de_standardize_path_no_inter_cad_10, mean_inputs_de_standardize_path_no_inter_cad_10,
                                                                                                                                                         std_label_de_standardize_path_cad_10, mean_label_de_standardize_path_cad_10, bm_bilinear)
    inputs_test_no_inter_cad_10_de_standardized_2, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized_2 = de_standardize(inputs_test_no_inter_cad_10, predictions_no_inter_cad_10_ts, label_test_cad_10, 
                                                                                                                                                         std_inputs_de_standardize_path_no_inter_cad_10, mean_inputs_de_standardize_path_no_inter_cad_10,
                                                                                                                                                         std_label_de_standardize_path_cad_10, mean_label_de_standardize_path_cad_10, bm_bilinear)

    # Convert the units from knots to meters per seconds
    inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized_ct, label_test_no_inter_cad_10_de_standardized, bm_bilinear_de_standardized = convert_node_msec(inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized, label_test_no_inter_cad_10_de_standardized, bm_bilinear_de_standardized, 1)
    inputs_test_no_inter_cad_10_de_standardized_1, predictions_no_inter_cad_10_tl_de_standardized_no_cp, label_test_no_inter_cad_10_de_standardized_1 = convert_node_msec(inputs_test_no_inter_cad_10_de_standardized_1, predictions_no_inter_cad_10_tl_de_standardized, label_test_no_inter_cad_10_de_standardized_1, bm_bilinear_de_standardized)
    inputs_test_no_inter_cad_10_de_standardized_2, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized_2 = convert_node_msec(inputs_test_no_inter_cad_10_de_standardized_2, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized_2, bm_bilinear_de_standardized)

    # Remove the padding pixels for the the input, prediction, label image and skip connection 
    inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized_pd, label_test_no_inter_cad_10_de_standardized, bm_bilinear_de_standardized = crop_result(inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized_ct, label_test_no_inter_cad_10_de_standardized, bm_bilinear_de_standardized, 1)
    inputs_test_no_inter_cad_10_de_standardized_1, predictions_no_inter_cad_10_tl_de_standardized_ct, label_test_no_inter_cad_10_de_standardized_1 = crop_result(inputs_test_no_inter_cad_10_de_standardized_1, predictions_no_inter_cad_10_tl_de_standardized_no_cp, label_test_no_inter_cad_10_de_standardized_1, bm_bilinear_de_standardized)
    inputs_test_no_inter_cad_10_de_standardized_2, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized_2 = crop_result(inputs_test_no_inter_cad_10_de_standardized_2, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized_2, bm_bilinear_de_standardized)

    # Initialize numpy array to store the metric values
    NUM_TRAIN_REGENERATE = len(inputs_test_no_inter_cad_10_de_standardized)
    MAE_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)
    RMSE_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_benchmark = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    # No interpolation general model
    MAE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    MSE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    RMSE_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    PSNR_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_ni = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    # No interpolation transfer learning
    MAE_tl = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    MSE_tl = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    RMSE_tl = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    PSNR_tl = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_tl = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    # No interpolation No transfer learning
    MAE_ts = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    MSE_ts = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    RMSE_ts = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    PSNR_ts = np.full(NUM_TRAIN_REGENERATE, 0.0)  
    SSIM_ts = np.full(NUM_TRAIN_REGENERATE, 0.0)  

    # Itterate over all the images predicted, analyse them, plot them and calculate metrics
    for images_prediction in range(NUM_TRAIN_REGENERATE):
        print(f'Currently analysing image: {images_prediction+1}')
        # _____Benchmark Metrics: 
        MAE_benchmark[images_prediction] = np.mean(np.abs(bm_bilinear_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        RMSE_benchmark[images_prediction] = np.sqrt(np.mean(np.square(bm_bilinear_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted = bm_bilinear_de_standardized[images_prediction].squeeze()
        ground_truth = label_test_no_inter_cad_10_de_standardized[images_prediction].squeeze()
        max_value = max(predicted.max(), ground_truth.max())
        min_value = min(predicted.min(), ground_truth.min())
        data_rg = max_value - min_value
        predicted_std = (predicted - min_value) / data_rg
        ground_truth_std = (ground_truth - min_value) / data_rg
        # Calculate SSIM
        SSIM_benchmark[images_prediction] = ssim(predicted_std, ground_truth_std, data_range=1.0)

        # ______No interpolation Cad 10 metrics:  
        MAE_ni[images_prediction] = np.mean(np.abs(predictions_no_inter_cad_10_de_standardized_pd[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        MSE_ni[images_prediction] = np.mean(np.square(predictions_no_inter_cad_10_de_standardized_pd[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        RMSE_ni[images_prediction] = np.sqrt(np.mean(np.square(predictions_no_inter_cad_10_de_standardized_pd[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted_nn = predictions_no_inter_cad_10_de_standardized_pd[images_prediction].squeeze()
        ground_truth_nn = label_test_no_inter_cad_10_de_standardized[images_prediction].squeeze()
        max_value_nn = max(predicted_nn.max(), ground_truth_nn.max())
        min_value_nn = min(predicted_nn.min(), ground_truth_nn.min())
        data_rg_nn = max_value_nn - min_value_nn
        predicted_std_nn = (predicted_nn - min_value_nn) / data_rg_nn
        ground_truth_std_nn = (ground_truth_nn - min_value_nn) / data_rg_nn
        # Calculate SSIM
        SSIM_ni[images_prediction] = ssim(predicted_std_nn, ground_truth_std_nn, data_range=1.0)
        # Standardize values to calculate psnr
        predicted_nn_p = predictions_no_inter_cad_10_de_standardized_pd[images_prediction]
        ground_truth_nn_p = label_test_no_inter_cad_10_de_standardized[images_prediction]
        max_value_nn_p = max(predicted_nn_p.max(), ground_truth_nn_p.max())
        min_value_nn_p = min(predicted_nn_p.min(), ground_truth_nn_p.min())
        data_rg_nn_p = max_value_nn_p - min_value_nn_p
        predicted_std_nn_p = (predicted_nn_p - min_value_nn_p) / data_rg_nn_p
        ground_truth_std_nn_p = (ground_truth_nn_p - min_value_nn_p) / data_rg_nn_p
        # Calculate PSNR
        PSNR_ni[images_prediction] = psnr(predicted_std_nn_p, ground_truth_std_nn_p, data_range=1.0)

        # ______No interpolation Cad 10 Transfer learning metrics:  
        MAE_tl[images_prediction] = np.mean(np.abs(predictions_no_inter_cad_10_tl_de_standardized_ct[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        MSE_tl[images_prediction] = np.mean(np.square(predictions_no_inter_cad_10_tl_de_standardized_ct[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        RMSE_tl[images_prediction] = np.sqrt(np.mean(np.square(predictions_no_inter_cad_10_tl_de_standardized_ct[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted_nn = predictions_no_inter_cad_10_tl_de_standardized_ct[images_prediction].squeeze()
        ground_truth_nn = label_test_no_inter_cad_10_de_standardized[images_prediction].squeeze()
        max_value_nn = max(predicted_nn.max(), ground_truth_nn.max())
        min_value_nn = min(predicted_nn.min(), ground_truth_nn.min())
        data_rg_nn = max_value_nn - min_value_nn
        predicted_std_nn = (predicted_nn - min_value_nn) / data_rg_nn
        ground_truth_std_nn = (ground_truth_nn - min_value_nn) / data_rg_nn
        # Calculate SSIM
        SSIM_tl[images_prediction] = ssim(predicted_std_nn, ground_truth_std_nn, data_range=1.0)
        # Standardize values to calculate psnr
        predicted_nn_p = predictions_no_inter_cad_10_tl_de_standardized_ct[images_prediction]
        ground_truth_nn_p = label_test_no_inter_cad_10_de_standardized[images_prediction]
        max_value_nn_p = max(predicted_nn_p.max(), ground_truth_nn_p.max())
        min_value_nn_p = min(predicted_nn_p.min(), ground_truth_nn_p.min())
        data_rg_nn_p = max_value_nn_p - min_value_nn_p
        predicted_std_nn_p = (predicted_nn_p - min_value_nn_p) / data_rg_nn_p
        ground_truth_std_nn_p = (ground_truth_nn_p - min_value_nn_p) / data_rg_nn_p
        # Calculate PSNR
        PSNR_tl[images_prediction] = psnr(predicted_std_nn_p, ground_truth_std_nn_p, data_range=1.0)

        # ______No interpolation Cad 10 No Transfer learning metrics:  
        MAE_ts[images_prediction] = np.mean(np.abs(predictions_no_inter_cad_10_ts_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        MSE_ts[images_prediction] = np.mean(np.square(predictions_no_inter_cad_10_ts_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction]))
        RMSE_ts[images_prediction] = np.sqrt(np.mean(np.square(predictions_no_inter_cad_10_ts_de_standardized[images_prediction] - label_test_no_inter_cad_10_de_standardized[images_prediction])))
        # Standardize the values to calculate ssim
        predicted_nn = predictions_no_inter_cad_10_ts_de_standardized[images_prediction].squeeze()
        ground_truth_nn = label_test_no_inter_cad_10_de_standardized[images_prediction].squeeze()
        max_value_nn = max(predicted_nn.max(), ground_truth_nn.max())
        min_value_nn = min(predicted_nn.min(), ground_truth_nn.min())
        data_rg_nn = max_value_nn - min_value_nn
        predicted_std_nn = (predicted_nn - min_value_nn) / data_rg_nn
        ground_truth_std_nn = (ground_truth_nn - min_value_nn) / data_rg_nn
        # Calculate SSIM
        SSIM_ts[images_prediction] = ssim(predicted_std_nn, ground_truth_std_nn, data_range=1.0)
        # Standardize values to calculate psnr
        predicted_nn_p = predictions_no_inter_cad_10_ts_de_standardized[images_prediction]
        ground_truth_nn_p = label_test_no_inter_cad_10_de_standardized[images_prediction]
        max_value_nn_p = max(predicted_nn_p.max(), ground_truth_nn_p.max())
        min_value_nn_p = min(predicted_nn_p.min(), ground_truth_nn_p.min())
        data_rg_nn_p = max_value_nn_p - min_value_nn_p
        predicted_std_nn_p = (predicted_nn_p - min_value_nn_p) / data_rg_nn_p
        ground_truth_std_nn_p = (ground_truth_nn_p - min_value_nn_p) / data_rg_nn_p
        # Calculate PSNR
        PSNR_ts[images_prediction] = psnr(predicted_std_nn_p, ground_truth_std_nn_p, data_range=1.0)

        if images_prediction < 10:
            # Create prediction image for general model
            create_prediction_image(images_prediction, inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_de_standardized_pd, label_test_no_inter_cad_10_de_standardized, 0,
                                    MAE_ni[images_prediction], MSE_ni[images_prediction], RMSE_ni[images_prediction], RMSE_benchmark[images_prediction], 
                                    PSNR_ni[images_prediction], SSIM_ni[images_prediction], SSIM_benchmark[images_prediction], HP_RESULT_PATH, 'no_interpolation_Cad_10')
            create_prediction_image(images_prediction, inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_tl_de_standardized_ct, label_test_no_inter_cad_10_de_standardized, 0,
                                    MAE_tl[images_prediction], MSE_tl[images_prediction], RMSE_tl[images_prediction], RMSE_benchmark[images_prediction], 
                                    PSNR_tl[images_prediction], SSIM_tl[images_prediction], SSIM_benchmark[images_prediction], HP_RESULT_PATH, 'no_interpolation_tl_Cad_10')
            create_prediction_image(images_prediction, inputs_test_no_inter_cad_10_de_standardized, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized, 0,
                                    MAE_ts[images_prediction], MSE_ts[images_prediction], RMSE_ts[images_prediction], RMSE_benchmark[images_prediction], 
                                    PSNR_ts[images_prediction], SSIM_ts[images_prediction], SSIM_benchmark[images_prediction], HP_RESULT_PATH, 'no_interpolation_ts_Cad_10')

            # Compute the power spectral graphic for the three interpolation strategies
            #power_spectral_density_graph_one_image(predictions_no_inter_cad_10_de_standardized_pd, predictions_no_inter_cad_10_tl_de_standardized_ct, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized, HP_RESULT_PATH, images_prediction, bm_bilinear_de_standardized)
            #pdf(predictions_no_inter_cad_10_de_standardized_pd, predictions_no_inter_cad_10_tl_de_standardized_ct, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized, HP_RESULT_PATH, images_prediction, bm_bilinear_de_standardized)

    # Compute the average Power spectal on the whole test set
    power_spectral_density_graph_average(predictions_no_inter_cad_10_de_standardized_pd, predictions_no_inter_cad_10_tl_de_standardized_ct, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized, HP_RESULT_PATH, bm_bilinear_de_standardized)
    average_pdf(predictions_no_inter_cad_10_de_standardized_pd, predictions_no_inter_cad_10_tl_de_standardized_ct, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized, HP_RESULT_PATH, bm_bilinear_de_standardized)
    combine_graphs(predictions_no_inter_cad_10_de_standardized_pd, predictions_no_inter_cad_10_tl_de_standardized_ct, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized, HP_RESULT_PATH, bm_bilinear_de_standardized)
    average_error_location(predictions_no_inter_cad_10_de_standardized_pd, predictions_no_inter_cad_10_tl_de_standardized_ct, predictions_no_inter_cad_10_ts_de_standardized, label_test_no_inter_cad_10_de_standardized, bm_bilinear_de_standardized, HP_RESULT_PATH)

    # Create a box plot to show the metrics instead of a table. 
    # MAE_________________________________________
    data = [MAE_ni, MAE_tl, MAE_ts, MAE_benchmark]
    labels = ['General', 'Specific', 'Zero', 'Baseline']
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
    means = [np.mean(d) for d in data]
    # Adding mean values to the plot
    for i, d in enumerate(data):
        mean_value = np.mean(d)
        median_value = np.median(d)
        q1_value = np.percentile(d, 25)
        q3_value = np.percentile(d, 75)

        plt.plot(i + 1, mean_value, 'D', color='black', markersize=5, label="Mean" if i == 0 else "")

        # # Annotating mean, median, Q1, and Q3 values
        # plt.text(i + 1.1, mean_value, f'Mean: {mean_value:.2f}', verticalalignment='center', color='black')
        # plt.text(i + 1.1, median_value, f'Median: {median_value:.2f}', verticalalignment='center', color=median_color)
        # plt.text(i + 1.1, q1_value, f'Q1: {q1_value:.2f}', verticalalignment='center', color='blue')
        # plt.text(i + 1.1, q3_value, f'Q3: {q3_value:.2f}', verticalalignment='center', color='blue')

    # Adding other plot elements
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel('m/s', fontsize=14)
    plt.grid(True)

    # Automatically determine y-axis limits
    plt.autoscale()

    # Get current y-axis limits
    y_min, y_max = plt.gca().get_ylim()

    # Set y-axis ticks to steps of 0.1 within the determined range
    y_ticks = np.arange(0.00, y_max, 0.2)
    plt.yticks(y_ticks)

    # Save the plot as an image file
    plt.savefig(f'{HP_RESULT_PATH}/mean_MAE_box_plot.png', bbox_inches='tight')

    # RMSE_________________________________________
    data = [RMSE_ni, RMSE_tl, RMSE_ts, RMSE_benchmark]
    labels = ['General', 'Specific', 'Zero', 'Baseline']
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

        # # Annotating mean, median, Q1, and Q3 values
        # plt.text(i + 1.1, mean_value, f'Mean: {mean_value:.2f}', verticalalignment='center', color='black')
        # plt.text(i + 1.1, median_value, f'Median: {median_value:.2f}', verticalalignment='center', color=median_color)
        # plt.text(i + 1.1, q1_value, f'Q1: {q1_value:.2f}', verticalalignment='center', color='blue')
        # plt.text(i + 1.1, q3_value, f'Q3: {q3_value:.2f}', verticalalignment='center', color='blue')

    # Adding other plot elements
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel('m/s', fontsize=14)
    plt.grid(True)
    
    # Automatically determine y-axis limits
    plt.autoscale()

    # Get current y-axis limits
    y_min, y_max = plt.gca().get_ylim()

    # Set y-axis ticks to steps of 0.1 within the determined range
    y_ticks = np.arange(0.10, y_max, 0.2)
    plt.yticks(y_ticks)

    # Save the plot as an image file
    plt.savefig(f'{HP_RESULT_PATH}/mean_RMSE_box_plot.png', bbox_inches='tight')

    # SSIM_________________________________________
    data = [SSIM_ni, SSIM_tl, SSIM_ts, SSIM_benchmark]
    labels = ['General', 'Specific', 'Zero', 'Baseline']
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

        # # Annotating mean, median, Q1, and Q3 values
        # plt.text(i + 1.1, mean_value, f'Mean: {mean_value:.2f}', verticalalignment='center', color='black')
        # plt.text(i + 1.1, median_value, f'Median: {median_value:.2f}', verticalalignment='center', color=median_color)
        # plt.text(i + 1.1, q1_value, f'Q1: {q1_value:.2f}', verticalalignment='center', color='blue')
        # plt.text(i + 1.1, q3_value, f'Q3: {q3_value:.2f}', verticalalignment='center', color='blue')

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
    plt.savefig(f'{HP_RESULT_PATH}/mean_SSIM_box_plot.png', bbox_inches='tight')

    # # Save the data to npy format.
    # print(f'Saving the data into npy format.')
    # np.save('/home/jfg000/ss5/CNN_results_article/non_interpolation_cad_10_transfer_learningUU/transferLearning_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_00000000001_Preliminary_search_domaine1_no_transfer_learning_HP_Search_HP_Search/results_nn_dom1', inputs_test_no_inter_cad_10_de_standardized)

    # Convert the numpy array to standard file____________________________________
    def get_sort_key_label(filename):
        """
        Put the input data into chronological order. From the latest date to the most recent date. 
        Input
        :Param filename: Names of the file that is currently being analysed.
        Output
        :Param sort_key: Names of the file in chronological order.
        """
        datetime_part = filename[:10]

        return datetime_part
    
    save_data = False
    if save_data:
        # Premiere etape, creer les fichiers stands avec les information pertinentes
        # Prendre un echantillon du fichier source de la mesoanalyse
        LABEL_DATA = '/home/jfg000/ss5/data_superResolution/label/domaine/non_interpolation/domaine_creation/east_canada_squential_test_allpasses_domaine1/crop_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE'
        output_dir = f"/home/jfg000/ss5/CNN_results_article/articles_testing/article_section_III_{ANALYSED_DOMAIN}/results_std"
        files_label = [file for file in os.listdir(LABEL_DATA)]
        files_label.sort(key=get_sort_key_label)

        # Loop throw all test files for mesoanalysis
        for i, file in enumerate(files_label):
            print(f'File number: {i+1}')
            print(f'{file}, type : {type(file)}')
            file_name = os.path.join(LABEL_DATA, file)
            # Selection TT a 1 niveau pour pouvoir copier le champ UV
            df = fstpy.StandardFileReader(file_name).to_pandas()
            # Retirer !! et P0
            df = df.loc[df.nomvar.isin(['UV', 'WD', '>>', '^^'])]
            print(f'This is the dataframe: {df}')
            # Ecrire le fichier resultant
            # Create the directory if it doesn't exists:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, file)
            fstpy.StandardFileWriter(output_file, df, overwrite=True).to_fst()

        for i, file in enumerate(files_label):
            print(f'File number: {i+1}')
            # Aller chercher les donnees du downscaling
            uv = predictions_no_inter_cad_10_de_standardized[i].copy()
            uv = np.squeeze(uv)
            uv = np.swapaxes(uv, -2, -1)
            print(f'The uv shape is : {uv.shape}')
            # Formater les enregistrements
            file_name = os.path.join(output_dir, file)
            df1 = fstpy.StandardFileReader(file_name).to_pandas()
            df1.drop(columns='d')
            # Renommer WD pour UV_final contenant les prédictions finales
            df1.loc[df1.nomvar=='WD', 'nomvar'] = 'UV_final'

            # Mettre ip1 a 0
            df1.loc[(df1.nomvar=='UV_final'), 'ip1'] = 0
            # Afficher
            df1.drop(columns='d')

            # Assigner les array numpy au bons enregistrements en utilisant les index de dataframe
            df1.at[3, 'd'] = uv

            # Ecrire le nouveau fichier
            name_output = f'{file}'
            output_dir_final = f"/home/jfg000/ss5/CNN_results_article/articles_testing/article_section_III_{ANALYSED_DOMAIN}/results_std_final"
            if not os.path.exists(output_dir_final):
                os.makedirs(output_dir_final)
            file_name_opt = os.path.join(output_dir_final, name_output)
            print(f'We are writting the new file at output: {file_name_opt}')
            fstpy.StandardFileWriter(file_name_opt, df1, overwrite=True).to_fst()