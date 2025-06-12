#!/usr/bin/env python

import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = '/home/jfg000/ss5/CNN_results_article/non_interpolation_qcnb_UU/NoInterpol_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE_ME_MG_Z0_000000000000_fineTune_HP_Search/image/CNN_Results_model_10/metrics_predicted_images.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)
# Display the first few rows to inspect the DataFrame
print(df.head())

# Remove rows where any of the specified columns have non-numeric values
numeric_columns = ['mae', 'mse', 'rmse', 'psnr', 'ssim']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=numeric_columns)

# Calculate the averages for the specified columns
mae_average = df['mae'].mean()
mse_average = df['mse'].mean()
rmse_average = df['rmse'].mean()
psnr_average = df['psnr'].mean()
ssim_average = df['ssim'].mean()

# Print the averages
print(f"MAE Average: {mae_average}")
print(f"MSE Average: {mse_average}")
print(f"RMSE Average: {rmse_average}")
print(f"PSNR Average: {psnr_average}")
print(f"SSIM Average: {ssim_average}")