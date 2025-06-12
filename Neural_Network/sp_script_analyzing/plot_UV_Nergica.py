import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_csv('/fs/site5/eccc/cmd/x/spb001/ramp_detection/data_ramps/Nergica/Wind/Nergica_pour_E_MMV1_refHT.csv')

# Convert the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Define the date range for filtering
start_date = '2023-06-30 00:00:00'
end_date = '2023-07-01 00:00:00'

# Filter the DataFrame based on the date range
mask = (df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)
filtered_df = df.loc[mask]

# Plot the data from the 'Avg' column
plt.plot(filtered_df['Timestamp'], filtered_df['Avg'])
plt.xlabel('DateTime')
plt.ylabel('Wind Velocity (m/s)')
plt.title('Measured wind velocity (80m) as a Function of DateTime')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
