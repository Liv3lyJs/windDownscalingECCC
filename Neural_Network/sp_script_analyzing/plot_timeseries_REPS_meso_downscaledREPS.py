import pandas as pd
import matplotlib.pyplot as plt


file1 = 'UV_48.994722_-64.457222_REPS_20231231_20240430_VOISIN.txt_UV_only_verLIN'
file2 = 'UV_48.994722_-64.457222_meso_2024010100_2024043023_VOISIN.txt_UV_only_verLIN'
file3 = 'UV_48.994722_-64.457222_downscaledREPS_2024010100_2024043023_VOISIN.txt_UV_only_verLIN'


# File paths for the three different files
# Function to read and process each file
def read_and_process(file):
    # Read the data from the file with a space delimiter and header
    data = pd.read_csv(file, delimiter=' ', header=0)
    # Convert the first column (date) to datetime format
    data['Date'] = pd.to_datetime(data.iloc[:, 0], format='%Y%m%d%H')
    return data

# Read and process the data from each file
data1 = read_and_process(file1)
data2 = read_and_process(file2)
data3 = read_and_process(file3)

# Plot the second column as a function of the converted date column for each file
plt.plot(data1['Date'], data1.iloc[:, 1], label='REPS')
plt.plot(data2['Date'], data2.iloc[:, 1], label='Meso')
plt.plot(data3['Date'], data3.iloc[:, 1], label='Downscaled REPS')

# Add labels and a legend
plt.xlabel('Date')
plt.ylabel(data1.columns[1])  # Using the header name for the second column
plt.legend()

# Format the x-axis for better date representation
plt.gcf().autofmt_xdate()

# Show the plot
plt.savefig('TimeSeries_REPS_meso_DownscaledREPS_VOISIN_010124_300424.png', dpi=300, bbox_inches='tight')
plt.show()

plt.close()  # This closes the figure to free up memory




