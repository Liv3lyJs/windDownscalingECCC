import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import pandas as pd
import math
#Lien vers figures qui donnent rampes selon données mesurées à 80m sur MMV1: 
#/fs/site5/eccc/cmd/x/jfg000/data_reconnaissance_automatique_de_patrons/ramp_event/nergica/mmv1

#Site experimental Nergica, position de mat MMV1.
#Info donnee par Nergica: 48 59 41N, 64 27 26 W, correspond à 48.994722, -64.457222
#lat=48.986708
#long=-64.455250
lat='48.994722'
long='-64.457222'


#date2extract=2022081814
date2extract=2022080200
syst_to_extract_values=["REPS", "meso"]
#INTERPOLATION='VOISIN'
INTERPOLATION='LINEAR'
#choice of approximation method for wind profile, log, power or noprof
metprof='log'
#metprof='noprof'
#metprof='power'


#Info for log wind profile: https://en.wikipedia.org/wiki/Log_wind_profile
#Zero displacement height in m
zerodisp=0.0   # After discussion with Ayrton Zadra on 080724, and From IFS DOCUMENTATION - Cy47r3 Operational implementation 12 Oct 2021
#PART IV: PHYSICAL PROCESSES at p.52, 
#a reasonable approximation is to consider that the velocity at 10m at an obs station is measured on open land. Zero displacement height is
#aproximated to zero, and the heights z are measured from the reference height z0, so that u(z2) = 0 when z2->0
#Roughness height in m
Z0=1.7
#alpha in wind profile law https://en.wikipedia.org/wiki/Wind_profile_power_law
#approximation for neutral stability
alpha_profile=0.143

#Power law
fact_UV_10_to_80_power=(80.0/10.0)**alpha_profile
#Log profile corrected following the comment above for zero displacement height which is set to 0
fact_UV_10_to_80_log=(math.log((80.0-zerodisp+Z0)/Z0))/(math.log((10.0-zerodisp+Z0)/Z0))
#No profile law
fact_UV_10_to_80_noprof=1


if metprof == "power":
    fact_UV_10_to_80=fact_UV_10_to_80_power
    print("chosen method is power")
    
elif metprof == "log":
    fact_UV_10_to_80=fact_UV_10_to_80_log
    print("chosen method is log")
elif metprof == "noprof":
    fact_UV_10_to_80=fact_UV_10_to_80_noprof
    print("no method chosen for profile")
else:
    # Code to execute if chosen_string is neither "A" nor "B"
    print("chosen method is neither log, power or noprof, there is a problem")


knots2ms = 0.5144

# Convert date2extract to a datetime object
start_date = datetime.strptime(str(date2extract), "%Y%m%d%H")
end_date = start_date + timedelta(hours=24)

# Function to plot UV data
def plot_UV(syst_to_extract):
    filename = f'UV_{lat}_{long}_{syst_to_extract}_{date2extract}_{INTERPOLATION}.txt'
    dates = []
    uv_values = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for i, line in enumerate(lines):
            columns = line.strip().split()
            
            if columns[1] == 'WD':  # Skip lines with 'WD' in the second column
                continue
            
            if columns[1] == 'UV':  # Process lines with 'UV' in the second column
                try:
                    date_str = columns[0]  # First column is the date
                    date = datetime.strptime(date_str, "%Y%m%d%H")
                    value = float(columns[4])  # Fifth column is the value
                    dates.append(date)
                    uv_values.append(value)
                except ValueError as e:
                    print(f"Error parsing line: {line}. Error: {e}")
                    
            if i % 2 == 1:  # Skip every second line
                continue
    
#converting in m/s and converting wind velocity from 10m to 80m the log profile law
    uv_values_converted = [value * knots2ms * fact_UV_10_to_80 for value in uv_values]
    #uv_values_converted = [value * knots2ms for value in uv_values]
    plt.plot(dates, uv_values_converted, marker='o', label=f'{syst_to_extract} Data')



# Function to plot additional data from the CSV file
def plot_additional_data():
    # Load the data into a DataFrame
    df = pd.read_csv('/fs/site5/eccc/cmd/x/spb001/ramp_detection/data_ramps/Nergica/Wind/Nergica_pour_E_MMV1_refHT.csv')

    # Convert the 'Timestamp' column to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Filter the DataFrame based on the calculated date range
    mask = (df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)
    filtered_df = df.loc[mask]

    # Plot the data from the 'Avg' column
    plt.plot(filtered_df['Timestamp'], filtered_df['Avg'], label='Measured Data')






for syst_to_extract in syst_to_extract_values:
    plot_UV(syst_to_extract)

# Plot the additional data
plot_additional_data()


plt.xlabel('Date')
plt.ylabel('Wind Velocity (m/s)')
title = f'Wind Velocity as a Function of Date at Point\n {lat},{long}, 80m height, {INTERPOLATION} Interpolation for model data\n {metprof} method to go from 10 to 80m for model data'
plt.title(title)
plt.gcf().autofmt_xdate()  # Rotate and format date labels on x-axis
plt.legend()  # Add a legend to differentiate the plots
plt.tight_layout()
# Save the figure as a PNG file with the desired title
filename = f'Wind_Velocity_{lat}_{long}_{date2extract}_{INTERPOLATION}_{metprof}.png'
plt.savefig(filename)

plt.show()




