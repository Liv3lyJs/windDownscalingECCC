import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import pandas as pd
import math
from dateutil.relativedelta import relativedelta
#Lien vers figures qui donnent rampes selon données mesurées à 80m sur MMV1: 
#/fs/site5/eccc/cmd/x/jfg000/data_reconnaissance_automatique_de_patrons/ramp_event/nergica/mmv1

#Takes as input the measured wind date from Nergica, and the mesoanalysis wind velocity at 10m extracted with
#pgsm at the same lat,long point, that was converted in m/s after being extracted. They are extracted
#with a script on a desired period, chosen usually as one year. Here, when plotting it
#we apply a method to convert the mesoanalysis velocity from 10m to 80m, and we can plot over several months if we want to.


#Site experimental Nergica, position de mat MMV1.
#Info donnee par Nergica: 48 59 41N, 64 27 26 W, correspond à 48.994722, -64.457222
#lat=48.986708
#long=-64.455250
lat='48.994722'
long='-64.457222'

#Je dois adapter, faire série temporelle avec données des passes du REPS avant de pouvoir faire les graphique du REPS
#J'ai juste deux systemes, je ne fais pas de boucle pour l'instant
syst_to_extract='meso'
syst_to_extract_2='REPS'

#INTERPOLATION='VOISIN'
INTERPOLATION='LINEAIR'
#choice of approximation method for wind profile, log, power or noprof
metprof='log'
#metprof='noprof'
#metprof='power'

# Define the start and end dates with specific hours
#start_date = datetime(2022, 7, 10, 14)  # July 10, 2023, 14:00 (2 PM)
#end_date = datetime(2022, 7, 15, 8)     # July 15, 2023, 08:00 (8 AM)
start_date = datetime(2024, 4, 1, 0)  #
end_date = datetime(2024, 4,1, 23)     #




#I have one datafile per year for the mesoanalysis, from 1st Jan to 31st Dec
year2extract=start_date.year
year2extract_str=str(year2extract)
#For REPS, when wanting to get one total year of data in one file, I had to start extracting one day
#before, because I keep only hours 4 to 9. For example if I want data from hour 0 the first of January 2022,
#I had to start extracting from the 31st December 2021 at the 18Z run.
one_year_before = start_date - relativedelta(years=1)
year2extract_2 = one_year_before.year
year2extract_2_str=str(year2extract_2)

#Might need to be adapted depending on the period of the year contained in each file. For 2022 and 2023 I wanted to have one year per file,
#but the other years are not complete. I could have put 2.5 years in one single file though..
filename_mod10m = f'UV_{lat}_{long}_{syst_to_extract}_{year2extract_str}010100_{year2extract_str}043023_{INTERPOLATION}.txt_UV_only_verLIN'
filename_mod10m_2 = f'UV_{lat}_{long}_{syst_to_extract_2}_{year2extract_2_str}1231_{year2extract_str}0430_{INTERPOLATION}.txt_UV_only_verLIN'
filename_obs80m='/fs/site5/eccc/cmd/x/spb001/ramp_detection/data_ramps/Nergica//Wind/Nergica_pour_E_MMV1_refHT.csv'





# Generate a list of hourly timestamps between start_date and end_date
date_range = pd.date_range(start=start_date, end=end_date, freq='H')

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


# Convert date2extract to  datetime object
#start_date = datetime.strptime(str(date2extract), "%Y%m%d%H")
#end_date = start_date + timedelta(hours=24)

# Function to plot UV data
#def plot_UV(syst_to_extract, start_date, end_date):


#meso
df1 = pd.read_csv(filename_mod10m, sep=' ', header=0)  # Assuming space-separated values with a header
# Convert the 'Date' column to datetime format
df1['Date'] = pd.to_datetime(df1['Date'], format='%Y%m%d%H')
# Multiply the wind speed values by the correcting factor for 80m
df1['Wind_Velocity_80m'] = df1['Wind_Velocity_10m(m/s)'] * fact_UV_10_to_80
# Filter the data between the start_date and end_date
filtered_df1 = df1[(df1['Date'] >= start_date) & (df1['Date'] <= end_date)]

# Debugging: Print the date range and number of rows in filtered_df1
print(f"Filtered df1 date range: {filtered_df1['Date'].min()} to {filtered_df1['Date'].max()}")
print(f"Filtered df1 number of rows: {len(filtered_df1)}")

#REPS
df1_REPS = pd.read_csv(filename_mod10m_2, sep=' ', header=0)  # Assuming space-separated values with a header
# Convert the 'Date' column to datetime format
df1_REPS['Date'] = pd.to_datetime(df1_REPS['Date'], format='%Y%m%d%H')
# Multiply the wind speed values by the correcting factor for 80m
df1_REPS['Wind_Velocity_80m'] = df1_REPS['Wind_Velocity_10m(m/s)'] * fact_UV_10_to_80
# Filter the data between the start_date and end_date
filtered_df1_REPS = df1_REPS[(df1_REPS['Date'] >= start_date) & (df1_REPS['Date'] <= end_date)]


# Debugging: Print the date range and number of rows in filtered_df1_REPS
print(f"Filtered df1_REPS date range: {filtered_df1_REPS['Date'].min()} to {filtered_df1_REPS['Date'].max()}")
print(f"Filtered df1_REPS number of rows: {len(filtered_df1_REPS)}")



#Obs
df2 = pd.read_csv(filename_obs80m)  # Assuming comma-separated values with a header
df2['Timestamp'] = pd.to_datetime(df2['Timestamp'])
# Filter the data between the start_date and end_date
filtered_df2 = df2[(df2['Timestamp'] >= start_date) & (df2['Timestamp'] <= end_date)]


# Plot the data
plt.figure(figsize=(12, 6))

# Plot wind speed from the first file
plt.plot(filtered_df1['Date'], filtered_df1['Wind_Velocity_80m'], marker='', linestyle='-', label=f'{syst_to_extract} data translated at 80m with {metprof} ')

# Plot wind speed from the second file (REPS)
plt.plot(filtered_df1_REPS['Date'], filtered_df1_REPS['Wind_Velocity_80m'], marker='', linestyle='-', label=f'{syst_to_extract_2} data translated at 80m with {metprof} ')
 

# Plot the fifth column (Avg) from the third file (obs)
plt.plot(filtered_df2['Timestamp'], filtered_df2['Avg'], marker='', linestyle='--', label='Wind speed obs at 80m')

# Configure the plot
plt.xlabel('Date and Time')
plt.ylabel('Values')
plt.title('Hourly Data Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

formatted_startdate = start_date.strftime('%Y%m%d_%H%M')
formatted_enddate=end_date.strftime('%Y%m%d_%H%M')

# Save the figure as a PNG file with the desired title
filename = f'Wind_Velocity_{lat}_{long}_{formatted_startdate}_{formatted_enddate}_{INTERPOLATION}_{metprof}.png'
plt.savefig(filename)

# Show the plot
plt.show()


