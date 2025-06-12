import matplotlib.pyplot as plt
from datetime import datetime
import argparse

lat='48.986708'
long='-64.455250'
date2extract=2023063000
syst_to_extract_values=["REPS", "meso"]
INTERPOLATION='VOISIN'
#INTERPOLATION='LINEAR'





knots2ms = 0.5144

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
    

    uv_values_converted = [value * knots2ms for value in uv_values]

    plt.plot(dates, uv_values_converted, marker='o', label=syst_to_extract)
  #  plt.xlabel('Date')
  #  plt.ylabel('Wind Velocity (m/s)')
  #  title = f'Wind Velocity as a Function of date from {syst_to_extract} at point\n {lat},{long}, {INTERPOLATION} interpolation'
  #  plt.title(title)
  #  plt.gcf().autofmt_xdate()  # Rotate and format date labels on x-axis
  #  plt.show()

for syst_to_extract in syst_to_extract_values:
    plot_UV(syst_to_extract)

plt.xlabel('Date')
plt.ylabel('Wind Velocity (m/s)')
title = f'Wind Velocity as a Function of Date at Point\n {lat},{long}, {INTERPOLATION} Interpolation'
plt.title(title)
plt.gcf().autofmt_xdate()  # Rotate and format date labels on x-axis
plt.legend()  # Add a legend to differentiate the plots
plt.show()




