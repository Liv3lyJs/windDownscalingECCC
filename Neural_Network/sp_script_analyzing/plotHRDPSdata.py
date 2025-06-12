

import json
import matplotlib.pyplot as plt

# Define the starting line number for UV data
start_line = 898
end_hour = 48

# Read the file and skip lines until the UV data starts
with open('/fs/site5/eccc/cmd/x/spb001/maestro/hrdps_icing_test_001/hub/product_dbase/windfarms/json/2024032306_0068.json') as f:
    lines = f.readlines()

# Extract UV data lines from the start line to the end line
uv_data_lines = lines[start_line - 1:]


# Join the lines to form a single JSON string
uv_data_str = ''.join(uv_data_lines)


# Find the start and end of the UV data block
start_json = uv_data_str.find('[')
end_json = uv_data_str.rfind(']')

if start_json == -1 or end_json == -1:
    raise ValueError("JSON array boundaries could not be found in the string.")

# Extract the UV data block
uv_data_str = uv_data_str[start_json:end_json + 1]

# Parse the JSON string
uv_data = json.loads(uv_data_str)

# Extract relevant data
times = []
wind_speeds = []

# Process each entry in the UV data
for entry in uv_data:
    time = entry['time'].strip()
    hour = int(time.split(':')[0])
    minute = int(time.split(':')[1])

    if hour >= end_hour:  # Stop if time is greater than or equal to 48:00
        break

    if (4 <= hour < 10) or (hour == 9 and minute == 30):
        times.append(time)
        wind_speeds.append(entry['data'][25])  # Assuming 80m is at index 25

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(times, wind_speeds, marker='o')
plt.title('Wind Velocity at 80m from 4:00 to 9:30')
plt.xlabel('Time')
plt.ylabel('Wind Speed (km/h)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
