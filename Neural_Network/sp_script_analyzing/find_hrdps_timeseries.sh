#!/bin/bash
cd /fs/site5/eccc/cmd/x/spb001/scripts/extract_data_meso
#hrpds_folder='/fs/site5/eccc/cmd/x/spb001/maestro/hrdps_icing_test_001/hub/product_dbase/windfarms/txt'



hrpds_folder='/fs/site5/eccc/cmd/x/spb001/maestro/hrdps_icing_test_001/hub/product_dbase/windfarms/txt'
input_file_name='2024011506_0010.txt'
input_file="${hrpds_folder}/${input_file_name}"
echo "${input_file}"

#input_file_name='2024011506_0010.txt'
#input_file=''${hrdps_folder}/${input_file_name}''
#echo ${input_file}

output_file="filtered_data.txt"
# Initialize line counter
line_count=0

start_line=883
end_line=957


# Height index for 80m (considering the first value is time, 80m is the 8th value)
height_index=8


# Convert time to minutes function
time_to_minutes() {
    IFS=":" read -r hour minute <<< "$1"
    echo $((hour * 60 + minute))
}

# Define the range of time to keep
start_time=$(time_to_minutes "4:00")
end_time=$(time_to_minutes "9:30")

# Read the header
header=$(sed -n "${start_line}p" "$input_file" | awk '{$1=""; print $0}')

# Extract data and filter the lines
filtered_data=$(sed -n "$((start_line+2)),$((end_line))p" "$input_file" | while read -r line; do
    time=$(echo "$line" | awk '{print $1}')
    time_minutes=$(time_to_minutes "$time")

    if ((time_minutes >= start_time && time_minutes <= end_time)); then
        echo "$line"
    fi
done)

# Write header and filtered data to the output file
{
    echo "Level:(m) $header"
    echo "$filtered_data"
} > "$output_file"

echo "Filtered data written to $output_file"
