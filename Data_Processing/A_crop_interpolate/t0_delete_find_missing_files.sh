#!/bin/bash

# RDPS
# Function to validate date
validate_date() {
    if ! date -d "$1" &>/dev/null; then
        echo "Error: invalid date '$1'" >&2
        exit 1
    fi
}

# Define the directory containing the files
directory="/home/jfg000/ss5/Data/input/domaine_new/east_canada_squential_valid_allpasses_domaine5/crop_UV_UU_VV_TT_P0_PN_H_CX_SD_WGE"

# Specify the starting and ending dates manually
start_date="2023-09-01"
end_date="2023-12-31"  # This will be checked for validity

# Validate the dates
validate_date "$start_date"
validate_date "$end_date"

# Define the hour passes
passes=("00" "06" "12") # "06" "12"
time_codes=$(seq -w 004 009)  # Generates time codes from 001 to 024 with leading zeros

# Change to the directory with the files
cd "$directory"

# Read all files into an array for efficient searching
declare -A file_map
for file in *; do
    file_map["$file"]=1
done

# Initialize the counter for missing files
missing_count=0

# Generate all expected filenames
for date in $(seq $(date -d "$start_date" +%s) 86400 $(date -d "$end_date" +%s)); do
    formatted_date=$(date -d @$date +%Y%m%d)
    for pass in "${passes[@]}"; do
        for time_code in $time_codes; do
            expected_filename="${formatted_date}${pass}_${time_code}_000"
            if [ -z "${file_map[$expected_filename]}" ]; then
                echo "Missing file: $expected_filename"
                ((missing_count++))
            fi
        done
    done
done

# Output the total number of missing files
echo "Total missing files: $missing_count"

# Return to the original directory
cd -


# # Mesoanalysis
# # Function to validate date
# validate_date() {
#     if ! date -d "$1" &>/dev/null; then
#         echo "Error: invalid date '$1'" >&2
#         exit 1
#     fi
# }

# # Define the directory containing the files
# directory="/home/jfg000/ss5/Data/label/initial_data/east_canada_sequential/test_sequential"

# # Specify the starting and ending dates manually
# start_date="2023-12-01"
# end_date="2024-02-29"

# # Validate the dates
# validate_date "$start_date"
# validate_date "$end_date"

# # Define the hour increments
# hours=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23")

# # Change to the directory with the files
# cd "$directory"

# # Initialize the counter for missing files
# missing_count=0

# # Generate all expected filenames
# while IFS= read -r date; do
#     for hour in "${hours[@]}"; do
#         expected_filename="${date}${hour}_000"
#         if ! ls | grep -qw "$expected_filename"; then
#             echo "Missing file: $expected_filename"
#             ((missing_count++))
#         fi
#     done
# done < <(seq $(date -d "$start_date" +%s) 86400 $(date -d "$end_date" +%s) | xargs -I{} date -d @{} +%Y%m%d)

# # Output the total number of missing files
# echo "Total missing files: $missing_count"

# # Return to the original directory
# cd -