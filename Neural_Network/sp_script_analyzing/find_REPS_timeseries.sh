#To extract time series from REPS forecats.
#As a first step, I concatenate forecasts from hour 4 to 9 of every successive REPS run
#Je dois ajouter un header au fichier



#meso or REPS
syst_to_extract="REPS"


#Start data should not contain hour, otherwise complet with the 4 runs per day
#start_date="2022010100"
#end_date="2022010200"

start_date="20231231"
end_date="20240430"






#Faire des tests avec interpolation
INTERPOLATION=VOISIN
#INTERPOLATION=LINEAIR
#Site experimental Nergica, position de mat MMV1.
#Info donnee par Nergica: 48 59 41N, 64 27 26 W, correspond à 48.994722, -64.457222
#48.986708,-64.455250
#lat=48.986708
#long=-64.455250
lat=48.994722
long=-64.457222

#I will convert here in m/s to get m/s in the final output file
knots2ms=0.5144


dd='WDUV_75597472'
vartoget=`echo $dd | cut -d _ -f 1`
ip1=`echo $dd | cut -d _ -f 2`

echo ${vartoget}
echo ${ip1}

outfile_path=/fs/site5/eccc/cmd/x/spb001/scripts/extract_data_meso

outfile=${outfile_path}/UV_${lat}_${long}_${syst_to_extract}_${start_date}_${end_date}_${INTERPOLATION}.txt


# Function to add hours to a given date and hour in the format YYYYMMDDHH
add_hours() {
    local date_hour="$1"
    local hours_to_add="$2"
    date -u -d "${date_hour:0:8} ${date_hour:8:2} +${hours_to_add} hours" +%Y%m%d%H
}


#folder2use=''/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel/hourly_files_par_passe/${REPS_pass}Z''
#Folder with files with data for 24 hours leadtime for every run
folder2use=''/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel
######
#A voir, j'ai déja les fichiers aux heures, soit je travaille avec ca en lisant
#une heure a la fois ou avec les fichiers de passe et que je lis heures de 4 a 9
#Mieux de jouer avec fichiers bruts car plus versatile

#####
echo $folder2use 
# Remove the initial deletion of the outfile
 rm -rf $outfile
  
current_date=$start_date 

while [ "$current_date" -le "$end_date" ]; do
    echo "Processing date: $current_date"
    
    for norun in 00 06 12 18; do
    
        

	for hourleadtime in {4..9}; do

        

        # Use a temporary file for the pgsm output
        temp_outfile=$(mktemp)
        temp_outfile_2=$(mktemp)
#       #Extracts data directly from dearchived reps file containing 24 hours leadtime for each run number	   
        date_hour="${current_date}${norun}"
	file2use=''${date_hour}_024_000''
           
    #See https://wiki.cmc.ec.gc.ca/w/images/c/ca/Pgsm.pdf for details.
    #101,102,103 is not important, IDENT is to decide what to output, and OUEST means that 
    #"the field identifiers will be printed to the left of the numerical values."
    #pgsm -iment ${folder2use}/${date%??}${formatted_hour}_000 -ozsrt ${temp_outfile} -i <<EOF
    pgsm -iment ${folder2use}/${file2use} -ozsrt ${temp_outfile} -i <<EOF
 VOIRENT=NON
 SORTIE(formatee)
 COORD([${lat},${long}],add)
 GRILLE(STATIONS,101,102,103)
 IDENT(OUEST,' ',[NOMVAR,IPUN,IPDEUX])
 FORMAT('F6.2')
 SETINTX(${INTERPOLATION})
 HEURE(${hourleadtime})
 CHAMP('${vartoget}',${ip1})
EOF
         #uses function defined below
         final_time_for_outfile=$(add_hours "$date_hour" "${hourleadtime}")

            #awk -v prepend_value="${date%??}${formatted_hour}" '{print prepend_value, $0}' "$temp_outfile" > "$temp_outfile_2"
        awk -v prepend_value=${final_time_for_outfile} '{print prepend_value, $0}' "$temp_outfile" > "$temp_outfile_2"
        # Append the temporary file to the main output file
        cat "$temp_outfile_2" >> "$outfile"

        # Optionally display the output (commented out for batch processing)
       # cat $temp_outfile

       # Remove the temporary file
       rm -f $temp_outfile
       rm -f $temp_outfile_2
       #loop on the six hours to use in each run
           done
   #Loop on the four runs
    done 
    #current_date=$(date -u -d "${current_date:0:8} ${current_date:8:2} + 1 hour" +%Y%m%d%H)
    #I go from one day to the next here, because I do 4 runs a day with 6 hours per run 
    current_date=$(date -u -d "$current_date + 1 day" +%Y%m%d)   
done



#Now I want to modify the file to only keep the date and the wind velocity, whereas the outfile that I get gives me data in
#the format
#2022080200 UV   75597472    0    4.57
#2022080200 WD   75597472    0  228.59
#2022080201 UV   75597472    0    4.51
#2022080201 WD   75597472    0  207.69

input_file=${outfile}
output_file=${outfile}_UV_only_verLIN



rm -f ${output_file}

# Initialize the output file
> $output_file


header="Date Wind_Velocity_10m(m/s)"
# Initialize the file with the header
echo "$header" > "$output_file"



# Read the input file line by line
while read -r line; do
    # Split the line into an array
    IFS=' ' read -r -a columns <<< "$line"

    # Check if the second column is 'UV'
    if [ "${columns[1]}" == "UV" ]; then
	  echo "Found UV value: ${columns[4]}"  # Debugging: Print the UV value
	 UV_ms=$(echo "${columns[4]} * $knots2ms" | bc -l)  
	 UV_ms_formatted=$(printf "%.3f" "$UV_ms")  # Ensure three decimal places with leading zero
	 echo "Converted UV to m/s: $UV_ms"  # Debugging: Print the converted value
	 #UV_ms=$(echo "${columns[4]} * ${knots2ms}" | bc -1)
        # Write the date and wind velocity to the output file
        echo "${columns[0]} ${UV_ms_formatted}" >> $output_file
    fi
done < "$input_file"

echo "Wind velocity data extracted to $output_file"




