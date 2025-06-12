#This file is only to extract wind velocity from mesoanalysis
syst_to_extract="downscaledREPS"


start_date="2024010100"
end_date="2024043023"
current_date=$start_date

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



folder2use='/fs/site5/eccc/cmd/x/jfg000/CNN_results_article/articles_testing/article_section_III_domaine1/results_std_final'

outfile_path=/home/jfg000/ss5/SuperResolution/Neural_Network/sp_script_analyzing/results
outfile=${outfile_path}/UV_${lat}_${long}_${syst_to_extract}_${start_date}_${end_date}_${INTERPOLATION}.txt

# Remove the initial deletion of the outfile
 rm -rf $outfile


while [ "$current_date" -le "$end_date" ]; do
    echo "Processing date: $current_date"
    
   
         
	#I do not have the level in the downscaled output file, only the variable name..
        for dd in UV_F
        do
            vartoget=`echo $dd`
            #ip1=`echo $dd | cut -d _ -f 2`

            # Use a temporary file for the pgsm output
            temp_outfile=$(mktemp)
	    temp_outfile_2=$(mktemp)
#	    echo 'date and formatted hour'
#            echo ${date%??}${formatted_hour}_000
            # Run the pgsm command and redirect output to the temporary file
#            if [[ ${syst_to_extract} == "meso" ]]; then
		    #file2use=''${date%??}${formatted_hour}_000''
		    file2use=${current_date}_000
            #needs to be adapted as there a 00_0 before the hour
 #	    elif [[ ${syst_to_extract} == "REPS" ]]; then
#		    file2use=''${date%??}00_0${formatted_hour}_000''
#            fi
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
 HEURE(all)
 CHAMP('${vartoget}')
EOF
            #awk -v prepend_value="${date%??}${formatted_hour}" '{print prepend_value, $0}' "$temp_outfile" > "$temp_outfile_2"
            awk -v prepend_value=${current_date} '{print prepend_value, $0}' "$temp_outfile" > "$temp_outfile_2"
	    # Append the temporary file to the main output file
            cat "$temp_outfile_2" >> "$outfile"

            # Optionally display the output (commented out for batch processing)
            # cat $temp_outfile

            # Remove the temporary file
            rm -f $temp_outfile
	    rm -f $temp_outfile_2 
            current_date=$(date -u -d "${current_date:0:8} ${current_date:8:2} + 1 hour" +%Y%m%d%H)
        done
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

    # Check if the second column is 'UV_F'
    if [ "${columns[1]}" == "UV_F" ]; then
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





