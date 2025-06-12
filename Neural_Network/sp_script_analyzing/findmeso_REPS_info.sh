#To choose: which system to extract data from: REPS or meso
#Je devrai remplacer ceci pour avoir une boucle sur les deux systemes dont j'extrais les données,
#pour ne pas devoir les faire à la main un après l'autre

syst_to_extract="REPS"

#date2extract=2022081814
date2extract=2022080200
#Faire des tests avec interpolation
INTERPOLATION=VOISIN
#INTERPOLATION=LINEAR



#Only for REPS, because meso has already hourly files
REPS_pass=00

if [[ ${syst_to_extract} == "meso" ]]; then
    folder2use='/fs/site5/eccc/cmd/x/spb001/archive_data/mesoanalysis'
elif [[ ${syst_to_extract} == "REPS" ]]; then
    folder2use=''/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel/hourly_files_par_passe/${REPS_pass}Z''
fi

echo $folder2use 

#For now for REPS I work with the files that I have already dearchived hourly
#REPS_fold=/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel//hourly_files_par_passe/${REPS_pass}Z

outfile_path=/home/jfg000/ss5/SuperResolution/Neural_Network/sp_script_analyzing/results

#Site experimental Nergica, position de mat MMV1.
#Info donnee par Nergica: 48 59 41N, 64 27 26 W, correspond à 48.994722, -64.457222
#48.986708,-64.455250
#lat=48.986708
#long=-64.455250
lat=48.994722
long=-64.457222

outfile=${outfile_path}/UV_${lat}_${long}_${syst_to_extract}_${date2extract}_${INTERPOLATION}.txt



# Remove the initial deletion of the outfile
 rm -rf $outfile

#for date in $(/home/spb001/site5/SuperResolution/Prep_REPS_hourly/gen_dates.sh 2024021400 2024021500 24);
for date in $(/home/spb001/site5/SuperResolution/Prep_REPS_hourly/gen_dates.sh ${date2extract} ${date2extract} 24);
do
    
 
    for hour in {0..23}
    
    do
        formatted_hour=$(printf "%02d" $hour)
       
         
	
        for dd in WDUV_75597472
        do
            vartoget=`echo $dd | cut -d _ -f 1`
            ip1=`echo $dd | cut -d _ -f 2`

            # Use a temporary file for the pgsm output
            temp_outfile=$(mktemp)
	    temp_outfile_2=$(mktemp)
	    echo 'date and formatted hour'
            echo ${date%??}${formatted_hour}_000
            # Run the pgsm command and redirect output to the temporary file
            if [[ ${syst_to_extract} == "meso" ]]; then
		    file2use=''${date%??}${formatted_hour}_000''
            elif [[ ${syst_to_extract} == "REPS" ]]; then
		    file2use=''${date%??}00_0${formatted_hour}_000''
            fi

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
 CHAMP('${vartoget}',${ip1})
EOF
            awk -v prepend_value="${date%??}${formatted_hour}" '{print prepend_value, $0}' "$temp_outfile" > "$temp_outfile_2"
            # Append the temporary file to the main output file
            cat "$temp_outfile_2" >> "$outfile"

            # Optionally display the output (commented out for batch processing)
            # cat $temp_outfile

            # Remove the temporary file
            rm -f $temp_outfile
	    rm -f $temp_outfile_2 
        done
    done
done
