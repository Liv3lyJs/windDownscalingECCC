#!/bin/bash

#echo 'Je suis ici'

mesoana_fold=/fs/site5/eccc/cmd/x/spb001/archive_data/mesoanalysis

outfile_path=/fs/site5/eccc/cmd/x/spb001/scripts/extract_data_meso
rm -rf $outfile

lat=45.0
long=-75.0

INTERPOLATION=VOISIN



for date in $(/home/spb001/site5/SuperResolution/Prep_REPS_hourly/gen_dates.sh 2024021400 2024021500 24);
 
 do
 outfile=${outfile_path}/${date%??}_${lat}_${long}.txt
 rm -rf $outfile
 echo 'Je suis ici'

 for hour in {0..23}
 do
     formatted_hour=$(printf "%02d" $hour)

     for dd in  UV_75597472 ;do 

     #outfile=${outfile_path}/${date%??}_${lat}_${long}.txt
     #rm -rf $outfile

    vartoget=`echo $dd | cut -d _ -f 1`
    ip1=`echo $dd | cut -d _ -f 2`
#Il ne contatène pas les donnees dans le fichier, il overwrite on dirait. Arranger ca.
#aussi, puis-je sortir seulement UV dans le fichier de sortie?
#si je mets UV_75597472, il sort UU et VV sur deux lignes séparées
    pgsm -iment ${mesoana_fold}/${date%??}${formatted_hour}_000 -ozsrt ${outfile} -i <<EOF
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
    cat $outfile
    #echo "Above is extraction for $dd for lat,long - ${lat}, ${long}"
   # echo "hit enter to continue";read
  done
 done
done

#IDENT(OUEST,' ',[NOMVAR,IPUN,IPDEUX])
