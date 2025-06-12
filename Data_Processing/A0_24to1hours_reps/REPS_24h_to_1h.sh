#!/bin/sh

#""" 
#    Environment and Climate Change Canada
#    Meteorological Service of Canada 
#    Canadian Centre for Meteorological and Environmental Prediction
#    Created by: Simon-Philippe Breton, Date: 2023-11-23
#    Inspired by: Oleksandr Huziy with the function gen_dates.sh
#"""


. ssmuse-sh -d /fs/ssm/eccc/mrd/rpn/utils/20230906
##On veut créer 24 fichiers aux heures à partir de chaque fichier *_024 du REPS qui contient les données de 1h à 24h.
#Pour l'instant, seulement la passe 00Z telle que choisie plus bas dans la variable passe,
#mais nous pourrons considerer les autres par la suite

#Dans hourly_files_niv_var_interet_v2 je vais chercher plus de variables que seulement UU et VV
#Dans hourly_files_niv_var_interet_v3 je vais chercher SD en plus

#Choix de passe entre 00, 06, 12, ou 18
passe=18


#Dossier avec les données brutes du REPS désarchivées
dir_REPS_24h=/home/jfg000/ss5/Data/input/initial_data_reps/combined_missing #/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel
#Dossier dans lequel on veut avoir les données aux heures seulement pour des variables d'intérêt, pour l'instant UU et VV
dir_REPS_1h=/home/jfg000/ss5/Data/input/initial_data_reps/reps_1h_initial_data_missing/${passe}Z

#NB: Variables utilisées du RDPS par Kevin Gauthier comme prédicteurs
#VARIABLE_NAME_ARRAY = ["MGD", "MED", "ZPD", "VGD", "TD", "TT", "PN", "NT", "H", "RT", "I4", "5P", "I6", "UU", "VV"]
#Pour l'instant, on choisit seulement UU et VV (certaines des variables ci-haut ne semblent pas être disponibles
#des fichiers de sortie du REPS). Ces variables sont écrites plus bas en entrée à editfst. Si on en vaut plus,
#il faudra considérer que les autres variables ne seront pas nécessairement au meme niveau vertical, et il faudra alors
#spécifier le niveau voulu. NB: le niveau ip1 75597472 correspond à 10m, tel qu'obtenu par la commande r.ip1 

#gen_dates.sh generates dates between a starting and an ending date in the format 2021100100
for date in $(/home/spb001/site5/SuperResolution/Prep_REPS_hourly/gen_dates.sh 2024040100 2024043000 24);
 do

 for hour in {4..9} 
 do
       formatted_hour=$(printf "%02d" $hour) 
       #Il faut prendre les tic-tacs de la grille séparément avec desire(-1,['>>', '^^'])
       #75597472 is 10m, 76696048 is 1.5m
       #Level 5 for SD is aggregated, obtained as 60268832
       rm ${dir_REPS_1h}/${date%??}${passe}_0${formatted_hour}_000
       editfst -s ${dir_REPS_24h}/${date%??}${passe}_024_000 -d ${dir_REPS_1h}/${date%??}${passe}_0${formatted_hour}_000 -e -n <<EOF
 desire(-1,['>>', '^^'])
 desire(-1,-1,-1,-1,-1,$hour,-1)
 END 
EOF
 done

done

#DESIRE((-1,['P0','H','CX'],-1,-1,-1,$hour,-1))
#DESIRE((-1,['P0'],-1,-1,-1,$hour,-1))

