#!/bin/bash

#Declaration of reed only variables (Constants)
"""
:Constant PATH_DP: Path that refer to the folder nammed Data_Processing/A_crop_interpolate. 
:Constant PATH_MAMBA: Path that refer to the mamba installation done localy. 
:Constant ENVNAME: Name of the conda environment that the user created for the project. 
:Constant INPUT_DATA: Path where are located the input data. 
:Constant LABEL_DATA: Path where are located the label data. 
:Constant DOMAINE: Path to the script that put data into the same domaine (Internal script). 
:Constant CROP_INPUT_DATA: Path where will be store the crop input data. 
:Constant CROP_LABEL_DATA: Path where will be store the crop label data. 
:Constant PGSM_PATH: Path where is located the PGSM script. 
:Constant SCRIPTNAME: Name of the script that will be run.
:Constant START_HOUR: The time at which you want to start to analyse the data.
:Constant END_HOUR: The time at which you want to end to analyse the data.
:Constant TOPOGRAPHIE: Flag used if the user whant to crop the topographie of the input standard file. 
:Constant: TOPO_PATH: The path where the topographie is located. 
:Constant: TOPO_STD_NANE: Name of the standard file that represent the topographie.
:Constant CROP_TOPO_DATA: Path where will be store the crop topographie data. 
"""
readonly PATH_DP='/home/jfg000/ss5/SuperResolution/Data_Processing/A_crop_interpolate'
readonly PATH_MAMBA="/fs/ssm/eccc/cmd/cmds/apps/mamba/master/mamba_2023.11.23_all"
readonly ENVNAME='MyEnv'
readonly INPUT_DATA="/home/jfg000/ss5/Data/input/temp_test_data" #'/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel/fs/site5/eccc/cmd/x/spb001/archive_data/operation.ensemble.ens.regmodel/hourly_files_niv_var_interet_v3'
readonly LABEL_DATA="/home/jfg000/ss5/Data/label/temp_test_data" #'/fs/site5/eccc/cmd/x/spb001/archive_data/mesoanalysis' #'/fs/site5/eccc/cmd/n/mav000/archives/mesoanalyis'
readonly DOMAINE='/home/jfg000/ss5/Data/Prep_Domain'
readonly CROP_INPUT_DATA='/home/jfg000/ss5/Data/input/crop'
readonly CROP_LABEL_DATA='/home/jfg000/ss5/Data/label/crop'
readonly PGSM_PATH='/home/jfg000/ss5/SuperResolution/Data_Processing/A_crop_interpolate'
readonly SCRIPTNAME='crop.py'
readonly START_HOUR=3 #To start at 4 am, because index start at 0.
readonly END_HOUR=10
readonly TOPOGRAPHIE=False #Flag to tell the script the user is now croping the topographie. Delete TOPOGRAPHIE on line 49 if False
readonly TOPO_PATH='/fs/site5/eccc/cmd/x/spb001/SuperResolution/get_HRinput_from_HRDPS'
readonly TOPO_STD_NANE='HR_input_from_HRDPS'
readonly CROP_TOPO_DATA='/home/jfg000/ss5/Data/topography/crop'


#Shell initialization to use cuda
. ssmuse-sh -x $PATH_MAMBA
conda config --set changeps1 false

#Activate virtual environment 
. activate $ENVNAME

#Run script crop.py
cd $PATH_DP
./$SCRIPTNAME --INPUT_DATA $INPUT_DATA --LABEL_DATA $LABEL_DATA --DOMAINE $DOMAINE --CROP_INPUT_DATA $CROP_INPUT_DATA --CROP_LABEL_DATA $CROP_LABEL_DATA --PGSM_PATH $PGSM_PATH --START_HOUR $START_HOUR --END_HOUR $END_HOUR --TOPO_PATH $TOPO_PATH --CROP_TOPO_DATA $CROP_TOPO_DATA --TOPO_STD_NANE $TOPO_STD_NANE