#!/usr/bin/env python

""" 
    Environment and Climate Change Canada
    Meteorological Service of Canada 
    Canadian Centre for Meteorological and Environmental Prediction
    Section: Products and Services 
    Created by: Jean-Sébastien Giroux, Date: 2024-05-22
"""

#.**************************************************************************************************************************************.#
#                                                          Code description                                                              #
#.**************************************************************************************************************************************.#
"""
    TASK NAME - Calculate the gradient of the selected variable (UU, VV, TT) between levels that are given to the program. 

     STATUS - experimental

     DESCRIPTION - TODO

    Before running this script you have to download the following items:
    . ~spst900/spooki/use_nb_master_python.dot
    
    Call example:
    ./TestGradient --levels 76696048,95370590,95353779 --inputFile  /home/spb001/site5/archive_data/operation.ensemble.ens.regmodel/2024040700_000_000 --outputFile TestFichier3Out.std
    ***IMPORTANT**** YOU MUST RUN THIS SCRIPT ON THE TERMINAL, DO NOT USE VSCODE OR ANY OTHER PROGRAMMING LANGUAGE SOFTWARE.
"""  

#.**************************************************************************************************************************************.#
#                                                               Imports                                                                  #
#.**************************************************************************************************************************************.#
import fstpy
import fnmatch
import os
import pandas as pd
import spookipy
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.append('/home/jfg000/ss5/SuperResolution/Data_Processing/A_crop_interpolate')
import crop as cr


class Calculate_gradient():
    def gradient(self, levels, input_file, output_file, variable, ajustment, level):
        # Lecture du fichier 
        df = fstpy.StandardFileReader(input_file).to_pandas()

        # Selection des metadatas
        meta_df = df.loc[df.nomvar.isin(["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])].reset_index(drop=True)

        # Selection des donnees
        df = fstpy.select_with_meta(df, [variable, 'GZ'])
        df = fstpy.add_columns(df, columns=['forecast_hour', 'ip_info'])
        df = df.loc[df['nomvar'].isin([variable, "GZ"])].reset_index(drop=True)

        # Former des groupes avec les memes infos 
        groups = df.groupby(['grid', 'datev', 'dateo', 'vctype'])

        df_list = []
        for key, current_group in groups:
            TT_GZ_current_group = current_group.loc[current_group['nomvar'].isin([variable, "GZ"])].reset_index(drop=True)

            if not TT_GZ_current_group.empty:
                TT_1_5m = TT_GZ_current_group.loc[(TT_GZ_current_group['nomvar'].isin([variable])) & (TT_GZ_current_group['level'].isin([level]))].reset_index(drop=True)
                if not TT_1_5m.empty:
                    GZ_1_5m = TT_GZ_current_group.loc[(TT_GZ_current_group['nomvar'].isin(["GZ"])) & (TT_GZ_current_group['level'].isin([1.0]))].reset_index(drop=True)
                    if GZ_1_5m.empty:
                        print(f'\n\n  ATTENTION GZ pour 1.5 m absent !!!!!!!! \n\n')
                    else:    
                        GZ_1_5m['d'] = GZ_1_5m['d'] + ajustment

                        GZ_1_5m['level'] = TT_1_5m['level']
                        GZ_1_5m['ip1'] = TT_1_5m['ip1']
                        GZ_1_5m['ip2'] = TT_1_5m['ip2']
                        GZ_1_5m['ip1_kind'] = TT_1_5m['ip1_kind']
                        GZ_1_5m['ip1_pkind'] = TT_1_5m['ip1_pkind']
                        GZ_1_5m['ascending'] = TT_1_5m['ascending']

                        TT_GZ_current_group = pd.concat([TT_GZ_current_group, GZ_1_5m]) 
                        TT_GZ_current_group = TT_GZ_current_group.sort_values(by='level', ascending=False).reset_index(drop=True)

                for i in range(0, len(levels) - 1):
                    pair = (levels[i], levels[i + 1])

                    first_level = pair[0]
                    second_level = pair[1] if len(pair) > 1 else None

                    TT_df = TT_GZ_current_group.loc[(TT_GZ_current_group['nomvar'].isin([variable])) & (TT_GZ_current_group['ip1'].isin([first_level, second_level]))].reset_index(drop=True)
                    GZ_df = TT_GZ_current_group.loc[(TT_GZ_current_group['nomvar'].isin(["GZ"])) & (TT_GZ_current_group['ip1'].isin([first_level, second_level]))].reset_index(drop=True)

                    if not TT_df.empty:
                        if (len(TT_df.index) != 2):
                            print(f'UN DES NIVEAUX EST MANQUANT! Seul {TT_df[["level"]]} est present \nON PASSE AU NIVEAU SUIVANT!')
                        elif TT_df.index.equals(GZ_df.index):
                            diffTT_df = spookipy.SubtractElementsVertically(TT_df, direction='ascending', reduce_df=True).compute()
                            diffGZ_df = spookipy.SubtractElementsVertically(GZ_df, direction='ascending', reduce_df=True).compute()

                            res_df2 = diffTT_df.copy()
                            res_df2['d'] = diffTT_df['d'].div(diffGZ_df['d'])
                            res_df2.loc[res_df2['nomvar'] == variable, 'nomvar'] = f'{variable}_G'
                            df_list.append(res_df2)
                        else:
                            print("Pas le meme nombre de niveaux pour les 2 champs !! On ignore  ")
                            print(TT_df[["nomvar", "ip1"]])
                            print(GZ_df[["nomvar", "ip1"]])

        df_list.append(meta_df)
        result_df = pd.concat(df_list, ignore_index=True)
        spookipy.WriterStd(result_df, output_file).compute()
        print(f' \n\n Fichier produit:  {output_file}\n\n')

    def process_file_uv(self, file, input_data, GRADIENTS_INPUT_DATA_UV, VARIABLE_NAME_ARRAY_INPUT_dict):
        input_file = os.path.join(input_data, file)
        output_file_uv = os.path.join(GRADIENTS_INPUT_DATA_UV, file)
        self.gradient(VARIABLE_NAME_ARRAY_INPUT_dict['UV'], input_file, output_file_uv, 'UV', 1.0, 10.0)

    def process_file(self, file, CROP_INPUT_DATA, GRADIENTS_INPUT_DATA_UU, GRADIENTS_INPUT_DATA_VV, GRADIENTS_INPUT_DATA_TT, VARIABLE_NAME_ARRAY_INPUT_dict):
        input_file = os.path.join(CROP_INPUT_DATA, file)
        output_file_uu = os.path.join(GRADIENTS_INPUT_DATA_UU, file)
        output_file_vv = os.path.join(GRADIENTS_INPUT_DATA_VV, file)
        output_file_tt = os.path.join(GRADIENTS_INPUT_DATA_TT, file)
        self.gradient(VARIABLE_NAME_ARRAY_INPUT_dict['UU'], input_file, output_file_uu, 'UU', 1.0, 10.0)
        self.gradient(VARIABLE_NAME_ARRAY_INPUT_dict['VV'], input_file, output_file_vv, 'VV', 1.0, 10.0)
        self.gradient(VARIABLE_NAME_ARRAY_INPUT_dict['TT'], input_file, output_file_tt, 'TT', 0.15, 1.5)

    def get_gradient_uv(self, PATH_EXTRACT_UV, input_data, GRADIENTS_INPUT_DATA_UV, VARIABLE_NAME_ARRAY_INPUT_dict):
        files_input = [file for file in os.listdir(input_data)
                       if os.path.isfile(os.path.join(input_data, file)) and
                       not fnmatch.fnmatch(file, '*.tgz') and
                       not fnmatch.fnmatch(file, '*.sh') and
                       file[11:14] == '024']

        cropping = cr.Crop()
        files_input.sort(key=cropping.get_sort_key_input)

        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.process_file_uv, file, input_data, GRADIENTS_INPUT_DATA_UV, VARIABLE_NAME_ARRAY_INPUT_dict) for file in files_input]
            for future in futures:
                future.result()

    def get_gradient(self, CROP_INPUT_DATA, VARIABLE_NAME_ARRAY_INPUT_dict, GRADIENTS_INPUT_DATA_UU, GRADIENTS_INPUT_DATA_VV, GRADIENTS_INPUT_DATA_TT):
        files_input = [file for file in os.listdir(CROP_INPUT_DATA)
                       if os.path.isfile(os.path.join(CROP_INPUT_DATA, file)) and
                       not fnmatch.fnmatch(file, '*.tgz') and
                       not fnmatch.fnmatch(file, '*.sh') and
                       file[11:14] == '024']

        cropping = cr.Crop()
        files_input.sort(key=cropping.get_sort_key_input)

        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.process_file, file, CROP_INPUT_DATA, GRADIENTS_INPUT_DATA_UU, GRADIENTS_INPUT_DATA_VV, GRADIENTS_INPUT_DATA_TT, VARIABLE_NAME_ARRAY_INPUT_dict) for file in files_input]
            for future in futures:
                future.result()


#.**************************************************************************************************************************************.#
#                                                                 Main                                                                   #
#.**************************************************************************************************************************************.#
if __name__ == '__main__':
    #List of constants 
    CROP_INPUT_DATA = '/home/spb001/site5/archive_data/operation.ensemble.ens.regmodel/'
    INPUT_DATA_UV = '/home/jfg000/ss5/Data/input/initial_data_reps/uv_values'
    PATH_EXTRACT_UV = '/home/jfg000/ss5/SuperResolution/Data_Processing/A00_calculate_gradients'
    VARIABLE_NAME_ARRAY_INPUT_dict_uv = {'UV': [95369342, 95339883]
                                      }
    VARIABLE_NAME_ARRAY_INPUT_dict = {'UU': [95369342, 95339883],  
                                      'VV': [95369342, 95339883],
                                      'TT': [95370590, 95344783],
                                      }   
    GRADIENTS_INPUT_DATA_UV = '/home/jfg000/ss5/Data/input/initial_data_reps/gradient/uv'
    GRADIENTS_INPUT_DATA_UU = '/home/jfg000/ss5/Data/input/initial_data_reps/gradient/uu'       
    GRADIENTS_INPUT_DATA_VV = '/home/jfg000/ss5/Data/input/initial_data_reps/gradient/vv'       
    GRADIENTS_INPUT_DATA_TT = '/home/jfg000/ss5/Data/input/initial_data_reps/gradient/tt'       

    #Folder used for gradients calculation
    if not os.path.exists(GRADIENTS_INPUT_DATA_UV):
        os.makedirs(GRADIENTS_INPUT_DATA_UV)
    if not os.path.exists(GRADIENTS_INPUT_DATA_UU):
        os.makedirs(GRADIENTS_INPUT_DATA_UU)
    if not os.path.exists(GRADIENTS_INPUT_DATA_VV):
        os.makedirs(GRADIENTS_INPUT_DATA_VV)
    if not os.path.exists(GRADIENTS_INPUT_DATA_TT):
        os.makedirs(GRADIENTS_INPUT_DATA_TT)

    gradient = Calculate_gradient()
    #For UV
    gradient.get_gradient_uv(PATH_EXTRACT_UV, INPUT_DATA_UV, GRADIENTS_INPUT_DATA_UV, VARIABLE_NAME_ARRAY_INPUT_dict_uv)
    #For the other values
    gradient.get_gradient(CROP_INPUT_DATA, VARIABLE_NAME_ARRAY_INPUT_dict, GRADIENTS_INPUT_DATA_UU, GRADIENTS_INPUT_DATA_VV, GRADIENTS_INPUT_DATA_TT)
