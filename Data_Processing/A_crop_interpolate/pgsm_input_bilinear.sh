#!/bin/sh

# Load environment setting or functions from specific paths. 
. ssmuse-sh -d /fs/ssm/eccc/mrd/rpn/utils/16.2
. ssmuse-sh -d /fs/ssm/eccc/mrd/rpn/diag/16.1
. r.load.dot rpn/diag/16.2.4

# Copy DOMAINE into OUTPUT_FILES_PATH_INPUT.
cp $2 $3

# -iment $1 = input file = INPUT_FILES_PATH_INPUT.
# -ozsrt $3 = output file = OUTPUT_FILES_PATH_INPUT.
# Intern documentation: https://wiki.cmc.ec.gc.ca/wiki/RPN-SI/RpnUtilities/pgsm$
pgsm -iment $1 -ozsrt $3 -l output <<FINPGSM
sortie(STD,100)
heure(tout)
grille(TAPE2, -1, -1, -1)
SETINTX(LINEAIR)

COMPAC=-32
COMPRESS=OUI

champ('UU', 75597472, 95369342, 95364364, 95357866, 95349708, 95339883)
champ('VV', 75597472, 95369342, 95364364, 95357866, 95349708, 95339883)
champ('TT', 76696048, 95370590, 95366850, 95361109, 95353779, 95344783)
champ('UU_G')
champ('VV_G')
champ('TT_G')
champ('UV_G')
champ('P0')
champ('PN')
champ('H')
champ('CX')
champ('SD', 60268832)
champ('WGE')
champ('ME')
champ('MG')
champ('Z0')
champ('GZ')
champ('WDUV', 75597472, 95369342, 95364364, 95357866, 95349708, 95339883)

FINPGSM

#A tester lequel est mieux.
# Can replace this:
# champ(UU, 75597472, 95369342, 95364364, 95357866)
# champ(VV, 75597472, 95369342, 95364364, 95357866)
# By this
# champ(UV, 75597472, 95369342, 95364364, 95357866)
# To get UU and VV on the x axis. maybee UV too, not sure.