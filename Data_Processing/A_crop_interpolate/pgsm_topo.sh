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
SETINTX(VOISIN)

COMPAC=-32
COMPRESS=OUI

champ(ME)
champ(MG)
champ('Z0LC')
FINPGSM

#LINEAIR     VOISIN