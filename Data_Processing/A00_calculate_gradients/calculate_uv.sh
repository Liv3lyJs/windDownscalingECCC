#!/bin/sh

# Load environment setting or functions from specific paths. 
. ssmuse-sh -d /fs/ssm/eccc/mrd/rpn/utils/16.2
. ssmuse-sh -d /fs/ssm/eccc/mrd/rpn/diag/16.1
. r.load.dot rpn/diag/16.2.4

pgsm -iment $1 -ozsrt $2 -l output <<FINPGSM
sortie(STD,100)
heure(tout)
grille(TAPE2, -1, -1, -1)
SETINTX(VOISIN)
COMPAC=-32
COMPRESS=OUI
champ(GZ, 93423264, 95369342, 95364364, 95357866, 95349708, 95339883)
champ(P0)
champ(WDUV, 75597472, 95369342, 95364364, 95357866, 95349708, 95339883)

FINPGSM