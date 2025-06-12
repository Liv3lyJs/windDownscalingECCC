#!/bin/sh

# Load environment setting or functions from specific paths. 
. r.load.dot rpn/utils/20231219
#To delete p0 from gradient file
editfst -s $2 $3 $4 $5 -d $6 -i <<FINEDITFST
 
exclure(-1,'P0',-1,-1,-1,-1,-1)
 
FINEDITFST

#Combine all file together
editfst -s $1 $6 -d $7 -i 0 