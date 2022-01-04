#!/bin/sh
echo "Learning byte-pair encoding and converting training csv..."
GOCSV=$1
TESTGOCSV=$2
NUMOPS=$3
./convert_annot_file_to_annot_bpe.sh $GOCSV $NUMOPS

echo "Applying byte-pair encoding and converting test csv..."
NAME=${GOCSV%%.*}
CODEFILE=${NAME}_codes_${NUMOPS}_ops.txt
./use_codes_file_to_convert_to_annot_bpe.sh $TESTGOCSV $CODEFILE $NUMOPS

