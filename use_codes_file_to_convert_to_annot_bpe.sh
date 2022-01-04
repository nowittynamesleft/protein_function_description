#!/bin/sh

TESTGOCSV=$1
NAME=${TESTGOCSV%%.*}
CODEFILE=$2
NUMOPS=$3 # just to specify for filenames

python get_corpus_from_go_csv.py $1 ${NAME}_descriptions_only.txt
subword-nmt apply-bpe -c $CODEFILE < ${NAME}_descriptions_only.txt > ${NAME}_codified_descriptions_${NUMOPS}_ops.txt
python replace_descriptions_w_bpe.py $1 ${NAME}_codified_descriptions_${NUMOPS}_ops.txt ${NAME}_codified_${NUMOPS}_ops.tsv

