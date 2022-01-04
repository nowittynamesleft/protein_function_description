#!/bin/sh
GOCSV=$1
NAME=${GOCSV%%.*}
NUMOPS=$2

python get_corpus_from_go_csv.py $1 ${NAME}_descriptions_only.txt
subword-nmt learn-bpe -s $NUMOPS < ${NAME}_descriptions_only.txt > ${NAME}_codes_${NUMOPS}_ops.txt
subword-nmt apply-bpe -c ${NAME}_codes_${NUMOPS}_ops.txt < ${NAME}_descriptions_only.txt > ${NAME}_codified_descriptions_${NUMOPS}_ops.txt
python replace_descriptions_w_bpe.py $1 ${NAME}_codified_descriptions_${NUMOPS}_ops.txt ${NAME}_codified_${NUMOPS}_ops.tsv

