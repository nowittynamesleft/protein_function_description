#!/bin/sh

fasta_file="$1"
gaf_file="$2"
output_fasta="$3"
id_list_fname="$4"

cut -f 2 $gaf_file > 'temp_id_list.txt'
grep -v ! 'temp_id_list.txt' > $id_list_fname
python scripts/align_fasta_to_ids.py $fasta_file $id_list_fname > $output_fasta
