#!/bin/sh

fasta_file="$1"
gaf_file="$2"
output_fasta="$3"

id_list=$(cut -f 2 $gaf_file | grep -v !)
python scripts/align_fasta_to_ids.py $fasta_file $id_list > $output_fasta
