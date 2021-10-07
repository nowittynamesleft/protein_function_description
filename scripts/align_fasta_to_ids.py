#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from useful_scripts.split_fasta import fasta_reader

def read_list(filename): 
    with open(filename, 'r') as f:
        elements = list(map( lambda line: line.strip(), f))
    return elements

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a FASTA where sequences IDs appear in the same order as supplied list.")
    parser.add_argument("fastafile", type=Path, help="input FASTA file")
    parser.add_argument("listfile", type=Path, help="IDs to align")
    #parser.add_argument("outfile", type=Path, help="Output FASTA")
    
    args = parser.parse_args()

    iterator = fasta_reader(args.fastafile)
    ordering = read_list(args.listfile)
    queries = set(ordering)

    query2seq = dict()
    for header, sequence in iterator:
        identifier = header.lstrip(">").split(" ")[0].split('|')[1]
        if identifier in queries:
            queries.remove(identifier)
            query2seq[identifier] = sequence

    for q in ordering:
        print(f">{q}\n{query2seq[q]}")
