import csv
import obonet
import argparse
import numpy as np
import networkx as nx
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_protein
from Bio.SeqRecord import SeqRecord


class Protein(object):
    def __init__(self, ID, name):
        self.ID = ID
        self.name = name

    def __repr__(self):
        return "Protein(%s, %s)" % (self.ID, self.name)

    def __eq__(self, other):
        if isinstance(other, Protein):
            return (self.ID == other.ID)
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())


class Function(object):
    def __init__(self, ID, name, def_str):
        self.ID = ID
        self.name = name
        self.definition = def_str

    def __repr__(self):
        return "Function(%s, %s)" % (self.ID, self.name)

    def __eq__(self, other):
        if isinstance(other, Function):
            return (self.ID == other.ID)
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())


class Annots(object):
    prot2funcs = {}
    func2prots = {}

    def __init__(self, prot_ID, prot_name, goterm_ID, goterm_name, goterm_def):
        if goterm_ID not in Annots.func2prots:
            Annots.func2prots[goterm_ID] = {'prots': set(), 'info': Function(goterm_ID, goterm_name, goterm_def)}
        Annots.func2prots[goterm_ID]['prots'].add(Protein(prot_ID, prot_name))
        if prot_ID not in Annots.prot2funcs:
            Annots.prot2funcs[prot_ID] = {'funcs': set(), 'info': Protein(prot_ID, prot_name)}
        Annots.prot2funcs[prot_ID]['funcs'].add(Function(goterm_ID, goterm_name, goterm_def))

    def get_functions():
        return set(Annots.func2prots.keys())

    def get_proteins():
        return set(Annots.prot2funcs.keys())


def load_prots(fn='uniprot_sprot_training.txt'):
    fRead = open(fn, 'r')
    prots = []
    for line in fRead:
        splitted = line.strip().split()
        prot = splitted[0]
        prots.append(prot)
    fRead.close()

    return prots


def load_fasta(fn_fasta, L_min=60, L_max=1200):
    aa = set(['U', 'O',
              'R', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K',
              'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    # 21st and 22nd amino acid
    # U -- Pyroglutamatic
    # O -- Hydroxyproline
    prot2seq = {}
    with open(fn_fasta, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            prot = record.id
            if len(seq) >= L_min and len(seq) <= L_max:
                if len((set(seq).difference(aa))) == 0:
                    prot2seq[prot] = seq

    return prot2seq


def load_annots(fn, go_graph, prots):
    root_terms = set(['GO:0008150', 'GO:0003674', 'GO:0005575'])
    with open(fn, 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        next(reader, None)
        for line in reader:
            prot_ID = 'sp|' + line[0] + '|' + line[1]
            prot_name = line[3]
            functions = line[-1].split(';')
            goterms = set()
            if functions != '' and prot_ID in prots:
                goterms = set([goterm.strip() for goterm in functions])

                all_goterms = set()
                for goterm in goterms:
                    if goterm in go_graph:
                        all_goterms.add(goterm)
                        parents = nx.descendants(go_graph, goterm)
                        all_goterms = all_goterms.union(parents)
                all_goterms = all_goterms.difference(root_terms)
                all_goterms = list(all_goterms)
                all_gonames = [go_graph.nodes[goterm]['name'] for goterm in all_goterms]
                goterm_defs = [go_graph.nodes[goterm]['def'] for goterm in all_goterms]

                # store annotations
                for i in range(len(all_goterms)):
                    Annots(prot_ID, prot_name, all_goterms[i], all_gonames[i], goterm_defs[i])

    return Annots


def write_output_file(data, prot2seq, out_fn='out.csv', min_prots=32, max_prots=40*32):
    # select GO terms (min_prots < #prots < max_prots)
    selected_goterms = []

    for goterm in data.get_functions():
        prots = data.func2prots[goterm]['prots']
        if len(prots) >= min_prots and len(prots) <= max_prots:
            go_obj = data.func2prots[goterm]['info']
            selected_goterms.append(go_obj)

    with open(out_fn, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["GO-term", "GO-name", "GO-def", "Prot-names", "Prot-seqs"])

        for goterm in selected_goterms:
            prots = data.func2prots[goterm.ID]['prots']
            prot_names = [p.ID for p in prots]
            prot_seqs = [prot2seq[pn] for pn in prot_names]
            tsv_writer.writerow([goterm.ID, goterm.name, goterm.definition.split('" ')[0].strip('"'), ','.join([p.split('|')[1] for p in prot_names]), ','.join(prot_seqs)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--annot', type=str, default='uniprot-reviewed_yes.tab', help="Input file with annotations stored in *.tab format. Obtained from: https://www.uniprot.org/")
    parser.add_argument('-f', '--fasta_fn', type=str, default='uniprot_sprot.fasta', help="Fasta file with uniprot sequnces. Obtained from: https://www.uniprot.org/")
    parser.add_argument('-go', '--obo_fn', type=str, default='go.obo', help="Gene Ontology file. Obtained from: http://geneontology.org/docs/download-ontology/")
    parser.add_argument('-min', '--min_prots', type=int, default=32, help="Minimum number pof proteins per GO term.")
    parser.add_argument('-max', '--max_prots', type=int, default=32*40, help="Maximum number pof proteins per GO term.")
    parser.add_argument('-p', '--prot_fn', type=str, required=True, help="List of protein IDs.")
    parser.add_argument('-o', '--out_fn', type=str, required=True, help="Output filename.")
    args = parser.parse_args()

    # load fasta file
    prot2seq = load_fasta(args.fasta_fn)

    # load proteins
    proteins = load_prots(args.prot_fn)
    proteins = [p for p in proteins if p in prot2seq]

    # load *.obo
    go_graph = obonet.read_obo(open(args.obo_fn, 'r'))

    # load annotations (uniprot *.tab file)
    data = load_annots(args.annot, go_graph, set(proteins))

    print ("## Number of functions: ", len(data.get_functions()))
    print ("## Number of proteins: ", len(data.get_proteins()))

    # write annotations
    write_output_file(data, prot2seq, out_fn=args.out_fn, min_prots=args.min_prots, max_prots=args.max_prots)
