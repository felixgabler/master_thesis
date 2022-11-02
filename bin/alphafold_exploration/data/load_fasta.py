from Bio import SeqIO


def load_fasta(file):
    fasta_sequences = SeqIO.parse(file, 'fasta')
    for fasta in fasta_sequences:
        yield str(fasta.seq)
