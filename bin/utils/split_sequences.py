#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import glob
import re
import pandas as pd

data_dir = '/tmp/global2/vikram/felix/master_thesis/data/alphafold/v2'

data_dir = '/Users/felixgabler/PycharmProjects/master_thesis/data/alphafold/v2'


def write_seq_to_file(accession: str, uniprot_id: str, seq_lines: str, existing_ids: set):
    if len(uniprot_id) == 0 or uniprot_id not in existing_ids:
        return
    with open(f'{data_dir}/sequences/{uniprot_id}.fasta', 'w') as handle:
        handle.write(accession + seq_lines)


def split_out_sequence_files():
    uniprot_ids = set()
    for proteome_file in glob.glob(f'{data_dir}/AA_scores/*.csv'):
        ids = pd.read_csv(proteome_file, usecols=['uniprot_id'])['uniprot_id'].array
        uniprot_ids = uniprot_ids.union(set(ids))
    print(f'Creating sequence files for {len(uniprot_ids)} sequences')
    with open(f'{data_dir}/sequences.fasta') as handle:
        accession = ''
        uniprot_id = ''
        seq_lines = ''
        for line in handle:
            if line.startswith('>'):
                write_seq_to_file(accession, uniprot_id, seq_lines, uniprot_ids)
                accession = line
                uniprot_id = re.search(r'AF-([A-Z0-9]+)-', line).group(1)
                seq_lines = ''
            else:
                seq_lines += line
    write_seq_to_file(accession, uniprot_id, seq_lines, uniprot_ids)


if __name__ == '__main__':
    split_out_sequence_files()
