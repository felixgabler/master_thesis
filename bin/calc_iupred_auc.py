#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import glob
import re
import pandas as pd
import numpy as np
from urllib.request import urlopen

from utils.extend_aa_scores import calculate_new_property_for_proteomes
from iupred3.iupred3_lib import iupred

data_dir = '/tmp/global2/vikram/felix/master_thesis/data/alphafold/v2'

data_dir = '/Users/felixgabler/PycharmProjects/master_thesis/data/alphafold/v2'


def load_sequence_from_uniprot(uniprot_id: str) -> str:
    res = ''
    try:
        with urlopen(f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta") as response:
            for decoded in response.read():
                if not decoded.startswith('>'):
                    res += decoded
    except Exception as e:
        print(f'Failed fetching sequence: {e.message if hasattr(e, "message") else e}')
        pass
    return res


def load_iupred_auc(uniprot_id: str):
    seq = load_sequence_from_uniprot(uniprot_id)
    if seq is None:
        return None
    elif len(seq) < 19:
        print(f'Sequence too short for {uniprot_id}')
        return None
    try:
        res = iupred(seq, 'long')
        return np.round(sum(res[0]) / len(seq), decimals=4)
    except:
        print(f'Error in {seq}')
    return None


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
    # split_out_sequence_files()
    calculate_new_property_for_proteomes(
        load_iupred_auc,
        prop='iupred_auc',
        ready_file=f"{data_dir}/proteomes_with_domain_count.csv",
        already_finished_file=f"{data_dir}/proteomes_with_iupred_auc.csv",
        result_folder=f"{data_dir}/AA_scores",
        filter_sequences=lambda df: df[df['seq_len'] > 19]
    )
