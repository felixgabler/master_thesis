#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import numpy as np

from utils.extend_aa_scores import calculate_new_property_for_proteomes, get_sequence_from_file
from iupred3.iupred3_lib import iupred

data_dir = '/tmp/global2/vikram/felix/master_thesis/data/alphafold/v2'

data_dir = '/Users/felixgabler/PycharmProjects/master_thesis/data/alphafold/v2'


def load_iupred_auc(uniprot_id: str):
    seq = get_sequence_from_file(f'{data_dir}/sequences', uniprot_id)
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


if __name__ == '__main__':
    calculate_new_property_for_proteomes(
        load_iupred_auc,
        prop='iupred_auc',
        ready_file=f"{data_dir}/proteomes_with_domain_count.csv",
        already_finished_file=f"{data_dir}/proteomes_with_iupred_auc.csv",
        result_folder=f"{data_dir}/AA_scores",
        filter_sequences=lambda df: df[df['seq_len'] > 19]
    )
