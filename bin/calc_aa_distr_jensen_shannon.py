#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import jensenshannon

from utils.extend_aa_scores import calculate_new_property_for_proteomes, get_sequence_from_file

data_dir = '/tmp/global2/vikram/felix/master_thesis/data/alphafold/v2'

df_aa_counts = pd.read_csv("/tmp/global2/vikram/felix/kmer_counts/trembl_aa_counts.csv")
uniprot_aa_order = df_aa_counts['aa'].values
uniprot_aa_distr = (df_aa_counts['count'] / df_aa_counts['count'].sum()).values


def calculate_aa_distr_jensen_shannon(uniprot_id: str):
    seq = get_sequence_from_file(f"{data_dir}/sequences", uniprot_id)
    if seq is None:
        return None
    seq = seq.replace('\n', '')
    try:
        seq_aa_counts = Counter(seq)
        seq_aa_count_sum = len(seq)
        seq_aa_distr = [seq_aa_counts[aa] / seq_aa_count_sum for aa in uniprot_aa_order]
        return np.round(jensenshannon(uniprot_aa_distr, seq_aa_distr), decimals=4)
    except:
        print(f'Error in {seq}')
    return None


if __name__ == '__main__':
    calculate_new_property_for_proteomes(
        calculate_aa_distr_jensen_shannon,
        prop='aa_distr_js',
        ready_file=f"{data_dir}/proteomes_with_domain_count.csv",
        already_finished_file=f"{data_dir}/proteomes_with_jensen_shannon.csv",
        result_folder=f"{data_dir}/AA_scores"
    )
