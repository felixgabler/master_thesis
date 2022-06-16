#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python
import numpy as np
from collections import Counter
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

from utils.extend_aa_scores import calculate_new_property_for_proteomes, get_sequence_from_file

data_dir = '/tmp/global2/vikram/felix/master_thesis/data/alphafold/v2'

data_dir = '/Users/felixgabler/PycharmProjects/master_thesis/data/alphafold/v2'

pos_charge_aas = ['K', 'R', 'H']
neg_charge_aas = ['D', 'E']
polar_aas = ['G', 'A', 'V', 'C', 'P', 'L', 'I', 'M', 'W', 'F']
non_polar_aas = ['S', 'T', 'Y', 'N', 'Q']


def calculate_charge_hydro_isoelectric(uniprot_id: str):
    seq = get_sequence_from_file(f"{data_dir}/sequences", uniprot_id)
    if seq is None:
        return None, None, None, None, None
    seq = seq.replace('\n', '')
    try:
        seq_aa_counts = Counter(seq)
        seq_aa_count_sum = len(seq)
        freq_pos_charge = sum(seq_aa_counts[aa] for aa in pos_charge_aas) / seq_aa_count_sum
        freq_neg_charge = sum(seq_aa_counts[aa] for aa in neg_charge_aas) / seq_aa_count_sum
        freq_polar = sum(seq_aa_counts[aa] for aa in polar_aas) / seq_aa_count_sum
        freq_non_polar = sum(seq_aa_counts[aa] for aa in non_polar_aas) / seq_aa_count_sum
        isoelectric_point = IsoelectricPoint(seq, aa_content=seq_aa_counts).pi()
        return np.round(
            freq_pos_charge, decimals=4), np.round(
            freq_neg_charge, decimals=4), np.round(
            freq_polar, decimals=4), np.round(
            freq_non_polar, decimals=4), np.round(
            isoelectric_point, decimals=4)
    except:
        print(f'Error in {seq}')
    return None, None, None, None, None


if __name__ == '__main__':
    # extend_proteome_features_with_new_property(
    #     calculate_charge_hydro_isoelectric,
    #     None,
    #     f"{data_dir}/AA_scores/9EURO1.csv",
    #     ['freq_pos_charge', 'freq_neg_charge', 'freq_polar', 'freq_non_polar', 'IEP'],
    #     f"{data_dir}/proteomes_with_charge_hydro_isoelectric.csv",
    #     f"{data_dir}/AA_scores"
    # )
    calculate_new_property_for_proteomes(
        calculate_charge_hydro_isoelectric,
        props=['freq_pos_charge', 'freq_neg_charge', 'freq_polar', 'freq_non_polar', 'IEP'],
        ready_file=f"{data_dir}/proteomes_with_domain_count.csv",
        already_finished_file=f"{data_dir}/proteomes_with_charge_hydro_isoelectric.csv",
        result_folder=f"{data_dir}/AA_scores"
    )
