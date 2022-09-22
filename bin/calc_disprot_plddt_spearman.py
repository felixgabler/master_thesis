#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import re

import numpy as np
from os.path import exists
from scipy.stats import spearmanr

from utils.extend_aa_scores import calculate_new_property_for_proteomes

plddt_data_dir = '/ebio/abt1_share/alphafold_data/alphafold_output/felix_2022_sequences_pdbs'
disorder_data_dir = '/tmp/global2/vikram/felix/master_thesis/data'
data_dir = '/tmp/global2/vikram/felix/master_thesis/data/alphafold/v2'


# data_dir = '/Users/felixgabler/PycharmProjects/master_thesis/data'


def get_plddt_scores(uniprot_id: str):
    pLDDT_scores = []
    if not exists(f'{plddt_data_dir}/{uniprot_id}_ranked_0.pdb'):
        return None
    with open(f'{plddt_data_dir}/{uniprot_id}_ranked_0.pdb', "rt") as handle:
        seen_res_i = 0
        for line in handle:
            # The pLDDT is the last number in each ATOM line
            if line.startswith('ATOM'):
                res_i = int(line[22:27])
                if res_i != seen_res_i:
                    seen_res_i = res_i
                    pLDDT = float(line[-20:-13])
                    pLDDT_scores.append(pLDDT)
    return np.asarray(pLDDT_scores)


def read_true_labels():
    items = {}
    with open(f'{disorder_data_dir}/disprot/2022/disprot-disorder-2022-unclustered.txt', 'r') as handle:
        i = -1
        acc = ''
        for line in handle:
            i += 1
            i_offset = i % 3
            if i_offset == 0:
                acc = re.search(r"full acc=([A-Z0-9]+)", line).group(1)
            elif i_offset == 2:
                items[acc] = line.strip()
    return items


def calculate_spearman(true_labels: dict):
    def internal(uniprot_id: str):
        try:
            plddts = get_plddt_scores(uniprot_id)
            if plddts is None:
                return None
            disorder = np.asarray([int(i) for i in list(true_labels[uniprot_id])])
            rho, p = spearmanr(plddts, disorder)
            return np.round(rho, decimals=4)
        except Exception as e:
            print(f'Error in {uniprot_id}: {e}')
            return None

    return internal


if __name__ == '__main__':
    true_labels = read_true_labels()
    calculate_new_property_for_proteomes(
        calculate_spearman(true_labels),
        prop='disorder_plddt_spearman',
        ready_file=f"{data_dir}/proteomes_with_domain_count.csv",
        already_finished_file=f"{data_dir}/proteomes_with_disorder_plddt_spearman.csv",
        result_folder=f"{data_dir}/AA_scores"
    )
