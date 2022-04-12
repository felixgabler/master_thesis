#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import glob
import gzip
import re
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from pqdm.threads import pqdm
from scipy.stats import spearmanr
from iupred3.iupred3_lib import iupred, read_seq

data_dir = '/tmp/global2/vikram/felix/master_thesis/data/alphafold/v2'

data_dir = '/Users/felixgabler/PycharmProjects/master_thesis/data/alphafold/v2'


def get_plddt_scores(uniprot_id: str, proteome: str):
    pLDDT_scores = []
    with gzip.open(glob.glob(f'{data_dir}/UP*{proteome}_v2/AF-{uniprot_id}-*.pdb.gz')[0], "rt") as handle:
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


def get_sequence_from_file(uniprot_id: str):
    try:
        return read_seq(f'{data_dir}/sequences/{uniprot_id}.fasta')
    except Exception as e:
        print(f'Failed reading fasta file {uniprot_id}: {e.message if hasattr(e, "message") else e}')
    return None


def get_iupred_scores(uniprot_id: str):
    seq = get_sequence_from_file(uniprot_id)
    if seq is None:
        return None
    try:
        res = iupred(seq, 'long')
        return res[0]
    except:
        print(f'Error in {seq}')


def calculate_spearman(proteome: str):
    def internal(uniprot_id: str):
        try:
            plddts = get_plddt_scores(uniprot_id, proteome)
            iupreds = get_iupred_scores(uniprot_id)
            rho, p = spearmanr(plddts, iupreds)
            return rho
        except Exception as e:
            print(f'Error in {uniprot_id}: {e}')
            return None

    return internal


def extend_proteome_features_with_iupred_spearman(proteome_file: str):
    proteome = re.search(r"/([A-Z0-9]+).csv", proteome_file).group(1)
    try:
        df_proteome = pd.read_csv(proteome_file, index_col=0)
    except pd.errors.EmptyDataError:
        print(f'Could not read file {proteome_file}')
        return
    if 'iupred_plddt_spearman' not in df_proteome:
        df_proteome['iupred_plddt_spearman'] = None
    df_proteome['iupred_plddt_spearman'] = pd.to_numeric(df_proteome['iupred_plddt_spearman'], errors='coerce')
    to_fill = df_proteome.loc[(df_proteome['iupred_plddt_spearman'].isnull()) & (df_proteome['seq_len'] > 19)].index
    if len(to_fill) == 0:
        proteomes_with_iupred_spearman = pd.read_csv(f"{data_dir}/proteomes_with_iupred_spearman.csv", header=None)
        proteomes_with_iupred_spearman.loc[len(proteomes_with_iupred_spearman.index)] = [proteome]
        proteomes_with_iupred_spearman.to_csv(f'{data_dir}/proteomes_with_iupred_spearman.csv', header=False,
                                              index=False)
        print('Already filled with iupred spearman')
        return
    df_proteome.loc[to_fill, 'iupred_plddt_spearman'] = pqdm(df_proteome['uniprot_id'][to_fill],
                                                             calculate_spearman(proteome),
                                                             n_jobs=10, desc=f'Sequences in {proteome}')
    df_proteome.to_csv(f'{data_dir}/AA_scores/{proteome}.csv')
    del df_proteome
    return


if __name__ == '__main__':
    # extend_proteome_features_with_iupred_spearman(f"{data_dir}/AA_scores/9EURO1.csv")
    proteomes_with_iupred_auc = pd.read_csv(f"{data_dir}/proteomes_with_iupred_auc.csv", header=None)
    proteomes_with_iupred_auc_list = proteomes_with_iupred_auc.squeeze('columns').to_list()
    proteomes_with_iupred_spearman = pd.read_csv(f"{data_dir}/proteomes_with_iupred_spearman.csv", header=None)
    proteomes_with_iupred_spearman_list = proteomes_with_iupred_spearman.squeeze('columns').to_list()
    proteomes_to_extend_with_spearman = set(proteomes_with_iupred_auc_list).difference(
        set(proteomes_with_iupred_spearman_list))
    all_proteome_files = glob.glob(f"{data_dir}/AA_scores/*.csv")
    proteome_files_without_spearman = list(
        filter(lambda file: re.search(r"/([A-Z0-9]+).csv", file).group(1) in proteomes_to_extend_with_spearman,
               all_proteome_files))
    for file in tqdm(proteome_files_without_spearman, desc='Files'):
        extend_proteome_features_with_iupred_spearman(file)
