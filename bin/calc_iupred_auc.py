#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import glob
import re
import pandas as pd
import numpy as np
from pqdm.threads import pqdm
from tqdm.auto import tqdm
from urllib.request import urlopen
from iupred3.iupred3_lib import iupred, read_seq

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


def get_sequence_from_file(uniprot_id: str):
    try:
        return read_seq(f'{data_dir}/sequences/{uniprot_id}.fasta')
    except Exception as e:
        print(f'Failed reading fasta file {uniprot_id}: {e.message if hasattr(e, "message") else e}')
    return None


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


def extend_proteome_features_with_iupred_auc(proteome_file: str):
    proteome = re.search(r"/([A-Z0-9]+).csv", proteome_file).group(1)
    try:
        df_proteome = pd.read_csv(proteome_file, index_col=0)
    except pd.errors.EmptyDataError:
        print(f'Could not read file {proteome_file}')
        return
    df_proteome = df_proteome[df_proteome['seq_len'] > 19]
    if 'iupred_auc' not in df_proteome:
        df_proteome['iupred_auc'] = None
    df_proteome['iupred_auc'] = pd.to_numeric(df_proteome['iupred_auc'], errors='coerce')
    to_fill = df_proteome[df_proteome['iupred_auc'].isnull()].index
    if len(to_fill) == 0:
        proteomes_with_iupred_auc = pd.read_csv(f"{data_dir}/proteomes_with_iupred_auc.csv", header=None)
        proteomes_with_iupred_auc.loc[len(proteomes_with_iupred_auc.index)] = [proteome]
        proteomes_with_iupred_auc.to_csv(f'{data_dir}/proteomes_with_iupred_auc.csv', header=False,
                                         index=False)
        print('Already filled with iupred score')
        return
    df_proteome.loc[to_fill, 'iupred_auc'] = pqdm(df_proteome['uniprot_id'][to_fill], load_iupred_auc, n_jobs=20,
                                               desc=f'Sequences in {proteome}')
    df_proteome.to_csv(f'{data_dir}/AA_scores/{proteome}.csv')
    # splits = np.array_split(to_fill, len(to_fill) / 1000 + 1)
    # for i, fill_split in enumerate(splits):
    #     df_proteome.loc[fill_split, 'iupred_auc'] = await tqdm.gather(
    #         *[load_iupred_auc(session, sem, uid) for uid in df_proteome['uniprot_id'][fill_split]],
    #         desc=f'Sequences in {proteome}, split {i + 1}/{len(splits)}')
    #     df_proteome.to_csv(f'{data_dir}/AA_scores/{proteome}.csv')
    del df_proteome
    return


def calculate_iupred_scores():
    proteomes_with_domain_count = pd.read_csv(f"{data_dir}/proteomes_with_domain_count.csv", header=None)
    proteomes_with_domain_count_list = proteomes_with_domain_count.squeeze('columns').to_list()
    proteomes_with_iupred_auc = pd.read_csv(f"{data_dir}/proteomes_with_iupred_auc.csv", header=None)
    proteomes_with_iupred_auc_list = proteomes_with_iupred_auc.squeeze('columns').to_list()
    proteomes_to_extend_with_iupred = set(proteomes_with_domain_count_list).difference(
        set(proteomes_with_iupred_auc_list))
    all_proteome_files = glob.glob(f"{data_dir}/AA_scores/*.csv")
    proteome_files_without_iupred = list(
        filter(lambda file: re.search(r"/([A-Z0-9]+).csv", file).group(1) in proteomes_to_extend_with_iupred,
               all_proteome_files))
    for file in tqdm(proteome_files_without_iupred, desc='Files'):
        extend_proteome_features_with_iupred_auc(file)
    return


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
    calculate_iupred_scores()
