#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import glob
import re
from collections import Counter

import pandas as pd
from pqdm.threads import pqdm
from tqdm.auto import tqdm

from iupred3.iupred3_lib import read_seq

data_dir = '/tmp/global2/vikram/felix/master_thesis/data'
af_data_dir = f'{data_dir}/alphafold/v2'
already_finished_list_file = f"{af_data_dir}/proteomes_with_6mer_counts_2000.csv"

df_counts = pd.read_csv(f'{data_dir}/uniprot/trembl_kmer_counts.csv', index_col=None).sort_values('count',
                                                                                                  ascending=False)
df_counts = df_counts[df_counts['kmer'] != 'XXXXXX']


def check_repeat(kmer: str, min_len=5):
    return any(all(c == kmer[start] for c in kmer[start + 1:start + min_len]) for start in range(len(kmer) - min_len))


most_freq_kmers = df_counts['kmer'][:2000].values
repeats = []
non_repeats = []
for kmer in most_freq_kmers:
    if check_repeat(kmer):
        repeats.append(kmer)
    else:
        non_repeats.append(kmer)


def get_sequence_from_file(uniprot_id: str):
    try:
        return read_seq(f'{af_data_dir}/sequences/{uniprot_id}.fasta')
    except Exception as e:
        print(f'Failed reading fasta file {uniprot_id}: {e.message if hasattr(e, "message") else e}')
    return None


def count_6mers_counter(seq: str, k=6):
    counter = Counter()
    n_kmers = len(seq) - k + 1
    for i in range(n_kmers):
        kmer = seq[i:i + k]
        counter[kmer] += 1
    return sum([counter[repeat] for repeat in repeats]), sum([counter[non_repeat] for non_repeat in non_repeats])


def load_6mer_counts(uniprot_id: str):
    seq = get_sequence_from_file(uniprot_id)
    if seq is None:
        return None, None
    try:
        return count_6mers_counter(seq)
    except:
        print(f'Error in {seq}')
    return None, None


def extend_proteome_features_with_6mer_counts(proteome_file: str):
    proteome = re.search(r"/([A-Z0-9]+).csv", proteome_file).group(1)
    try:
        df_proteome = pd.read_csv(proteome_file, index_col=0)
    except pd.errors.EmptyDataError:
        print(f'Could not read file {proteome_file}')
        return
    if 'repeat_6mers_2000' not in df_proteome:
        df_proteome['repeat_6mers_2000'] = None
    if 'non_repeat_6mers_2000' not in df_proteome:
        df_proteome['non_repeat_6mers_2000'] = None
    df_proteome['repeat_6mers_2000'] = pd.to_numeric(df_proteome['repeat_6mers_2000'], errors='coerce')
    df_proteome['non_repeat_6mers_2000'] = pd.to_numeric(df_proteome['non_repeat_6mers_2000'], errors='coerce')
    to_fill = df_proteome[df_proteome['repeat_6mers_2000'].isnull()].index
    if len(to_fill) == 0:
        proteomes_already_finished = pd.read_csv(already_finished_list_file)
        proteomes_already_finished.loc[len(proteomes_already_finished.index)] = [proteome]
        proteomes_already_finished.to_csv(already_finished_list_file, index=False)
        print('Already filled')
        return
    counts = pqdm(df_proteome['uniprot_id'][to_fill], load_6mer_counts, n_jobs=20, desc=f'Sequences in {proteome}')
    df_proteome.loc[to_fill, ['repeat_6mers_2000', 'non_repeat_6mers_2000']] = counts
    df_proteome.to_csv(f'{af_data_dir}/AA_scores/{proteome}.csv')
    del df_proteome
    return


def calculate_6mer_counts():
    proteomes_ready = pd.read_csv(f"{af_data_dir}/proteomes_with_domain_count.csv", header=None)
    proteomes_ready_list = proteomes_ready.squeeze('columns').to_list()
    proteomes_already_finished = pd.read_csv(already_finished_list_file)
    proteomes_already_finished_list = proteomes_already_finished.squeeze('columns').to_list()
    proteomes_to_process = set(proteomes_ready_list).difference(
        set(proteomes_already_finished_list))
    all_proteome_files = glob.glob(f"{af_data_dir}/AA_scores/*.csv")
    files_to_process = list(
        filter(lambda file: re.search(r"/([A-Z0-9]+).csv", file).group(1) in proteomes_to_process,
               all_proteome_files))
    for file in tqdm(files_to_process, desc='Files'):
        extend_proteome_features_with_6mer_counts(file)
    return


if __name__ == '__main__':
    calculate_6mer_counts()
