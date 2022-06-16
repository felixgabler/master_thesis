import glob
import re
from urllib.request import urlopen

import pandas as pd
from pqdm.threads import pqdm
from tqdm.auto import tqdm
from os.path import exists


def read_seq(fasta_file):
    _seq = ""
    with open(fasta_file) as file_handler:
        for _line in file_handler:
            if _line.startswith(">"):
                continue
            _seq += _line.strip()
    return _seq


def get_sequence_from_file(sequences_folder: str, uniprot_id: str):
    try:
        return read_seq(f'{sequences_folder}/{uniprot_id}.fasta')
    except Exception as e:
        print(f'Failed reading fasta file {uniprot_id}: {e.message if hasattr(e, "message") else e}')
    return None


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


def extend_proteome_features_with_new_property(fun, filter_sequences, proteome_file: str, props: [str],
                                               already_finished_file: str, result_folder: str):
    proteome = re.search(r"/([A-Z0-9]+).csv", proteome_file).group(1)
    try:
        df_proteome = pd.read_csv(proteome_file, index_col=0)
    except pd.errors.EmptyDataError:
        print(f'Could not read file {proteome_file}')
        return
    if filter_sequences is not None:
        df_proteome = filter_sequences(df_proteome)
    for prop in props:
        if prop not in df_proteome:
            df_proteome[prop] = None
        df_proteome[prop] = pd.to_numeric(df_proteome[prop], errors='coerce')
    to_fill = df_proteome[df_proteome[props[0]].isnull()].index
    if len(to_fill) == 0:
        proteomes_already_finished = pd.read_csv(already_finished_file)
        proteomes_already_finished.loc[len(proteomes_already_finished.index)] = [proteome]
        proteomes_already_finished.to_csv(already_finished_file, index=False)
        print('Already finished file')
        return
    df_proteome.loc[to_fill, props] = pqdm(df_proteome['uniprot_id'][to_fill], fun, n_jobs=20,
                                           desc=f'Sequences in {proteome}')
    df_proteome.to_csv(f'{result_folder}/{proteome}.csv')
    del df_proteome
    return


def calculate_new_property_for_proteomes(fun,
                                         ready_file: str,
                                         already_finished_file: str,
                                         result_folder: str,
                                         props: [str] = None,
                                         prop: str = None,
                                         filter_sequences=None):
    proteomes_ready = pd.read_csv(ready_file)
    proteomes_ready_list = proteomes_ready.squeeze('columns').to_list()
    if not exists(already_finished_file):
        with open(already_finished_file, 'w') as file:
            file.write('proteome\n')
    proteomes_already_finished = pd.read_csv(already_finished_file)
    proteomes_already_finished_list = proteomes_already_finished.squeeze('columns').to_list()
    proteomes_to_process = set(proteomes_ready_list).difference(
        set(proteomes_already_finished_list))
    all_proteome_files = glob.glob(f"{result_folder}/*.csv")
    files_to_process = list(
        filter(lambda file: re.search(r"/([A-Z0-9]+).csv", file).group(1) in proteomes_to_process,
               all_proteome_files))
    print(f'Files to process: {files_to_process}')
    for file in tqdm(files_to_process, desc='Files'):
        extend_proteome_features_with_new_property(
            fun,
            filter_sequences,
            file,
            props if props is not None else [prop],
            already_finished_file,
            result_folder
        )
    return
