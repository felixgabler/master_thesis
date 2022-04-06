#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import glob
import re
import pandas as pd
from pqdm.threads import pqdm
from tqdm.auto import tqdm

from iupred3.iupred3_lib import read_seq

data_dir = '/tmp/global2/vikram/felix/master_thesis/data/alphafold/v2'

repeats_regex = '(?=(XXXXXX|QQQQQQ|SSSSSS|AAAAAA|GGGGGG|PPPPPP|EEEEEE|LLLLSL|NNNNNN|DDDDDD|TTTTTT|KKKKKK))'
non_repeats_regex = '(?=(GGFGNW|DMAFPR|TVYPPL|WTVYPP|GWTVYP|PDMAFP|GFGNWL|LLLSLP|NNMSFW|SLPVLA|GTGWTV|TGWTVY|LSLPVL|IGGFGN|LLSLPV|FFMVMP|MIFFMV|IFFMVM|IFSLHL|SLHLAG|APDMAF|FSLHLA|LPVLAG|SFWLLP|FPRMNN|FWLLPP|VTAHAF|PVLAGA|SSILGA|VLAGAI|NMSFWL|MNNMSF|AFPRMN|GAPDMA|RMNNMS|LAGAIT|MSFWLL|MAFPRM|PRMNNM|AGAITM|FMVMPI|ITMLLT|VYPPLS|WLLPPS|TMLLTD|GAITML|AITMLL|MIGGFG|IVTAHA|GNWLVP|FGNWLV|GAGTGW|AGTGWT|MLLTDR|AIFSLH|NWLVPL|AGISSI|WLVPLM|MVMPIM|LLTDRN|LGAPDM|MLGAPD|GSGKST|GISSIL|PIMIGG|ISSILG|IMIGGF|YPPLSS|MPIMIG|VMPIMI|LAIFSL|LLPPSL|DLAIFS|PLMLGA|LMLGAP|VDLAIF|HTGEKP|FIMIFF|AFIMIF|HAFIMI|IMIFFM|SVDLAI|LAGISS|LHLAGI|HLAGIS|YNVIVT|LVPLML|PLFVWS))'


def get_sequence_from_file(uniprot_id: str):
    try:
        return read_seq(f'{data_dir}/sequences/{uniprot_id}.fasta')
    except Exception as e:
        print(f'Failed reading fasta file {uniprot_id}: {e.message if hasattr(e, "message") else e}')
    return None


def load_6mer_counts(uniprot_id: str):
    seq = get_sequence_from_file(uniprot_id)
    if seq is None:
        return None, None
    try:
        return sum(1 for _ in re.finditer(repeats_regex, seq)), sum(1 for _ in re.finditer(non_repeats_regex, seq))
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
    if 'repeat_6mers_100' not in df_proteome:
        df_proteome['repeat_6mers_100'] = None
    if 'non_repeat_6mers_100' not in df_proteome:
        df_proteome['non_repeat_6mers_100'] = None
    df_proteome['repeat_6mers_100'] = pd.to_numeric(df_proteome['repeat_6mers_100'], errors='coerce')
    df_proteome['non_repeat_6mers_100'] = pd.to_numeric(df_proteome['non_repeat_6mers_100'], errors='coerce')
    to_fill = df_proteome[df_proteome['repeat_6mers_100'].isnull()].index
    if len(to_fill) == 0:
        proteomes_with_6mer_counts = pd.read_csv(f"{data_dir}/proteomes_with_6mer_counts.csv", header=None)
        proteomes_with_6mer_counts.loc[len(proteomes_with_6mer_counts.index)] = [proteome]
        proteomes_with_6mer_counts.to_csv(f'{data_dir}/proteomes_with_6mer_counts.csv', header=False,
                                          index=False)
        print('Already filled with counted 6mers')
        return
    counts = pqdm(df_proteome['uniprot_id'][to_fill], load_6mer_counts, n_jobs=20, desc=f'Sequences in {proteome}')
    df_proteome.loc[to_fill, ['repeat_6mers_100', 'non_repeat_6mers_100']] = counts
    df_proteome.to_csv(f'{data_dir}/AA_scores/{proteome}.csv')
    del df_proteome
    return


def calculate_6mer_counts():
    proteomes_with_domain_count = pd.read_csv(f"{data_dir}/proteomes_with_domain_count.csv", header=None)
    proteomes_with_domain_count_list = proteomes_with_domain_count.squeeze('columns').to_list()
    proteomes_with_6mer_counts = pd.read_csv(f"{data_dir}/proteomes_with_6mer_counts.csv", header=None)
    proteomes_with_6mer_counts_list = proteomes_with_6mer_counts.squeeze('columns').to_list()
    proteomes_to_extend_with_6mer_counts = set(proteomes_with_domain_count_list).difference(
        set(proteomes_with_6mer_counts_list))
    all_proteome_files = glob.glob(f"{data_dir}/AA_scores/*.csv")
    proteome_files_without_6mer_counts = list(
        filter(lambda file: re.search(r"/([A-Z0-9]+).csv", file).group(1) in proteomes_to_extend_with_6mer_counts,
               all_proteome_files))
    for file in tqdm(proteome_files_without_6mer_counts, desc='Files'):
        extend_proteome_features_with_6mer_counts(file)
    return


if __name__ == '__main__':
    calculate_6mer_counts()
