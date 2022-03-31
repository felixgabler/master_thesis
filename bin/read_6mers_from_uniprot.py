#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import gzip
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter


def count_kmers_for_sequence(counter: Counter, seq: str, k: int):
    n_kmers = len(seq) - k + 1
    for i in range(n_kmers):
        kmer = seq[i:i + k]
        counter[kmer] += 1


k = 6
kmer_counts = Counter()
with gzip.open("/tmp/global2/vikram/felix/uniprot_trembl.fasta.gz", "rt") as f:
    _seq_parts = list()
    for line in tqdm(f):
        if line.startswith('>'):
            if len(_seq_parts) > 0:
                count_kmers_for_sequence(kmer_counts, ''.join(_seq_parts), k)
            _seq_parts.clear()
            continue
        _seq_parts.append(line.rstrip())
    count_kmers_for_sequence(kmer_counts, ''.join(_seq_parts), k)

# Save to dataframe
df = pd.DataFrame(data=kmer_counts.most_common(n=5000), columns=['kmer', 'count'])
df.to_csv("/tmp/global2/vikram/felix/trembl_kmer_counts.csv", index=False)
