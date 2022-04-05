#!/ebio/abt1_share/toolkit_support1/sources/anaconda3/bin/python

import gzip
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter

aa_counts = Counter()
with gzip.open("/tmp/global2/vikram/felix/kmer_counts/uniprot_trembl.fasta.gz", "rt") as f:
    for line in tqdm(f):
        if not line.startswith('>'):
            aa_counts += Counter(line.rstrip())

# Save to dataframe
df = pd.DataFrame(data=aa_counts.most_common(n=35), columns=['aa', 'count'])
df.to_csv("/tmp/global2/vikram/felix/kmer_counts/trembl_aa_counts.csv", index=False)
