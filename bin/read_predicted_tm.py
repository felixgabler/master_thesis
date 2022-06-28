import glob
import pickle
import re

import pandas as pd
from pqdm.threads import pqdm


def get_predicted_tm(path):
    with open(path, "rb") as file:
        return pickle.load(file)['ptm']


files = glob.glob('/ebio/abt1_share/alphafold_data/alphafold_output/aln_safety_preds/merged_smpls_pdbs/*_clean_*.pkl')
uniprot_ids = [re.search(r"/*_([A-Z0-9]+)_ranked_0.pkl", f).group(1) for f in files]
ptms = pqdm(files, get_predicted_tm, n_jobs=10)
df = pd.DataFrame({'uniprot_id': uniprot_ids, 'ptm': ptms})
df.to_csv('/tmp/global2/vikram/felix/master_thesis/data/suboptimal/ptms.csv', index=False)
