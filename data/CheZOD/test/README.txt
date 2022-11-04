This is the syntax of the files:

zscores<bmrID>.txt
col1: amino acid 1-letter
col2: residue number
col3: CheZOD Z-score

shifts<bmrID>.txt
col1: residue number
col2: atom type
col3: observed chemical shift (not offset corrected)
col4: pentapeptide sequence (middle letter is current residue, n = N-term, c = C-term)
col5: estimated secondary chemical shift

The estimated offset corrections are found at the bottom of the shift file (atom type, correcton)
