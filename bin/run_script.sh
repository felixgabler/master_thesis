#!/usr/bin/bash

# Running it
# To run it on a node with 32 cores, 256GB RAM, and a GPU,\
# use `qlogin -pe parallel 64 -l h_vmem=8G,gpu=1`

# To submit a job via qsub, use `qsub -l h_vmem=8G,gpu=1 -pe parallel 32 -cwd -V run_script.sh`

python disorder_language_model.py --precision 16 --accelerator="auto" --devices="auto" --strategy="dp" \
       --max_epochs=100 --loader_workers=4 &> out_bert_crf_dp_16

# --accumulate_grad_batches=64
