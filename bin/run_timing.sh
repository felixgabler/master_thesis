#!/bin/bash -l
python disorder/runtime_benchmark_chezod.py --accelerator="gpu" --devices="auto" --loader_workers=0 --batch_size=32 --max_length=1500 \
       --predict_file="/Users/felixgabler/PycharmProjects/master_thesis/data/uniprot/UP000005640_9606.fasta" \
       --checkpoint="/Users/felixgabler/PycharmProjects/master_thesis/data/models/timing/epoch=6-val_loss=23.01-val_spearman=0.63-val_auroc=0.71.ckpt" \
       --hparams_file="/Users/felixgabler/PycharmProjects/master_thesis/data/models/timing/hparams.yaml" \
       --out_file="/Users/felixgabler/PycharmProjects/master_thesis/data/uniprot/human_predictions.out"
