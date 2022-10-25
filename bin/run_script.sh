#!/usr/bin/bash

# Running it
# To run it on a node with 32 cores, 256GB RAM, and a GPU,\
# use `qlogin -pe parallel 64 -l h_vmem=8G,gpu=1`

# To submit a job via qsub, use `qsub -l h_vmem=8G,gpu=1 -pe parallel 32 -cwd -V run_script.sh`

# Some models: "facebook/esm-1b", "prot_t5_xl_half_uniref50-enc", "prot_t5_xl_uniref50", "prot_bert_bfd" (default)

python disorder/train.py --architecture="rnn_crf" --model_name="Rostlab/prot_bert_bfd" --precision=16 \
       --accelerator="gpu" --devices="1" --max_epochs=100 --loader_workers=4 \
       &> metrics_out

#python disorder/predict.py --checkpoint="logs/lightning_logs/version_04-06-2022--12-29-18/checkpoints/epoch=0-val_loss=93.68-val_acc=0.75-val_f1=0.00.ckpt" \
#        --hparams_file="logs/lightning_logs/04-06-2022--12-29-18/hparams.yaml" \
#        --precision=16 --accelerator="auto" --devices="auto" --strategy="dp" &> predict_bert_crf

# Make sure PL doesn't occupy all the GPUs from ray
#python disorder/hyperparam_tune_disprot_esm1.py --precision=16 --accelerator="gpu" --devices="1" --min_epochs=15 --max_epochs=30 --loader_workers=4 \
#       --train_file="/tmp/global2/vikram/felix/master_thesis/data/disprot/flDPnn_Training_Annotation.txt" \
#       --val_file="/tmp/global2/vikram/felix/master_thesis/data/disprot/flDPnn_Validation_Annotation.txt" \
#       &> out_tune_bert
