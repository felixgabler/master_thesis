import time
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from transformers import logging

from speed_chezod_data_module import SpeedCheZODDataModule
from speed_continuous_disorder_classifier import SpeedContinuousDisorderClassifier

# Silence the warnings about transformers not loading correctly (i.e. decoder missing)
logging.set_verbosity_error()

parser = ArgumentParser()
parser.add_argument(
    "--checkpoint",
    required=True,
    type=str,
    help="Path to the model checkpoint to predict with",
)
parser.add_argument(
    "--hparams_file",
    default=None,
    type=str,
    help="Path to the model parameters",
)
parser.add_argument(
    "--out_file",
    default="./predict.out",
    type=str,
    help="Path to write to",
)

# add all the available trainer options to argparse
parser = SpeedCheZODDataModule.add_data_specific_args(parser)
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
trainer = Trainer.from_argparse_args(args)

tic = time.perf_counter()

dm = SpeedCheZODDataModule(args)
model = SpeedContinuousDisorderClassifier.load_from_checkpoint(args.checkpoint, hparams_file=args.hparams_file)

predictions = trainer.predict(model, dm)
accessions = dm.dataset['acc']

with open(args.out_file, 'w') as out_file:
    for i, p in enumerate(predictions):
        acc = accessions[i]
        pred = "".join([str(s) for s in p.flatten().tolist()])
        out_file.writelines([acc, '\n', pred, '\n'])

toc = time.perf_counter()
print(f"Predicting {len(predictions)} sequences took {toc - tic:0.4f} seconds")
