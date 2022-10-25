import time
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from transformers import logging

from chezod_data_module import CheZODDataModule
from continuous_disorder_classifier import ContinuousDisorderClassifier

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
parser = CheZODDataModule.add_data_specific_args(parser)
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
trainer = Trainer.from_argparse_args(args)

tic = time.perf_counter()

dm = CheZODDataModule(args)
model = ContinuousDisorderClassifier.load_from_checkpoint(args.checkpoint, hparams_file=args.hparams_file)

predictions = trainer.predict(model, dm)

with open(args.out_file, 'w') as out_file:
    for p in predictions:
        pred = "".join([str(s) for s in p.flatten().tolist()])
        out_file.writelines([pred, '\n'])

toc = time.perf_counter()
print(f"Predicting {len(predictions)} sequences took {toc - tic:0.4f} seconds")
