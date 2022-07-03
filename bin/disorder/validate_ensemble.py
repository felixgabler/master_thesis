from argparse import ArgumentParser

from pytorch_lightning import Trainer
from transformers import logging

from ensemble_model import EnsembleDisorderPredictor

# Silence the warnings about transformers not loading correctly (i.e. decoder missing)
logging.set_verbosity_error()

parser = ArgumentParser()
parser.add_argument(
    "--batch_size", default=1, type=int, help="Batch size to be used."
)
# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)
parser = EnsembleDisorderPredictor.add_model_specific_args(parser)
args = parser.parse_args()

model = EnsembleDisorderPredictor(args)

trainer = Trainer.from_argparse_args(args)

trainer.validate(model)
