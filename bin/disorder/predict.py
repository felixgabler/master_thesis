import torch

from pytorch_lightning import Trainer
from transformers import logging

from argparse import ArgumentParser

from disorder_language_model import ProtTransDisorderPredictor

# Silence the warnings about transformers not loading correctly (i.e. decoder missing)
logging.set_verbosity_error()

parser = ArgumentParser()
# remove this for newer checkpoints
parser.add_argument(
    "--model_name",
    default="Rostlab/prot_bert_bfd",
    type=str,
    help="ProtTrans language model to use as embedding encoder",
)
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

# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

model = ProtTransDisorderPredictor.load_from_checkpoint(args.checkpoint, hparams_file=args.hparams_file)

trainer = Trainer.from_argparse_args(args)

predictions = trainer.predict(model)

for p in predictions:
    pred = "".join([str(i) for i in p.flatten().tolist()])
    print(pred)
