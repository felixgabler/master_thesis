from argparse import ArgumentParser

from pytorch_lightning import Trainer
from transformers import logging

from disorder_language_model import DisorderPredictor

# ##### ONLY RUN THIS ONCE THE MODEL IS DECIDED UPON #####

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

# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

model = DisorderPredictor.load_from_checkpoint(args.checkpoint, hparams_file=args.hparams_file)

trainer = Trainer.from_argparse_args(args, profiler="simple")

metrics = trainer.test(model)

print(f"Resulting metrics: {metrics}")
