from argparse import ArgumentParser

from pytorch_lightning import Trainer
from transformers import logging

from data_utils import load_dataset
from disorder_language_model import DisorderPredictor

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
    default=None,
    type=str,
    help="Path to write to. If not provided, write to console",
)
parser.add_argument(
    "--predict_file",
    default="../data/disprot/flDPnn_Validation_Annotation.txt",
    type=str,
    help="Path to the file containing the prediction data.",
)

# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

model = DisorderPredictor.load_from_checkpoint(args.checkpoint, hparams_file=args.hparams_file)

trainer = Trainer.from_argparse_args(args)

predictions = trainer.predict(model)

dataset = load_dataset(args.predict_file)
accessions = dataset['acc']

out_file = open(args.out_file, 'w') if args.out_file is not None else None
for i, p in enumerate(predictions):
    acc = accessions[i]
    pred = "".join([str(s) for s in p.flatten().tolist()])
    if out_file is None:
        print(pred)
    else:
        out_file.writelines([acc, '\n', pred, '\n'])

if out_file is not None:
    out_file.close()
