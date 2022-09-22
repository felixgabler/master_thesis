import glob
import os
from argparse import ArgumentParser
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import logging

from chezod_data_module import CheZODDataModule
from continuous_disorder_classifier import ContinuousDisorderClassifier

# Silence the warnings about transformers not loading correctly (i.e. decoder missing)
logging.set_verbosity_error()

parser = ArgumentParser(conflict_handler='resolve')

# Checkpointing and Early Stopping
parser.add_argument("--monitor", default="val_spearman", type=str, help="Quantity to monitor.")
parser.add_argument(
    "--metric_mode",
    default="max",
    type=str,
    help="If we want to min/max the monitored quantity.",
    choices=["min", "max"],
)
parser.add_argument(
    "--patience",
    default=5,
    type=int,
    help="Number of epochs with no improvement after which training will be stopped.",
)
parser.add_argument(
    "--save_top_k",
    default=3,
    type=int,
    help="The best k models according to the quantity monitored will be saved.",
)

# add model and data module specific args
parser = ContinuousDisorderClassifier.add_model_specific_args(parser)
parser = CheZODDataModule.add_data_specific_args(parser)
# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

dm = CheZODDataModule(args)
model = ContinuousDisorderClassifier(args)

logger = TensorBoardLogger(
    save_dir="../logs/",
    version=datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
)

# Init model checkpoint path and saver
ckpt_path = os.path.join(
    logger.save_dir,
    logger.name,
    f"{logger.version}",
    "checkpoints",
)
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_path,
    filename="{epoch}-{val_loss:.2f}-{val_spearman:.2f}-{val_auroc:.2f}",
    save_top_k=args.save_top_k,
    monitor=args.monitor,
    every_n_epochs=1,
    mode=args.metric_mode,
)

early_stop_callback = EarlyStopping(
    monitor=args.monitor,
    min_delta=0.0,
    patience=args.patience,
    verbose=True,
    mode=args.metric_mode,
)

trainer = Trainer.from_argparse_args(
    args,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback],
)

trainer.fit(model, dm)

best_checkpoints_paths = glob.glob(ckpt_path + "/*")
print(f"Best checkpoints: {best_checkpoints_paths}")
