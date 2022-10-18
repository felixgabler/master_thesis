import glob
import os
from argparse import ArgumentParser
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import logging

from binary_disorder_classifier import BinaryDisorderClassifier
from disprot_data_module import DisprotDataModule


def main(hparams):
    dm = DisprotDataModule(hparams)
    model = BinaryDisorderClassifier(hparams)

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
        filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}-{val_bac:.2f}-{val_mcc:.2f}-{val_f1:.2f}",
        save_top_k=hparams.save_top_k,
        monitor=hparams.monitor,
        every_n_epochs=1,
        mode=hparams.metric_mode,
    )

    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, dm)

    best_checkpoints_paths = glob.glob(ckpt_path + "/*")
    print(f"Best checkpoints: {best_checkpoints_paths}")


if __name__ == "__main__":
    # Silence the warnings about transformers not loading correctly (i.e. decoder missing)
    logging.set_verbosity_error()

    parser = ArgumentParser(conflict_handler='resolve')

    # Checkpointing and Early Stopping
    parser.add_argument("--monitor", default="val_mcc", type=str, help="Quantity to monitor.")
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
    parser = BinaryDisorderClassifier.add_model_specific_args(parser)
    parser = DisprotDataModule.add_data_specific_args(parser)
    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
