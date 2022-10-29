from argparse import ArgumentParser

from pytorch_lightning import Trainer
from transformers import logging

from chezod_data_module import CheZODDataModule
from continuous_disorder_classifier import ContinuousDisorderClassifier


# ##### ONLY RUN THIS ONCE THE MODEL IS DECIDED UPON #####

def main(hparams):
    dm = CheZODDataModule(hparams)
    model = ContinuousDisorderClassifier.load_from_checkpoint(hparams.checkpoint, hparams_file=hparams.hparams_file)
    model.save_hyperparameters(hparams)

    trainer = Trainer.from_argparse_args(hparams, profiler="simple")

    trainer.test(model, dm)


if __name__ == "__main__":
    # Silence the warnings about transformers not loading correctly (i.e. decoder missing)
    logging.set_verbosity_error()

    parser = ArgumentParser(conflict_handler='resolve')

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
    parser = CheZODDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    main(args)
