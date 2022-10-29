from argparse import ArgumentParser

from pytorch_lightning import Trainer
from transformers import logging

from disprot_data_module import DisprotDataModule
from ensemble_model import EnsembleDisorderPredictor


def main(hparams):
    dm = DisprotDataModule(hparams)
    model = EnsembleDisorderPredictor(hparams)

    trainer = Trainer.from_argparse_args(hparams)

    trainer.test(model, dm)


if __name__ == "__main__":
    # Silence the warnings about transformers not loading correctly (i.e. decoder missing)
    logging.set_verbosity_error()

    parser = ArgumentParser(conflict_handler='resolve')

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    parser = EnsembleDisorderPredictor.add_model_specific_args(parser)
    parser = DisprotDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    main(args)
