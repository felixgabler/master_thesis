import time
from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import Trainer
from transformers import logging

from speed_chezod_data_module import SpeedCheZODDataModule
from speed_continuous_disorder_classifier import SpeedContinuousDisorderClassifier


def main(hparams):
    trainer = Trainer.from_argparse_args(hparams)

    tic = time.perf_counter()

    dm = SpeedCheZODDataModule(hparams)
    model = SpeedContinuousDisorderClassifier.load_from_checkpoint(hparams.checkpoint,
                                                                   hparams_file=hparams.hparams_file)

    prediction_batches = trainer.predict(model, dm)
    accessions = dm.dataset['acc']

    with open(hparams.out_file, 'w') as out_file:
        i = 0
        for b in prediction_batches:
            for p in b:
                acc = accessions[i]
                i += 1
                pred = ",".join([str(np.round(s, decimals=4)) for s in p.tolist()])
                out_file.writelines([acc, '\n', pred, '\n'])

    toc = time.perf_counter()
    print(f"Predicting {i} sequences took {toc - tic:0.4f} seconds")


if __name__ == "__main__":
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

    main(args)
