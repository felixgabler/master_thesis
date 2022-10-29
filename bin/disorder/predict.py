from argparse import ArgumentParser

from pytorch_lightning import Trainer
from transformers import logging

from binary_disorder_classifier import BinaryDisorderClassifier
from disprot_data_module import DisprotDataModule


def main(hparams):
    dm = DisprotDataModule(hparams)
    model = BinaryDisorderClassifier.load_from_checkpoint(hparams.checkpoint, hparams_file=hparams.hparams_file)

    trainer = Trainer.from_argparse_args(hparams)

    prediction_batches = trainer.predict(model, dm)

    accessions = dm.dataset['acc']

    out_file = open(hparams.out_file, 'w') if hparams.out_file is not None else None
    i = 0
    for pb in prediction_batches:
        for p in pb:
            acc = accessions[i]
            i += 1
            pred = "".join([str(s) for s in p.tolist()])
            if out_file is None:
                print(pred)
            else:
                out_file.writelines([acc, '\n', pred, '\n'])

    if out_file is not None:
        out_file.close()


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
        default=None,
        type=str,
        help="Path to write to. If not provided, write to console",
    )

    # add all the available trainer options to argparse
    parser = DisprotDataModule.add_data_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
