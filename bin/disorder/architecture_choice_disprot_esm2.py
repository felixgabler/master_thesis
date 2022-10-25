from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from transformers import logging

from binary_disorder_classifier import BinaryDisorderClassifier
from disprot_data_module import DisprotDataModule


def main(hparams):
    tuning_callback = TuneReportCallback({
        "loss": "val_loss", "acc": "val_acc", "bac": "val_bac", "mcc": "val_mcc", "f1": "val_f1"
    })

    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        mode=hparams.metric_mode,
    )

    trainer = Trainer.from_argparse_args(
        hparams,
        callbacks=[early_stop_callback, tuning_callback],
    )

    config = {
        # "model_name": tune.choice(['esm2_t33_650M_UR50D']),
        "architecture": tune.choice(['rnn_crf', 'cnn', 'rnn', 'linear']),
        # "rnn": tune.choice(['lstm', 'gru']),
        # "rnn_layers": tune.choice([1, 2]),
        # "crf_after_rnn": tune.choice([True, False]),
        # "hidden_features": tune.choice([1024, 2048]),
        # "accumulate_grad_batches": tune.choice([4, 20, 64]),
        # "learning_rate": tune.loguniform(1e-5, 1e-2),
        # "encoder_learning_rate": tune.loguniform(1e-6, 1e-3),
        # "nr_frozen_epochs": tune.choice([1, 30]),
    }

    def train(c):
        # Silence the warnings about transformers not loading correctly (i.e. decoder missing)
        logging.set_verbosity_error()
        args_dict = vars(hparams)
        args_dict.update(c)
        if args_dict['model_name'] == 'facebook/esm-1b':
            args_dict['max_length'] = 1024
        params = Namespace(**args_dict)
        dm = DisprotDataModule(params)
        model = BinaryDisorderClassifier(params)
        trainer.fit(model, dm)

    trainable = tune.with_parameters(train)

    reporter = tune.CLIReporter(
        metric_columns=["loss", "acc", "bac", "mcc", "f1", "time_this_iter_s", "training_iteration"])

    # We are not using a cluster launcher but just one node for now
    # https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html#checkpointing-examples
    sync_config = tune.SyncConfig(syncer=None)

    analysis = tune.run(
        trainable,
        resources_per_trial={"cpu": 16, "gpu": 1},
        metric="loss",
        mode="min",
        config=config,
        num_samples=16,
        local_dir='./raytune_result',
        name="architecture_choice_disprot_esm2",
        sync_config=sync_config,
        checkpoint_score_attr="max-mcc",
        keep_checkpoints_num=8,
        # a very useful trick! this will resume from the last run specified by
        # sync_config (if one exists), otherwise it will start a new tuning run
        resume="AUTO",
        progress_reporter=reporter,
    )

    print(analysis.best_config)


if __name__ == "__main__":
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
        default=1,
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
