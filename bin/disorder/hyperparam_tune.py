from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from transformers import logging

from argparse import ArgumentParser, Namespace

from disorder_language_model import DisorderPredictor

# Silence the warnings about transformers not loading correctly (i.e. decoder missing)
logging.set_verbosity_error()

parser = ArgumentParser()

# Checkpointing and Early Stopping
parser.add_argument("--monitor", default="val_acc", type=str, help="Quantity to monitor.")
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

# Batching
parser.add_argument(
    "--batch_size", default=1, type=int, help="Batch size to be used."
)

# add model specific args
parser = DisorderPredictor.add_model_specific_args(parser)
# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

tuning_callback = TuneReportCallback({
    "loss": "val_loss", "acc": "val_acc", "bac": "val_bac", "mcc": "val_mcc", "f1": "val_f1"
})

early_stop_callback = EarlyStopping(
    monitor=args.monitor,
    min_delta=0.0,
    patience=args.patience,
    mode=args.metric_mode,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[early_stop_callback, tuning_callback],
)

config = {
    "rnn": tune.choice(['lstm', 'gru']),
    "rnn_layers": tune.choice([1, 2]),
    "crf_after_rnn": tune.choice([True, False]),
    "hidden_features": tune.choice([1024, 2048]),
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "nr_frozen_epochs": tune.choice([1, 3])
}


def train_prottrans(c):
    args_dict = vars(args)
    args_dict.update(c)
    model = DisorderPredictor(Namespace(**args_dict))
    trainer.fit(model)


trainable = tune.with_parameters(train_prottrans)

reporter = tune.CLIReporter(metric_columns=["loss", "acc", "bac", "mcc", "f1", "training_iteration"])
analysis = tune.run(
    trainable,
    resources_per_trial={"cpu": 1, "gpu": 1},
    metric="loss",
    mode="min",
    config=config,
    num_samples=10,
    local_dir='../raytune_result',
    name="tune_prottrans",
    progress_reporter=reporter,
)

print(analysis.best_config)
