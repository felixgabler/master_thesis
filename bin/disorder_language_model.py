import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, LightningModule
from torchmetrics import Accuracy

from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import logging

from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors

from bi_lstm_crf import CRF

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from argparse import ArgumentParser
import os
import re
from datetime import datetime
import logging as log
import glob

# Select the model
# model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"  # only encoder part, half-precision
# model_name = "Rostlab/prot_t5_xl_uniref50"
model_name = "Rostlab/prot_bert_bfd"

# Silence the warnings about transformers not loading correctly (i.e. decoder missing)
logging.set_verbosity_error()


class NLPDataset(Dataset):
    """ A copy of torchnlp.datasets.dataset.Dataset where the equal check first checks whether the type is correct
    """

    def __init__(self, rows):
        self.columns = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Row must be a dict.')
            self.columns.update(row.keys())
        self.rows = rows

    def __getitem__(self, key):
        """
        Get a column or row from the dataset.

        Args:
            key (str or int): String referencing a column or integer referencing a row
        Returns:
            :class:`list` or :class:`dict`: List of column values or a dict representing a row
        """
        # Given an column string return list of column values.
        if isinstance(key, str):
            if key not in self.columns:
                raise AttributeError('Key not in columns.')
            return [row[key] if key in row else None for row in self.rows]
        # Given an row integer return a object of row values.
        elif isinstance(key, (int, slice)):
            return self.rows[key]
        else:
            raise TypeError('Invalid argument type.')

    def __setitem__(self, key, item):
        """
        Set a column or row for a dataset.

        Args:
            key (str or int): String referencing a column or integer referencing a row
            item (list or dict): Column or rows to set in the dataset.
        """
        if isinstance(key, str):
            column = item
            self.columns.add(key)
            if len(column) > len(self.rows):
                for i, value in enumerate(column):
                    if i < len(self.rows):
                        self.rows[i][key] = value
                    else:
                        self.rows.append({key: value})
            else:
                for i, row in enumerate(self.rows):
                    if i < len(column):
                        self.rows[i][key] = column[i]
                    else:
                        self.rows[i][key] = None
        elif isinstance(key, slice):
            rows = item
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError('Row must be a dict.')
                self.columns.update(row.keys())
            self.rows[key] = rows
        elif isinstance(key, int):
            row = item
            if not isinstance(row, dict):
                raise ValueError('Row must be a dict.')
            self.columns.update(row.keys())
            self.rows[key] = row
        else:
            raise TypeError('Invalid argument type.')

    def __len__(self):
        return len(self.rows)

    def __contains__(self, key):
        return key in self.columns

    def __str__(self):
        return str(self.rows)

    def __eq__(self, other):
        # This line would fail in DDP because [other] was inspect._empty
        return isinstance(other, NLPDataset) and self.columns == other.columns and self.rows == other.rows

    def __add__(self, other):
        return NLPDataset(self.rows + other.rows)


def load_dataset(path, max_length=1536):
    seqs = []
    labels = []
    with open(path) as file_handler:
        i = -1
        for line in file_handler:
            i += 1
            if i < 10 or len(line) > max_length:
                continue
            i_offset = i - 10
            if i_offset % 7 == 1:
                # Map rare amino acids
                seqs.append(" ".join(list(re.sub(r"[UZOB]", "X", line.strip()))))
            elif i_offset % 7 == 2:
                labels.append(line.strip())

    assert len(seqs) == len(labels)
    return NLPDataset([{"seq": seqs[i], "label": labels[i]} for i in range(len(seqs))])


class ProtTransDisorderPredictor(LightningModule):
    """
    ProtTrans model to predict intrinsic disorder in sequences.

    :param params: parsed hyperparameters from ArgumentParser
    """

    def __init__(self, params) -> None:
        super().__init__()
        # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html#lightningmodule-hyperparameters
        self.save_hyperparameters(params)

        self.model_name = model_name

        self._loss = nn.CrossEntropyLoss()

        num_classes = len(self.hparams.label_set.split(","))
        self.metric_acc = Accuracy(num_classes=num_classes, ignore_index=-100)
        # self.metric_f1 = F1Score(num_classes=num_classes, mdmc_average='global', ignore_index=-100)
        # self.metric_mcc = MatthewsCorrCoef(num_classes=num_classes)

        # build model
        self.__build_model()

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        if "t5" in self.model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.LM = T5EncoderModel.from_pretrained(self.model_name)
        elif "albert" in self.model_name:
            self.tokenizer = AlbertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.LM = AlbertModel.from_pretrained(self.model_name)
        elif "bert" in self.model_name:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.LM = BertModel.from_pretrained(self.model_name)
        elif "xlnet" in self.model_name:
            self.tokenizer = XLNetTokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.LM = XLNetModel.from_pretrained(self.model_name)
        else:
            print("Unkown model name")

        if self.hparams.gradient_checkpointing:
            self.LM.gradient_checkpointing_enable()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        # Label Encoder
        self.label_encoder = LabelEncoder(self.hparams.label_set.split(","), reserved_labels=[], unknown_index=None)

        self.hidden_features = 1024
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # https://www.dotlayer.org/en/training-rnn-using-pytorch/
        self.lstm = nn.LSTM(
            input_size=self.LM.config.hidden_size,
            hidden_size=self.hidden_features,
            num_layers=2,
            bidirectional=self.hparams.bidirectional_lstm,
            batch_first=True,
        )

        self.hidden1 = nn.Linear(
            2 * self.hidden_features if self.hparams.bidirectional_lstm else self.hidden_features,
            self.hidden_features
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.3)

        # TODO remove
        self.output = nn.Linear(
            self.hidden_features,
            self.label_encoder.vocab_size
        )

        self.crf = CRF(
            self.hidden_features,
            self.label_encoder.vocab_size
        )

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.LM.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.LM.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            predicted_labels = [
                self.label_encoder.index_to_token[prediction]
                for prediction in model_out
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    def forward(self, input_ids, attention_mask, length, token_type_ids=None):
        """ Usual pytorch forward function.
        input ids is already padded
        Returns:
            model outputs
        """
        # Should not be necessary anymore after we use return_tensors in batch encode
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)

        padded_word_embeddings = self.LM(input_ids, attention_mask)[0]

        # We pack the padded sequence to improve the computational speed during training
        # https://github.com/pytorch/pytorch/issues/43227
        pack_padded_sequences_vectors = pack_padded_sequence(padded_word_embeddings, length.cpu(),
                                                             batch_first=True,
                                                             # not that relevant for batch size 1
                                                             enforce_sorted=False)

        lstm_out, _ = self.lstm(pack_padded_sequences_vectors)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.hparams.max_length)

        x = self.hidden1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        scores, tag_seq = self.crf(x, attention_mask)

        tag_space = self.output(x)

        tag_scores = F.softmax(tag_space, dim=1)

        print(f'scores: {scores}')
        print(f'tag_seq: {tag_seq}')
        print(f'tag_space: {tag_space}')
        print(f'tag_scores: {tag_scores}')

        return tag_scores.transpose(-1, 1)

    def loss(self, predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: a tensor [batch_size x 1] with model predictions
        :param targets: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions, targets)

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param prepare_target: also load label
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        seq, label = sample['seq'], sample['label']

        inputs = self.tokenizer(seq,
                                add_special_tokens=True,
                                padding='max_length',
                                return_length=True,
                                truncation=True,
                                return_tensors='pt',
                                max_length=self.hparams.max_length)

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            labels = [self.label_encoder.batch_encode(l) for l in label]
            labels.append(torch.empty(self.hparams.max_length))
            padded_sequences_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            return inputs, padded_sequences_labels[:-1]
        except RuntimeError:
            print(label)
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss = self.loss(model_out, targets)

        return {'loss': loss}

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, y = batch
        model_out = self.forward(**inputs)
        y_hat = torch.argmax(model_out, dim=1)

        self.log('val_loss', self.loss(model_out, y))
        self.log('val_acc', self.metric_acc(y_hat, y))
        # self.log('val_f1', self.metric_f1(y_hat, y))
        # self.log('val_mcc', self.metric_mcc(y_hat, y))

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, y = batch
        model_out = self.forward(**inputs)
        y_hat = torch.argmax(model_out, dim=1)

        self.log('test_loss', self.loss(model_out, y))
        self.log('test_acc', self.metric_acc(y_hat, y))
        # self.log('test_f1', self.metric_f1(y_hat, y))
        # self.log('test_mcc', self.metric_mcc(y_hat, y))

    def predict_step(self, batch, batch_idx: int, *args, **kwargs):
        inputs, y = batch
        model_out = self.forward(**inputs)
        return torch.argmax(model_out, dim=1)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.output.parameters()},
            {"params": self.dropout.parameters()},
            {"params": self.relu.parameters()},
            {"params": self.hidden1.parameters()},
            {"params": self.lstm.parameters()},
            {
                "params": self.LM.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        if self.hparams.strategy.endswith('_offload'):
            return DeepSpeedCPUAdam(parameters, lr=self.hparams.learning_rate)
        elif self.hparams.strategy == 'deepspeed_stage_3':
            return FusedAdam(parameters, lr=self.hparams.learning_rate)
        else:
            return torch.optim.Adam(parameters, lr=self.hparams.learning_rate)

    def on_train_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.hparams.nr_frozen_epochs:
            self.unfreeze_encoder()

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        train_dataset = load_dataset(self.hparams.train_file, self.hparams.max_length)
        return DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        dev_dataset = load_dataset(self.hparams.val_file, self.hparams.max_length)
        return DataLoader(
            dataset=dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        test_dataset = load_dataset(self.hparams.test_file, self.hparams.max_length)
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parent_parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """
        parser = parent_parser.add_argument_group("ProtTransDisorderPredictor")
        parser.add_argument(
            "--max_length",
            default=1536,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=5e-06,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        # Data Args:
        parser.add_argument(
            "--label_set",
            default="0,1",
            type=str,
            help="Classification labels set.",
        )
        parser.add_argument(
            "--train_file",
            default="../data/disprot/flDPnn_Training_Annotation.txt",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--val_file",
            default="../data/disprot/flDPnn_Validation_Annotation.txt",
            type=str,
            help="Path to the file containing the validation data.",
        )
        parser.add_argument(
            "--test_file",
            default="../data/disprot/flDPnn_Test_Annotation.txt",
            type=str,
            help="Path to the file containing the test data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=4,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            default=True,
            type=bool,
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )
        parser.add_argument(
            "--bidirectional_lstm",
            default=True,
            type=bool,
            help="Enable bidirectional LSTM in the decoder.",
        )
        return parent_parser


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
parser = ProtTransDisorderPredictor.add_model_specific_args(parser)
# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

model = ProtTransDisorderPredictor(args)

logger = TensorBoardLogger(
    save_dir="logs/",
    version=datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
)

# Init model checkpoint path and saver
ckpt_path = os.path.join(
    logger.save_dir,
    logger.name,
    f"version_{logger.version}",
    "checkpoints",
)
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_path,
    filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}-{val_f1:.2f}",
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

trainer.fit(model)

best_checkpoint_path = glob.glob(ckpt_path + "/*")[0]
print(f"Best checkpoint: {best_checkpoint_path}")
