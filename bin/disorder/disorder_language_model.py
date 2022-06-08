import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import ESMModel, ESMTokenizer

from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors

from bi_lstm_crf import CRF

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from argparse import ArgumentParser
import logging as log

from data_utils import load_dataset


class DisorderPredictor(LightningModule):
    """
    ProtTrans model to predict intrinsic disorder in sequences.

    :param params: parsed hyperparameters from ArgumentParser
    """

    def __init__(self, params) -> None:
        super().__init__()
        # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html#lightningmodule-hyperparameters
        self.save_hyperparameters(params)

        num_classes = len(self.hparams.label_set.split(","))
        self.metric_acc = Accuracy(num_classes=num_classes, ignore_index=-100)
        # self.metric_f1 = F1Score(num_classes=num_classes, mdmc_average='global', ignore_index=-100)
        # self.metric_mcc = MatthewsCorrCoef(num_classes=num_classes)

        self.__build_model()

    def __build_model(self) -> None:
        model_name = self.hparams.model_name
        """ Init BERT model + tokenizer + classification head."""
        if "t5" in model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
            self.LM = T5EncoderModel.from_pretrained(model_name)
        elif "albert" in model_name:
            self.tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
            self.LM = AlbertModel.from_pretrained(model_name)
        elif "bert" in model_name:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
            self.LM = BertModel.from_pretrained(model_name)
        elif "xlnet" in model_name:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False)
            self.LM = XLNetModel.from_pretrained(model_name)
        elif "esm" in model_name:
            self.tokenizer = ESMTokenizer.from_pretrained(model_name, do_lower_case=False )
            self.LM = ESMModel.from_pretrained(model_name)
        else:
            print("Unkown model name")

        if self.hparams.gradient_checkpointing and "esm" not in model_name:
            self.LM.gradient_checkpointing_enable()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        # Label Encoder
        self.label_encoder = LabelEncoder(self.hparams.label_set.split(","), reserved_labels=[], unknown_index=None)

        hidden_features = self.hparams.hidden_features

        RNN = nn.LSTM if self.hparams.rnn == 'lstm' else nn.GRU
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # https://www.dotlayer.org/en/training-rnn-using-pytorch/
        self.lstm = RNN(
            input_size=self.LM.config.hidden_size,
            hidden_size=hidden_features,
            num_layers=self.hparams.rnn_layers,
            bidirectional=self.hparams.bidirectional_rnn,
            batch_first=True,
        )

        rnn_out = 2 * hidden_features if self.hparams.bidirectional_rnn else hidden_features

        if self.hparams.crf_after_rnn:
            self.crf = CRF(
                rnn_out,
                self.label_encoder.vocab_size
            )
        else:
            self.hidden1 = nn.Linear(rnn_out, hidden_features)

            self.relu = nn.ReLU()

            self.dropout = nn.Dropout(p=0.3)

            self.crf = CRF(
                hidden_features,
                self.label_encoder.vocab_size
            )

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
                                # Special tokens not useful for CRF return values
                                add_special_tokens=False,
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
            unpadded = torch.cat(labels).unsqueeze(0)
            labels.append(torch.empty(self.hparams.max_length))
            padded_sequences_labels = pad_sequence(labels, batch_first=True)
            return inputs, unpadded, padded_sequences_labels[:-1]
        except RuntimeError:
            print(label)
            raise Exception("Label encoder found an unknown label.")

    def __build_features(self, input_ids, attention_mask, length):
        """
        All inputs will already be tensors on the right device thanks to BatchEncoding.to(device)
        https://github.com/jidasheng/bi-lstm-crf/blob/master/bi_lstm_crf/model/model.py
        :param input_ids:
        :param attention_mask:
        :param length:
        :return:
        """
        padded_word_embeddings = self.LM(input_ids, attention_mask)[0]

        # We pack the padded sequence to improve the computational speed during training
        # https://github.com/pytorch/pytorch/issues/43227
        pack_padded_sequences_vectors = pack_padded_sequence(padded_word_embeddings, length.cpu(),
                                                             batch_first=True,
                                                             # not that relevant for batch size 1
                                                             enforce_sorted=False)

        lstm_out, _ = self.lstm(pack_padded_sequences_vectors)
        x, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.hparams.max_length)

        if not self.hparams.crf_after_rnn:
            x = self.hidden1(x)
            x = self.relu(x)
            x = self.dropout(x)

        return x, attention_mask

    def forward(self, input_ids, attention_mask, length, token_type_ids=None):
        """ Usual pytorch forward function.
        input ids is already padded
        Returns:
            model outputs
        """
        # Get the emission scores from the BiLSTM
        x, attention_mask = self.__build_features(input_ids, attention_mask, length)
        _, tag_seq = self.crf(x, attention_mask)  # ignore the scores
        tag_seq = torch.tensor(tag_seq, device=self.device)
        return tag_seq

    def loss(self, xs: torch.tensor, attention_mask, length, targets: torch.tensor) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param xs: a tensor [batch_size x max_length] with data input
        :param attention_mask: Attention mask [batch_size x max_length]
        :param length: length of the sequences in this batch [batch_size]
        :param targets: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        features, masks = self.__build_features(xs, attention_mask, length)
        loss = self.crf.loss(features, targets, masks=masks)
        return loss

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets, padded_targets = batch
        inputs = inputs.to(self.device)
        padded_targets = padded_targets.to(self.device)

        # seqs = self.forward(**inputs)
        loss = self.loss(inputs['input_ids'], inputs['attention_mask'], inputs['length'], padded_targets)
        return {'loss': loss}

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, y, padded_y = batch
        inputs = inputs.to(self.device)
        y = y.to(self.device)
        padded_y = padded_y.to(self.device)

        y_hat = self.forward(**inputs)
        # y_hat = torch.argmax(model_out, dim=1)
        loss = self.loss(inputs['input_ids'], inputs['attention_mask'], inputs['length'], padded_y)

        self.log('val_loss', loss, batch_size=self.hparams.batch_size)
        self.log('val_acc', self.metric_acc(y_hat, y), batch_size=self.hparams.batch_size)
        # self.log('val_f1', self.metric_f1(y_hat, y))
        # self.log('val_mcc', self.metric_mcc(y_hat, y))

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, y, padded_y = batch
        inputs = inputs.to(self.device)
        y = y.to(self.device)

        y_hat = self.forward(**inputs)
        # y_hat = torch.argmax(model_out, dim=1)
        loss = self.loss(inputs['input_ids'], inputs['attention_mask'], inputs['length'], padded_y)

        self.log('test_loss', loss, batch_size=self.hparams.batch_size)
        self.log('test_acc', self.metric_acc(y_hat, y), batch_size=self.hparams.batch_size)
        # self.log('test_f1', self.metric_f1(y_hat, y))
        # self.log('test_mcc', self.metric_mcc(y_hat, y))

    def predict_step(self, batch, batch_idx: int, *args, **kwargs):
        inputs, y, padded_y = batch
        inputs = inputs.to(self.device)
        return self.forward(**inputs)
        # torch.argmax(model_out, dim=1)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.crf.parameters()},
            {"params": self.lstm.parameters()},
            {
                "params": self.LM.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        if not self.hparams.crf_after_rnn:
            parameters = parameters + [
                {"params": self.dropout.parameters()},
                {"params": self.relu.parameters()},
                {"params": self.hidden1.parameters()},
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

    def predict_dataloader(self) -> DataLoader:
        """ Function that loads the prediction set. """
        predict_dataset = load_dataset(self.hparams.predict_file, self.hparams.max_length)
        return DataLoader(
            dataset=predict_dataset,
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
            "--model_name",
            default="Rostlab/prot_bert_bfd",
            type=str,
            help="Language model to use as embedding encoder (ProtTrans or ESM)",
        )
        parser.add_argument(
            "--rnn",
            default="lstm",
            type=str,
            help="Type of RNN architecture to use",
            choices=['lstm', 'gru']
        )
        parser.add_argument(
            "--rnn_layers",
            default=2,
            type=int,
            help="Number of layers for the rnn.",
        )
        parser.add_argument(
            "--bidirectional_rnn",
            default=True,
            type=bool,
            help="Enable bidirectional RNN in the encoder.",
        )
        parser.add_argument(
            "--crf_after_rnn",
            default=False,
            type=bool,
            help="Whether to directly feed the RNN output into the CRF or to add linear net and dropout.",
        )
        parser.add_argument(
            "--hidden_features",
            default=1024,
            type=int,
            help="Number of neurons in the hidden linear net.",
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
            "--max_length",
            default=1536,
            type=int,
            help="Maximum sequence length.",
        )
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
            "--predict_file",
            default="../data/disprot/flDPnn_Validation_Annotation.txt",
            type=str,
            help="Path to the file containing the prediction data.",
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
                with the gpu memory to store the model. (Does not apply to ESM models)",
        )
        return parent_parser
