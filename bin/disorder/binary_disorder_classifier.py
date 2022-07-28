import logging as log
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from bi_lstm_crf import CRF
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef
from transformers import AlbertModel, BertModel, ESMModel, T5EncoderModel, XLNetModel


class BinaryDisorderClassifier(LightningModule):
    """
    pLM-based model to predict binary intrinsic disorder for sequences.

    :param params: parsed hyperparameters from ArgumentParser
    """

    def __init__(self, params) -> None:
        super().__init__()
        # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html#lightningmodule-hyperparameters
        self.save_hyperparameters(params)

        self.num_classes = len(self.hparams.label_set.split(","))
        # We do not use ignore index because by the time we use the metrics, the predictions will already be unpadded
        # We have to set multiclass because we want to transform binary data to multi class format.
        self.metric_acc = Accuracy(num_classes=self.num_classes, multiclass=True)
        # Behaves like sklearn.metrics.balanced_accuracy_score
        self.metric_bac = Accuracy(num_classes=self.num_classes, average='macro', multiclass=True)
        # We only want to see the f1 score with regard to misclassifying disorder. Will therefore extract the second dim
        # https://github.com/Lightning-AI/metrics/issues/629
        self.metric_f1 = F1Score(
            num_classes=self.num_classes,
            average='none',
            multiclass=True
        )
        self.metric_mcc = MatthewsCorrCoef(num_classes=self.num_classes)

        self.build_model()

    def build_model(self) -> None:
        model_name = self.hparams.model_name
        """ Init BERT model + tokenizer + classification head."""
        if "t5" in model_name:
            self.LM = T5EncoderModel.from_pretrained(model_name)
        elif "albert" in model_name:
            self.LM = AlbertModel.from_pretrained(model_name)
        elif "bert" in model_name:
            self.LM = BertModel.from_pretrained(model_name)
        elif "xlnet" in model_name:
            self.LM = XLNetModel.from_pretrained(model_name)
        elif "esm" in model_name:
            self.LM = ESMModel.from_pretrained(model_name)
        else:
            print("Unkown model name")

        if self.hparams.gradient_checkpointing and "esm" not in model_name:
            self.LM.gradient_checkpointing_enable()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        if self.hparams.architecture == 'rnn_crf':
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

            self.hidden1 = nn.Linear(rnn_out, hidden_features)

            self.crf = CRF(hidden_features, self.num_classes)

        elif self.hparams.architecture == 'cnn':
            # We want the CNN to return logits for BCEWithLogitsLoss
            self.cnn = SETH_CNN(1, self.LM.config.hidden_size)

        elif self.hparams.architecture == 'linear':
            self.lin1 = nn.Linear(self.LM.config.hidden_size, 1)

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

        x = self.hidden1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)

        return x, attention_mask

    def forward(self, input_ids, attention_mask, length, token_type_ids=None):
        """ Usual pytorch forward function.
        input ids is already padded
        Returns:
            model outputs (unpadded, important for metrics since they can't handle padding)
        """
        if self.hparams.architecture == 'rnn_crf':
            # Get the emission scores from the BiLSTM
            x, attention_mask = self.__build_features(input_ids, attention_mask, length)
            _, tag_seq = self.crf(x, attention_mask)  # ignore the scores
            tag_seq = torch.tensor(tag_seq, device=self.device)
            return tag_seq

        elif self.hparams.architecture in ['cnn', 'linear']:
            padded_word_embeddings = self.LM(input_ids, attention_mask)[0]

            # We need to make preds and target have same dimension for loss and metrics
            if self.hparams.architecture == 'cnn':
                scores = self.cnn(padded_word_embeddings).squeeze(dim=1)
            else:
                scores = self.lin1(padded_word_embeddings).squeeze(dim=-1)

            # WARNING: This will break if the batch size becomes larger than 1 !!!!
            scores_unpadded = scores[:, :length[0]]
            return scores_unpadded

    def loss(self, xs: torch.tensor, attention_mask, length, padded_targets: torch.tensor, targets: torch.tensor,
             preds=None) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param xs: a tensor [batch_size x max_length] with data input
        :param attention_mask: Attention mask [batch_size x max_length]
        :param length: length of the sequences in this batch [batch_size]
        :param padded_targets: Label values with padding [batch_size x max_length]
        :param targets: Label values without padding [batch_size x individual_length]
        :param preds: potentially precomputed predictions (CRF does not need them for training)
                        [batch_size x num_classes x individual_length]
        Returns:
            torch.tensor with loss value.
        """
        if self.hparams.architecture == 'rnn_crf':
            features, masks = self.__build_features(xs, attention_mask, length)
            loss = self.crf.loss(features, padded_targets, masks=masks)
            return loss

        elif self.hparams.architecture in ['cnn', 'linear']:
            preds = preds if preds is not None else self.forward(xs, attention_mask, length)
            # We need to make preds and target have same dimension,
            # cast targets to float, and also ignore the padded values
            loss = F.binary_cross_entropy_with_logits(preds, targets.float())
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
        loss = self.loss(inputs['input_ids'], inputs['attention_mask'], inputs['length'], padded_targets, targets)
        return {'loss': loss}

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_step_end function.
        """
        return self._evaluate(batch)

    def validation_step_end(self, outputs):
        self._log_metrics(outputs, 'val')

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the test_step_end function.
        """
        return self._evaluate(batch)

    def test_step_end(self, outputs):
        self._log_metrics(outputs, 'test')

    def _evaluate(self, batch: tuple):
        inputs, targets, padded_targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        padded_targets = padded_targets.to(self.device)

        preds = self.forward(**inputs)
        # y_hat = torch.argmax(model_out, dim=1)
        loss = self.loss(inputs['input_ids'], inputs['attention_mask'], inputs['length'], padded_targets, targets,
                         preds)
        return {'loss': loss, 'targets': targets, 'preds': preds}

    def _log_metrics(self, outputs, prefix):
        preds, targets = outputs['preds'], outputs['targets']
        # tp, fp, tn, fn, _ = stat_scores(preds, targets)
        # sensitivity = recall = tp / (tp + fn)
        # specificity = tn / (tn + fp)
        # precision = tp / (tp + fp)
        # our_acc = (tp + tn) / (tp + fp + tn + fn)
        # our_bac = (sensitivity + specificity) / 2
        # our_f1 = 2 * (precision * recall) / (precision + recall)
        # our_mcc = (tn * tp - fp * fn) / np.sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))
        # WARNING: THIS WOULD NOT WORK WITH BATCH SIZE LARGER 1 !!!
        preds = preds[0]
        targets = targets[0]
        self.metric_acc(preds, targets)
        self.metric_bac(preds, targets)
        # The f1 score was very low and should instead behave like binary in sklearn
        # https://github.com/Lightning-AI/metrics/issues/629
        f1 = self.metric_f1(preds, targets)[1]
        self.metric_mcc(preds, targets)
        self.log(f'{prefix}_loss', outputs['loss'], batch_size=self.hparams.batch_size)
        self.log(f'{prefix}_acc', self.metric_acc, batch_size=self.hparams.batch_size)
        self.log(f'{prefix}_bac', self.metric_bac, batch_size=self.hparams.batch_size)
        self.log(f'{prefix}_f1', f1, batch_size=self.hparams.batch_size)
        self.log(f'{prefix}_mcc', self.metric_mcc, batch_size=self.hparams.batch_size)

    def predict_step(self, batch, batch_idx: int, *args, **kwargs):
        inputs, y, padded_y = batch
        inputs = inputs.to(self.device)
        preds = self.forward(**inputs)
        if self.hparams.architecture == 'rnn_crf':
            return preds
        else:
            return (preds > 0.5).int()

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {
                "params": self.LM.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        if self.hparams.architecture == 'rnn_crf':
            parameters += [
                {"params": self.crf.parameters()},
                {"params": self.hidden1.parameters()},
                {"params": self.lstm.parameters()},
            ]
        elif self.hparams.architecture == 'cnn':
            parameters += [
                {"params": self.cnn.parameters()},
            ]
        elif self.hparams.architecture == 'linear':
            parameters += [
                {"params": self.lin1.parameters()},
            ]

        if self.hparams.strategy is not None and self.hparams.strategy.endswith('_offload'):
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

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parent_parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """
        parser = parent_parser.add_argument_group("BinaryDisorderClassifier")
        parser.add_argument(
            "--model_name",
            default="Rostlab/prot_t5_xl_half_uniref50-enc",
            type=str,
            help="Language model to use as embedding encoder (ProtTrans or ESM)",
        )
        parser.add_argument(
            "--architecture",
            default="rnn_crf",
            type=str,
            help="Architecture to use after embedding",
            choices=['rnn_crf', 'cnn', 'linear']
        )
        parser.add_argument(
            "--rnn",
            default="gru",
            type=str,
            help="Type of RNN architecture to use",
            choices=['lstm', 'gru']
        )
        parser.add_argument(
            "--rnn_layers",
            default=1,
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
            "--hidden_features",
            default=1024,
            type=int,
            help="Number of neurons in the hidden linear net.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=2e-04,
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
            "--gradient_checkpointing",
            default=True,
            type=bool,
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model. (Does not apply to ESM models)",
        )
        return parent_parser


# Taken from https://github.com/DagmarIlz/SETH/blob/main/SETH_1.py
class SETH_CNN(nn.Module):
    def __init__(self, n_classes, n_features):
        super(SETH_CNN, self).__init__()
        self.n_classes = n_classes
        bottleneck_dim = 28
        self.classifier = nn.Sequential(
            # summarize information from 5 neighbouring amino acids (AAs)
            # padding: dimension corresponding to AA number does not change
            nn.Conv2d(n_features, bottleneck_dim, kernel_size=(5, 1), padding=(2, 0)),
            nn.Tanh(),
            nn.Conv2d(bottleneck_dim, self.n_classes, kernel_size=(5, 1), padding=(2, 0))
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of output nodes (1 for disorder, since predict one continuous number)
        """
        # IN: X = (B x L x F); OUT: (B x F x L x 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        Yhat = self.classifier(x)  # OUT: Yhat_consurf = (B x N x L x 1)
        # IN: (B x N x L x 1); OUT: ( B x N x L )
        Yhat = Yhat.squeeze(dim=-1)
        return Yhat
