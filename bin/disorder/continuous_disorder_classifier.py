import logging as log
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchmetrics import AUROC, SpearmanCorrCoef
from transformers import AlbertModel, BertModel, ESMModel, T5EncoderModel, XLNetModel


class ContinuousDisorderClassifier(LightningModule):
    """
    pLM-based model to predict continuous intrinsic disorder for sequences. Comparable to SETH and ODiNPred

    :param params: parsed hyperparameters from ArgumentParser
    """

    def __init__(self, params) -> None:
        super().__init__()
        self.save_hyperparameters(params)

        # For comparability to the SETH paper, we compute Spearman's rho and AUROC (with cutoff 8)
        self.metric_spearman = SpearmanCorrCoef()
        self.metric_auroc = AUROC()

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

        if self.hparams.architecture == 'cnn':
            # We want the CNN to return logits for BCEWithLogitsLoss
            self.cnn = SETH_CNN(1, self.LM.config.hidden_size)

        elif self.hparams.architecture == 'rnn':
            hidden_features = self.hparams.hidden_features
            self.lstm = nn.GRU(
                input_size=self.LM.config.hidden_size,
                hidden_size=hidden_features,
                num_layers=self.hparams.rnn_layers,
                bidirectional=self.hparams.bidirectional_rnn,
                batch_first=True,
            )
            rnn_out = 2 * hidden_features if self.hparams.bidirectional_rnn else hidden_features
            self.hidden1 = nn.Linear(rnn_out, 1)

        elif self.hparams.architecture == 'linear':
            self.lin1 = nn.Linear(self.LM.config.hidden_size, 10)
            self.lin2 = nn.Linear(10, 1)

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
        if self.hparams.architecture == 'rnn':
            # Get the emission scores from the BiLSTM
            scores, _ = self.__build_features(input_ids, attention_mask, length)
            scores_unpadded = scores.squeeze(dim=-1)[:, :length[0]]
            return scores_unpadded

        elif self.hparams.architecture in ['cnn', 'linear']:
            padded_word_embeddings = self.LM(input_ids, attention_mask)[0]

            # We need to make preds and target have same dimension for loss and metrics
            if self.hparams.architecture == 'cnn':
                scores = self.cnn(padded_word_embeddings).squeeze(dim=1)
            else:
                scores = self.lin2(torch.tanh(self.lin1(padded_word_embeddings))).squeeze(dim=-1)

            # WARNING: This will break if the batch size becomes larger than 1 !!!!
            scores_unpadded = scores[:, :length[0]]
            return scores_unpadded

    def loss(self, targets: torch.tensor, preds: torch.tensor) -> torch.tensor:
        """
        Computes the MSE loss function because it is a regression problem. Ignores all labels with 999.
        :param targets: Continuous disorder values without padding [batch_size x individual_length]
        :param preds: model predictions [batch_size x num_classes x individual_length]
        Returns:
            torch.tensor with loss value.
        """
        mask = targets == 999
        return F.mse_loss(preds[~mask], targets[~mask])

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
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        preds = self.forward(**inputs)
        loss = self.loss(targets, preds)
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
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        preds = self.forward(**inputs)
        loss = self.loss(targets, preds)
        return {'loss': loss, 'targets': targets, 'preds': preds}

    def _log_metrics(self, outputs, prefix):
        preds, targets = outputs['preds'], outputs['targets']
        # WARNING: this code only works for batch size 1
        # We cannot compare them directly because we first need to remove the padded areas
        mask = targets[0] == 999
        preds = preds[0][~mask].float()
        targets = targets[0][~mask]
        self.metric_spearman(preds, targets)
        auroc = self.metric_auroc((preds <= 8).float(), (targets <= 8).int())
        self.log(f'{prefix}_loss', outputs['loss'], batch_size=self.hparams.batch_size)
        self.log(f'{prefix}_spearman', self.metric_spearman, batch_size=self.hparams.batch_size)
        # We also need to replace AUROC 0 with 1 because a perfect prediction should get perfect score
        # https://torchmetrics.readthedocs.io/en/stable/classification/auroc.html
        self.log(f'{prefix}_auroc', torch.tensor(1.) if auroc.item() == 0 else auroc, batch_size=self.hparams.batch_size)

    def predict_step(self, batch, batch_idx: int, *args, **kwargs):
        inputs, y = batch
        inputs = inputs.to(self.device)
        preds = self.forward(**inputs)
        return preds

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {
                "params": self.LM.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        if self.hparams.architecture == 'cnn':
            parameters += [
                {"params": self.cnn.parameters()},
            ]
        elif self.hparams.architecture == 'rnn':
            parameters += [
                {"params": self.lstm.parameters()},
                {"params": self.hidden1.parameters()},
            ]
        elif self.hparams.architecture == 'linear':
            parameters += [
                {"params": self.lin1.parameters()},
                {"params": self.lin2.parameters()},
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
        parser = parent_parser.add_argument_group("ContinuousDisorderClassifier")
        parser.add_argument(
            "--model_name",
            default="Rostlab/prot_t5_xl_half_uniref50-enc",
            type=str,
            help="Language model to use as embedding encoder (ProtTrans or ESM)",
        )
        parser.add_argument(
            "--architecture",
            default="cnn",
            type=str,
            help="Architecture to use after embedding",
            choices=['cnn', 'rnn', 'linear']
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