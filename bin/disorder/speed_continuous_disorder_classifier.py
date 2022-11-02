import torch

from continuous_disorder_classifier import ContinuousDisorderClassifier


class SpeedContinuousDisorderClassifier(ContinuousDisorderClassifier):

    def forward(self, input_ids, attention_mask, length, token_type_ids=None):
        """ Usual pytorch forward function.
        input ids is already padded
        Returns:
            model outputs (unpadded, important for metrics since they can't handle padding)
        """
        if self.hparams.architecture == 'rnn':
            # Get the emission scores from the BiLSTM
            scores, _ = self.__build_features(input_ids, attention_mask, length)
            return scores.squeeze(dim=-1)

        elif self.hparams.architecture in ['cnn', 'linear']:
            if 'esm2' in self.hparams.model_name:
                # TODO: replace end-of-word-padding with padding_idx to get real masking (once ESM-2 released on HF)
                padded_word_embeddings = self.LM(input_ids, repr_layers=[33])["representations"][33]
            else:
                padded_word_embeddings = self.LM(input_ids, attention_mask)[0]

            # We need to make preds and target have same dimension for loss and metrics
            if self.hparams.architecture == 'cnn':
                scores = self.cnn(padded_word_embeddings).squeeze(dim=1)
            else:
                scores = self.lin2(torch.tanh(self.lin1(padded_word_embeddings))).squeeze(dim=-1)

            return scores

    def _log_metrics(self, outputs, prefix):
        preds, targets = outputs['preds'], outputs['targets']
        # TODO: re-look at how to best handle multiple batches
        mask = targets == 999
        preds = preds[~mask].float()
        targets = targets[~mask]
        self.metric_spearman(preds, targets)
        auroc = self.metric_auroc((preds <= 8).float(), (targets <= 8).int())
        self.log(f'{prefix}_loss', outputs['loss'],
                 sync_dist=self.hparams.strategy is not None and 'deepspeed' in self.hparams.strategy)
        self.log(f'{prefix}_spearman', self.metric_spearman,
                 sync_dist=self.hparams.strategy is not None and 'deepspeed' in self.hparams.strategy)
        # We also need to replace AUROC 0 with 1 because a perfect prediction should get perfect score
        # https://torchmetrics.readthedocs.io/en/stable/classification/auroc.html
        self.log(f'{prefix}_auroc',
                 torch.tensor(1.) if ((preds <= 8).int() == (targets <= 8).int()).all() else auroc,
                 sync_dist=self.hparams.strategy is not None and 'deepspeed' in self.hparams.strategy)
