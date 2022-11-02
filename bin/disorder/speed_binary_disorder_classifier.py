import torch

from binary_disorder_classifier import BinaryDisorderClassifier


class SpeedBinaryDisorderClassifier(BinaryDisorderClassifier):

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

        elif self.hparams.architecture == 'rnn':
            # Get the emission scores from the BiLSTM
            scores, _ = self.__build_features(input_ids, attention_mask, length)
            return scores

        elif self.hparams.architecture in ['cnn', 'linear']:
            if 'esm2' in self.hparams.model_name:
                # TODO: replace end-of-word-padding with padding_idx to get real masking
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
        # tp, fp, tn, fn, _ = stat_scores(preds, targets)
        # sensitivity = recall = tp / (tp + fn)
        # specificity = tn / (tn + fp)
        # precision = tp / (tp + fp)
        # our_acc = (tp + tn) / (tp + fp + tn + fn)
        # our_bac = (sensitivity + specificity) / 2
        # our_f1 = 2 * (precision * recall) / (precision + recall)
        # our_mcc = (tn * tp - fp * fn) / np.sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))
        # WARNING: THIS WOULD NOT WORK WITH BATCH SIZE LARGER 1 !!!
        self.metric_acc(preds, targets)
        self.metric_bac(preds, targets)
        # The f1 score was very low and should instead behave like binary in sklearn
        # https://github.com/Lightning-AI/metrics/issues/629
        self.metric_f1(preds, targets)
        self.metric_mcc(preds, targets)
        self.log(f'{prefix}_loss', outputs['loss'], batch_size=self.hparams.batch_size,
                 sync_dist=self.hparams.strategy is not None and 'deepspeed' in self.hparams.strategy)
        self.log(f'{prefix}_acc', self.metric_acc, batch_size=self.hparams.batch_size,
                 sync_dist=self.hparams.strategy is not None and 'deepspeed' in self.hparams.strategy)
        self.log(f'{prefix}_bac', self.metric_bac, batch_size=self.hparams.batch_size,
                 sync_dist=self.hparams.strategy is not None and 'deepspeed' in self.hparams.strategy)
        self.log(f'{prefix}_f1', self.metric_f1, batch_size=self.hparams.batch_size,
                 sync_dist=self.hparams.strategy is not None and 'deepspeed' in self.hparams.strategy)
        self.log(f'{prefix}_mcc', self.metric_mcc, batch_size=self.hparams.batch_size,
                 sync_dist=self.hparams.strategy is not None and 'deepspeed' in self.hparams.strategy)
