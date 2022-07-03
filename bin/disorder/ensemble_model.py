from argparse import ArgumentParser

import torch

from disorder_language_model import DisorderPredictor


class EnsembleDisorderPredictor(DisorderPredictor):
    """
    Runs multiple DisorderPredictor models and combines their predictions by majority vote

    :param params: parsed hyperparameters from ArgumentParser
    """

    def build_model(self) -> None:
        self.models = []
        for checkpoint in self.hparams.checkpoints[0]:
            self.models.append(DisorderPredictor.load_from_checkpoint(checkpoint))

    def forward(self, inputs):
        """ Usual pytorch forward function.
        Run through all models and return combined result
        Returns:
            model outputs
        """
        outs = [model(**inputs.to(model.device)) for model in self.models]
        return torch.median(torch.cat(outs), dim=0).values.view(1, -1)

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets, padded_y = batch
        targets = targets.to(self.device)

        preds = self.forward(inputs).to(self.device)
        return {'targets': targets, 'preds': preds}

    def validation_step_end(self, outputs):
        self._log_metrics(outputs, 'val')

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets, padded_y = batch
        targets = targets.to(self.device)

        preds = self.forward(inputs).to(self.device)
        return {'targets': targets, 'preds': preds}

    def test_step_end(self, outputs):
        self._log_metrics(outputs, 'test')

    def _log_metrics(self, outputs, prefix):
        preds, targets = outputs['preds'], outputs['targets']
        self.metric_acc(preds, targets)
        self.metric_bac(preds, targets)
        self.metric_f1(preds, targets)
        self.metric_mcc(preds, targets)
        self.log(f'{prefix}_acc', self.metric_acc, batch_size=self.hparams.batch_size)
        self.log(f'{prefix}_bac', self.metric_bac, batch_size=self.hparams.batch_size)
        self.log(f'{prefix}_f1', self.metric_f1, batch_size=self.hparams.batch_size)
        self.log(f'{prefix}_mcc', self.metric_mcc, batch_size=self.hparams.batch_size)

    def predict_step(self, batch, batch_idx: int, *args, **kwargs):
        inputs, y, padded_y = batch
        inputs = inputs.to(self.device)
        return self.forward(**inputs)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parent_parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """
        parent_parser = DisorderPredictor.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("EnsembleDisorderPredictor")
        parser.add_argument(
            "--checkpoints",
            type=str,
            required=True,
            nargs="+",
            action="append",
            help="List of model checkpoint files to use",
        )
        return parent_parser
