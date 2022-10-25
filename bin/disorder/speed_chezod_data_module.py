import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchnlp.utils import collate_tensors

from chezod_data_module import CheZODDataModule
from data_utils import load_prediction_fasta


class SpeedCheZODDataModule(CheZODDataModule):
    def prepare_sample(self, sample: list) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)

        inputs = self.tokenizer(sample['seq'],
                                # Special tokens not useful for CRF return values and also make it harder
                                add_special_tokens=False,
                                padding='max_length',
                                return_length=True,
                                truncation=True,
                                return_tensors='pt',
                                max_length=self.hparams.max_length)

        if "scores" in sample:
            scores = sample["scores"]
            scores.append(torch.empty(self.hparams.max_length))
            # We pad with 999 which will be removed when calculating metrics
            padded_scores = pad_sequence(scores, batch_first=True, padding_value=999)
            return inputs, padded_scores[:-1]
        else:
            return inputs, None

    def predict_dataloader(self) -> DataLoader:
        """ Function that loads the prediction set. """
        predict_dataset = load_prediction_fasta(self.hparams.predict_files,
                                                self.hparams.max_length)
        return DataLoader(
            dataset=predict_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            pin_memory=True,
        )
