from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.utils import collate_tensors
from transformers import AlbertTokenizer, BertTokenizer, ESMTokenizer, T5Tokenizer, XLNetTokenizer

from data_utils import load_chezod_dataset, load_chezod_dataset_two_files, load_prediction_fasta


class CheZODDataModule(LightningDataModule):
    """
    Read CheZOD files
    """

    def __init__(self, params) -> None:
        super().__init__()
        # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html#lightningmodule-hyperparameters
        self.hparams.update(vars(params))

        model_name = self.hparams.model_name
        """ Tokenizer and label encoder are needed for prepare_sample """
        if "t5" in model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        elif "albert" in model_name:
            self.tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        elif "bert" in model_name:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        elif "xlnet" in model_name:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False)
        elif "esm" in model_name:
            self.tokenizer = ESMTokenizer.from_pretrained('facebook/esm-1b', do_lower_case=False)
        else:
            print("Unkown model name")

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
            scores = torch.tensor(sample["scores"]).permute(1, 0)

            return inputs, scores
        else:
            return inputs, None

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        train_dataset = load_chezod_dataset_two_files(self.hparams.train_seqs_file,
                                                      self.hparams.train_scores_file,
                                                      self.hparams.max_length)
        return DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        val_dataset = load_chezod_dataset_two_files(self.hparams.val_seqs_file,
                                                    self.hparams.val_scores_file,
                                                    self.hparams.max_length)
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        test_dataset = load_chezod_dataset(self.hparams.test_files,
                                           self.hparams.max_length)
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """ Function that loads the prediction set. """
        predict_dataset = load_prediction_fasta(self.hparams.predict_files,
                                                self.hparams.max_length)
        return DataLoader(
            dataset=predict_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Parser for data module specific arguments/hyperparameters.
        :param parent_parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """
        parser = parent_parser.add_argument_group("CheZODDataModule")
        parser.add_argument(
            "--model_name",
            default="esm2_t33_650M_UR50D",
            type=str,
            help="Language model to use as embedding encoder (ProtTrans or ESM)",
        )
        parser.add_argument(
            "--batch_size",
            default=1,
            type=int,
            help="Batch size to be used."
        )
        parser.add_argument(
            "--max_length",
            default=3000,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--train_seqs_file",
            default="../data/CheZOD/train/CheZOD998_training_set_sequences.fasta.txt",
            type=str,
            help="Path to the files containing the train data sequences.",
        )
        parser.add_argument(
            "--train_scores_file",
            default="../data/CheZOD/train/CheZOD998_training_set_CheZOD_scores.txt",
            type=str,
            help="Path to the files containing the train data scores.",
        )
        parser.add_argument(
            "--val_seqs_file",
            default="../data/CheZOD/val/CheZOD176_val_set_sequences.fasta.txt",
            type=str,
            help="Path to the files containing the train data sequences.",
        )
        parser.add_argument(
            "--val_scores_file",
            default="../data/CheZOD/val/CheZOD176_val_set_CheZOD_scores.txt",
            type=str,
            help="Path to the files containing the train data scores.",
        )
        parser.add_argument(
            "--test_files",
            default="../data/CheZOD/test/zscores*.txt",
            type=str,
            help="Path to the files containing the test data (Use glob).",
        )
        parser.add_argument(
            "--predict_files",
            default="../data/CheZOD/predict/zscores*.txt",
            type=str,
            help="Path to the files containing the prediction data (Use glob).",
        )
        parser.add_argument(
            "--loader_workers",
            default=4,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main "
                 "process.",
        )
        return parent_parser
