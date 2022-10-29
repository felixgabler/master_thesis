from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors
from transformers import AlbertTokenizer, BertTokenizer, ESMTokenizer, T5Tokenizer, XLNetTokenizer

from data_utils import load_disprot_dataset, load_prediction_fasta


class DisprotDataModule(LightningDataModule):
    """
    Read files in FASTA-format with accession, sequence, and binary labels.
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
        elif "esm" in model_name: # Use esm-1b tokenizer for all esm models
            self.tokenizer = ESMTokenizer.from_pretrained('facebook/esm-1b', do_lower_case=False)
        else:
            print("Unkown model name")

        # Label Encoder
        self.label_encoder = LabelEncoder(self.hparams.label_set.split(","), reserved_labels=[], unknown_index=None)

    def prepare_sample(self, sample: list) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)

        inputs = self.tokenizer(sample["seq"],
                                # Special tokens not useful for CRF return values
                                add_special_tokens=False,
                                padding='max_length',
                                return_length=True,
                                truncation=True,
                                return_tensors='pt',
                                max_length=self.hparams.max_length)

        if "label" in sample:
            label = sample["label"]
            try:
                # Prepare target:
                labels = [self.label_encoder.batch_encode(lab) for lab in label]
                unpadded = torch.cat(labels).unsqueeze(0)
                labels.append(torch.empty(self.hparams.max_length))
                # We pad with 0 but when computing the metrics and loss, the padding will already be removed or a mask is used
                padded_sequences_labels = pad_sequence(labels, batch_first=True)
                return inputs, unpadded, padded_sequences_labels[:-1]
            except RuntimeError:
                raise Exception(f"Label encoder found unknown label: {label}")
        else:
            return inputs, None, None


    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        train_dataset = load_disprot_dataset(self.hparams.train_file,
                                             self.hparams.max_length,
                                             self.hparams.skip_first_lines,
                                             self.hparams.lines_per_entry)
        return DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        dev_dataset = load_disprot_dataset(self.hparams.val_file,
                                           self.hparams.max_length,
                                           self.hparams.skip_first_lines,
                                           self.hparams.lines_per_entry)
        return DataLoader(
            dataset=dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        test_dataset = load_disprot_dataset(self.hparams.test_file,
                                            self.hparams.max_length,
                                            self.hparams.skip_first_lines,
                                            self.hparams.lines_per_entry)
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """ Function that loads the prediction set. """
        predict_dataset = load_prediction_fasta(self.hparams.predict_file,
                                                self.hparams.max_length,
                                                self.hparams.skip_first_lines)
        self.dataset = predict_dataset
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
        parser = parent_parser.add_argument_group("DisprotDataModule")
        parser.add_argument(
            "--model_name",
            default="esm2_t33_650M_UR50D",
            type=str,
            help="Language model to use as embedding encoder (ProtTrans or ESM)",
        )
        parser.add_argument(
            "--batch_size",  # WARNING: LOOK AT forward FUNCTION OF BINARY DISORDER CLASSIFIER BEFORE CHANGING
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
            "--label_set",
            default="0,1",
            type=str,
            help="Classification labels set.",
        )
        parser.add_argument(
            "--skip_first_lines",
            default=0,
            type=int,
            help="Number of lines to skip in the data file.",
        )
        parser.add_argument(
            "--lines_per_entry",
            default=3,
            type=int,
            help="How many lines each entry in the data file has.",
        )
        parser.add_argument(
            "--train_file",
            default="../data/disprot/2022/disprot-disorder-2022-train.txt",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--val_file",
            default="../data/disprot/2022/disprot-disorder-2022-val.txt",
            type=str,
            help="Path to the file containing the validation data.",
        )
        parser.add_argument(
            "--test_file",
            default="../data/disprot/2022/disprot-disorder-2022-test.txt",
            type=str,
            help="Path to the file containing the test data.",
        )
        parser.add_argument(
            "--predict_file",
            default="../data/disprot/2022/disprot-disorder-2022-val.txt",
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
        return parent_parser
