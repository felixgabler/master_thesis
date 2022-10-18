import glob
import re

from torch.utils.data import Dataset


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


def load_disprot_dataset(path, max_length=1536, skip_first=0, lines_per_entry=3):
    items = []
    with open(path) as file_handler:
        i = -1
        item = {}
        for line in file_handler:
            i += 1
            if i < skip_first:
                continue
            i_offset = (i - skip_first) % lines_per_entry
            if i_offset == 0:
                item["acc"] = line.strip()
            elif i_offset == 1:
                # Map rare amino acids
                item["seq"] = " ".join(list(re.sub(r"[UZOB]", "X", line.strip())))
            elif i_offset == 2:
                item["label"] = line.strip()
                if len(item["label"]) <= max_length:
                    items.append(item)
                    item = {}

    return NLPDataset(items)


def load_chezod_dataset(glob_path, max_length=1536):
    items = []
    for path in glob.glob(glob_path):
        with open(path) as file_handler:
            item = {"seq": "", "scores": []}
            number = '-1'
            for line in file_handler:
                parts = line.strip().split()
                if parts[1] == number:
                    print(f'Warning: Received same number for two residues in file: {path}')
                    break
                number = parts[1]

                item["seq"] += f" {parts[0]}"
                item["scores"].append(float(parts[2]))

            if len(item["scores"]) <= max_length:
                item["seq"] = item["seq"].strip()
                items.append(item)

    return NLPDataset(items)


def load_chezod_dataset_two_files(seqs_path, scores_path, max_length=1536):
    """
    The training data from ODiNPred has a different format with two files.
    """
    seqs = []
    scores = []
    with open(seqs_path) as seqs_handle:
        for line in seqs_handle:
            if line.startswith('>') or len(line.strip()) == 0:
                continue
            masked = list(re.sub(r"[UZOB]", "X", line.strip()))
            if len(masked) <= max_length:
                seqs.append(" ".join(masked))

    with open(scores_path) as scores_handle:
        for line in scores_handle:
            if len(line.strip()) == 0:
                continue
            score = line.strip().split(':	')[1].split(', ')
            if len(score) <= max_length:
                score = list(map(float, score))
                scores.append(score)

    return NLPDataset([{"seq": seq, "scores": score} for seq, score in zip(seqs, scores)])
