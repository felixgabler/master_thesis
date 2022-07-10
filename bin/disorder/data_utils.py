from torch.utils.data import Dataset

import re


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


def load_dataset(path, max_length=1536, skip_first=0, lines_per_entry=3):
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
