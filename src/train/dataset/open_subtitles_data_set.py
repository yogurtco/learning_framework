import json
import os
import numpy as np
from torch.distributions import ComposeTransform
from torch.utils.data.dataset import IterableDataset
from typing import List

from learning_framework.src.train.sample.text_sample import TextSample
from learning_framework.src.train.preprocess.i_preprocess import IPreprocess


class OpenSubtitlesDataSet(IterableDataset):
    """
    Open
    """
    index_file_name = 'index.json'

    def __init__(self, data_path: str, num_lines: int, preprocess: IPreprocess):
        """
        :param data_path: path to data
        :param num_lines: lines to read
        :param preprocess: additional transforms
        """

        self._data_path = data_path
        self._preprocess = preprocess

        self._index_file_path = os.path.join(self._data_path, self.index_file_name)
        index_file_exists = os.path.isfile(self._index_file_path)
        if not index_file_exists:
            self._index_dataset(self._data_path, self._index_file_path)

        with open(self._index_file_path, 'r') as f:
            self._index = json.load(f)

        self._file_names = list(self._index.keys())
        self._num_lines = num_lines
        self._sample_start_range = np.cumsum(np.array([0] + [self._index[k]-self._num_lines for k in self._file_names]))

    def __getitem__(self, index):
        file_index = np.where(self._sample_start_range >= index)[0][0]
        current_index = index - self._sample_start_range[file_index-1]
        with open(os.path.join(self._data_path, self._file_names[file_index])) as f:
            lines = f.readlines()

        sample = TextSample(lines[current_index: current_index+self._num_lines])

        if self._preprocess:
            sample = self._preprocess(sample)

        sample.convert_to_torch()
        return sample

    def __iter__(self):
        total_num_samples = self._sample_start_range[-1]
        for index in range(total_num_samples):
            yield self[index]

    @staticmethod
    def _index_dataset(data_path: str, index_file_path: str):
        """
        Create an index file to access dataset files efficiently
        """
        rows_per_file = {}
        all_files = os.listdir(data_path)
        for index, file in enumerate(all_files):
            num_lines = sum(1 for line in open(os.path.join(data_path, file)))
            rows_per_file[file] = num_lines
            print("finished {}%, file: {}".format(index / len(all_files), file))

        with open(index_file_path, 'w') as f:
            json.dump(rows_per_file, f)
            print("Index saved to {}".format(index_file_path))
