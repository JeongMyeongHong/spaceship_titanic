# dname, fname, train, test, id, label
from dataclasses import dataclass

import pandas


@dataclass()
class Dataset:
    data_path: str
    save_path: str
    file_name: str
    train: pandas.core.frame.DataFrame
    test: pandas.core.frame.DataFrame
    id: str
    label: str
    
    @property
    def data_path(self) -> str: return self._data_path

    @data_path.setter
    def data_path(self, value): self._data_path = value

    @property
    def save_path(self) -> str: return self._save_path

    @save_path.setter
    def save_path(self, save_path): self._save_path = save_path

    @property
    def file_name(self) -> str: return self._file_name

    @file_name.setter
    def file_name(self, value): self._file_name = value

    @property
    def train(self) -> pandas.core.frame.DataFrame: return self._train

    @train.setter
    def train(self, value): self._train = value

    @property
    def test(self) -> pandas.core.frame.DataFrame: return self._test

    @test.setter
    def test(self, value): self._test = value

    @property
    def id(self) -> str: return self._id

    @id.setter
    def id(self, value): self._id = value

    @property
    def label(self) -> str: return self._label

    @label.setter
    def label(self, value): self._label = value
