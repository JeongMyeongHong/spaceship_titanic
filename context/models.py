import pandas

from context.domains import Dataset
import pandas as pd


class Model:
    def __init__(self):
        self.dataset = Dataset()
        self.dataset.data_path = './data/'
        self.dataset.save_path = './save/'

    def load_dataset(self, file_name) -> pandas.core.frame.DataFrame:
        return pd.read_csv(f'{self.dataset.data_path}{file_name}')

    @staticmethod
    def save_model(file_name, df):
        df.to_csv(f'./save/{file_name}', sep=',', na_rep='NaN')
