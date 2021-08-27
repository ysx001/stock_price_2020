from math import sqrt
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class StockData():
    def __init__(self, data_path: str):
        self._init_df(data_path)
        self.attribute_list = ['open', 'close', 'low', 'high', 'average']
        self._init_first_deri()
        self._init_second_deri()

    def _init_df(self, data_path: str):
        df = pd.read_csv(data_path)
        self.symbol_list = np.sort(df.symbol.unique())
        self.time_list = df.time.unique()
        self.day_list = df.day.unique()
        # important, do this after getting the time and days
        df = df.set_index(['day', 'time'])
        self.all = df
        self.open = self._get_filled_df('open')
        self.close = self._get_filled_df('close')
        self.low = self._get_filled_df('low')
        self.high = self._get_filled_df('high')
        self.average = self._get_filled_df('average')

    def _init_first_deri(self):
        for attribute in self.attribute_list:
            self._add_first_deri(attribute)

    def _init_second_deri(self):
        for attribute in self.attribute_list:
            self._add_second_deri(attribute)

    def _get_index(self):
        # create multi index of both day and time
        return pd.MultiIndex.from_product(
            [self.day_list, self.time_list],
            names=["day", "time"])

    def _get_filled_df(self, column_name: str):
        new_df = pd.DataFrame(0,
                              index=self._get_index(),
                              columns=self.symbol_list)
        for sym in self.symbol_list:
            new_df[sym] = self.all[self.all['symbol'] == sym][column_name]
        new_df = new_df.fillna(method='ffill')
        new_df = new_df.fillna(method='bfill')
        return new_df

    def get_slice(self,
                  attr_name: str,
                  day_slice: Tuple):
        return getattr(self, attr_name).loc[day_slice[0]: day_slice[1], :]

    def plot_data(self,
                  attr_name: str,
                  symbol_list: List[str]):
        for sym in symbol_list:
            y = getattr(self, attr_name)[sym].to_list()
            plt.plot(np.arange(len(y)), y, label=sym)
        plt.legend()

    def plot_corr(self, attr_name):
        """
        Calculate correlation matrix for each stock.
        """
        corrMatrix = getattr(self, attr_name)[self.symbol_list].corr()
        ax = sns.heatmap(corrMatrix, annot=True, cmap='PiYG')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.savefig(f'{attr_name}_corrlation.png')

    def _add_first_deri(self, attr_name):
        for sym in self.symbol_list:
            symbol_list = getattr(self, attr_name)[sym].to_list()
            first = np.gradient(symbol_list)
            getattr(self, attr_name)[sym + '_first_deri'] = first

    def _add_second_deri(self, attr_name):
        for sym in self.symbol_list:
            symbol_list = getattr(self, attr_name)[sym].to_list()
            second = np.gradient(symbol_list, 2)
            getattr(self, attr_name)[sym + '_second_deri'] = second