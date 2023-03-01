# -*- coding: utf-8 -*-
"""
Created on 2023/2/14 8:53

@author: Susan
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from pt_utils.OUProcess import OUProcess
from pt_utils.function import timer, calculate_single_distance_value, TradingFrequency, start_end_period, \
    visualize_price_spread, createFolder
from pt_utils.get_data_sql import PairTradingData


# %% PairOn
class PairOn(object):

    def __init__(self, end_date: str, freq: TradingFrequency = TradingFrequency.month, freq_num: int = -1,
                 out_folder: str = 'fig/', pair_bar: float = 0.02, pair_pct_bar: float = 0.9):
        self.out_folder = out_folder
        createFolder(out_folder)
        self.start_date, self.end_date = start_end_period(end_date, freq, freq_num)
        self.pairs = None
        self.pair_bar = pair_bar
        self.pair_pct_bar = pair_pct_bar

    def stock_pool(self) -> list:
        """
        股票池
        :return:
        """
        start_date, end_date = self.start_date, self.end_date
        stock_list = set(PairTradingData.valid_status_stock(start_date, end_date).values)
        stock_list = stock_list.intersection(
            set(PairTradingData.valid_quote_stock(start_date, end_date).values))
        stock_list = stock_list.intersection(
            set(PairTradingData.valid_mkt_stock(start_date, end_date).values))

        return list(stock_list)

    def _get_ind_pairs(self, stock_list):
        start_date = self.start_date
        end_date = self.end_date
        sk_tuple = tuple(stock_list)
        sid_ind = PairTradingData.get_stock_industry(sk_tuple, start_date, end_date).set_index('sid')
        ind_pair_series = sid_ind.groupby(['industry']).apply(lambda x: list(itertools.combinations(x.index, 2)))
        ind_pair = [p for sub in ind_pair_series.values for p in sub]
        return ind_pair

    def _distance_date_series(self, date_series: pd.Series, sk_tuple: tuple):
        date = date_series.name
        cov_matrix = np.matrix(PairTradingData.get_cov_data(date).set_index('factor').sort_index())
        factor = PairTradingData.get_stock_factor(sk_tuple, date).set_index('sid')

        distance_v = pd.Series(data=date_series.index,index=date_series.index)
        distance_v = distance_v.apply(lambda x: calculate_single_distance_value(x, factor, cov_matrix))
        return distance_v

    def _pair_on(self, stock_list: list, tds_list: list) -> list:
        """
        配对组
        :param stock_list:
        :param tds_list: trading timeseries
        :return: list of tuples
        TODO: 1. get industry; 2. get daily distance data; 3. two bar: daily bar-smaller than 0.03 and pct_bar-0.9
        """
        sk_tuple = tuple(stock_list)
        ind_pairs = self._get_ind_pairs(stock_list)
        distance_df = pd.DataFrame(columns=tds_list, index=ind_pairs)
        tqdm.pandas(desc='formation')

        distance_df = distance_df.progress_apply(lambda x: self._distance_date_series(x, sk_tuple), axis=0)
        distance_tf = (distance_df <= self.pair_bar).mean(axis=1)
        pairs = distance_tf[distance_tf >= self.pair_pct_bar].index.tolist()

        return pairs

    @timer
    def run_pairOn(self) -> (list, float):
        """
        list of tuples
        :return:
        """
        stock_list = self.stock_pool()
        tds_list, mdate = PairTradingData.trading_date_between(self.start_date, self.end_date)
        pairs = self._pair_on(stock_list, tds_list)
        self.pairs = pairs

        return pairs

    @staticmethod
    def _calculate_cc_lmda(y: pd.Series, x: pd.Series) -> float:
        reg = LinearRegression(fit_intercept=False)
        X = x.values.reshape(-1, 1)
        reg.fit(X, y)
        return reg.coef_[0]

    def _single_entry_level(self, price_pair: pd.DataFrame, c: float = 0.0015):
        price_pivot = price_pair.pivot(index='date', columns='sid', values='vwap_adj')
        cc = price_pivot.columns.tolist()
        lmda = self._calculate_cc_lmda(price_pivot.loc[:, cc[0]], price_pivot.loc[:, cc[1]])
        spread = price_pivot.apply(lambda x: np.log(x[cc[0]]) - np.log(lmda * x[cc[1]]), axis=1)
        spread_array = spread.values
        ou = OUProcess(spread_array)

        price_pivot[cc[1]] = lmda * price_pivot[cc[1]]
        if ou.exists:
            a, cycle = ou.run_ouOpt(c)
            visualize_price_spread(price_pivot, spread, self.out_folder + f'form_{str(cc)}.jpg', str(cc), [a, 0, -a])
        else:
            a, cycle = None, None

        return lmda, a, cycle

    def run_opt_pair_entry_level(self, pairs: list = None, c=0.0015):
        """
        配对组，配对组系数，对数价差买入阈值，一次平仓的期望时间长度
        :param pairs:
        :param c:
        :return:
        """
        price = PairTradingData.get_pair_origin_data(pairs, self.start_date, self.end_date)
        pairs_entry_dict = {}
        for cc in tqdm(pairs):
            price_pair = price[price['sid'].isin(list(cc))]
            lmda, best_a, cycle = self._single_entry_level(price_pair, c)
            if best_a is not None:
                pairs_entry_dict.update({cc: {'lmda': lmda, 'entry': best_a, 'cycle': cycle}})
        self.pairs = list(pairs_entry_dict.keys())
        return pairs_entry_dict


if __name__ == '__main__':
    end_date = '2022-01-01'
    form_freq = TradingFrequency.year
    form_freq_num = -1
    c = 0.0015
    pt_db = PairOn(end_date, form_freq, form_freq_num)
    pair = pt_db.run_pairOn()
    # pair_entry_dict = pt_db.run_opt_pair_entry_level(pair, c)
