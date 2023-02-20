# -*- coding: utf-8 -*-
"""
Created on 2023/2/14 8:53

@author: Susan
"""
from pt_utils.function import TradingFrequency
from pt_utils.get_data_sql import PairTradingData
import numpy as np
from pt_utils.function import timer, calculate_single_distance_value, TradingFrequency, start_end_period, \
    visualize_price_spread
from CommonUse.funcs import createFolder
from sklearn.linear_model import LinearRegression
from OUProcess import OUProcess
import pandas as pd
from math import ceil
import itertools


# %% PairOn
class PairOn(object):

    def __init__(self, end_date: str, freq: TradingFrequency = TradingFrequency.month, freq_num: int = -1,
                 out_folder: str = 'fig/'):
        self.out_folder = out_folder
        createFolder(out_folder)
        self.start_date, self.end_date = start_end_period(end_date, freq, freq_num)
        self.pairs = None
        self.bar = None

    @timer
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

    def _pair_on(self, stock_list: list, mdate: str, pair_num: int = 20) -> (list, float):
        """
        配对组
        :param stock_list:
        :param mdate:
        :param pair_num: list of tuples
        :return:
        """
        sk_tuple = tuple(stock_list)
        cov = PairTradingData.get_cov_data(mdate).set_index('factor').sort_index()
        factor = PairTradingData.get_stock_factor(sk_tuple, mdate).set_index('sid')
        cov_matrix = np.matrix(cov)
        combinations = list(itertools.combinations(stock_list, 2))
        # distance_df = pd.DataFrame(index=stock_list, columns=stock_list)
        distance_dict = {}

        for sk_com in combinations:
            distance = calculate_single_distance_value(sk_com, factor, cov_matrix)
            # distance_df.loc[sk_com] = distance
            distance_dict.update({sk_com: distance})

        distance_series = pd.Series(distance_dict)

        pair_num_bar = ceil(pair_num * 1.5)

        pairs = distance_series.nsmallest(pair_num).index.tolist()
        bar = distance_series.nsmallest(pair_num_bar).iloc[-1]

        return pairs, bar

    @timer
    def run_pairOn(self, pair_num: int = 20) -> (list, float):
        """
        list of tuples
        :param pair_num:
        :return:
        """
        stock_list = self.stock_pool()
        _, mdate = PairTradingData.trading_date_between(self.start_date, self.end_date)
        pairs, bar = self._pair_on(stock_list, mdate, pair_num)
        self.pairs, self.bar = pairs, bar

        return pairs, bar

    @staticmethod
    def _calculate_cc_lmda(y: pd.Series, x: pd.Series) -> float:
        reg = LinearRegression(fit_intercept=False)
        X = x.values.reshape(-1, 1)
        reg.fit(X, y)
        return reg.coef_[0]

    def _single_entry_level(self, price_pair: pd.DataFrame, c: float = 0.0015):
        price_pivot = price_pair.pivot(index='date', columns='sid', values='vwap_adj')
        cc = price_pivot.columns.tolist()
        # TODO: draw log price and spread during formation
        lmda = self._calculate_cc_lmda(price_pivot.loc[:, cc[0]], price_pivot.loc[:, cc[1]])
        spread = price_pivot.apply(lambda x: np.log(x[cc[0]]) - np.log(lmda * x[cc[1]]), axis=1)
        spread_array = spread.values
        ou = OUProcess(spread_array)
        a, cycle = ou.run_ouOpt(c)

        visualize_price_spread(price_pivot, spread, self.out_folder + f'form_{str(cc)}.jpg', str(cc), [a, 0, -a])
        return lmda, a, cycle

    @timer
    def run_opt_pair_entry_level(self, pairs: list = None, c=0.0015):
        """
        配对组，配对组系数，对数价差买入阈值，一次平仓的期望时间长度
        :param pairs:
        :param c:
        :return:
        """
        price = PairTradingData.get_pair_origin_data(pairs, self.start_date, self.end_date)
        pairs_entry_dict = {}
        for cc in pairs:
            price_pair = price[price['sid'].isin(list(cc))]
            lmda, best_a, cycle = self._single_entry_level(price_pair, c)
            pairs_entry_dict.update({cc: {'lmda': lmda, 'entry': best_a, 'cycle': cycle}})
        return pairs_entry_dict


if __name__ == '__main__':
    end_date = '2018-07-01'
    pair_num = 20
    form_freq = TradingFrequency.month
    form_freq_num = -1
    c = 0.0015
    pt_db = PairOn(end_date, form_freq, form_freq_num)
    pair, bar = pt_db.run_pairOn(pair_num)
    pair_entry_dict = pt_db.run_opt_pair_entry_level(pair, c)
