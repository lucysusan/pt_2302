# -*- coding: utf-8 -*-
"""
Created on 2023/2/14 8:44

@author: Susan
TODO:
 - 每日检测：超过 check_day_length 天数低于pair_bar则清空这两支股票
"""
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import warnings
from pt_utils.function import TradingFrequency, start_end_period, pairs_sk_set, calculate_single_distance_value
from pt_utils.get_data_sql import PairTradingData
from math import floor

warnings.filterwarnings('ignore')

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('trading_log_file.log')
logger.addHandler(file_handler)
file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class Trading(object):

    def __init__(self, pairs_entry_dict: dict, pair_bar: float, start_date: str,
                 freq: TradingFrequency = TradingFrequency.month,
                 freq_num: int = 1):
        """
        初始化，得到trading_time_list以及最后一个交易日
        :param start_date:
        :param freq: DEFAULT，按月交易
        TODO: 按月每日检查是否仍为配对组，每日交易是否发生，交易状态是否改变，每个交易日的revenue是多少；
         - 每个交易日顺次遍历，再按照配对组的顺序顺次遍历
            - 得到因子在该交易日的数据，和因子的协方差数据，使用distance_value得到配对组的距离值，若低于bar则将配对组距离dict的value值+1
                - 剔除value值>=3的配对组(后续不再出现)，并且跳过下一步操作的执行
            - 查看对数价格差的关系，是否达到了敲入敲出的阈值点，若达到了，执行买入卖出操作，并修改该配对组的交易状态，同时记录下该笔交易的发生
        """
        self.zero_ratio = None
        self.pairs_status = None
        self.pairs = list(pairs_entry_dict.keys())
        self.pairs_entry_dict = pairs_entry_dict
        self.pair_bar = pair_bar
        self.start_date, self.end_date = start_end_period(start_date, freq, freq_num)
        self.price = PairTradingData.get_pair_origin_data(self.pairs, start_date, self.end_date)
        self.tds_list, _ = PairTradingData.trading_date_between(start_date, self.end_date)

    @staticmethod
    def _series_trading_status(series: pd.Series, a: float, zero_ratio: float = 1e-1) -> pd.Series:
        """
        对每一个对数价差序列, 返回交易状态序列
        :param series:
        :param a:
        :param zero_ratio: 判定平仓bar = a * zero_ratio
        :return:
        """
        zero_bar = a * zero_ratio
        index_list = series.index
        status = pd.Series(index=index_list, data=0)
        abs_series = abs(series)
        status_open = np.where(abs_series >= abs(a))
        status_close = np.where(abs_series <= abs(zero_bar))
        status[status_open] = 1
        status[status_close] = -1

        for end in range(len(index_list)):
            if abs(sum(status[:index_list[end + 1]])) > 1:
                status[end] = 0

        return status

    def _trading_status(self, pairs: list, zero_ratio: float = 1e-1):
        """
        每个交易日配对组的交易状态(open,close)
        :param pairs:
        :param zero_ratio:
        :return:
        """
        price = self.price
        pairs_entry_dict = self.pairs_entry_dict
        pairs_status = pd.DataFrame(index=self.tds_list)
        for cc in pairs:
            lmda = pairs_entry_dict[cc]['lmda']
            a = pairs_entry_dict[cc]['entry']
            price_pair = price[price['sid'].isin(list(cc))]

            price_pivot = price_pair.pivot(index='date', columns='sid', values='vwap_adj')
            # TODO: price_pivot here account for latter position control

            spread = price_pivot.apply(lambda x: np.log(x[cc[0]]) - np.log(lmda * x[cc[1]]), axis=1)
            status = self._series_trading_status(spread, a, zero_ratio)
            pairs_status[cc] = status
        return pairs_status

    @staticmethod
    def daily_pair_checking(mdate: str, pair_bar: float, pairf_dict: dict) -> (list, dict):
        """
        当天不再满足配对条件的配对组(大于pair_bar持续超过两天), 记录pair配对情况的dict
        :param mdate: YYYY-MM-DD 00:00:00
        :param pair_bar:
        :param pairf_dict: {pair: num(continuous not pair)}
        :return:
        """
        pairs = list(pairf_dict.keys())
        pair_close = []

        sk_tuple = tuple(pairs_sk_set(pairs))
        cov = PairTradingData.get_cov_data(mdate).set_index('factor').sort_index()
        factor = PairTradingData.get_stock_factor(sk_tuple, mdate).set_index('sid')
        cov_matrix = np.matrix(cov)

        for cc in pairs:
            distance = calculate_single_distance_value(cc, factor, cov_matrix)
            if distance > pair_bar:
                pairf_dict[cc] += 1
                if pairf_dict[cc] >= 2:
                    pairf_dict.pop(cc)
                    pair_close.append(cc)
            else:
                pairf_dict[cc] = 0

        return pair_close, pairf_dict

    @staticmethod
    def daily_pair_position_control(sk_price_1: float, sk_price_2: float, lmda: float, amount: float = 1e5):
        """

        :param sk_price_1:
        :param sk_price_2:
        :param lmda:
        :param amount:
        :return: False-卖s1, 买s2; True-买s1, 卖s2
        """
        sk_1 = sk_price_1 * 100
        sk_2 = sk_price_2 * 100
        am_1 = floor(amount / sk_1)
        am_2 = floor(amount / sk_2)
        if am_1 * lmda <= am_2:
            return -1, am_1, floor(am_1 * lmda)
        else:
            return 1, floor(am_2 / lmda), am_2

    def run_open_close(self, zero_ratio: float = 1e-1):
        pairs = self.pairs
        pair_bar = self.pair_bar
        pairf_dict = dict(zip(pairs, [0] * len(pairs)))

        pairs_status = self._trading_status(pairs, zero_ratio)
        tds_list = self.tds_list

        for i, td in enumerate(tds_list):
            pairs_close, pairf_dict = self.daily_pair_checking(td, pair_bar, pairf_dict)
            for cc in pairs_close:
                cc_status = pairs_status[cc]

                if sum(cc_status[:td]):
                    cc_status[td] = -1

                if i + 1 < len(tds_list):
                    cc_status[tds_list[i + 1]:] = 0

                pairs_status[cc] = cc_status

        self.pairs_status = pairs_status
        self.zero_ratio = zero_ratio

        return pairs_status

    def position_control(self, amount: float = 1e5, zero_ratio: float = 1e-1):
        pairs = self.pairs
        pairs_status = self.pairs_status
        price = self.price
        tds_list = self.tds_list

        close = PairTradingData.get_pair_close_data(pairs, self.start_date, self.end_date)
        close_pivot = close.pivot(index='date', columns='sid', values='close_adj')
        price_pivot = price.pivot(index='date', columns='sid', values='vwap_adj')
        pct = close_pivot.pct_change()

        columns = pd.MultiIndex.from_product(
            [pairs, ['sk_1_close', 'sk_1_pct', 'sk_1_volume', 'sk_2_close', 'sk_2_pct', 'sk_2_volume', 'cash']])
        am_df = pd.DataFrame(columns=columns, index=tds_list)

        # fill in close data and the first trading date
        for cc in pairs:
            am_df[cc]['sk_1_close'] = close[cc[0]]
            am_df[cc]['sk_2_close'] = close[cc[1]]
            am_df[cc]['sk_1_pct'] = pct[cc[0]]
            am_df[cc]['sk_2_pct'] = pct[cc[1]]
            am_df.loc[tds_list[0], (cc, 'sk_1_volume')] = 0
            am_df.loc[tds_list[0], (cc, 'sk_2_volume')] = 0
            am_df.loc[tds_list[0], (cc, 'cash')] = 0
            am_df.loc[tds_list[0], (cc, 'sk_1_pct')] = (close_pivot.loc[tds_list[0], cc[0]] - price_pivot.loc[
                tds_list[0], cc[0]]) / price_pivot.loc[tds_list[0], cc[0]]
            am_df.loc[tds_list[0], (cc, 'sk_2_pct')] = (close_pivot.loc[tds_list[0], cc[1]] - price_pivot.loc[
                tds_list[0], cc[1]]) / price_pivot.loc[tds_list[0], cc[1]]

        for cc in pairs:
            status = pairs_status[cc]
            open = status == 1
            close = status == -1

            for td in open.index.tolist():
                price_1 = price.loc[td, cc[0]]
                price_2 = price.loc[td, cc[1]]
                sig, am_1, am_2 = self.daily_pair_position_control(price_1, price_2, self.pairs_entry_dict[cc]['lmda'],
                                                                   amount)
                in_flow = sig * (-am_1 * price_1 * 100 + am_2 * price_2 * 100)
                cash_use_all = am_1 * price_1 * 100 + am_2 * price_2 * 100
                am_df.loc[td,(cc,'sk_1_volume')] = am_1
                am_df.loc[td,(cc,'sk_2_volume')] = am_2
                am_df.loc[td,(cc,)]


    def daily_trading_info_log(self):
        ...

    def daily_revenue_calculation(self):
        ...
