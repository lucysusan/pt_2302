# -*- coding: utf-8 -*-
"""
Created on 2023/2/14 8:44

@author: Susan
"""
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio as pf
from tqdm import tqdm

from pt_utils.function import TradingFrequency, start_end_period, pairs_sk_set, calculate_single_distance_value, timer, \
    visualize_spread, createFolder
from pt_utils.get_data_sql import PairTradingData

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


class Trading(object):

    def __init__(self, pairs_entry_dict: dict, pair_bar: float, start_date: str,
                 freq: TradingFrequency = TradingFrequency.month,
                 freq_num: int = 1, out_folder: str = None):
        """
        初始化，得到trading_time_list以及最后一个交易日
        :param start_date:
        :param freq: DEFAULT，按月交易
        按月每日检查是否仍为配对组，每日交易是否发生，交易状态是否改变，每个交易日的revenue是多少；
         - 每个交易日顺次遍历，再按照配对组的顺序顺次遍历
            - 得到因子在该交易日的数据，和因子的协方差数据，使用distance_value得到配对组的距离值，若低于bar则将配对组距离dict的value值+1
                - 剔除value值>=2的配对组(后续不再出现)，并且跳过下一步操作的执行
            - 查看对数价格差的关系，是否达到了敲入敲出的阈值点，若达到了，执行买入卖出操作，并修改该配对组的交易状态，同时记录下该笔交易的发生
        """
        self.pairs_status = None
        self.pairs = list(pairs_entry_dict.keys())
        self.pairs_entry_dict = pairs_entry_dict
        self.pair_bar = pair_bar
        self.start_date, self.end_date = start_end_period(start_date, freq, freq_num)
        self.price = PairTradingData.get_pair_origin_data(self.pairs, start_date, self.end_date)
        self.tds_list, _ = PairTradingData.trading_date_between(start_date, self.end_date)
        self.out_folder = out_folder
        createFolder(out_folder)

    @staticmethod
    def _series_trading_status(series: pd.Series, a: float) -> pd.Series:
        """
        对每一个对数价差序列, 返回交易状态序列
        :param series:
        :param a:
        :return:
        """
        index_list = series.index
        status = pd.Series(index=index_list, data=0)
        abs_series = abs(series)
        status_open = abs_series >= abs(a)

        sig_series = np.sign(series) * np.sign(series.shift(1))
        status_close = sig_series <= 0

        status[status_open] = 1
        status[status_close] = -1

        for end in range(len(index_list)):
            if not sum(status[:index_list[end]]) in [0, 1]:
                status[end] = 0

        # if positions exists, last day will close all positions compulsory
        sum_status = sum(status[:-1])

        status[index_list[-1]] = -1 if sum_status else 0

        return status

    def _trading_status(self, pairs: list):
        """
        每个交易日配对组的交易状态(open,close)
        :param pairs:
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

            spread = price_pivot.apply(lambda x: np.log(x[cc[0]]) - np.log(lmda * x[cc[1]]), axis=1)
            visualize_spread(cc, spread, self.out_folder + f'trans_{str(cc)}.jpg', [a, 0, -a])
            status = self._series_trading_status(spread, a)
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
    def daily_pair_position_control(sk_price_1: float, sk_price_2: float, lmda: float):
        """
        sk_1持仓，sk_2持仓
        :param sk_price_1:
        :param sk_price_2:
        :param lmda:
        :return: 买入-正持仓，卖出-负持仓
        """
        if sk_price_2 * lmda <= sk_price_1:
            return -1, 1
        else:
            return 1, -1

    def run_open_close(self):
        pairs = self.pairs
        pair_bar = self.pair_bar
        pairf_dict = dict(zip(pairs, [0] * len(pairs)))

        pairs_status = self._trading_status(pairs)
        tds_list = self.tds_list

        for i, td in tqdm(enumerate(tds_list)):
            pairs_close, pairf_dict = self.daily_pair_checking(td, pair_bar, pairf_dict)
            for cc in pairs_close:
                cc_status = pairs_status[cc]

                cc_status[td] = -1 if sum(cc_status[:tds_list[i - 1]]) else 0

                if i + 1 < len(tds_list):
                    cc_status[tds_list[i + 1]:] = 0

                pairs_status[cc] = cc_status

        self.pairs_status = pairs_status

        return pairs_status

    def write_sheet_pairs(self, pairs_df: pd.DataFrame, out_folder: str, file_name: str = '持仓明细'):
        writer = pd.ExcelWriter(out_folder + file_name + '.xlsx')
        for cc in self.pairs:
            pairs_df.loc[:, (cc,)].to_excel(writer, sheet_name=str(cc))
        writer.save()
        writer.close()

    def position_control(self, logger):
        pairs = self.pairs
        pairs_status = self.pairs_status
        price = self.price
        tds_list = self.tds_list

        close = PairTradingData.get_pair_close_data(pairs, self.start_date, self.end_date)
        close_pivot = close.pivot(index='date', columns='sid', values='close_adj')
        price_pivot = price.pivot(index='date', columns='sid', values='vwap_adj')
        pct = close_pivot.pct_change()

        columns = pd.MultiIndex.from_product(
            [pairs,
             ['sk_1_pct', 'sk_1_position', 'sk_2_pct', 'sk_2_position', 'w_1', 'w_2', 'ret']])
        am_df = pd.DataFrame(columns=columns, index=tds_list)
        # TODO: 根据status，将1--1间填充pct_change

        # fill in close data and the first trading date
        for cc in pairs:
            am_df.loc[:, (cc, 'sk_1_pct')] = pct[cc[0]]
            am_df.loc[:, (cc, 'sk_2_pct')] = pct[cc[1]]
            am_df.loc[tds_list[0], (cc, 'sk_1_position')] = 0
            am_df.loc[tds_list[0], (cc, 'sk_2_position')] = 0
            am_df.loc[tds_list[0], (cc, 'sk_1_pct')] = (close_pivot.loc[tds_list[0], cc[0]] - price_pivot.loc[
                tds_list[0], cc[0]]) / price_pivot.loc[tds_list[0], cc[0]]
            am_df.loc[tds_list[0], (cc, 'sk_2_pct')] = (close_pivot.loc[tds_list[0], cc[1]] - price_pivot.loc[
                tds_list[0], cc[1]]) / price_pivot.loc[tds_list[0], cc[1]]

        # TODO: lmda 和 ret 无关
        for cc in tqdm(pairs):
            status = pairs_status[cc]
            open = status[status == 1].index.tolist()
            close = status[status == -1].index.tolist()

            for td in open:
                price_1 = price_pivot.loc[td, cc[0]]
                price_2 = price_pivot.loc[td, cc[1]]

                am_1, am_2 = self.daily_pair_position_control(price_1, price_2, self.pairs_entry_dict[cc]['lmda'])
                am_df.loc[td, (cc, 'sk_1_position')] = am_1
                am_df.loc[td, (cc, 'sk_2_position')] = am_2
                am_df.loc[td, (cc, 'sk_1_pct')] = (close_pivot.loc[td, cc[0]] - price_pivot.loc[td, cc[0]]) / \
                                                  price_pivot.loc[td, cc[0]]
                am_df.loc[td, (cc, 'sk_2_pct')] = (close_pivot.loc[td, cc[1]] - price_pivot.loc[td, cc[1]]) / \
                                                  price_pivot.loc[td, cc[1]]

                logger.info(f'OPEN - {cc} - {td} - {(am_1, am_2)}')

            for i, td in enumerate(close):
                am_1 = am_df.loc[open[i], (cc, 'sk_1_position')].values[0]
                am_2 = am_df.loc[open[i], (cc, 'sk_2_position')].values[0]

                i_list = tds_list.index(td)
                am_df.loc[td, (cc, 'sk_1_pct')] = (price_pivot.loc[td, cc[0]] - close_pivot.loc[
                    tds_list[i_list-1], cc[0]]) / close_pivot.loc[tds_list[i_list-1], cc[0]]
                am_df.loc[td, (cc, 'sk_2_pct')] = (price_pivot.loc[td, cc[1]] - close_pivot.loc[
                    tds_list[i_list-1], cc[1]]) / close_pivot.loc[tds_list[i_list-1], cc[1]]

                am_df.loc[td, (cc, 'sk_1_position')] = 0
                am_df.loc[td, (cc, 'sk_2_position')] = 0

                nv_df = am_df.loc[open[i]:td, (cc,)][(cc,)]
                nv_df['w_1'] = (nv_df['sk_1_pct'] + 1).cumprod()
                nv_df['w_2'] = (nv_df['sk_2_pct'] + 1).cumprod()
                nv_df['w_1_pre'] = nv_df['w_1'].shift(1)
                nv_df['w_2_pre'] = nv_df['w_2'].shift(1)
                nv_df['w_1_pre'] = nv_df['w_1_pre'].fillna(1)
                nv_df['w_2_pre'] = nv_df['w_2_pre'].fillna(1)

                am_df.loc[open[i]:td, (cc, 'w_1')] = nv_df['w_1']
                am_df.loc[open[i]:td, (cc, 'w_2')] = nv_df['w_2']
                am_df.loc[open[i]:td, (cc, 'ret')] = nv_df['w_1_pre'] * nv_df['sk_1_pct'] * am_1 + nv_df['w_2_pre'] * nv_df['sk_2_pct'] * am_2

                logger.info(f'CLOSE - {cc} - {td} - {(-am_1, -am_2)}')

            am_df.loc[:, (cc, 'sk_1_position')] = am_df.loc[:, (cc, 'sk_1_position')].fillna(method='ffill')
            am_df.loc[:, (cc, 'sk_2_position')] = am_df.loc[:, (cc, 'sk_2_position')].fillna(method='ffill')

            am_df.loc[:, (cc, 'ret')] = am_df.loc[:, (cc, 'ret')].fillna(0)

        return am_df

    def calculate_pairs_rev(self, am_df, out_folder: str, rev_filename: str = 'rev_pair'):
        pairs = self.pairs
        ret_df = pd.DataFrame(index=self.tds_list, columns=pairs)
        for cc in pairs:
            df = am_df[(cc,)]
            ret_df[cc] = df['ret']

        ret_df['mean'] = ret_df.mean(axis=1)
        ret_df.to_csv(out_folder + rev_filename + '.csv')

        return ret_df

    @timer
    def run(self, out_folder: str):
        createFolder(out_folder)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(out_folder + '../trading_log_file.log')
        file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(
            f'---------------------------START_DATE\t{self.start_date}\tEND_DATE\t{self.end_date}---------------------------')
        self.run_open_close()
        am_df = self.position_control(logger)
        self.write_sheet_pairs(am_df, out_folder)
        rev_df = self.calculate_pairs_rev(am_df, out_folder)
        logger.info(f'---------------------------END---------------------------')
        return rev_df

    @staticmethod
    def evaluation(flow_table: pd.DataFrame, index_df: pd.DataFrame, index_sid: str = '000906.SH', disp: bool = False):
        """
        调用pyfolio生成可视化策略评价
        :param flow_table: date为index, 含有rev列
        :param index_df: 基准指数数据表
        :param index_sid: 基准指数名称
        :param disp: 是否显示该期图表, DEFAULT 不显示
        :return:
        """
        flow_table.index = pd.to_datetime(flow_table.index.tolist())
        cum_netv_mean_ret = (flow_table + 1).cumprod().mean(axis=1).pct_change().dropna()
        cum_netv_mean_ret.name = 'ret'
        bm_data = pd.merge(cum_netv_mean_ret.to_frame(), index_df, 'left', left_index=True, right_index=True)
        if disp:
            pf.create_returns_tear_sheet(bm_data['ret'], benchmark_rets=bm_data[index_sid])
        return bm_data

    @staticmethod
    def pkl_evaluation(pkl_route: str, index_sid: str = '000906.SH', note: str = None):
        print('\n\n--------------------------------------------------------------------------\n', note)
        data = pd.read_pickle(pkl_route)
        data.plot(figsize=(30, 18))
        plt.title('return')
        plt.show()
        pf.create_full_tear_sheet(data['ret'], benchmark_rets=data[index_sid])
        return
