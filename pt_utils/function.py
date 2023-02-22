# -*- coding: utf-8 -*-
"""
Created on 2023/2/13 15:26

@author: Susan
"""
import datetime
import warnings
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

factor_name = ['beta', 'btop', 'divyild', 'earnqlty', 'earnvar', 'earnyild', 'growth', 'invsqlty', 'leverage',
               'liquidty', 'ltrevrsl', 'midcap', 'momentum', 'profit', 'resvol', 'size']
factor_name.sort()
start_date = '2010-01-04'
end_date = '2022-11-29'


def timer(func):
    """
    计时装饰器
    :param func:
    :return:
    """

    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result

    return func_wrapper


def get_current_date():
    today = datetime.datetime.today()
    return today.strftime("%m%d")


def date_Opt_year(date: str, years: int):
    """
    date过去years
    :param date:
    :param years:
    :return:
    """
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    d_y = (d - relativedelta(years=years)).strftime('%Y-%m-%d')
    return d_y


def date_Opt_month(date: str, months: int):
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    d_y = (d - relativedelta(months=months)).strftime('%Y-%m-%d')
    return d_y


def date_Opt_day(date: str, days: int):
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    d_y = (d - relativedelta(days=days)).strftime('%Y-%m-%d')
    return d_y


def date_Opt_quarter(date: str, quarters: int):
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    d_y = (d - relativedelta(months=quarters * 3)).strftime('%Y-%m-%d')
    return d_y


# %%
class TradingFrequency(Enum):
    month = 1
    quarter = 2
    year = 3


def start_end_period(date: str, freq: TradingFrequency = TradingFrequency.month, freq_num: int = 1):
    """
    给出起止日期, start_date < end_date, DEFAULT后推
    :param date: YYYY-MM-DD
    :param freq: month, quarter, year; if none of these options, set it to be month
    :param freq_num: number of freq
    :return: start_date, end_date
    """
    if freq == TradingFrequency.quarter:
        date_2 = date_Opt_quarter(date, -freq_num)
    elif freq == TradingFrequency.year:
        date_2 = date_Opt_year(date, -freq_num)
    else:
        date_2 = date_Opt_month(date, -freq_num)
    date_list = [date, date_2]
    date_list.sort()
    return date_list[0], date_list[1]


def calculate_single_distance_value(sk_com: tuple, factor: pd.DataFrame, cov_matrix: np.matrix) -> float:
    """
    一个配对组的配对距离
    :param sk_com: 配对组tuple
    :param factor: 包含配对组的因子数据
    :param cov_matrix: 因子协方差矩阵
    :return: distance
    """
    fac_sk = factor.loc[sk_com, :]
    fac_minus = abs(fac_sk.loc[sk_com[0], :] - fac_sk.loc[sk_com[1], :]).values
    return np.dot(np.dot(fac_minus, cov_matrix), fac_minus.T)[0, 0]


def pairs_sk_set(pairs: list) -> set:
    """
    pairs中所有不同的股票
    :param pairs: list of tuples
    :return:
    """
    pairs_set_list = [set(sk) for sk in pairs]
    sk_set = pairs_set_list[0].union(*pairs_set_list[1:]) if len(pairs_set_list) else {}
    return sk_set


def visualize_price_spread(price, spread, outer_fig, name=None, hline=None):
    """
    formation, transaction分隔画图
    :param price: 待画图数据, tuple(form,trans) or pd.DataFrame or pd.Series
    :param spread:
    :param outer_fig: fig地址
    :param name: fig title
    :param hline: 是否需要添加横线，默认None不需要
    :return:
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    a, b = axes[0], axes[1]
    a.set_title('log_vwap')
    b.set_title('spread')
    price.plot(ax=a)
    spread.plot(ax=b)
    if hline:
        for i, y in enumerate(hline, start=1):
            if i % 2:
                plt.axhline(y=y, color="black", linestyle="--")
            else:
                plt.axhline(y=y, color="grey", linestyle=":")
    if name:
        plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(outer_fig, dpi=300)
    plt.close()


def visualize_spread(cc, spread, outer_fig, hline=None):
    spread.plot()
    plt.title(str(cc) + '_spread')
    if hline:
        for i, y in enumerate(hline, start=1):
            if i % 2:
                plt.axhline(y=y, color="black", linestyle="--")
            else:
                plt.axhline(y=y, color="grey", linestyle=":")
    plt.savefig(outer_fig, dpi=300)
    plt.close()


# if __name__ == '__main__':
#     date_str = '2018-01-01'
#     print(date_Opt_day(date_str, 1))
#     print(date_Opt_month(date_str, 1))
#     print(date_Opt_year(date_str, 1))
#     print(date_Opt_quarter(date_str, 1))
