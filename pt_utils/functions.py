# -*- coding: utf-8 -*-
"""
Created on 2023/1/30 18:26

@author: Susan
"""
import datetime
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CommonUse.funcs import read_pkl, createFolder
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from scipy.special import erfi
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['FangSong']


# %% COMMON FUNCTIONS
def date_Opt_year(date: str, years: int):
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    d_y = (d - relativedelta(years=years)).strftime('%Y-%m-%d')
    return d_y


def date_Opt_month(date: str, months: int):
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    d_y = (d - relativedelta(months=months)).strftime('%Y-%m-%d')
    return d_y


# %% COMMON VARIABLES
start_date = '2018-01-01'
end_date = '2022-11-30'
form_end = '2022-01-01'
form_y = 1
trans_m = 6
form_start = date_Opt_year(form_end, form_y)
trans_start = form_end
trans_end = date_Opt_month(trans_start, -trans_m)
print(
    f'formation start\t{form_start}\tformation end\t{form_end}\ntransaction start\t{trans_start}\ttransaction end\t{trans_end}')

in_folder = 'use_data/'
out_folder = f'result/{form_end}_{form_y}Y_{trans_start}_{trans_m}M/'
fig_folder = out_folder + 'fig/'
createFolder(fig_folder)
# TODO: stock pool, log_vwap_aft_price_pivot, origin_price_pivot(vwap_aft_price_pivot)
stock_pool = read_pkl('data/output_process/' + 'stock_pool.pkl')
bar_tolerance = 1e-4

c = 0.01
c_ratio = 1e-4


# TODO: add function separately deals with data splitting
def split_form_trans(data, form_start=form_start, form_end=form_end, trans_start=trans_start, trans_end=trans_end):
    data_form = data[(data.index >= form_start) & (data.index < form_end)]
    data_trans = data[(data.index >= trans_start) & (data.index <= trans_end)]
    return data_form, data_trans


# %% functions

def plt_split_time(data, outer_fig, name=None, hline=None):
    """
    formation, transaction分隔画图
    :param hline: 是否需要添加横线，默认None不需要
    :param data: 待画图数据, tuple(form,trans) or pd.DataFrame or pd.Series
    :param outer_fig: fig地址
    :param name: fig title
    :return:
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey='row', figsize=(20, 10))
    a, b = axes[0], axes[1]
    a.set_title('formation')
    b.set_title('transaction')
    if isinstance(data, tuple):
        data_form, data_trans = data
    else:
        data_form, data_trans = split_form_trans(data)
    data_form.plot(ax=a)
    data_trans.plot(ax=b)
    if hline:
        for i, y in enumerate(hline, start=1):
            if i % 2:
                plt.axhline(y=y, color="black", linestyle="--")
            else:
                plt.axhline(y=y, color="grey", linestyle=":")
    if name:
        plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(outer_fig)
    plt.show()


def plt_log_price_pair(cc_top_list, price_pivot, figFolder: str = fig_folder + 'log_price/'):
    """
    画出给定match组的股票对数价格的走势图
    :param cc_top_list: list中每个元素为tuple,类似 [('002040.SZ', '600190.SH'), ('603233.SH', '603883.SH')]
    :param price_pivot: log_price_pivot
    :param figFolder:
    :return:
    """
    createFolder(figFolder)
    for c_code in tqdm(cc_top_list):
        c_df = price_pivot.loc[:, c_code]
        plt_split_time(c_df, figFolder + c_code[0] + '-' + c_code[1] + '.png')


# %% OU PROCESS
@dataclass
class OUParams:
    alpha: float  # mean reversion parameter
    gamma: float  # asymptotic mean
    beta: float  # Brownian motion scale (standard deviation)


def estimate_OU_params(X_t: np.ndarray) -> OUParams:
    """
    Estimate OU params from OLS regression.
    - X_t is a 1D array.
    Returns instance of OUParams.
    """
    y = np.diff(X_t)
    X = X_t[:-1].reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    # regression coeficient and constant
    alpha = -reg.coef_[0]
    gamma = reg.intercept_ / alpha
    # residuals and their standard deviation
    y_hat = reg.predict(X)
    beta = np.std(y - y_hat)
    return OUParams(alpha, gamma, float(beta))


def expect_return_optimize_function(a: float or list, alpha: float, beta: float, c: float):
    """
    用期望收益优化进出价格
    :param a: entry level
    :param alpha: OU Param
    :param beta: OU Param
    :param c: trading cost
    :return:
    """
    if not isinstance(a, float):
        a = a[0]
    l = np.exp((alpha * a ** 2) / beta ** 2) * (2 * a + c)
    r = beta * np.sqrt(np.pi / alpha) * erfi((a * np.sqrt(alpha)) / beta)
    return abs(l - r)


def a_m_best_expected_return(cc, price_pivot, c=0.01, a_init=0.08, show_fig=False,
                             figFolder=fig_folder + 'spread/'):
    """
    给定配对组和对数价格，返回entry,exit level,交易时间序列,配对系数
    :param cc:
    :param price_pivot: log_price pivot
    :param c:
    :param a_init:
    :param show_fig:
    :param figFolder: entry,exit,trade_time_list,lmda
    :return:
    """
    createFolder(figFolder)
    cc_price = price_pivot.loc[:, cc]
    price_form, price_trans = split_form_trans(cc_price)

    y = price_form.loc[:, cc[0]]
    X = price_form.loc[:, cc[1]]
    reg = LinearRegression(fit_intercept=False)
    X = X.values.reshape(-1, 1)
    reg.fit(X, y)
    lmda = reg.coef_[0]
    print(f'\t trading\tbuy per {cc[0]}, sell {lmda}\t{cc[1]}')

    spread_form = price_form.loc[:, cc[0]] - lmda * price_form.loc[:, cc[1]]
    spread_trans = price_trans.loc[:, cc[0]] - lmda * price_trans.loc[:, cc[1]]
    # 计算最优a,m
    spread_array = spread_form.values
    ou_params = estimate_OU_params(spread_array)
    r = minimize(expect_return_optimize_function, a_init, args=(ou_params.alpha, ou_params.beta, c))
    a_best = r.x[0]
    m_best = -a_best
    print(f'{cc}\n\ta\t{a_best}\n\tm\t{m_best}')
    # draw plus entry and exit level
    if show_fig:
        plt_split_time((spread_form, spread_trans), f'{figFolder}spread_hline_{cc[0]}-{cc[1]}.png', f'{cc[0]}-{cc[1]}',
                       hline=[a_best, m_best])

    return a_best, m_best, spread_trans, lmda


# %% trading
def touch_target(data: pd.Series, target: float, method: str = 'larger'):
    """
    data中最早比target大(小)的元素的index
    :param data:
    :param target:
    :param method: larger or smaller, default: larger
    :return:
    """
    if method == 'larger':
        for i, v in data.iteritems():
            if v >= target:
                return i
    elif method == 'smaller':
        for i, v in data.iteritems():
            if v <= target:
                return i
    return None


def transaction_time_list(a, m, spread_trans):
    """
    交易时间点列表，[买入,卖出,...]
    :param a: entry level
    :param m: exit level
    :param spread_trans: 交易对数价差序列
    :return: trade_tlist
    """
    if -a < bar_tolerance:
        return None
    trade_time_list = []
    i_start = spread_trans.index[0]
    i_end = spread_trans.index[-1]
    i_len = 0

    while i_start <= i_end:
        if not i_len % 2:
            i = touch_target(spread_trans[i_start:], a, 'smaller')
        else:
            i = touch_target(spread_trans[i_start:], m, 'larger')
        if i:
            trade_time_list.append(i)
            i_len += 1
            i_start = i
        else:
            break
    print('\t trading_time_list', trade_time_list)
    return trade_time_list


def calculate_pair_revenue(cc, origin_price, trade_time_list, lmda):
    """
    计算配对组净收益
    :param cc: 配对组tuple
    :param origin_price: 股票价格pivot
    :param trade_time_list: 交易时间序列,[买入,卖出,...]
    :param lmda:
    :return:
    """
    print(cc, '\t TRADING REVENUE-------------------------------------------------')
    if not trade_time_list:
        print('NO PROPER TRADING OCCURRING')
        return 0, 0
    rev = 0
    price_use = origin_price.loc[trade_time_list, cc]
    price_delta = price_use.iloc[:, 0] - lmda * price_use.iloc[:, 1]
    price_ccost = (price_use.iloc[:, 0] + lmda * price_use.iloc[:, 1]) * c_ratio
    ccost_all = sum(price_ccost)
    t_len = len(trade_time_list)
    if not t_len % 2:
        ccost = ccost_all
        print('已平仓')
    else:
        ccost = sum(price_ccost[:-1])
        print(f'未平仓仓位\t{cc[0]}\t1{cc[1]}\t{-lmda}\tSPREAD\t{price_delta[-1]}')
    for i, delta in enumerate(price_delta.iloc[:t_len - (t_len % 2)]):
        if not i % 2:
            rev -= delta
        else:
            rev += delta
    rev_all = rev - price_delta[-1] if t_len % 2 else rev
    print(f'已平仓实现收益\t{rev - ccost}\t总盈亏\t{rev_all - ccost_all}')
    return rev - ccost, rev_all - ccost_all
