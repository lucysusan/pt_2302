# -*- coding: utf-8 -*-
"""
Created on 2023/1/30 18:26

@author: Susan
"""
import datetime
import warnings

import matplotlib.pyplot as plt
from CommonUse.funcs import read_pkl, createFolder
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import dataclasses
import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from scipy.special import erfi
from scipy.optimize import minimize

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
form_start = date_Opt_year(form_end, 1)
trans_start = form_end
trans_end = date_Opt_month(trans_start, -6)
print(
    f'formation start\t{start_date}\tformation end\t{form_end}\ntransaction start\t{trans_start}\ttransaction end\t{end_date}')

in_folder = 'use_data/'
out_folder = 'data/output_process/'
fig_folder = out_folder + 'fig/'
createFolder(fig_folder)
stock_pool = read_pkl(out_folder + 'stock_pool.pkl')


# %% functions

def plt_split_time(data, split_index, outer_fig, name=None, hline=None):
    """
    分隔index画图
    :param hline: 是否需要添加横线，默认None不需要
    :param data: 待画图数据
    :param split_index: 分隔时间，将与data index比较
    :param outer_fig: fig地址
    :param name: fig title
    :return:
    """
    fig = plt.figure(figsize=(36, 18))
    a = fig.add_subplot(1, 2, 1)
    plt.title('formation')
    b = fig.add_subplot(1, 2, 2)
    plt.title('transaction')
    data[data.index < split_index].plot(ax=a)
    data[data.index >= split_index].plot(ax=b)
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


def plt_log_price_pair_split_trans(cc_top_list, trans_start: str = trans_start,
                                   log_price_pivot_route: str = out_folder + 'log_vwap_aft_price_pivot.pkl',
                                   fig_folder: str = fig_folder):
    """
    画出给定match组的股票对数价格的走势图，以trans_start分隔
    :param trans_start: 策略中交易开始的日期，例 '2022-01-01'
    :param cc_top_list: list中每个元素为tuple,类似 [('002040.SZ', '600190.SH'), ('603233.SH', '603883.SH')]
    :param log_price_pivot_route: log_price_pivot文件地址
    :param fig_folder:
    :return:
    """
    log_price_pivot = read_pkl(log_price_pivot_route)
    for c_code in tqdm(cc_top_list):
        c_df = log_price_pivot.loc[:, c_code]
        plt_split_time(c_df, trans_start, fig_folder + c_code[0] + '-' + c_code[1] + '.png')
        # fig, axs = plt.subplots(1, 2)
        # a = axs[0]
        # b = axs[1]
        # c_df[c_df.index < trans_start].plot(ax=a)
        # c_df[c_df.index >= trans_start].plot(ax=b)
        # plt.savefig(fig_folder + c_code[0] + '-' + c_code[1] + '.png')
        # plt.show()


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
                             figFolder=out_folder + 'fig/spread/'):
    createFolder(figFolder)
    cc_price = price_pivot.loc[:, cc]
    lmda = sum(cc_price.iloc[:, 0]) / sum(cc_price.iloc[:, 1])
    spread = cc_price.iloc[:, 0] - lmda * cc_price.iloc[:, 1]
    spread_form = spread.loc[:trans_start]
    spread_trans = spread.loc[trans_start:]
    # 计算最优a,m
    spread_array = spread_form.values
    ou_params = estimate_OU_params(spread_array)
    r = minimize(expect_return_optimize_function, a_init, args=(ou_params.alpha, ou_params.beta, c))
    a_best = r.x[0]
    m_best = -a_best
    print(f'{cc}\n\ta\t{a_best}\n\tm\t{m_best}')
    # draw plus entry and exit level
    if show_fig:
        plt_split_time(spread, trans_start, f'{figFolder}spread_hline_{cc[0]}-{cc[1]}.png', f'{cc[0]}-{cc[1]}',
                       hline=[a_best, m_best])

    return a_best, m_best, spread_trans, lmda
