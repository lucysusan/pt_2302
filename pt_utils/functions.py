# -*- coding: utf-8 -*-
"""
Created on 2023/1/30 18:26

@author: Susan
"""
import datetime
import warnings
from dataclasses import dataclass
from numpy.linalg.linalg import norm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CommonUse.funcs import read_pkl, createFolder
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from scipy.special import erfi
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import itertools

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
form_end = '2021-07-01'
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
bar_tolerance = 1e-4

c = 0.01
c_ratio = 1e-4


def split_form_trans(data, form_start=form_start, form_end=form_end, trans_start=trans_start, trans_end=trans_end):
    """
    function separately deals with data splitting
    :param data:
    :param form_start:
    :param form_end:
    :param trans_start:
    :param trans_end:
    :return: formation_data, transaction_data
    """
    data_form = data[(data.index >= form_start) & (data.index < form_end)]
    data_trans = data[(data.index >= trans_start) & (data.index <= trans_end)]
    return data_form, data_trans


# %% 筛选股票池 formation和transaction阶段都非新股，非ST，无停牌复牌行为
def _partial_stock_pool(stock_status) -> set:
    """
    get_stock_pool的辅助函数
    :param stock_status:
    :return:
    """
    stock_status.set_index(['date', 'sid'], inplace=True)
    stock_status['normal'] = stock_status.apply(lambda x: sum(x) > 0, axis=1)
    stock_status.reset_index(inplace=True)
    status_abnormal = set(stock_status[stock_status['normal']]['sid'].unique().tolist())
    status_all = set(stock_status['sid'].unique().tolist())
    return status_all - status_abnormal


def get_stock_pool(stock_status_route: str = 'use_data/ashare_universe_status.pkl') -> list:
    """
    股票池，在formation和transaction阶段都没有异常行为(新上市、退市、停复牌)的股票
    :param stock_status_route: ashare_universe_status.pkl所在路径
    :return: 满足条件的股票池list
    """
    stock_status = read_pkl(stock_status_route)
    status_form, status_trans = split_form_trans(stock_status.set_index('date'))

    stock_form = _partial_stock_pool(status_form.reset_index())
    stock_trans = _partial_stock_pool(status_trans.reset_index())
    return list(stock_trans.intersection(stock_form))


def get_price_log_origin(stock_pool, form_start=form_start, trans_end=trans_end):
    """
    日期和股票池筛选过的log_price_pivot和origin_price_pivot
    :param stock_pool: 股票池
    :param form_start: 形成期开始日期
    :param trans_end: 交易期结束日期
    :return: log_price_pivot, origin_price_pivot
    """
    price_pivot_log_origin = read_pkl('data/output_process/price_pivot_log_origin.pkl')
    price_pivot, origin_price = price_pivot_log_origin['log_vwap_aft'], price_pivot_log_origin['vwap_aft']
    price_pivot = price_pivot[(price_pivot.index >= form_start) & (price_pivot.index <= trans_end)][stock_pool]
    origin_price = origin_price[(origin_price.index >= form_start) & (origin_price.index <= trans_end)][stock_pool]
    return price_pivot, origin_price


def get_cc_top_list(stock_pool: list, norm_bar: float = 10, pair_num: int = 10, form_start=form_start,
                    form_end=form_end) -> list:
    cne_exposure = read_pkl('use_data/' + 'cne6_exposure.pkl')

    cne_factor = cne_exposure[
        (cne_exposure['date'] >= form_start) & (cne_exposure['date'] <= form_end) & (
            cne_exposure['sid'].isin(stock_pool))]

    factor_list = ['beta', 'btop', 'divyild', 'earnqlty', 'earnvar', 'earnyild',
                   'growth', 'invsqlty', 'leverage', 'liquidty', 'ltrevrsl', 'midcap',
                   'momentum', 'profit', 'resvol', 'size']
    ind_list = cne_factor['industry'].unique().tolist()
    cne_factor.reset_index(inplace=True)
    ind_group = cne_factor.groupby('industry')
    topn_df = pd.DataFrame()
    for ind in tqdm(ind_list):
        ind_data = ind_group.get_group(ind)
        security_list = ind_data['sid'].unique().tolist()
        cc = list(itertools.combinations(security_list, 2))
        ind_df = pd.DataFrame(index=cc)
        for val in factor_list:
            ind_pivot = ind_data.pivot(index='date', columns='sid', values=val)
            ind_val_corr = abs(ind_pivot.corr(method='pearson'))
            val_list = [ind_val_corr.loc[c] for c in cc]
            ind_df[val] = val_list

        ind_df['norm'] = ind_df.apply(lambda x: norm(x, ord=1), axis=1)
        topn_ind_df = ind_df[ind_df['norm'] > norm_bar]
        topn_ind_df['industry'] = ind
        topn_df = pd.concat([topn_df, topn_ind_df])

    topn_df.sort_values('norm', ascending=False, inplace=True)
    topn_df.to_excel(out_folder + f'industry_norm_l{norm_bar}.xlsx')
    topn_df_largest = topn_df.nlargest(pair_num, 'norm')
    cc_top_list = topn_df_largest.index.to_list()
    return cc_top_list


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
    # plt.show()


def plt_log_price_pair(cc_top_list, price_pivot, figFolder: str = fig_folder + 'log_price/'):
    """
    画出给定match组的股票对数价格的走势图
    :param cc_top_list: list中每个元素为tuple,类似 [('002040.SZ', '600190.SH'), ('603233.SH', '603883.SH')]
    :param price_pivot: log_price_pivot
    :param figFolder:
    :return:
    """
    createFolder(figFolder)
    for c_code in cc_top_list:
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


def a_m_best_expected_return(cc, price_pivot, c=0.01, a_init=0.08, show_fig=False, verbose=False,
                             figFolder=fig_folder + 'spread/'):
    """
    给定配对组和对数价格，返回entry,exit level,交易时间序列,配对系数
    :param cc:
    :param price_pivot: log_price pivot
    :param c:
    :param a_init:
    :param show_fig:
    :param verbose: 是否print中间值, default False
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

    spread_form = price_form.loc[:, cc[0]] - lmda * price_form.loc[:, cc[1]]
    spread_trans = price_trans.loc[:, cc[0]] - lmda * price_trans.loc[:, cc[1]]
    # 计算最优a,m
    spread_array = spread_form.values
    ou_params = estimate_OU_params(spread_array)
    r = minimize(expect_return_optimize_function, a_init, args=(ou_params.alpha, ou_params.beta, c))
    a_best = r.x[0]
    m_best = -a_best
    if verbose:
        print(f'\t trading\tbuy per {cc[0]}, sell {lmda}\t{cc[1]}')
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


def transaction_time_list(a, m, spread_trans, verbose=False):
    """
    交易时间点列表，[买入,卖出,...]
    :param a: entry level
    :param m: exit level
    :param spread_trans: 交易对数价差序列
    :param verbose: 是否显示中间值
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
    print('\t trading_time_list', trade_time_list) if verbose else None
    return trade_time_list


def calculate_pair_revenue(cc, origin_price, trade_time_list, lmda, verbose=False):
    """
    计算配对组净收益
    :param cc: 配对组tuple
    :param origin_price: 股票价格pivot
    :param trade_time_list: 交易时间序列,[买入,卖出,...]
    :param lmda:
    :param verbose:
    :return:
    """
    print(cc, '\t TRADING REVENUE-------------------------------------------------') if verbose else None
    if not trade_time_list:
        print('NO PROPER TRADING OCCURRING') if verbose else None
        return 0, 0
    rev = 0
    price_use = origin_price.loc[trade_time_list, cc]
    price_delta = price_use.iloc[:, 0] - lmda * price_use.iloc[:, 1]
    price_ccost = (price_use.iloc[:, 0] + lmda * price_use.iloc[:, 1]) * c_ratio
    ccost_all = sum(price_ccost)
    t_len = len(trade_time_list)
    if not t_len % 2:
        ccost = ccost_all
        print('已平仓') if verbose else None
    else:
        ccost = sum(price_ccost[:-1])
        print(f'未平仓仓位\t{cc[0]}\t1{cc[1]}\t{-lmda}\tSPREAD\t{price_delta[-1]}') if verbose else None
    for i, delta in enumerate(price_delta.iloc[:t_len - (t_len % 2)]):
        if not i % 2:
            rev -= delta
        else:
            rev += delta
    rev_all = rev - price_delta[-1] if t_len % 2 else rev
    print(f'已平仓实现收益\t{rev - ccost}\t总盈亏\t{rev_all - ccost_all}') if verbose else None
    return rev - ccost, rev_all - ccost_all


# %%
def run_once():
    stock_pool = get_stock_pool()
    price_pivot, origin_price = get_price_log_origin(stock_pool)
    cc_top_list = get_cc_top_list(stock_pool)

    col = ['stock_0', 'stock_1', '配对系数', '已平仓实现收益', '总盈亏', 'entry_level', 'exit_level', 'trading_tlist']
    res_df = pd.DataFrame(columns=col)
    nrev, nrev_all = 0, 0
    plt_log_price_pair(cc_top_list, price_pivot)
    for cc in tqdm(cc_top_list):
        a, m, spread_trans, lmda = a_m_best_expected_return(cc, price_pivot, show_fig=True)
        t_list = transaction_time_list(a, m, spread_trans)
        c_nrev, c_nreva = calculate_pair_revenue(cc, origin_price, t_list, lmda)
        res_r = [cc[0], cc[1], lmda, c_nrev, c_nreva, a, m, t_list]
        res_dict = dict(zip(col, res_r))
        res_df = res_df.append(res_dict, ignore_index=True)
        nrev += c_nrev
        nrev_all += c_nreva

    res_r = ['formation', form_start, form_end, nrev, nrev_all, 'transaction', trans_start, trans_end]
    res_dict = dict(zip(col, res_r))
    res_df = res_df.append(res_dict, ignore_index=True)
    res_df.to_csv(out_folder + '配对组交易结果.csv', index=False)
    print(res_df)
    print(
        f'所有挑选出的配对组===================================================================\n已平仓实现收益\t{nrev}\t总盈亏\t{nrev_all}')


print('RESULTS saved at ' + out_folder)
