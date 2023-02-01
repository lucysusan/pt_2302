# -*- coding: utf-8 -*-
"""
Created on 2023/1/31 14:14

@author: Susan
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from scipy.special import erfi
# from math import exp, sqrt
import matplotlib.pyplot as plt
import warnings
from CommonUse.funcs import read_pkl, write_pkl, createFolder
from pt_utils.functions import out_folder, start_date, end_date, form_start, form_end, trans_start, trans_end, \
    plt_split_time, a_m_best_expected_return
from sympy import *
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['FangSong']
round_num = 10

#
cc_top_list = read_pkl(out_folder + 'cc_top10_list.pkl')
price_pivot = read_pkl(out_folder + 'log_vwap_aft_price_pivot.pkl')
origin_price = read_pkl(out_folder + 'vwap_aft_price_pivot.pkl')
bar_tolerate = 1e-4

c = 0.01
c_ratio = 1e-4

# a_init = 0.08
# cc = cc_top_list[0]


# fig_folder=out_folder + 'fig/spread/'

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


def transaction_time_list(cc, price_pivot):
    """
    cc配对组交易时间点列表，[买入,卖出,...]
    :param cc: 配对组tuple
    :param price_pivot: 股票对数价格pivot
    :return:
    """
    a, m, spread_trans, lmda = a_m_best_expected_return(cc, price_pivot, c=c)
    if abs(a) < bar_tolerate:
        return None, lmda
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
    print(f'\t trading\tbuy per {cc[0]}, sell {lmda} {cc[1]}')
    print('\t trading_time_list', trade_time_list)
    return trade_time_list, lmda


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
    print(f'已平仓实现收益\t{rev-ccost}\t总盈亏\t{rev_all-ccost_all}')
    return rev-ccost, rev_all-ccost_all


nrev, nrev_all = 0, 0
for cc in cc_top_list:
    t_list, lmda = transaction_time_list(cc, price_pivot)
    c_nrev, c_nreva = calculate_pair_revenue(cc, origin_price, t_list, lmda)
    nrev += c_nrev
    nrev_all += c_nreva
print(f'所有挑选出的配对组=============================================================\n已平仓实现收益\t{nrev}\t总盈亏\t{nrev_all}')
# # %%
# # TODO: 基于formation，计算每个配对的a,m。分form,trans画出spread图 - 【为使均值为0，交易股份数volume_b =
# #  lmda为小数】【交易手续费，price_a,lmda * price_b为c=0.01】 ！考虑这里的成本表达   得到trans每个经过a,m的时间对，计算收益情况
#
#
# col = ['code_0', 'code_1', 'a', 'm']
# trade_a_m = pd.DataFrame(columns=col)
# for cc in cc_top_list:
#     a, m, spread_trans = a_m_best_expected_return(cc, price_pivot)
#     dict_r = [cc[0], cc[1], a, m]
#     dict_tmp = dict(zip(col, dict_r))
#     trade_a_m.append(dict_tmp, ignore_index=True)
#
# trade_a_m.to_csv(out_folder + 'cc_top10_a_m.csv', index=False)
# # %% check correctness
# cc = ('600269.SH', '600717.SH')
# trade_time_list = ['2022-01-04', '2022-03-23', '2022-04-01', '2022-07-12', '2022-09-22']
# lmda = 0.7744789379344803
# price_use = origin_price.loc[trade_time_list, cc]
# price_use[f'{cc[0]}_log'] = price_use[cc[0]].apply(lambda x: np.log(x))
# price_use[f'{cc[1]}_log'] = price_use[cc[1]].apply(lambda x: np.log(x))
# price_use[f'spread_log'] = price_use[f'{cc[0]}_log'] - lmda * price_use[f'{cc[1]}_log']
#
# price_use['spread'] = price_use.iloc[:, 0] - lmda * price_use.iloc[:, 1]
#
# # spread = price_use['spread']
# # for i, v in enumerate(spread[:len(spread)]):
# #     print(i, '\t', v)
