# -*- coding: utf-8 -*-
"""
Created on 2022/11/30 17:22

@author: Susan
TODO:
 - sw一级行业市值最大的代码，三级 or 二级 同组代码 list
 - APT portfolio


TIPS:
 - [已解决]实际以每日均价成交(vwap - 涉及复权处理): vwap * adj_factor
 - 每次成交金额不超过股票市值的 1/? ，待定-询问
MEMO:
 - check_file_route: output_process
"""
from typing import Tuple
from CommonUse.funcs import createFolder, read_pkl, write_pkl
import pandas as pd
import numpy as np
from pandas import DataFrame
from math import log
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from random import randint
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False


out_process = 'data/output_process/'
fig_folder = 'figs/'
createFolder(fig_folder)
createFolder(out_process)


def data_timeFilter(df_pkl: str, file_name: str = None, start_date: str = '2018-01-01', end_date: str = '2022-11-30',
                    t_cname: str = 'date', out_folder: str = 'use_data/') -> \
        Tuple[DataFrame, DataFrame]:
    df = pd.read_pickle(df_pkl)
    df[t_cname] = df[t_cname].dt.floor('D')
    out_df = df[(df[t_cname] >= start_date) & (df[t_cname] <= end_date)]
    if file_name:
        out_df.to_csv(out_folder + file_name + f'{start_date}_{end_date}.csv', index=None)
    return out_df, pd.DataFrame(out_df.columns, columns=['col'])


def pre_read(file_list: list, in_folder: str = 'data/', start_date: str = '2018-01-01', end_date: str = '2022-11-30',
             t_cname: str = 'date', out_folder: str = 'use_data/'):
    for f in file_list:
        df, col = data_timeFilter(in_folder + f + '.csv', f, start_date=start_date, end_date=end_date, t_cname=t_cname,
                                  out_folder=out_folder)


data_folder = 'use_data/'
createFolder(data_folder)
file_list = ['quote', 'mkt_cap', 'industry_sw']
# pre_read(file_list)
# quote, qcol = data_timeFilter('data/quote')
# mkt_cap, mcol = data_timeFilter('data/mkt_cap')
# industry_sw, swcol = data_timeFilter('data/industry_sw')
# # DONE: 输出行业分布表
# time_new = max(industry_sw.date)
# sw_new = industry_sw.loc[industry_sw.date == time_new, :].drop_duplicates(subset=['sid'])
# sw_new_vc = sw_new.groupby(['sw_1', 'sw_2', 'sw_3'], as_index=False)['sid'].count().rename(columns={'sid': 'num'})
# # sw_new_vc.sort_values('num', ascending=False).to_csv(out_process + f'sw_vc_{str(time_new)[:10]}.csv', index=False)

# %%
quote = pd.read_csv(data_folder + file_list[0] + f'{start_date}_{end_date}.csv')
mkt_cap = pd.read_csv(data_folder + file_list[1] + f'{start_date}_{end_date}.csv')
industry_sw = pd.read_csv(data_folder + file_list[2] + f'{start_date}_{end_date}.csv')

quote['vwap_aft'] = quote['vwap'] * quote['adj_factor']
quote_pivot = quote.pivot(index='date', columns='sid', values='vwap_aft')
ret_pivot = quote_pivot.applymap(lambda x: log(x)).diff().iloc[1:]
mkt_pivot = mkt_cap.pivot(index='date', columns='sid', values='float')
sw_pivot = industry_sw.pivot(index='date', columns='sid', values=['sw_1', 'sw_2', 'sw_3'])
sw1_vc = sw_pivot['sw_1'].iloc[0, :].value_counts()
# print(quote_pivot.head())
# print(mkt_pivot.head())
# print(sw_pivot.head())
# print(sw1_vc)
# %% 数据探索
# TODO: 行业输出
#  - LEVEL, NAME, TOTALNUM,
#  - 计算相关性，corr
#  对每一支股票给出其余股票对其的相关性list，按从大到小排序，
#  计算前五的平均相关性，挑出平均相关性最高的那支股票及这五只股票，plt
#  输出 股票名称 这五只股票的名称， 相关性list， 后二者顺序匹配
# PARAMS:
# 1. 使用过去几个月的时间作为形成期时间 form_num = 12month
# 2. 多少个月作为交易月进行检测  trans_num = 3month

level = 'sw_3'
level_list = ['sw_1', 'sw_2', 'sw_3']
print(level, end='\t')
# TODO: 从行业中筛选股票交集
sw_use = industry_sw[(industry_sw['date'] >= form_start) & (industry_sw['date'] <= trans_end)]
sw_use_pivot = sw_use.pivot(index='date', columns='sid', values=level_list)
sw_use_level = sw_use_pivot[level].dropna(axis=1)
sw_form = sw_use_level[(sw_use_level.index >= form_start) & (sw_use_level.index < form_end)].mode().T
sw_form_vc = sw_form.value_counts(ascending=False)
sw_form_vc.name = 'formation'
sw_trans = sw_use_level[(sw_use_level.index > trans_start) & (sw_use_level.index <= trans_end)].mode().T
sw_trans_vc = sw_trans.value_counts(ascending=False)
sw_trans_vc.name = 'transaction'
sw_vc = pd.merge(sw_form_vc, sw_trans_vc, 'inner', left_index=True, right_index=True)
sw_uni = list(set(sw_form.iloc[:, 0]).intersection(sw_trans.iloc[:, 0]))
nlargest = 5
for industry in sw_uni:
    # industry = sw_uni[0]
    print(industry, end='\t')
    # sk_list 为trans和form的交集
    sk_form_list = sw_form[(sw_form == industry).values].index
    sk_trans_list = sw_trans[(sw_trans == industry).values].index
    sk_list = list(set(sk_form_list).intersection(sk_trans_list))
    sk_list = list(set(sk_list).intersection(ret_pivot.columns))
    print(len(sk_list), end='\t')
    if sk_list:
        ret_use = ret_pivot.loc[(form_start <= ret_pivot.index) & (ret_pivot.index < form_end), sk_list]
        ret_corr = ret_use.corr()
        for sk in sk_list:
            vi_exp = ret_corr.nlargest(min(nlargest + 1, len(sk_list)), sk).index
            vi_list = list(set(ret_corr.index) - set(vi_exp))
            ret_corr.loc[vi_list, sk] = np.nan
        ret_mean = ret_corr.mean()
        ret_nlarge = ret_mean.nlargest(1).index[0]
        ret_portfolio_corr = ret_corr.loc[~(ret_corr[ret_nlarge].isna()), ret_nlarge]
        ret_list = ret_portfolio_corr.index
        mkt_use = mkt_pivot[mkt_pivot.index < form_end].iloc[-1][ret_list]
        mkt_use.name = 'mkt_cap'
        industry_df = pd.merge(ret_portfolio_corr, mkt_use, 'inner', left_index=True, right_index=True)
        sk_list = industry_df.index.to_list()
        ret_sk = ret_use.loc[:, sk_list]
        ret_sk.plot()
        plt.legend()
        title = industry + '-' + industry_df.columns[0]
        plt.title(title)
        plt.savefig(fig_folder + title + '.png')
        plt.show()
        print(industry_df)
    else:
        print()
