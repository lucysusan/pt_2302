# -*- coding: utf-8 -*-
"""
Created on 2023/1/18 16:02

@author: Susan
TODO:
 - 查看因子数据状态
"""
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from CommonUse.funcs import read_pkl, write_pkl, createFolder
from pprint import pprint
from numpy.linalg.linalg import norm
import itertools
from tqdm import tqdm
from math import log

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['FangSong_GB2312']


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
    f'formation start\t{form_start}\tformation end\t{form_end}\ntransaction start\t{trans_start}\ttransaction end\t{trans_end}')

in_folder = 'use_data/'
out_folder = 'data/output_process/'
fig_folder = out_folder + 'fig/'
createFolder(fig_folder)
stock_pool = read_pkl(out_folder + 'stock_pool.pkl')
# %% cne_exp_use info
# cne_exposure = read_pkl(in_folder + 'cne6_exposure.pkl')
# cne_exposure.reset_index(inplace=True)

cne_exposure = read_pkl(out_folder + 'cne_exposure_use.pkl')
cne_exp_use = cne_exposure[
    (cne_exposure['date'] >= form_start) & (cne_exposure['date'] <= form_end) & (cne_exposure['sid'].isin(stock_pool))]
exp_head = cne_exp_use.head()
# date,sid,country,industry
len_date = cne_exp_use['date'].nunique()  # 1193
len_sid = cne_exp_use['sid'].nunique()  # 4976
len_country = cne_exp_use['country'].nunique()  # 1
len_industry = cne_exp_use['industry'].nunique()  # 32
industry_list = cne_exp_use['industry'].unique().tolist()
pprint(industry_list)
write_pkl(out_folder + 'cne_exposure_use.pkl', cne_exp_use)
# # %% [NO USE]cne_specific_use info
# cne_specific = read_pkl(in_folder + 'cne6l_specific_risk.pkl')
#
# cne_specific_use = cne_specific[(cne_specific['date'] >= start_date) & (cne_specific['date'] <= end_date)]
# specific_head = cne_specific_use.head(10)
# write_pkl(out_folder + 'cne_specific_use.pkl', cne_specific_use)
# cne_specific.info()
# %% 两张表结合
# cne_exposure = read_pkl(out_folder + 'cne_exposure_use.pkl')
# cne_exposure.set_index(['date', 'sid'], inplace=True)
# cne_specific = read_pkl(out_folder + 'cne_specific_use.pkl')
# cne_specific.set_index(['date', 'sid'], inplace=True)
# print(len(cne_exposure), '\t', len(cne_specific))
# cne_factor = pd.merge(cne_exposure, cne_specific, 'outer', left_index=True, right_index=True)
# print(len(cne_factor))
cne_factor = cne_exp_use
factor_head = cne_factor.head(10)
factor_list = ['beta', 'btop', 'divyild', 'earnqlty', 'earnvar', 'earnyild',
               'growth', 'invsqlty', 'leverage', 'liquidty', 'ltrevrsl', 'midcap',
               'momentum', 'profit', 'resvol', 'size']
ind_list = cne_factor['industry'].unique().tolist()
cne_factor.reset_index(inplace=True)
ind_group = cne_factor.groupby('industry')
topn = 5
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
    # topn_ind_df = ind_df.nlargest(topn, 'norm')
    # SECOND WAY: SELECT NORM > 10
    topn_ind_df = ind_df[ind_df['norm'] > 10]
    topn_ind_df['industry'] = ind
    topn_df = pd.concat([topn_df, topn_ind_df])

topn_df.sort_values('norm', ascending=False, inplace=True)
topn_df.to_excel(out_folder + 'industry_norm_l10.xlsx')
write_pkl(out_folder + 'industry_norm_l10.pkl', topn_df)

topn_df_largest = topn_df.nlargest(10, 'norm')
cc_top_list = topn_df_largest.index.to_list()
pprint(cc_top_list)
write_pkl(out_folder + 'cne_factor.pkl', cne_factor)
# %%
# 筛选股票池 上市满三个月，非ST，无停牌复牌行为
stock_status = read_pkl(in_folder + 'ashare_universe_status.pkl')
stock_status = stock_status[(stock_status['date'] >= start_date) & (stock_status['date'] <= end_date)]
write_pkl(out_folder + 'stock_status.pkl', stock_status)
stock_status.set_index(['date', 'sid'], inplace=True)
stock_status['normal'] = stock_status.apply(lambda x: sum(x) > 0, axis=1)

stock_status.reset_index(inplace=True)
status_abnormal = set(stock_status[stock_status['normal']]['sid'].unique().tolist())
status_all = set(stock_status['sid'].unique().tolist())
stock_pool = list(status_all - status_abnormal)

write_pkl(out_folder + 'stock_pool.pkl', stock_pool)
# %% TODO: for all 两只股票，给出股票形成期对数价格的画图
# 给出股票对数价格
quote = pd.read_csv(in_folder + 'quote' + f'{start_date}_{end_date}.csv')
quote_use = quote[quote['sid'].isin(stock_pool)]
quote_use['vwap_aft'] = quote_use['vwap'] * quote_use['adj_factor']
quote_use['log_vwap_aft'] = quote_use['vwap_aft'].apply(lambda x: log(x))
quote_head = quote_use.head(100)
quote_use_pivot = quote_use.pivot(index='date', columns='sid', values='log_vwap_aft')
write_pkl(out_folder + 'log_vwap_aft_price_pivot.pkl', quote_use_pivot)


for c_code in tqdm(cc_top_list):
    c_df = quote_use_pivot.loc[:, c_code]
    fig, axs = plt.subplots(1, 2)
    a = axs[0]
    b = axs[1]
    c_df[c_df.index < trans_start].plot(ax=a)
    c_df[c_df.index >= trans_start].plot(ax=b)
    plt.savefig(fig_folder + c_code[0] + '-' + c_code[1] + '.png')
    plt.show()


write_pkl(out_folder + 'cc_top10_list.pkl', cc_top_list)
