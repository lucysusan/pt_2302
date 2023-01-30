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


# %% functions

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
        fig, axs = plt.subplots(1, 2)
        a = axs[0]
        b = axs[1]
        c_df[c_df.index < trans_start].plot(ax=a)
        c_df[c_df.index >= trans_start].plot(ax=b)
        plt.savefig(fig_folder + c_code[0] + '-' + c_code[1] + '.png')
        plt.show()
