# -*- coding: utf-8 -*-
"""
Created on 2023/1/31 14:14

@author: Susan
"""
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
from CommonUse.funcs import read_pkl

from pt_utils.functions import a_m_best_expected_return, transaction_time_list, calculate_pair_revenue, \
    plt_log_price_pair
from pt_utils.functions import out_folder, form_start, form_end, trans_start, trans_end

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['FangSong']

in_folder = 'data/output_process/'
cc_top_list = read_pkl(in_folder + 'cc_top10_list.pkl')
price_pivot = read_pkl(in_folder + 'log_vwap_aft_price_pivot.pkl')
origin_price = read_pkl(in_folder + 'vwap_aft_price_pivot.pkl')

col = ['stock_0', 'stock_1', '配对系数', '已平仓实现收益', '总盈亏', 'entry_level', 'exit_level', 'trading_tlist']
res_df = pd.DataFrame(columns=col)
nrev, nrev_all = 0, 0
plt_log_price_pair(cc_top_list, price_pivot)
for cc in cc_top_list:
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
pprint(res_df)
print(f'所有挑选出的配对组===================================================================\n已平仓实现收益\t{nrev}\t总盈亏\t{nrev_all}')
