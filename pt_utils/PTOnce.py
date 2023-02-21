# -*- coding: utf-8 -*-
"""
Created on 2023/2/16 13:32

@author: Susan
# TODO: Backtest收益计算
"""
from pt_utils.PairOn import PairOn
from pt_utils.Trading import Trading, TradingFrequency
from pt_utils.function import get_current_date
import os

project_path = os.getcwd()
today = get_current_date()

form_end = '2018-07-01'

form_freq = TradingFrequency.month
form_freq_num = -1

trans_start = '2018-07-01'
trans_freq = TradingFrequency.month
trans_freq_num = 1

pair_num = 20
c = 0.0015
pair_bar = 0.02

trade_out_folder = f'{project_path}/result_bar_{today}/{form_end}_{form_freq_num}{form_freq}_{trans_start}_{trans_freq_num}{trans_freq}/'

amount = 1e7
index_sid = '000906.SH'

pt_db = PairOn(form_end, form_freq, form_freq_num, out_folder=trade_out_folder + 'formation/', pair_bar=pair_bar)
pair = pt_db.run_pairOn(pair_num)
pair_entry_dict = pt_db.run_opt_pair_entry_level(pair, c)

# %%
trade = Trading(pair_entry_dict, pair_bar, trans_start, trans_freq, trans_freq_num,
                out_folder=trade_out_folder + 'transaction/')
rev_df = trade.run(trade_out_folder, amount)
# %%
trade.evaluation(rev_df, index_sid)
