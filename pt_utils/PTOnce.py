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

trade_out_folder = f'{project_path}/result_{today}/{form_end}_{form_freq_num}{form_freq}_{trans_start}_{trans_freq_num}{trans_freq}/'

amount = 1e6
zero_ratio = 1e-1
index_sid = '000906.SH'

pt_db = PairOn(form_end, form_freq, form_freq_num, out_folder=trade_out_folder + 'formation/',pair_bar=pair_bar)
pair = pt_db.run_pairOn(pair_num)
pair_entry_dict = pt_db.run_opt_pair_entry_level(pair, c)

# pair_entry_dict = {('000333.SZ', '000858.SZ'): {'lmda': 0.17614018291862346, 'entry': -0.008223697644841155, 'cycle': 2}, ('000333.SZ', '000651.SZ'): {'lmda': 0.033251525608624176, 'entry': -0.008063697035637866, 'cycle': 5}, ('000858.SZ', '600887.SH'): {'lmda': 0.6377654033189237, 'entry': -0.008715263858267597, 'cycle': 2}, ('000858.SZ', '000651.SZ'): {'lmda': 5.296876133296109, 'entry': -0.009093490484146996, 'cycle': 3}, ('000858.SZ', '600519.SH'): {'lmda': 0.24723413617327944, 'entry': -0.007662087213324259, 'cycle': 2}, ('601318.SH', '000002.SZ'): {'lmda': 25.605381767525248, 'entry': -0.010015119996521285, 'cycle': 1}, ('600519.SH', '600887.SH'): {'lmda': 2.5795879494706364, 'entry': -0.006060149670264338, 'cycle': 3}, ('000333.SZ', '600519.SH'): {'lmda': 0.043549997718422925, 'entry': -0.008966813650483678, 'cycle': 4}, ('000333.SZ', '601318.SH'): {'lmda': 1.6609073265426002, 'entry': -0.006666011906279426, 'cycle': 1}, ('600276.SH', '600887.SH'): {'lmda': 1.348316235490017, 'entry': -0.012555428326729623, 'cycle': 2}, ('601318.SH', '000651.SZ'): {'lmda': 49.941156699665946, 'entry': -0.009688331829137115, 'cycle': 2}, ('600519.SH', '000651.SZ'): {'lmda': 1.309744023043101, 'entry': -0.006172099979521714, 'cycle': 3}, ('000651.SZ', '600887.SH'): {'lmda': 3.378630981720803, 'entry': -0.00820535421409073, 'cycle': 4}, ('600276.SH', '600519.SH'): {'lmda': 0.5227446584544228, 'entry': -0.011293077438483567, 'cycle': 2}, ('000651.SZ', '000002.SZ'): {'lmda': 0.5125533807681248, 'entry': -0.015179924374444875, 'cycle': 4}, ('000333.SZ', '000002.SZ'): {'lmda': 15.414380426315162, 'entry': -0.010936284594849903, 'cycle': 1}, ('601318.SH', '000858.SZ'): {'lmda': 9.427213153500372, 'entry': -0.009360423418788764, 'cycle': 1}, ('000333.SZ', '600887.SH'): {'lmda': 0.11234432919539672, 'entry': -0.008623157207210366, 'cycle': 3}, ('600276.SH', '000858.SZ'): {'lmda': 0.4725251166677588, 'entry': -0.014553119634440524, 'cycle': 2}, ('601318.SH', '600519.SH'): {'lmda': 38.128180712394204, 'entry': -0.009305971487260742, 'cycle': 2}}
# bar = 0.04584744010178275

# %%
trade = Trading(pair_entry_dict, pair_bar, trans_start, trans_freq, trans_freq_num,
                out_folder=trade_out_folder + 'transaction/')
rev_df = trade.run(trade_out_folder, amount, zero_ratio)
# %%
trade.evaluation(rev_df, index_sid)
