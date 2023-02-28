# -*- coding: utf-8 -*-
"""
Created on 2023/2/16 13:32

@author: Susan
TODO: 1. 把evaluate作图单独拿出来; 2. PairOn方法改进
TODO: 重新open时的amount是否应该变换，o.w.Trading's c seems useless;
TODO: 把收益率重新改下计算方式 - 不涉及成本方法
Q: 手续费，配对组
"""
import configparser
import os

import pandas as pd
import pyfolio as pf

from pt_utils.PairOn import PairOn
from pt_utils.Trading import Trading, TradingFrequency
from pt_utils.function import get_current_date, start_end_period, timer
from pt_utils.get_data_sql import PairTradingData

# project_path = os.getcwd()
project_path = 'D:\\courses\\2022-2024FDU\\COURCES\\Dlab\\intern\\PairTrading\\pt\\'

config_file = project_path + '/input/parameter.ini'

config = configparser.ConfigParser()
config.read(config_file)
freq_dict = {'month': 1, 'quarter': 2, 'year': 3}

start_date = config['general']['start_date']
end_date = config['general']['end_date']
c = config['general'].getfloat('c')

form_end = config['formation']['form_end']
form_freq = TradingFrequency(freq_dict[config['formation']['form_freq']])
form_freq_num = config['formation'].getint('form_freq_num')
pair_bar = config['formation'].getfloat('pair_bar')

trans_start = config['transaction']['trans_start']
trans_freq = TradingFrequency(freq_dict[config['transaction']['trans_freq']])
trans_freq_num = config['transaction'].getint('trans_freq_num')
amount = config['transaction'].getfloat('amount')
index_sid = config['transaction']['index_sid']

index_df = PairTradingData.get_index_data(start_date, end_date)
index_df.index = pd.to_datetime(index_df.index)


def PairTrading_once(form_end=form_end, form_freq=form_freq, form_freq_num=form_freq_num, project_path=project_path,
                     pair_bar=pair_bar, c=c, trans_start=trans_start, trans_freq=trans_freq,
                     trans_freq_num=trans_freq_num,
                     amount=amount, index_df=index_df, index_sid=index_sid, disp: bool = True):
    today = get_current_date()
    trade_out_folder = f"""{project_path}/output/{today}/{form_freq_num}{str(form_freq).split('.')[-1]}_{trans_freq_num}{str(trans_freq).split('.')[-1]}/{form_end}_{trans_start}/"""

    pt_db = PairOn(form_end, form_freq, form_freq_num, out_folder=trade_out_folder + 'formation/', pair_bar=pair_bar)
    pair = pt_db.run_pairOn()
    pair_entry_dict = pt_db.run_opt_pair_entry_level(pair, c)

    trade = Trading(pair_entry_dict, pair_bar, trans_start, trans_freq, trans_freq_num,
                    out_folder=trade_out_folder + 'transaction/')
    if pair_entry_dict:
        rev_df = trade.run(trade_out_folder, amount, c)
        bm_data = trade.evaluation(rev_df, index_df, index_sid, disp)
    else:
        bm_data = pd.DataFrame(columns=['ret', index_sid])

    return bm_data, trade.end_date


@timer
def PairTrading(start_date=start_date, end_date=end_date, form_freq=form_freq, form_freq_num=form_freq_num,
                trans_freq=trans_freq, trans_freq_num=trans_freq_num, pair_bar=pair_bar, c=c, amount=amount,
                index_df=index_df, index_sid=index_sid, project_path=project_path, disp: bool = False):
    _, form_end_start = start_end_period(start_date, form_freq, -form_freq_num)
    trans_start_end, _ = start_end_period(end_date, trans_freq, -trans_freq_num)

    form_end = form_end_start
    trans_start = form_end_start
    bm_df = pd.DataFrame(columns=['ret', index_sid])

    if trans_start > trans_start_end:
        return None

    while trans_start <= trans_start_end:
        print(trans_start, '-------------------------------------------------------')
        bm_data, trans_start = PairTrading_once(form_end, form_freq, form_freq_num, project_path, pair_bar, c,
                                                trans_start,
                                                trans_freq, trans_freq_num, amount, index_df, index_sid, disp)
        bm_df = pd.concat([bm_df, bm_data])
        form_end = trans_start

    today = get_current_date()
    out_folder = f"""{project_path}/output/{today}/{form_freq_num}{str(form_freq).split('.')[-1]}_{trans_freq_num}{str(trans_freq).split('.')[-1]}/"""
    ret_file_name = f"""{start_date}_{end_date}_{form_freq_num}{str(form_freq).split('.')[-1]}_{trans_freq_num}{str(trans_freq).split('.')[-1]}_ret_index"""
    bm_df.to_csv(f"""{out_folder}{ret_file_name}.csv""")
    bm_df.to_pickle(f"""{out_folder}{ret_file_name}""")
    print(out_folder, ret_file_name, ' DONE!')

    pf.create_returns_tear_sheet(bm_df['ret'], benchmark_rets=bm_df[index_sid])

    return bm_df


if __name__ == '__main__':
    bm_data = PairTrading_once()
