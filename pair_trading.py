# -*- coding: utf-8 -*-
"""
Created on 2023/1/31 14:14

@author: Susan
TODO:
 0. 风险限额
"""
import matplotlib.pyplot as plt
import pandas as pd
from CommonUse.funcs import read_pkl

from pt_utils.PairTrading import PairTrading

trans_start = '2020-01-01'
end_date = '2020-01-01'
out_folder = 'rev_result_2Y_11M/'

col = ['cash', 'value', 'rev']
flow_table = pd.DataFrame(columns=col)

pair_num = 20
invest_amount = 2e6
invest_num = invest_amount / pair_num
index_rev = read_pkl('raw/000906_ret.pkl')
# %% 策略执行
while trans_start <= end_date:
    pt = PairTrading(trans_start, trans_start, out_folder, form_y=2, trans_m=11, c=0.0015, c_ratio=0.0015,
                     pair_num=pair_num, norm_bar=0.007)
    flow_df = pt.run(invest_num)
    # pt.evaluation(flow_df, index_rev)
    try:
        pt.evaluation(flow_df, index_rev)
    except:
        print(f'ERR:TRANSACTION - {pt.trans_start} TO {pt.trans_end} evaluation NaT')
    plt.show()
    flow_table = pd.concat([flow_table, flow_df[col]])
    trans_start = pt.trans_end
PairTrading.clear_object_data()
flow_table.index.name = 'date'
flow_table.reset_index().to_csv(out_folder + '收益率交易收益汇总.csv', index=False)
plt.figure(figsize=(20, 10))
flow_table['rev'].plot()
plt.savefig(out_folder + f'收益率_{trans_start}_{end_date}.png')
plt.show()

# PairTrading.evaluation(flow_table, index_rev)
# %% 业绩评价
# rev_data = pd.read_csv(out_folder + '收益率交易收益汇总.csv', index_col='date')
# rev_data.index = pd.to_datetime(rev_data.index)
# rev = rev_data['rev']
# index_rev.name = '000906.SH'
# index_frame = index_rev.to_frame()
# index_data_df = pd.merge(rev, index_frame, 'left', left_index=True, right_index=True)
# # pf.create_returns_tear_sheet(rev)
# pf.create_full_tear_sheet(rev, benchmark_rets=index_data_df['000906.SH'])
