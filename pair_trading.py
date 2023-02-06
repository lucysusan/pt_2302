# -*- coding: utf-8 -*-
"""
Created on 2023/1/31 14:14

@author: Susan
"""
import pandas as pd

from pt_utils.PairTrading import PairTrading

import matplotlib.pyplot as plt

trans_start = '2019-01-01'
end_date = '2020-01-01'
out_folder = 'value_result_0206/'

col = ['cash', 'value']
flow_table = pd.DataFrame(columns=col)

pair_num = 20
invest_amount = 2e6
invest_num = invest_amount/pair_num

while trans_start <= end_date:
    pt = PairTrading(trans_start, trans_start, out_folder, c=0.0015, c_ratio=0.0015, pair_num=pair_num, norm_bar=0.007)
    flow_df = pt.run(invest_num)
    flow_table = pd.concat([flow_table, flow_df[col]])
    trans_start = pt.trans_end
PairTrading.clear_object_data()
flow_table['per_value'] = flow_table['value']/invest_amount
flow_table.to_csv(out_folder + '净值交易收益汇总.csv', index=False)
flow_table['per_value'].plot()
plt.show()
