# -*- coding: utf-8 -*-
"""
Created on 2023/1/31 14:14

@author: Susan
TODO: 1. 截面or均值数据的选取怎么思考 2. L2阈值选取 3. 询问净值出现这张情况的原因，感觉是cash加了两遍，可能还是成本的问题 4. 有了净值曲线后各种指标的计算，有没有现成的代码
"""
import pandas as pd

from pt_utils.PairTrading import PairTrading

import matplotlib.pyplot as plt

trans_start = '2019-01-01'
end_date = '2022-01-01'
out_folder = 'value_result_0207/'

col = ['cash', 'value']
flow_table = pd.DataFrame(columns=col)

pair_num = 20
invest_amount = 2e6
invest_num = invest_amount / pair_num

while trans_start <= end_date:
    pt = PairTrading(trans_start, trans_start, out_folder, c=0.0015, c_ratio=0.0015, pair_num=pair_num, norm_bar=0.007)
    flow_df = pt.run(invest_num)
    flow_table = pd.concat([flow_table, flow_df[col]])
    trans_start = pt.trans_end
PairTrading.clear_object_data()
flow_table['per_value'] = flow_table['value'] / invest_amount
flow_table.to_csv(out_folder + '净值交易收益汇总.csv', index=False)
plt.figure(figsize=(20, 10))
flow_table['per_value'].plot()
plt.savefig(out_folder + f'净值_{trans_start}_{end_date}.png')
plt.show()
