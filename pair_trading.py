# -*- coding: utf-8 -*-
"""
Created on 2023/1/31 14:14

@author: Susan
TODO:
 0. 风险限额
 2. 截面or均值数据的选取怎么思考, 距离向量为什么可以看成类似方差, 其阈值的选取
 3. 有了净值曲线后各种指标的计算，有没有现成的代码
"""
import pandas as pd

from pt_utils.PairTrading import PairTrading

import matplotlib.pyplot as plt

trans_start = '2019-01-01'
end_date = '2022-01-01'
out_folder = 'rev_result_0207/'

col = ['cash', 'value', 'rev']
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
flow_table.to_csv(out_folder + '收益率交易收益汇总.csv', index=False)
plt.figure(figsize=(20, 10))
flow_table['rev'].plot()
plt.savefig(out_folder + f'收益率_{trans_start}_{end_date}.png')
plt.show()
