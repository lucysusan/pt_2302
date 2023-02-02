# -*- coding: utf-8 -*-
"""
Created on 2023/1/31 14:14

@author: Susan
"""
import pandas as pd

from pt_utils.PairTrading import PairTrading

trans_start = '2019-01-01'
end_date = '2022-01-01'
n_rev, n_reva = 0, 0
col = ['stock_0', 'stock_1', '配对系数', '已平仓实现收益', '总盈亏', 'entry_level', 'exit_level', 'trading_tlist']
nres_df = pd.DataFrame(
    columns=col)
while trans_start <= end_date:
    pt = PairTrading(trans_start, trans_start, 'pt_result/', c=0.0015, c_ratio=0.0015)
    res_df, rev, reva = pt.run()
    nres_df = pd.concat([nres_df, res_df], ignore_index=True)
    n_rev += rev
    n_reva += reva
    trans_start = pt.trans_end
res_r = [None, None, None, n_rev, n_reva, None, None, None]
res_dict = dict(zip(col, res_r))
nres_df = nres_df.append(res_dict, ignore_index=True)
print(nres_df)
