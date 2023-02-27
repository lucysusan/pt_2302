# -*- coding: utf-8 -*-
"""
Created on 2023/2/27 10:50

@author: Susan
"""
from pt_utils.PairTrading import PairTrading, TradingFrequency

bm_df = PairTrading(start_date='2016-12-01', form_freq=TradingFrequency.month, trans_freq=TradingFrequency.month)
