# -*- coding: utf-8 -*-
"""
Created on 2023/2/27 10:40

@author: Susan
"""
from pt_utils.PairTrading import PairTrading, TradingFrequency

bm_df = PairTrading(start_date='2016-01-01', form_freq=TradingFrequency.year, trans_freq=TradingFrequency.year)
