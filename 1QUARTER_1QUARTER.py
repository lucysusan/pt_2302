# -*- coding: utf-8 -*-
"""
Created on 2023/2/27 10:49

@author: Susan
"""
from pt_utils.PairTrading import PairTrading, TradingFrequency

bm_df = PairTrading(start_date='2016-10-01', form_freq=TradingFrequency.quarter, trans_freq=TradingFrequency.quarter)
