# -*- coding: utf-8 -*-
"""
Created on 2023/2/27 10:47

@author: Susan
"""
from pt_utils.PairTrading import PairTrading, TradingFrequency

bm_df = PairTrading(start_date='2016-07-01', form_freq=TradingFrequency.quarter,form_freq_num=-2 ,trans_freq=TradingFrequency.year)

