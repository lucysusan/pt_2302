# -*- coding: utf-8 -*-
"""
Created on 2023/2/10 18:31

@author: Susan
"""
import os
import sqlite3

import pandas as pd
from CommonUse.funcs import read_pkl

base_route = '../raw/base/'
print(file_list := os.listdir(base_route))

quote = pd.read_pickle(base_route + 'quote')
conn = sqlite3.connect('PairTrading.db')
quote.to_sql('quote', conn, if_exists='replace')
status = read_pkl(base_route + 'ashare_universe_status.pkl')


def raw2sql(file_name: str, table_name: str, route: str = base_route):
    if '.pkl' in file_name:
        df = read_pkl(route + file_name)
    else:
        df = pd.read_pickle(route + file_name)
    df.to_sql(table_name, conn, if_exists='replace')
    print(f'{file_name} to {table_name}\t Done!')


raw2sql('ashare_universe_status.pkl', 'status')
raw2sql('cne6_exposure.pkl', 'cne6_exposure')
raw2sql('factor_cov.pkl', 'factor_cov')
raw2sql('index_quote.pkl', 'index_quote')
raw2sql('mkt_cap', 'mkt_cap')
raw2sql('quote', 'quote')
raw2sql('stock_sid_name.pkl', 'stock_sid_name')

conn.close()
