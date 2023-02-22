# -*- coding: utf-8 -*-
"""
Created on 2023/2/10 18:31

@author: Susan
"""
import os
import sqlite3

import pandas as pd
from CommonUse.funcs import read_pkl

#
base_route = './raw/'
tds = read_pkl(base_route + 'tds.pkl')
tds_df = pd.DataFrame({'trading_date': tds})

# %%
project_route = 'D:\\courses\\2022-2024FDU\\COURCES\\Dlab\\intern\\PairTrading\\pt'

conn = sqlite3.connect(project_route+'\\input\\PairTrading.db')
tds_df.to_sql('tds', conn, if_exists='replace')


def raw2sql(file_name: str, table_name: str, route: str = base_route):
    if '.pkl' in file_name:
        df = read_pkl(route + file_name)
    else:
        df = pd.read_pickle(route + file_name)
    df.to_sql(table_name, conn, if_exists='replace')
    print(f'{file_name} to {table_name}\t Done!')


raw2sql('tds.pkl', 'tds')
# raw2sql('ashare_universe_status.pkl', 'status')
# raw2sql('cne6_exposure.pkl', 'cne6_exposure')
# raw2sql('factor_cov.pkl', 'factor_cov')
# raw2sql('index_quote.pkl', 'index_quote')
# raw2sql('mkt_cap', 'mkt_cap')
# raw2sql('quote', 'quote')
# raw2sql('stock_sid_name.pkl', 'stock_sid_name')

conn.close()
