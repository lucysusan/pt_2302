import pandas as pd
import os
import pyfolio as pf

index_sid = '000906.SH'
data_folder = 'OUT_DATA/'
file_list = os.listdir(data_folder)

file_pkl = [f for f in file_list if not f.endswith('.csv')]

for file in file_pkl:
    data = pd.read_pickle(data_folder + file)

    pf.create_returns_tear_sheet(data['ret'], benchmark_rets=data[index_sid])
