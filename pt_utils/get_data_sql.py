# -*- coding: utf-8 -*-
"""
Created on 2023/2/13 12:20

@author: Susan
"""
import sqlite3
import warnings

import pandas as pd

from pt_utils.function import factor_name, pairs_sk_set

warnings.filterwarnings('ignore')

project_path = 'D:/courses/2022-2024FDU/COURCES/Dlab/intern/PairTrading/pt/'


def create_sql_engine(db_name: str = project_path + 'input/PairTrading.db'):
    conn = sqlite3.connect(db_name)
    return conn


class PairTradingData(object):
    conn = create_sql_engine()

    @staticmethod
    def trading_date_between(start_date: str, end_date: str):
        """
        start_date-end_date间所有交易日期列表以及最大的交易日期
        :param start_date: YYYY-MM-DD
        :param end_date: YYYY-MM-DD
        :return:
        """
        tds_sql = f"""
            select trading_date
            from tds
            where trading_date between '{start_date}' and '{end_date}';
            """
        tds_df = pd.read_sql(tds_sql, PairTradingData.conn)
        tds_list = tds_df['trading_date'].values.tolist()
        return tds_list, max(tds_list)

    # %% StockPool
    @staticmethod
    def init_stock_list(start_date: str, end_date: str) -> pd.Series:
        """
        所有的股票
        :param start_date:
        :param end_date:
        :return:
        """
        all_stock_sql = pd.read_sql(f"""
                    select distinct sid from status where date between '{start_date}' and '{end_date}'
                    """, PairTradingData.conn)
        return pd.read_sql(all_stock_sql, PairTradingData.conn)['sid']

    @staticmethod
    def valid_status_stock(start_date: str, end_date: str) -> pd.Series:
        """
        无停复牌，非新上市，非退市股票
        :param start_date:
        :param end_date:
        :return:
        """
        all_stock_sql = f"""
            SELECT DISTINCT s1.sid
            FROM status s1
            LEFT JOIN (
              SELECT DISTINCT sid
              FROM status
              WHERE date BETWEEN '{start_date}' AND '{end_date}'
                AND (stop OR long_stop OR new OR st)
            ) s2 ON s1.sid = s2.sid
            WHERE s1.date BETWEEN '{start_date}' AND '{end_date}'
              AND s2.sid IS NULL;
            """
        return pd.read_sql(all_stock_sql, PairTradingData.conn)['sid']

    @staticmethod
    def valid_quote_stock(start_date: str, end_date: str, quantile: float = 0.8) -> pd.Series:
        """
        成交量满足条件的stock_list
        :param start_date:
        :param end_date:
        :param quantile:
        :return:
        """
        amount_stock_sql = f"""
            SELECT sid
            FROM (
              SELECT sid,
                     ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS row_number,
                     COUNT(*) AS total_rows
              FROM quote
              WHERE date BETWEEN '{start_date}' AND '{end_date}'
              GROUP BY sid
            ) derived
            WHERE row_number <= {quantile} * total_rows;
            """
        return pd.read_sql(amount_stock_sql, PairTradingData.conn)['sid']

    @staticmethod
    def valid_mkt_stock(start_date: str, end_date: str, quantile: float = 0.8) -> pd.Series:
        """
        市值满足条件的stock_list
        :param start_date:
        :param end_date:
        :param quantile:
        :return:
        """
        mkt_sql = f"""
            SELECT sid
            FROM (
              SELECT sid,
                     ROW_NUMBER() OVER (ORDER BY SUM(free) DESC) AS row_number,
                     COUNT(*) AS total_rows
              FROM mkt_cap
              WHERE date BETWEEN '{start_date}' AND '{end_date}'
              GROUP BY sid
            ) derived
            WHERE row_number <= {quantile} * total_rows;
            """
        return pd.read_sql(mkt_sql, PairTradingData.conn)['sid']

    # %% PairOn
    @staticmethod
    def get_cov_data(mdate) -> pd.DataFrame:
        """
        因子在最大的交易日的协方差矩阵
        :param mdate:
        :return:
        """
        col_list_str = ','.join(['factor'] + factor_name)
        factor_tuple = tuple(factor_name)

        cov_sql = f"""
            select {col_list_str}
            from factor_cov sort 
            where date = '{mdate}'
            and factor in {str(factor_tuple)}
            order by factor;
            """

        return pd.read_sql(cov_sql, PairTradingData.conn)

    @staticmethod
    def get_stock_factor(stock_list: tuple, mdate) -> pd.DataFrame:
        """
        在最大交易日的各股票的因子值
        :param stock_list:
        :param mdate:
        :return:
        """
        select_stmt = ','.join(factor_name)
        if not len(stock_list):
            return pd.DataFrame()
        elif len(stock_list) == 1:
            stock_tuple_str = f"""('{stock_list[0]}')"""
        else:
            stock_tuple_str = str(stock_list)
        factor_sql = f"""
            select sid, {select_stmt}, industry
            from cne6_exposure
            where sid in {stock_tuple_str}
              and date = '{mdate}';
            """

        return pd.read_sql(factor_sql, PairTradingData.conn)

    @staticmethod
    def get_stock_vwap_adj(stock_list: tuple, start_date, end_date) -> pd.DataFrame:
        vwap_sql = f"""
        select date, sid, adj_factor * vwap AS vwap_adj
        from quote
        where date between '{start_date}' and '{end_date}'
          and sid in {str(stock_list)};
        """

        return pd.read_sql(vwap_sql, PairTradingData.conn)

    @staticmethod
    def get_stock_close_adj(stock_list: tuple, start_date, end_date) -> pd.DataFrame:
        close_sql = f"""
        select date, sid, close AS close_adj
        from quote
        where date between '{start_date}' and '{end_date}'
          and sid in {str(stock_list)};
        """

        return pd.read_sql(close_sql, PairTradingData.conn)

    @staticmethod
    def get_index_data(start_date: str, end_date: str, sid: str = '000906.SH'):
        index_sql = f"""
        select date,sid,(close-preclose)/preclose as pct from index_quote where date between '{start_date}' and '{end_date}' and sid = '{sid}'
        """
        data = pd.read_sql(index_sql, PairTradingData.conn)
        return data.pivot(index='date', columns='sid', values='pct')

    @staticmethod
    def get_pair_origin_data(pairs: list = None, start_date: str = None, end_date: str = None):
        sk_set = pairs_sk_set(pairs)
        vwap_data = PairTradingData.get_stock_vwap_adj(tuple(sk_set), start_date, end_date)
        return vwap_data

    @staticmethod
    def get_pair_close_data(pairs: list = None, start_date: str = None, end_date: str = None):
        sk_set = pairs_sk_set(pairs)
        close_data = PairTradingData.get_stock_close_adj(tuple(sk_set), start_date, end_date)
        return close_data

    @staticmethod
    def get_data_tmp(sql):
        return pd.read_sql(sql, PairTradingData.conn)
