# -*- coding: utf-8 -*-
"""
Created on 2023/2/2 14:56

@author: Susan
"""
import datetime
import itertools
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CommonUse.funcs import read_pkl, createFolder
from dateutil.relativedelta import relativedelta
from scipy.special import erfi
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['FangSong']

start_date = '2010-01-04'
end_date = '2022-11-29'


def timer(func):
    """
    计时装饰器
    :param func:
    :return:
    """

    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result

    return func_wrapper


def date_Opt_year(date: str, years: int):
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    d_y = (d - relativedelta(years=years)).strftime('%Y-%m-%d')
    return d_y


def date_Opt_month(date: str, months: int):
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    d_y = (d - relativedelta(months=months)).strftime('%Y-%m-%d')
    return d_y


# %% OU PROCESS
@dataclass
class OUParams:
    alpha: float  # mean reversion parameter
    gamma: float  # asymptotic mean
    beta: float  # Brownian motion scale (standard deviation)


def estimate_OU_params(X_t: np.ndarray) -> OUParams:
    """
    Estimate OU params from OLS regression.
    - X_t is a 1D array.
    Returns instance of OUParams.
    """
    y = np.diff(X_t)
    X = X_t[:-1].reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    # regression coeficient and constant
    alpha = -reg.coef_[0]
    gamma = reg.intercept_ / alpha
    # residuals and their standard deviation
    y_hat = reg.predict(X)
    beta = np.std(y - y_hat)
    return OUParams(alpha, gamma, float(beta))


class PairTrading:
    stock_status_route = 'raw/status.pkl'
    price_route = 'raw/price_log_origin.pkl'
    cne_exposure_route = 'raw/barra_exposure.pkl'
    mkt_cap_route = 'raw/mkt.pkl'
    amount_route = 'raw/amount.pkl'
    close_route = 'raw/close.pkl'

    stock_status = pd.DataFrame()
    price_pivot_log_origin = pd.DataFrame()
    cne_exposure = pd.DataFrame()
    mkt_cap = pd.DataFrame()
    amount = pd.DataFrame()
    close = pd.DataFrame()

    def __init__(self, form_end: str, trans_start: str, out_route: str = 'result/', form_y: int = 1, trans_m: int = 6,
                 bar_tolerance: float = 1e-4,
                 c: float = 0.0015, c_ratio: float = 0.0015, norm_bar: float = 10, pair_num: int = 20):
        self.cc_top_list = None
        self.stock_pool = None
        self.form_end = form_end
        self.form_y = form_y
        self.trans_start = trans_start
        self.trans_m = trans_m

        self.form_start = date_Opt_year(form_end, form_y)
        self.trans_end = date_Opt_month(trans_start, -trans_m)
        print(
            f'formation start\t{self.form_start}\tformation end\t{self.form_end}\ntransaction start\t{self.trans_start}\ttransaction end\t{self.trans_end}')

        out_folder = f'{out_route}/{form_end}_{form_y}Y_{trans_start}_{trans_m}M/'
        self.out_folder = out_folder
        self.fig_folder = out_folder + 'fig/'
        createFolder(self.fig_folder)

        self.bar_tolerance = bar_tolerance
        self.c = c
        self.c_ratio = c_ratio
        self.norm_bar = norm_bar
        self.pair_num = pair_num

        print('RESULTS will be saved at ' + self.out_folder)

    @staticmethod
    def clear_object_data():
        PairTrading.stock_status = pd.DataFrame()
        PairTrading.price_pivot_log_origin = pd.DataFrame()
        PairTrading.cne_exposure = pd.DataFrame()
        PairTrading.mkt_cap = pd.DataFrame()
        PairTrading.amount = pd.DataFrame()
        return

    @staticmethod
    def update_object_params(stock_status_route: str or None = None, price_route: str or None = None,
                             cne_exposure_route: str or None = None, mkt_cap_route: str or None = None,
                             amount_route: str or None = None):
        PairTrading.stock_status_route = stock_status_route if stock_status_route else PairTrading.stock_status_route
        PairTrading.price_route = price_route if price_route else PairTrading.price_route
        PairTrading.cne_exposure_route = cne_exposure_route if cne_exposure_route else PairTrading.cne_exposure_route
        PairTrading.mkt_cap_route = mkt_cap_route if mkt_cap_route else PairTrading.mkt_cap_route
        PairTrading.amount_route = amount_route if amount_route else PairTrading.amount_route
        return

    def split_form_trans(self, data):
        """
        function separately deals with data splitting
        :param data:
        :return: formation_data, transaction_data
        """
        data_form = data[(data.index >= self.form_start) & (data.index <= self.form_end)]
        data_trans = data[(data.index >= self.trans_start) & (data.index <= self.trans_end)]
        return data_form, data_trans

    # %% 筛选股票池 formation和transaction阶段都非新股，非ST，无停牌复牌行为, 市值和成交量前80%
    @staticmethod
    def _partial_stock_pool(stock_status) -> set:
        """
        get_stock_pool的辅助函数
        :param stock_status:
        :return:
        """
        stock_status.set_index(['date', 'sid'], inplace=True)
        stock_status['normal'] = stock_status.apply(lambda x: sum(x) > 0, axis=1)
        stock_status.reset_index(inplace=True)
        status_abnormal = set(stock_status[stock_status['normal']]['sid'].unique().tolist())
        status_all = set(stock_status['sid'].unique().tolist())
        return status_all - status_abnormal

    def _stock_pool_status(self) -> list:
        """
        股票池，在formation和transaction阶段都没有异常行为(新上市、退市、停复牌)的股票
        :return: 满足条件的股票池list
        """
        PairTrading.stock_status = read_pkl(
            PairTrading.stock_status_route) if PairTrading.stock_status.empty else PairTrading.stock_status

        status_form, status_trans = self.split_form_trans(PairTrading.stock_status.set_index('date'))

        stock_form = self._partial_stock_pool(status_form.reset_index())
        stock_trans = self._partial_stock_pool(status_trans.reset_index())
        self.stock_pool = list(stock_trans.intersection(stock_form))
        return self.stock_pool

    def _filter_quantile(self, data: pd.DataFrame, quantile: float = 0.2) -> set:
        """
        辅助函数，筛选mkt_cap和amount
        :param data:
        :param quantile:
        :return:
        """
        data_form = data[(data.index >= self.form_start) & (data.index <= self.form_end)]
        data_mean = data_form.mean()
        return set(data_mean[data_mean > data_mean.quantile(quantile)].index)

    def _stock_pool_mkt_cap(self) -> set:
        """
        筛选市值，剔除形成期阶段后20%的股票
        :return:
        """
        PairTrading.mkt_cap = read_pkl(PairTrading.mkt_cap_route) if PairTrading.mkt_cap.empty else PairTrading.mkt_cap
        stock_use = self._filter_quantile(PairTrading.mkt_cap)
        self.stock_pool = list(stock_use.intersection(self.stock_pool))
        return stock_use

    def _stock_pool_amount(self) -> set:
        """
        筛选成交金额，剔除形成期阶段后20%的股票
        :return:
        """
        PairTrading.amount = read_pkl(PairTrading.amount_route) if PairTrading.amount.empty else PairTrading.amount
        stock_use = self._filter_quantile(PairTrading.amount)
        self.stock_pool = list(stock_use.intersection(self.stock_pool))
        return stock_use

    def get_stock_pool(self) -> list:
        """
        配对交易股票池
        :return:
        """
        self._stock_pool_status()
        self._stock_pool_mkt_cap()
        self._stock_pool_amount()
        return self.stock_pool

    def get_price_log_origin(self, stock_pool):
        """
        日期和股票池筛选过的log_price_pivot和origin_price_pivot
        :param stock_pool: 股票池
        :return: log_price_pivot, origin_price_pivot
        """
        PairTrading.price_pivot_log_origin = read_pkl(
            PairTrading.price_route) if PairTrading.price_pivot_log_origin.empty else PairTrading.price_pivot_log_origin

        price_pivot, origin_price = PairTrading.price_pivot_log_origin['log_vwap_aft'], \
            PairTrading.price_pivot_log_origin['vwap_aft']
        price_pivot = price_pivot[(price_pivot.index >= self.form_start) & (price_pivot.index <= self.trans_end)][
            stock_pool]
        origin_price = origin_price[(origin_price.index >= self.form_start) & (origin_price.index <= self.trans_end)][
            stock_pool]
        return price_pivot, origin_price

    def get_factor_cov(self):
        """
        form_end的因子协方差
        :return: cov和因子顺序
        """
        form_end = self.form_end
        factor_cov = read_pkl('raw/barra_cov.pkl')
        factor_cov = factor_cov[factor_cov.index <= form_end].sort_index(ascending=True)
        t_cov = factor_cov.index.unique().tolist()[-1]
        cov = factor_cov[factor_cov.index == t_cov]
        cov = cov.reset_index(drop=True).set_index('factor').sort_index(ascending=True)
        factor_list = cov.index.tolist()
        cov_matrix = cov[factor_list].values
        return cov_matrix, factor_list

    def get_cc_top_list(self, stock_pool: list) -> list:
        """
        得到配对组, 距离变量 - 加权L2
        :param stock_pool:
        :return:
        """
        PairTrading.cne_exposure = read_pkl(
            PairTrading.cne_exposure_route) if PairTrading.cne_exposure.empty else PairTrading.cne_exposure

        cov_matrix, factor_list = self.get_factor_cov()

        cne_exposure = PairTrading.cne_exposure
        cne_factor = cne_exposure[
            (cne_exposure['date'] >= self.form_start) & (cne_exposure['date'] <= self.form_end) & (
                cne_exposure['sid'].isin(stock_pool))]

        ind_list = cne_factor['industry'].unique().tolist()
        cne_factor.reset_index(inplace=True)
        ind_group = cne_factor.groupby('industry')
        topn_df = pd.DataFrame()
        for ind in tqdm(ind_list):
            ind_data = ind_group.get_group(ind)
            security_list = ind_data['sid'].unique().tolist()
            cc = list(itertools.combinations(security_list, 2))
            ind_df = pd.DataFrame(index=cc)
            for val in factor_list:
                ind_pivot = ind_data.pivot(index='date', columns='sid', values=val)
                ind_val_corr = abs(ind_pivot.corr(method='pearson'))
                val_list = [ind_val_corr.loc[c] for c in cc]
                ind_df[val] = val_list

            ind_df['norm'] = ind_df.apply(lambda x: x.values.T @ cov_matrix @ x.values, axis=1)
            # TODO: 阈值的设定仍然没有实际含义，在千分之一到百分之一之间
            topn_ind_df = ind_df[ind_df['norm'] > self.norm_bar]
            topn_ind_df['industry'] = ind
            topn_df = pd.concat([topn_df, topn_ind_df])

        topn_df.sort_values('norm', ascending=False, inplace=True)
        topn_df.to_excel(self.out_folder + f'industry_norm_l{self.norm_bar}.xlsx')
        topn_df_largest = topn_df.nlargest(self.pair_num, 'norm')
        cc_top_list = topn_df_largest.index.to_list()
        self.cc_top_list = cc_top_list
        return cc_top_list

    def plt_split_time(self, data, outer_fig, name=None, hline=None):
        """
        formation, transaction分隔画图
        :param data: 待画图数据, tuple(form,trans) or pd.DataFrame or pd.Series
        :param outer_fig: fig地址
        :param name: fig title
        :param hline: 是否需要添加横线，默认None不需要
        :return:
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey='row', figsize=(20, 10))
        a, b = axes[0], axes[1]
        a.set_title('formation')
        b.set_title('transaction')
        if isinstance(data, tuple):
            data_form, data_trans = data
        else:
            data_form, data_trans = self.split_form_trans(data)
        data_form.plot(ax=a)
        data_trans.plot(ax=b)
        if hline:
            for i, y in enumerate(hline, start=1):
                if i % 2:
                    plt.axhline(y=y, color="black", linestyle="--")
                else:
                    plt.axhline(y=y, color="grey", linestyle=":")
        if name:
            plt.suptitle(name)
        plt.tight_layout()
        plt.savefig(outer_fig)
        plt.close()

    def plt_log_price_pair(self, cc_top_list, price_pivot):
        """
        画出给定match组的股票对数价格的走势图
        :param cc_top_list: list中每个元素为tuple,类似 [('002040.SZ', '600190.SH'), ('603233.SH', '603883.SH')]
        :param price_pivot: log_price_pivot
        :return:
        """
        figFolder = self.fig_folder + 'log_price/'
        createFolder(figFolder)
        for c_code in cc_top_list:
            c_df = price_pivot.loc[:, c_code]
            self.plt_split_time(c_df, figFolder + c_code[0] + '-' + c_code[1] + '.png')

    # one pair
    @staticmethod
    def _expect_return_optimize_function(a: float or list, alpha: float, beta: float, c: float):
        """
        用期望收益优化进出价格
        :param a: entry level
        :param alpha: OU Param
        :param beta: OU Param
        :param c: trading cost
        :return:
        """
        if not isinstance(a, float):
            a = a[0]
        l = np.exp((alpha * a ** 2) / beta ** 2) * (2 * a + c)
        r = beta * np.sqrt(np.pi / alpha) * erfi((a * np.sqrt(alpha)) / beta)
        return abs(l - r)

    def _opt(self, a_list, alpha, beta, c):
        min_v = 100
        min_a = 0
        for a in a_list:
            v = self._expect_return_optimize_function(a, alpha, beta, c)
            if v < min_v:
                min_v = v
                min_a = a
        return min_a

    def a_m_best_expected_return(self, cc, origin_price, a_init=0.08, show_fig=False, verbose=False):
        """
        给定配对组和对数价格，返回entry,exit level,交易时间序列,配对系数
        :param cc:
        :param origin_price: log_price pivot
        :param a_init:
        :param show_fig:
        :param verbose: 是否print中间值, default False
        :return:
        """
        figFolder = self.fig_folder + 'spread/'
        createFolder(figFolder)
        cc_price = origin_price.loc[:, cc]
        price_form, price_trans = self.split_form_trans(cc_price)

        y = price_form.loc[:, cc[0]]
        X = price_form.loc[:, cc[1]]
        reg = LinearRegression(fit_intercept=False)
        X = X.values.reshape(-1, 1)
        reg.fit(X, y)
        lmda = reg.coef_[0]

        spread_form = price_form.apply(lambda x: np.log(x[cc[0]]) - np.log(lmda * x[cc[1]]), axis=1)
        spread_trans = price_trans.apply(lambda x: np.log(x[cc[0]]) - np.log(lmda * x[cc[1]]), axis=1)
        # 计算最优a,m
        spread_array = spread_form.values
        ou_params = estimate_OU_params(spread_array)

        min_a = min(spread_form)
        n_iter = 1000
        a_list = np.arange(min_a, 0, -min_a / n_iter)
        a_best = self._opt(a_list, ou_params.alpha, ou_params.beta, self.c)
        # r = minimize(self._expect_return_optimize_function, a_init, args=(ou_params.alpha, ou_params.beta, self.c))
        # a_best = r.x[0]
        m_best = -a_best
        if verbose:
            print(f'\t trading\tbuy per {cc[0]}, sell {lmda}\t{cc[1]}')
            print(f'{cc}\n\ta\t{a_best}\n\tm\t{m_best}')
        # draw plus entry and exit level
        if show_fig:
            self.plt_split_time((spread_form, spread_trans), f'{figFolder}spread_hline_{cc[0]}-{cc[1]}.png',
                                f'{cc[0]}-{cc[1]}',
                                hline=[a_best, m_best])

        return a_best, m_best, spread_trans, lmda

    # %% trading
    @staticmethod
    def touch_target(data: pd.Series, target: float, method: str = 'larger'):
        """
        data中最早比target大(小)的元素的index
        :param data:
        :param target:
        :param method: larger or smaller, default: larger
        :return:
        """
        if method == 'larger':
            # i = np.argmax(data >= target)
            # t = data.index[i]
            # if not i:
            #     return None if data[t] < target else t
            # return t
            for i, v in data.iteritems():
                if v >= target:
                    return i
        elif method == 'smaller':
            # i = np.argmin(data <= target) - 1
            # t = data.index[i]
            # return t if i >= 0 else None
            for i, v in data.iteritems():
                if v <= target:
                    return i
        return None

    def transaction_time_list(self, a, m, spread_trans, verbose=False):
        """
        交易时间点列表，[买入,卖出,...]
        :param a: entry level
        :param m: exit level
        :param spread_trans: 交易对数价差序列
        :param verbose: 是否显示中间值
        :return: trade_tlist
        """
        if -a < self.bar_tolerance:
            return None
        trade_time_list = []
        i_start = spread_trans.index[0]
        i_end = spread_trans.index[-1]
        i_len = 0

        while i_start <= i_end:
            if not i_len % 2:
                i = self.touch_target(spread_trans[i_start:], a, 'smaller')
            else:
                i = self.touch_target(spread_trans[i_start:], m, 'larger')
            if i:
                trade_time_list.append(i)
                i_len += 1
                i_start = i
            else:
                break
        print('\t trading_time_list', trade_time_list) if verbose else None
        return trade_time_list

    def calculate_pair_revenue(self, cc, origin_price, trade_time_list, lmda, invest_amount=1e5, verbose=False):
        """
        计算配对组净收益, 返回配对组在transaction阶段的仓位，收盘浮动价值 - 成交价:日内均价后复权，浮动价值:日收盘价后复权
        :param cc: 配对组tuple
        :param origin_price: 股票价格pivot
        :param trade_time_list: 交易时间序列,[买入,卖出,...]
        :param lmda:
        :param invest_amount: 投入该组合的投资金额总和
        :param verbose:
        :return:
        TODO: 收益率曲线，净值曲线
        """
        print(cc, '\t TRADING REVENUE-------------------------------------------------') if verbose else None
        # TODO:
        #  1. trade_list在check_table中提取vwap_1,vwap_2
        #  2. 整除买入，余项卖出，补充flow_table对应的volume_1,volume_2,cash,outflow; 向后填充
        #  3. 通过check_table中的close_1,close_2, 以及flow_table中的volume_1,volume_2,cash计算value
        PairTrading.close = read_pkl(PairTrading.close_route) if PairTrading.close.empty else PairTrading.close
        close = PairTrading.close.loc[self.trans_start:self.trans_end, cc]
        col = [f'{cc[0]}_volume_1', f'{cc[1]}_volume_2', 'cash', 'outflow', 'value']

        flow_table = pd.DataFrame(columns=col, index=close.index)
        flow_dict = dict(zip(col, [0, 0, invest_amount, 0, 0]))

        def _update_flow_table(dv: dict, df: pd.DataFrame, t_i):
            for k, v in dv.items():
                df.loc[t_i, k] = v
            return df

        # 首行
        flow_table = _update_flow_table(flow_dict, flow_table, flow_table.index[0])

        for i, t in enumerate(trade_time_list):
            if not i % 2:
                # long 1, short 2
                num_1 = np.floor(invest_amount / (100 * origin_price.loc[t, cc[0]])) * 100
                num_2 = - np.floor((invest_amount / (lmda * 100 * origin_price.loc[t, cc[1]])) * lmda) * 100
                out_flow = num_1 * 100 * origin_price.loc[t, cc[0]] + num_2 * 100 * origin_price.loc[t, cc[1]]
                cash = invest_amount - out_flow
            else:
                # 平仓
                t_pre = trade_time_list[i - 1]
                num_1 = - flow_table.loc[t_pre, f'{cc[0]}_volume_1']
                num_2 = - flow_table.loc[t_pre, f'{cc[1]}_volume_2']
                cash = flow_table.loc[t_pre, 'cash']
                out_flow = num_1 * 100 * origin_price.loc[t, cc[0]] + num_2 * 100 * origin_price.loc[t, cc[1]]
                cash -= out_flow
                num_1, num_2 = 0, 0
            flow_table = _update_flow_table(
                {f'{cc[0]}_volume_1': num_1 * 100, f'{cc[1]}_volume_2': -num_2 * 100, 'cash': cash,
                 'outflow': out_flow}, flow_table,
                t)
        flow_table = flow_table.fillna(method='ffill')
        flow_table['value'] = flow_table[f'{cc[0]}_volume_1'] * close[cc[0]] + flow_table[f'{cc[1]}_volume_2'] * close[
            cc[1]] + flow_table['cash']
        print(flow_table) if verbose else None
        return flow_table

    @timer
    def run(self, invest_amount=1e5, verbose=False):

        PairTrading.close = read_pkl(PairTrading.close_route) if PairTrading.close.empty else PairTrading.close
        flow_table = pd.DataFrame(index=PairTrading.close.loc[self.trans_start:self.trans_end].index,
                                  columns=['cash', 'value'])
        flow_table.fillna(0, inplace=True)

        stock_pool = self.get_stock_pool()

        if not stock_pool:
            return None

        price_pivot, origin_price = self.get_price_log_origin(stock_pool)
        cc_top_list = self.get_cc_top_list(stock_pool)

        self.plt_log_price_pair(cc_top_list, price_pivot)
        for cc in tqdm(cc_top_list):
            a, m, spread_trans, lmda = self.a_m_best_expected_return(cc, origin_price, show_fig=True, verbose=verbose)
            t_list = self.transaction_time_list(a, m, spread_trans, verbose=verbose)
            flow_df = self.calculate_pair_revenue(cc, origin_price, t_list, lmda, invest_amount=invest_amount,
                                                  verbose=verbose)
            flow_table = flow_table.merge(flow_df[[f'{cc[0]}_volume_1', f'{cc[1]}_volume_2']], 'outer', left_index=True,
                                          right_index=True)
            flow_table['value'] = flow_table['value'] + flow_df['value']
            flow_table['cash'] = flow_table['cash'] + flow_df['cash']

        # col_name = flow_table.columns.tolist()
        flow_table['per_value'] = flow_table['value'] / (invest_amount * self.pair_num)
        # col_name.insert(0, 'per_value')
        # flow_table = flow_table[col_name]
        plt.figure(figsize=(20, 10))
        flow_table['per_value'].plot()
        plt.savefig(self.out_folder + f'配对组净值表现_{self.trans_start}_{self.trans_end}.png')
        flow_table.to_csv(self.out_folder + '净值配对组交易结果.csv')

        return flow_table
