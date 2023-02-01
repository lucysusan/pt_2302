# -*- coding: utf-8 -*-
"""
Created on 2023/1/31 14:14

@author: Susan
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from scipy.special import erfi
# from math import exp, sqrt
import matplotlib.pyplot as plt
import warnings
from CommonUse.funcs import read_pkl, write_pkl, createFolder
from pt_utils.functions import out_folder, start_date, end_date, form_start, form_end, trans_start, trans_end
from sympy import *
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['FangSong_GB2312']
round_num = 10


# %% OU_process
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
    return OUParams(alpha, gamma, beta)


def expect_return_optimize_function(a: float, alpha: float, beta: float, c: float):
    """
    用期望收益优化进出价格
    :param round_num: 数值显示的小数位数
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


# %%
cc_top_list = read_pkl(out_folder + 'cc_top10_list.pkl')
price_pivot = read_pkl(out_folder + 'log_vwap_aft_price_pivot.pkl')


# c = 0.01

# cc = cc_top_list[0]
# TODO: 基于formation，计算每个配对的a,m。分form,trans画出spread图
#  ！考虑这里的成本表达   得到trans每个经过a,m的时间对，计算收益情况
def a_m_best_expected_return(cc, price_pivot, c=0.01):
    cc_price = price_pivot.loc[:trans_start, cc]
    lmda = sum(cc_price.iloc[:, 0]) / sum(cc_price.iloc[:, 1])
    spread = cc_price.iloc[:, 0] - lmda * cc_price.iloc[:, 1]
    spread_array = spread.values
    ou_params = estimate_OU_params(spread_array)

    r = minimize(expect_return_optimize_function, 0.08, args=(ou_params.alpha, ou_params.beta, c))
    a_best = r.x[0]
    m_best = -a_best
    print(f'{cc}\n\ta\t{a_best}\n\tm\t{m_best}')
    return a_best, m_best
