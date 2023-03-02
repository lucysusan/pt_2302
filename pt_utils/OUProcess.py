# -*- coding: utf-8 -*-
"""
Created on 2023/2/14 10:48

@author: Susan
"""
from dataclasses import dataclass
from math import floor

import numpy as np
from scipy.special import erfi
from sklearn.linear_model import LinearRegression


@dataclass
class OUParams:
    alpha: float  # mean reversion parameter
    gamma: float  # asymptotic mean
    beta: float  # Brownian motion scale (standard deviation)


class OUProcess(object):

    def __init__(self, X_t: np.ndarray):
        self.X_t = X_t
        self.ou_param = None
        self.exists = False
        ou_param = self._estimate_OU_params()
        if ou_param.alpha > 0 and ou_param.beta > 0:
            self.ou_param = ou_param
            self.exists = True

    def _estimate_OU_params(self) -> OUParams:
        """
        Estimate OU params from OLS regression.
        - X_t is a 1D array.
        Returns instance of OUParams.
        """
        X_t = self.X_t
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

    def calculate_ou_cycle(self, a):
        """
        计算得到最优参数后的OU过程的时间周期
        :param a: entry_param
        :return:
        """
        alpha = self.ou_param.alpha
        beta = self.ou_param.beta
        cycle_all = np.pi / alpha * (erfi((-a * np.sqrt(alpha)) / beta) - erfi(a * np.sqrt(alpha) / beta))
        return floor(cycle_all / 4)

    # one pair
    @staticmethod
    def _expect_return_optimize_function(a: float or list, alpha: float, beta: float, c: float) -> float:
        """
        用期望收益优化进出价格
        :param a: entry level
        :param alpha: OU Param
        :param beta: OU Param
        :param c: trading cost in percent type
        :return:
        """
        if not isinstance(a, float):
            a = a[0]
        l = np.exp((alpha * a ** 2) / beta ** 2) * (2 * a + c)
        r = beta * np.sqrt(np.pi / alpha) * erfi((a * np.sqrt(alpha)) / beta)
        return abs(l - r)

    def _opt(self, a_list, c) -> float:
        min_v = 100
        min_a = 0
        alpha = self.ou_param.alpha
        beta = self.ou_param.beta
        for a in a_list:
            v = self._expect_return_optimize_function(a, alpha, beta, c)
            if v < min_v:
                min_v = v
                min_a = a
        return min_a

    def run_ouOpt(self, c=0.0015):
        """
        价差序列，返回最优进入解及最优解下的OU过程的期望周期时间长度
        :param c: 手续费 percent, default: 千一点五
        :return: 最优进入点(负)，期望周期时间长度
        """
        min_a = min(self.X_t)
        n_iter = 1000
        a_list = np.arange(min_a, 0, -min_a / n_iter)
        a = self._opt(a_list, c)
        cycle = self.calculate_ou_cycle(a)
        return a, cycle
