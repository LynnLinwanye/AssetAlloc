"""
教材 8.4 节 n=1 无约束 CARA 解析解
"""

import numpy as np


class AnalyticalSolution:
    """
    最优分配 (dollar amount):
        x_t* = (μ - r) / (σ² · a · (1+r)^{T-t-1})

    最优分配比例:
        p_t* = x_t* / W_t

    最优价值函数:
        V_t*(W) = -exp(-(μ-r)²(T-t)/(2σ²)) / a · exp(-a(1+r)^{T-t} · W)
    """

    def __init__(self, T, r, mu, sigma2, risk_aversion):
        self.T = T
        self.r = r
        self.mu = mu
        self.sigma2 = sigma2
        self.a = risk_aversion

    def optimal_dollar_allocation(self, t):
        """x_t* (投入风险资产的金额)"""
        return (self.mu - self.r) / (
            self.sigma2 * self.a * (1 + self.r) ** (self.T - t - 1)
        )

    def optimal_proportion(self, t, W):
        """p_t* = x_t* / W (风险资产比例)"""
        return self.optimal_dollar_allocation(t) / W

    def optimal_value(self, t, W):
        """V_t*(W)"""
        exp1 = -(self.mu - self.r) ** 2 * (self.T - t) / (2 * self.sigma2)
        exp2 = -self.a * (1 + self.r) ** (self.T - t) * W
        return -np.exp(exp1) / self.a * np.exp(exp2)

    def print_summary(self, W0=1.0):
        print(f"  参数: T={self.T}, r={self.r}, μ={self.mu}, "
              f"σ²={self.sigma2}, a={self.a}")
        print()
        for t in range(self.T):
            x_opt = self.optimal_dollar_allocation(t)
            p_opt = self.optimal_proportion(t, W0)
            V_opt = self.optimal_value(t, W0)
            print(f"  t={t}: x*={x_opt:.6f}, p*={p_opt:.6f}, V*(W0)={V_opt:.6f}")

class MultiAssetAnalyticalSolution:
    """
    n>1 无约束 CARA 解析解

    x_t* = 1 / (a (1+r)^{T-t-1}) · Σ^{-1}(μ - r1)

    V_t*(W) =
        -1/a · exp(-½ θᵀ Σ^{-1} θ (T-t))
        · exp(-a (1+r)^{T-t} W)
    """

    def __init__(self, T, r, mu_vec, Sigma, risk_aversion):
        self.T = T
        self.r = r
        self.mu = np.array(mu_vec)
        self.Sigma = np.array(Sigma)
        self.a = risk_aversion

        self.n = len(mu_vec)

        self.ones = np.ones(self.n)
        self.theta = self.mu - self.r * self.ones
        self.Sigma_inv = np.linalg.inv(self.Sigma)

        # 常数项 θᵀ Σ^{-1} θ
        self.sharpe_quad = self.theta.T @ self.Sigma_inv @ self.theta

    def optimal_dollar_allocation(self, t):
        """
        x_t* (风险资产金额向量)
        """
        factor = 1 / (self.a * (1 + self.r) ** (self.T - t - 1))
        return factor * (self.Sigma_inv @ self.theta)

    def optimal_proportion(self, t, W):
        """
        风险资产比例向量
        """
        return self.optimal_dollar_allocation(t) / W

    def optimal_value(self, t, W):
        """
        V_t*(W)
        """
        exp1 = -0.5 * self.sharpe_quad * (self.T - t)
        exp2 = -self.a * (1 + self.r) ** (self.T - t) * W
        return -1 / self.a * np.exp(exp1) * np.exp(exp2)

    def print_summary(self, W0=1.0):
        print(f"参数: T={self.T}, r={self.r}, a={self.a}")
        print("μ =", self.mu)
        print("Σ =")
        print(self.Sigma)
        print()

        for t in range(self.T):
            x_opt = self.optimal_dollar_allocation(t)
            p_opt = self.optimal_proportion(t, W0)
            V_opt = self.optimal_value(t, W0)

            print(f"t={t}")
            print("  x* =", x_opt)
            print("  p* =", p_opt)
            print(f"  V*(W0) = {V_opt:.6f}")
            print()