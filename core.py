"""
Fitted Value Iteration (ADP) 求解器
核心改进:
1. 参数化拟合: log(-V) 用线性回归 (精确匹配 CARA 结构)
2. 网格搜索优化: 避免 SLSQP 在噪声目标上失效
3. 预生成 MC 样本: 消除优化过程中的随机性
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class ValueFunctionApproximator:
    """
    拟合 log(-V) 作为 (W, p_1, ..., p_n) 的函数
    对于 CARA: log(-V) ≈ const - c*W, 线性回归足够
    对于多资产: 用 degree=2 多项式捕捉交叉项
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
            ('ridge', Ridge(alpha=1e-4))
        ])
        self.fitted = False

    def fit(self, features, V_values):
        """
        拟合 log(-V)
        features: (n_samples, 1+n_assets) -> [W, p1, ..., pn]
        V_values: (n_samples,) -> V < 0
        """
        neg_V = -V_values
        neg_V = np.maximum(neg_V, 1e-30)  # 安全下界
        log_neg_V = np.log(neg_V)

        # 检查并处理无穷大或NaN
        if np.any(~np.isfinite(log_neg_V)):
            print(f"[WARNING: {np.sum(~np.isfinite(log_neg_V))} invalid log_neg_V values, clipping]")
            log_neg_V = np.clip(log_neg_V, -700, 700)  # 防止溢出

        self.model.fit(features, log_neg_V)
        self.fitted = True

        # 计算拟合质量
        pred_log = self.model.predict(features)
        pred_V = -np.exp(pred_log)
        ss_res = np.sum((V_values - pred_V) ** 2)
        ss_tot = np.sum((V_values - np.mean(V_values)) ** 2)
        self.r2 = 1 - ss_res / (ss_tot + 1e-30)

        return self.r2

    def predict_V(self, features):
        """从 log(-V) 模型恢复 V"""
        log_neg_V = self.model.predict(features)
        return -np.exp(log_neg_V)


class AssetAllocationADP:
    """
    State:  (W_t, p_1, ..., p_n)   p_0 = 1 - sum(p_k) 为现金
    Action: (Δp_1, ..., Δp_n)      sum|Δp_k| <= max_adjust
    Terminal Reward: U(W_T) = -e^{-a W_T} / a
    """

    def __init__(self, n_assets, T, r, means, variances,
                 risk_aversion, max_adjust=0.10, n_mc=5000):
        self.n = n_assets
        self.T = T
        self.r = r
        self.means = np.array(means, dtype=float)
        self.variances = np.array(variances, dtype=float)
        self.stds = np.sqrt(self.variances)
        self.a = risk_aversion
        self.max_adjust = max_adjust
        self.n_mc = n_mc

        self.V_approx = {}   # t -> ValueFunctionApproximator
        self.policies = {}

    # -------- 效用函数 --------
    def utility(self, W):
        return -np.exp(-self.a * W) / self.a

    # -------- 模拟下一步 --------
    def _compute_next_states(self, W, props_risky, delta_p, returns_samples):
        new_props = props_risky + delta_p
        p0 = 1.0 - np.sum(new_props)

        growth = p0 * (1.0 + self.r) + np.sum(
            new_props[np.newaxis, :] * (1.0 + returns_samples), axis=1
        )
        growth = np.maximum(growth, 1e-8)
        W_next = W * growth

        props_next = np.zeros_like(returns_samples)
        for k in range(self.n):
            props_next[:, k] = new_props[k] * (1.0 + returns_samples[:, k]) / growth

        return W_next, props_next

    # -------- 评估动作 --------
    def _evaluate_action(self, t, W, props_risky, delta_p, returns_samples):
        W_next, props_next = self._compute_next_states(
            W, props_risky, delta_p, returns_samples
        )

        if t + 1 == self.T:
            values = self.utility(W_next)
        else:
            features = self._make_features(W_next, props_next)
            values = self.V_approx[t + 1].predict_V(features)

        return np.mean(values)

    # -------- 生成候选动作（网格搜索）--------
    def _generate_candidate_actions(self, props_risky, n_grid=21):
        """
        生成满足约束的候选动作集合
        n=1: 细网格
        n>1: 组合网格
        """
        n = self.n
        max_adj = self.max_adjust

        if n == 1:
            # 1D 细网格
            lb = max(-props_risky[0], -max_adj)
            ub = min(1.0 - props_risky[0], max_adj)
            # 还要确保 cash = 1 - (props+dp) >= 0
            ub = min(ub, 1.0 - props_risky[0])
            grid = np.linspace(lb, ub, n_grid * 2)
            candidates = grid.reshape(-1, 1)
            return candidates

        # 多资产: 在约束内均匀采样
        candidates = [np.zeros(n)]  # 零动作

        # 各维度独立扫描
        for k in range(n):
            lb_k = max(-props_risky[k], -max_adj)
            ub_k = min(1.0 - props_risky[k], max_adj)
            for v in np.linspace(lb_k, ub_k, n_grid):
                dp = np.zeros(n)
                dp[k] = v
                if np.sum(np.abs(dp)) <= max_adj:
                    new_p = props_risky + dp
                    if np.all(new_p >= 0) and np.sum(new_p) <= 1.0:
                        candidates.append(dp.copy())

        # 随机组合
        for _ in range(500):
            dp = np.random.uniform(-max_adj / n, max_adj / n, n)
            l1 = np.sum(np.abs(dp))
            if l1 > max_adj:
                dp *= max_adj / l1
            new_p = props_risky + dp
            if np.all(new_p >= 0) and np.sum(new_p) <= 1.0:
                candidates.append(dp.copy())

        return np.array(candidates)

    # -------- 求最优动作（网格搜索 + 精炼）--------
    def optimize_action(self, t, W, props_risky, returns_samples=None):
        """网格搜索找最优动作，然后用 SLSQP 局部精炼"""
        n = self.n

        if returns_samples is None:
            returns_samples = np.random.normal(
                self.means[np.newaxis, :],
                self.stds[np.newaxis, :],
                size=(self.n_mc, self.n)
            )

        # 第一步: 网格搜索
        candidates = self._generate_candidate_actions(props_risky)
        best_val = -np.inf
        best_dp = np.zeros(n)

        for dp in candidates:
            val = self._evaluate_action(t, W, props_risky, dp, returns_samples)
            if val > best_val:
                best_val = val
                best_dp = dp.copy()

        # 第二步: 局部精炼
        def neg_value(delta_p):
            return -self._evaluate_action(
                t, W, props_risky, delta_p, returns_samples
            )

        bounds = []
        for k in range(n):
            lb = max(-props_risky[k], -self.max_adjust)
            ub = min(1.0 - props_risky[k], self.max_adjust)
            bounds.append((lb, ub))

        constraints = [
            {'type': 'ineq',
             'fun': lambda dp: self.max_adjust - np.sum(np.abs(dp))},
            {'type': 'ineq',
             'fun': lambda dp: 1.0 - np.sum(props_risky + dp)},
        ]

        try:
            result = minimize(
                neg_value, best_dp, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if result.success and -result.fun > best_val:
                best_val = -result.fun
                best_dp = result.x.copy()
        except Exception:
            pass

        return best_dp, best_val

    # -------- 辅助 --------
    def _make_features(self, W, props):
        if np.isscalar(W) or (isinstance(W, np.ndarray) and W.ndim == 0):
            W = np.array([W])
            props = props.reshape(1, -1)
        return np.column_stack([W.reshape(-1, 1), props])

    def _sample_states(self, t, n_samples, W0, p_init):
        """
        采样覆盖宽范围的 (W, props)
        关键: W 范围要足够宽，覆盖优化器可能探索的区域
        """
        # W 范围: 根据时间步，覆盖多个标准差
        # 即使在 t=0 也要覆盖宽范围，因为后续的 V 拟合要在宽 W 上准确
        log_W_center = np.log(W0) + self.r * t
        log_W_std = max(0.8, 0.3 * (t + 1))

        W_samples = np.exp(
            np.random.normal(log_W_center, log_W_std, n_samples)
        )
        W_samples = np.maximum(W_samples, 0.01)

        # 比例: 覆盖全范围
        p_samples = np.random.dirichlet(
            np.ones(self.n + 1) * 2.0, n_samples
        )[:, 1:]  # 去掉 cash 列，得到风险资产比例

        # 混入初始比例附近的样本
        n_near = n_samples // 3
        p_near = np.tile(p_init, (n_near, 1))
        noise = np.random.normal(0, 0.1, (n_near, self.n))
        p_near = np.clip(p_near + noise, 0.01, None)
        for i in range(n_near):
            if np.sum(p_near[i]) > 0.98:
                p_near[i] *= 0.98 / np.sum(p_near[i])
        p_samples[:n_near] = p_near

        return W_samples, p_samples

    # -------- 主求解 --------
    def solve(self, n_train=1000, W0=1.0, p_init=None, verbose=True):
        if p_init is None:
            p_init = np.ones(self.n) / (self.n + 1)

        for t in range(self.T - 1, -1, -1):
            if verbose:
                print(f"    Solving t={t} ...", end=" ", flush=True)

            W_samples, p_samples = self._sample_states(t, n_train, W0, p_init)

            # 预生成一组大的 MC 样本供所有训练点共用
            # 每个训练点用不同的子集
            master_returns = np.random.normal(
                self.means[np.newaxis, :],
                self.stds[np.newaxis, :],
                size=(self.n_mc, self.n)
            )

            values = np.zeros(n_train)
            actions = np.zeros((n_train, self.n))

            for i in range(n_train):
                dp_opt, val_opt = self.optimize_action(
                    t, W_samples[i], p_samples[i], master_returns
                )
                values[i] = val_opt
                actions[i] = dp_opt

            # 安全检查: V 必须 < 0
            if np.any(values >= 0):
                n_bad = np.sum(values >= 0)
                if verbose:
                    print(f"[WARNING: {n_bad} non-negative V values, clipping]",
                          end=" ")
                values = np.minimum(values, -1e-20)

            # 拟合 log(-V) 
            features = self._make_features(W_samples, p_samples)
            approx = ValueFunctionApproximator(degree=2)
            r2 = approx.fit(features, values)
            self.V_approx[t] = approx
            self.policies[t] = (W_samples, p_samples, actions, values)

            if verbose:
                print(f"V mean={np.mean(values):.6f}, "
                      f"std={np.std(values):.6f}, R²={r2:.4f}")

    # -------- 前向模拟 --------
    def simulate_optimal(self, W0, p_init, n_sims=500, verbose=False):
        W = np.full(n_sims, W0)
        props = np.tile(p_init, (n_sims, 1))

        for t in range(self.T):
            if verbose:
                print(f"    Simulating t={t}, mean W={np.mean(W):.4f}")
            for i in range(n_sims):
                dp_opt, _ = self.optimize_action(t, W[i], props[i])
                new_props = props[i] + dp_opt
                p0 = 1.0 - np.sum(new_props)

                returns = np.random.normal(self.means, self.stds)
                growth = p0 * (1 + self.r) + np.sum(new_props * (1 + returns))
                growth = max(growth, 1e-8)

                W[i] *= growth
                for k in range(self.n):
                    props[i, k] = new_props[k] * (1 + returns[k]) / growth
                props[i] = np.clip(props[i], 0, 1)

        return W, np.mean(self.utility(W))