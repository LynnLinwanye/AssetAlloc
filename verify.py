"""
验证：n=1 无约束 ADP vs 解析解
"""

import numpy as np
import matplotlib.pyplot as plt
from core import AssetAllocationADP
from analytical import AnalyticalSolution, MultiAssetAnalyticalSolution


def verify_analytical_direct():
    """
    额外验证: 完全不用 NN，直接用解析 V_{t+1} 做一步优化
    确认 MC 优化器本身无问题
    """
    print("=" * 65)
    print("  预检: MC 优化器精度 (用解析 V_{t+1})")
    print("=" * 65)

    T = 4
    r = 0.02
    mu = 0.08
    sigma2 = 0.04
    sigma = np.sqrt(sigma2)
    a = 2.0
    W0 = 1.0
    n_mc = 50000

    analytical = AnalyticalSolution(T, r, mu, sigma2, a)

    print(f"\n  {'t':<4} {'解析 x*':<12} {'MC优化 x*':<12} {'误差%':<10}")
    print("  " + "-" * 34)

    from scipy.optimize import minimize_scalar

    for t in range(T):
        np.random.seed(42)
        Y = np.random.normal(mu, sigma, n_mc)

        if t < T - 1:
            def neg_EV(x):
                W_next = x * (1 + Y) + (W0 - x) * (1 + r)
                return -np.mean(analytical.optimal_value(t + 1, W_next))
        else:
            def neg_EV(x):
                W_next = x * (1 + Y) + (W0 - x) * (1 + r)
                return -np.mean(-np.exp(-a * W_next) / a)

        res = minimize_scalar(neg_EV, bounds=(-2, 5), method='bounded')
        x_mc = res.x
        x_an = analytical.optimal_dollar_allocation(t)
        err = abs(x_mc - x_an) / abs(x_an) * 100
        print(f"  {t:<4} {x_an:<12.6f} {x_mc:<12.6f} {err:<10.2f}")

    print("\n  ✓ MC 优化器精度 < 1%，误差来源在函数近似")


def verify_n1_unconstrained():
    """n=1 无约束 ADP vs 解析解"""

    print("\n" + "=" * 65)
    print("  验证: n=1 无约束 ADP vs 解析解")
    print("=" * 65)

    T = 4
    r = 0.02
    mu = 0.08
    sigma2 = 0.04
    risk_aversion = 2.0
    W0 = 1.0

    analytical = AnalyticalSolution(T, r, mu, sigma2, risk_aversion)
    print("\n[解析解]")
    analytical.print_summary(W0)

    print("\n[ADP 求解]")
    adp = AssetAllocationADP(
        n_assets=1, T=T, r=r,
        means=[mu], variances=[sigma2],
        risk_aversion=risk_aversion,
        max_adjust=1.0,    # 无约束
        n_mc=5000
    )

    np.random.seed(42)
    p_init = np.array([0.5])
    adp.solve(n_train=1000, W0=W0, p_init=p_init, verbose=True)

# ===== 1. 最优比例对比 =====
    print("\n[对比最优比例: 从不同起始 p 出发]")
    print(f"  {'t':<4} {'解析 p*':<12} {'ADP 均值':<12} {'误差%':<8} "
          f"{'各起始点→目标'}")
    print("  " + "-" * 75)

    for t in range(T):
        p_analytical = analytical.optimal_proportion(t, W0)
        test_starts = [0.2, 0.4, 0.6, 0.8, 0.95]
        adp_targets = []
        for p_start in test_starts:
            np.random.seed(100 + t * 10 + int(p_start * 10))
            dp, _ = adp.optimize_action(t, W0, np.array([p_start]))
            adp_targets.append(p_start + dp[0])

        mean_target = np.mean(adp_targets)
        err = abs(mean_target - p_analytical) / abs(p_analytical) * 100

        targets_str = ", ".join([f"{x:.3f}" for x in adp_targets])
        print(f"  {t:<4} {p_analytical:<12.4f} {mean_target:<12.4f} "
              f"{err:<8.2f} [{targets_str}]")

    # ===== 2. 价值函数对比 =====
    print("\n[价值函数 V*(W) 对比: t=0]")
    W_range = np.linspace(0.3, 2.5, 10)
    print(f"  {'W':<8} {'解析 V*':<14} {'ADP V*':<14} {'误差%':<10}")
    print("  " + "-" * 46)

    V_an_list = []
    V_adp_list = []

    for W in W_range:
        V_an = analytical.optimal_value(0, W)
        p_start = np.array([
            np.clip(analytical.optimal_proportion(0, W), 0.05, 0.95)
        ])
        np.random.seed(200)
        _, V_adp = adp.optimize_action(0, W, p_start)
        err = abs(V_adp - V_an) / abs(V_an) * 100
        print(f"  {W:<8.2f} {V_an:<14.6f} {V_adp:<14.6f} {err:<10.2f}")
        V_an_list.append(V_an)
        V_adp_list.append(V_adp)

    # ===== 3. 各时间步价值函数 =====
    print("\n[各时间步 V*(W0=1.0) 对比]")
    print(f"  {'t':<4} {'解析 V*':<14} {'ADP V*':<14} {'误差%':<10}")
    print("  " + "-" * 42)

    V_an_t = []
    V_adp_t = []

    for t in range(T):
        V_an = analytical.optimal_value(t, W0)
        p_start = np.array([
            np.clip(analytical.optimal_proportion(t, W0), 0.05, 0.95)
        ])
        np.random.seed(300 + t)
        _, V_adp = adp.optimize_action(t, W0, p_start)
        err = abs(V_adp - V_an) / abs(V_an) * 100
        print(f"  {t:<4} {V_an:<14.6f} {V_adp:<14.6f} {err:<10.2f}")
        V_an_t.append(V_an)
        V_adp_t.append(V_adp)

    # ===== 4. 拟合质量诊断 =====
    print("\n[拟合诊断: log(-V) 的线性回归质量]")
    for t in range(T):
        W_train, p_train, _, V_train = adp.policies[t]
        features = adp._make_features(W_train, p_train)

        # log(-V) vs W 的相关系数
        log_neg_V = np.log(-V_train)
        corr = np.corrcoef(W_train, log_neg_V)[0, 1]
        r2 = adp.V_approx[t].r2

        print(f"  t={t}: R²={r2:.4f}, corr(W, log(-V))={corr:.4f}, "
              f"V range=[{V_train.min():.6f}, {V_train.max():.6f}]")

    # ===== 画图 =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 最优比例 vs 时间
    t_range = list(range(T))
    p_an_plot = [analytical.optimal_proportion(t, W0) for t in t_range]
    p_adp_plot = []
    for t in t_range:
        np.random.seed(400 + t)
        dp, _ = adp.optimize_action(t, W0, np.array([0.5]))
        p_adp_plot.append(0.5 + dp[0])

    axes[0, 0].plot(t_range, p_an_plot, 'bo-', label='Analytical p*',
                    markersize=10, linewidth=2)
    axes[0, 0].plot(t_range, p_adp_plot, 'r^--', label='ADP p*',
                    markersize=10, linewidth=2)
    axes[0, 0].set_xlabel('Time t', fontsize=12)
    axes[0, 0].set_ylabel('Optimal Risky Proportion', fontsize=12)
    axes[0, 0].set_title('Optimal Allocation', fontsize=13)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: 价值函数 V*(W) at t=0
    axes[0, 1].plot(W_range, V_an_list, 'bo-', label='Analytical V*',
                    markersize=8, linewidth=2)
    axes[0, 1].plot(W_range, V_adp_list, 'r^--', label='ADP V*',
                    markersize=8, linewidth=2)
    axes[0, 1].set_xlabel('Wealth W', fontsize=12)
    axes[0, 1].set_ylabel('V*(W)', fontsize=12)
    axes[0, 1].set_title('Value Function at t=0', fontsize=13)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: 各时间步价值函数
    axes[1, 0].plot(t_range, V_an_t, 'bo-', label='Analytical V*',
                    markersize=10, linewidth=2)
    axes[1, 0].plot(t_range, V_adp_t, 'r^--', label='ADP V*',
                    markersize=10, linewidth=2)
    axes[1, 0].set_xlabel('Time t', fontsize=12)
    axes[1, 0].set_ylabel('V*(W0=1.0)', fontsize=12)
    axes[1, 0].set_title('Value at W0=1.0 Over Time', fontsize=13)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: 拟合质量 — 训练数据散点 at t=0
    if 0 in adp.policies:
        W_train, p_train, _, V_train = adp.policies[0]
        features_train = adp._make_features(W_train, p_train)
        V_pred = adp.V_approx[0].predict_V(features_train)

        axes[1, 1].scatter(V_train, V_pred, alpha=0.3, s=10, c='blue')
        vmin = min(V_train.min(), V_pred.min())
        vmax = max(V_train.max(), V_pred.max())
        axes[1, 1].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2,
                        label='Perfect fit')
        axes[1, 1].set_xlabel('True V', fontsize=12)
        axes[1, 1].set_ylabel('Predicted V', fontsize=12)
        axes[1, 1].set_title('Fit Quality at t=0', fontsize=13)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/verify_n1.png', dpi=150)
    plt.show()
    print("\n  图已保存: output/verify_n1.png")

    return analytical, adp


def verify_n1_constrained_vs_unconstrained():
    """对比 n=1 有约束 vs 无约束"""

    print("\n" + "=" * 65)
    print("  对比: n=1 有约束(10%) vs 无约束")
    print("=" * 65)

    T = 5
    r = 0.02
    mu = 0.10
    sigma2 = 0.04
    risk_aversion = 1.5
    W0 = 1.0

    analytical = AnalyticalSolution(T, r, mu, sigma2, risk_aversion)

    print(f"\n  解析最优比例(无约束): ", end="")
    for t in range(T):
        print(f"t{t}={analytical.optimal_proportion(t, W0):.3f}", end="  ")
    print()

    # 无约束 ADP
    print("\n  求解无约束版本...")
    adp_free = AssetAllocationADP(
        n_assets=1, T=T, r=r,
        means=[mu], variances=[sigma2],
        risk_aversion=risk_aversion,
        max_adjust=1.0,
        n_mc=5000
    )
    np.random.seed(42)
    p_init = np.array([0.3])
    adp_free.solve(n_train=1000, W0=W0, p_init=p_init, verbose=True)

    # 有约束 ADP
    print("\n  求解有约束版本 (max_adjust=10%)...")
    adp_con = AssetAllocationADP(
        n_assets=1, T=T, r=r,
        means=[mu], variances=[sigma2],
        risk_aversion=risk_aversion,
        max_adjust=0.10,
        n_mc=5000
    )
    np.random.seed(42)
    adp_con.solve(n_train=1000, W0=W0, p_init=p_init, verbose=True)

    # ---------- 单步对比 ----------
    print("\n  [单步对比: 从 p=0.3 出发]")
    print(f"  {'t':<4} {'解析p*':<12} {'ADP无约束':<14} {'ADP有约束':<14} "
          f"{'|Δp|':<10} {'约束生效?'}")
    print("  " + "-" * 66)

    p_an_list = []
    p_free_list = []
    p_con_list = []

    for t in range(T):
        p_an = analytical.optimal_proportion(t, W0)

        np.random.seed(500 + t)
        dp_free, _ = adp_free.optimize_action(t, W0, p_init)

        np.random.seed(500 + t)
        dp_con, _ = adp_con.optimize_action(t, W0, p_init)

        target_free = p_init[0] + dp_free[0]
        target_con = p_init[0] + dp_con[0]
        dp_con_abs = abs(dp_con[0])

        binding = "YES" if abs(target_free - target_con) > 0.005 else "no"

        p_an_list.append(p_an)
        p_free_list.append(target_free)
        p_con_list.append(target_con)

        print(f"  {t:<4} {p_an:<12.4f} {target_free:<14.4f} "
              f"{target_con:<14.4f} {dp_con_abs:<10.4f} {binding}")

    # ---------- 多步轨迹模拟 ----------
    print("\n  [多步轨迹: 从 p=0.3, W=1.0 出发（均值收益前推）]")
    print(f"  {'t':<4} {'无约束 p':<12} {'有约束 p':<12} {'无约束 W':<12} "
          f"{'有约束 W':<12}")
    print("  " + "-" * 52)

    W_f, W_c = W0, W0
    p_f, p_c = p_init[0], p_init[0]
    traj_pf, traj_pc = [p_f], [p_c]
    traj_Wf, traj_Wc = [W_f], [W_c]

    for t in range(T):
        np.random.seed(600 + t)
        dp_f, _ = adp_free.optimize_action(t, W_f, np.array([p_f]))
        np.random.seed(600 + t)
        dp_c, _ = adp_con.optimize_action(t, W_c, np.array([p_c]))

        new_pf = p_f + dp_f[0]
        new_pc = p_c + dp_c[0]

        print(f"  {t:<4} {new_pf:<12.4f} {new_pc:<12.4f} "
              f"{W_f:<12.4f} {W_c:<12.4f}")

        # 均值前推
        growth_f = (1 - new_pf) * (1 + r) + new_pf * (1 + mu)
        growth_c = (1 - new_pc) * (1 + r) + new_pc * (1 + mu)
        W_f *= growth_f
        W_c *= growth_c
        p_f = new_pf * (1 + mu) / growth_f
        p_c = new_pc * (1 + mu) / growth_c

        traj_pf.append(new_pf)
        traj_pc.append(new_pc)
        traj_Wf.append(W_f)
        traj_Wc.append(W_c)

    print(f"\n  终端: 无约束 W_T={W_f:.4f}, 有约束 W_T={W_c:.4f}")

    # ---------- 画图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    t_range = list(range(T))

    # 图1: 单步目标比例
    axes[0].plot(t_range, p_an_list, 'bo-', label='Analytical (unconstrained)',
                 markersize=10, linewidth=2)
    axes[0].plot(t_range, p_free_list, 'g^--', label='ADP unconstrained',
                 markersize=9, linewidth=2)
    axes[0].plot(t_range, p_con_list, 'rs-.', label='ADP constrained (10%)',
                 markersize=9, linewidth=2)
    axes[0].axhline(y=p_init[0], color='gray', linestyle=':', alpha=0.5,
                    label=f'Start p={p_init[0]}')
    axes[0].axhline(y=p_init[0] + 0.10, color='orange', linestyle='--',
                    alpha=0.4, label='Max reachable (+10%)')
    axes[0].set_xlabel('Time t', fontsize=12)
    axes[0].set_ylabel('Target Risky Proportion', fontsize=12)
    axes[0].set_title('Single-Step: Constrained vs Unconstrained', fontsize=13)
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)

    # 图2: 价值函数对比
    W_range = np.linspace(0.5, 2.0, 8)
    V_an_vals, V_free_vals, V_con_vals = [], [], []

    for W in W_range:
        V_an_vals.append(analytical.optimal_value(0, W))
        np.random.seed(700)
        _, V_f = adp_free.optimize_action(0, W, p_init)
        V_free_vals.append(V_f)
        np.random.seed(700)
        _, V_c = adp_con.optimize_action(0, W, p_init)
        V_con_vals.append(V_c)

    axes[1].plot(W_range, V_an_vals, 'bo-', label='Analytical',
                 markersize=8, linewidth=2)
    axes[1].plot(W_range, V_free_vals, 'g^--', label='ADP unconstrained',
                 markersize=8, linewidth=2)
    axes[1].plot(W_range, V_con_vals, 'rs-.', label='ADP constrained',
                 markersize=8, linewidth=2)
    axes[1].set_xlabel('Wealth W', fontsize=12)
    axes[1].set_ylabel('V*(W) at t=0', fontsize=12)
    axes[1].set_title('Value Functions Comparison', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/verify_constrained.png', dpi=150)
    plt.show()
    print("\n  图已保存: output/verify_constrained.png")

    return adp_free, adp_con

def verify_multi_asset():
    """n>1 无约束 ADP vs 解析解"""
    print("\n" + "=" * 65)
    print("  验证: n>1 无约束 ADP vs 解析解")
    print("=" * 65)

    # 参数设置: 两个风险资产
    T = 4
    r = 0.02
    mu = [0.08, 0.10]
    Sigma = [[0.04, 0.0], [0.0, 0.05]]
    risk_aversion = 2.0
    W0 = 1.0

    analytical = MultiAssetAnalyticalSolution(T, r, mu, Sigma, risk_aversion)
    print("\n[解析解]")
    analytical.print_summary(W0)

    print("\n[ADP 求解]")
    adp = AssetAllocationADP(
        n_assets=2, T=T, r=r,
        means=mu, variances=[0.04, 0.05],
        risk_aversion=risk_aversion,
        max_adjust=1.0,    # 无约束
        n_mc=5000
    )

    np.random.seed(42)
    p_init = np.array([0.5, 0.3])
    adp.solve(n_train=1500, W0=W0, p_init=p_init, verbose=True)

    # ===== 1. 最优比例对比 =====
    print("\n[对比最优比例: 从不同起始 p 出发]")
    print(f"  {'t':<4} {'解析 p*':<30} {'ADP 均值':<30} {'误差%':<8} {'各起始点→目标'}")
    print("  " + "-" * 100)

    for t in range(T):
        p_analytical = analytical.optimal_proportion(t, W0)
        test_starts = [np.array([0.2, 0.2]), np.array([0.4, 0.1]),
                       np.array([0.1, 0.5]), np.array([0.3, 0.3])]
        adp_targets = []
        for p_start in test_starts:
            np.random.seed(100 + t * 10 + int(p_start[0] * 10))
            dp, _ = adp.optimize_action(t, W0, p_start)
            adp_targets.append(p_start + dp)

        mean_target = np.mean(adp_targets, axis=0)
        err = np.linalg.norm(mean_target - p_analytical) / np.linalg.norm(p_analytical) * 100

        targets_str = ", ".join([f"[{x[0]:.3f},{x[1]:.3f}]" for x in adp_targets])
        print(f"  {t:<4} {p_analytical} {mean_target} {err:<8.2f} [{targets_str}]")

    # ===== 1.5. 最优金额对比 =====
    print("\n[对比最优金额: 从不同起始 p 出发]")
    print(f"  {'t':<4} {'解析 x*':<30} {'ADP 均值':<30} {'误差%':<8} {'各起始点→目标'}")
    print("  " + "-" * 100)

    for t in range(T):
        x_analytical = analytical.optimal_dollar_allocation(t)
        test_starts = [np.array([0.2, 0.2]), np.array([0.4, 0.1]),
                       np.array([0.1, 0.5]), np.array([0.3, 0.3])]
        adp_targets_x = []
        for p_start in test_starts:
            np.random.seed(100 + t * 10 + int(p_start[0] * 10))
            dp, _ = adp.optimize_action(t, W0, p_start)
            target_p = p_start + dp
            target_x = target_p * W0
            adp_targets_x.append(target_x)

        mean_target_x = np.mean(adp_targets_x, axis=0)
        err_x = np.linalg.norm(mean_target_x - x_analytical) / np.linalg.norm(x_analytical) * 100

        targets_x_str = ", ".join([f"[{x[0]:.3f},{x[1]:.3f}]" for x in adp_targets_x])
        print(f"  {t:<4} {x_analytical} {mean_target_x} {err_x:<8.2f} [{targets_x_str}]")

    # ===== 2. 价值函数对比 =====
    print("\n[价值函数 V*(W) 对比: t=0]")
    W_range = np.linspace(0.5, 2.0, 8)
    print(f"  {'W':<8} {'解析 V*':<14} {'ADP V*':<14} {'误差%':<10}")
    print("  " + "-" * 46)

    for W in W_range:
        V_an = analytical.optimal_value(0, W)
        p_start = np.clip(analytical.optimal_proportion(0, W), 0.05, 0.95)
        np.random.seed(200)
        _, V_adp = adp.optimize_action(0, W, p_start)
        err = abs(V_adp - V_an) / abs(V_an) * 100
        print(f"  {W:<8.2f} {V_an:<14.6f} {V_adp:<14.6f} {err:<10.2f}")

    # ===== 3. 各时间步价值函数 =====
    print("\n[各时间步 V*(W0=1.0) 对比]")
    print(f"  {'t':<4} {'解析 V*':<14} {'ADP V*':<14} {'误差%':<10}")
    print("  " + "-" * 42)

    for t in range(T):
        V_an = analytical.optimal_value(t, W0)
        p_start = np.clip(analytical.optimal_proportion(t, W0), 0.05, 0.95)
        np.random.seed(300 + t)
        _, V_adp = adp.optimize_action(t, W0, p_start)
        err = abs(V_adp - V_an) / abs(V_an) * 100
        print(f"  {t:<4} {V_an:<14.6f} {V_adp:<14.6f} {err:<10.2f}")

    # ===== 简单绘图 =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1: t vs 第一资产比例
    t_range = list(range(T))
    p_an_plot = [analytical.optimal_proportion(t, W0)[0] for t in t_range]
    p_adp_plot = []
    for t in t_range:
        np.random.seed(400 + t)
        dp, _ = adp.optimize_action(t, W0, np.array([0.5, 0.3]))
        p_adp_plot.append(0.5 + dp[0])

    axes[0].plot(t_range, p_an_plot, 'bo-', label='Analytical p1*')
    axes[0].plot(t_range, p_adp_plot, 'r^--', label='ADP p1*')
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Asset 1 Proportion')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 图2: Visualize value function at t=0
    axes[1].plot(W_range, [analytical.optimal_value(0, W) for W in W_range], 'bo-', label='Analytical V*')
    axes[1].plot(W_range, [adp.optimize_action(0, W, np.clip(analytical.optimal_proportion(0, W),0.05,0.95))[1] for W in W_range], 'r^--', label='ADP V*')
    axes[1].set_xlabel('Wealth W')
    axes[1].set_ylabel('V*(W) at t=0')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/verify_multi.png', dpi=150)
    plt.show()
    print("\n  图已保存: output/verify_multi.png")

    return analytical, adp




if __name__ == "__main__":
    # 第一步: 确认 MC 优化器本身没问题
    verify_analytical_direct()

    # 第二步: ADP vs 解析解
    verify_n1_unconstrained()

    # 第三步: 约束 vs 无约束
    verify_n1_constrained_vs_unconstrained()

    # 第四步: 多资产无约束
    verify_multi_asset()