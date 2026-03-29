"""
多资产演示: n=2, 3, 4; T<10; n<5
展示程序对任意合理参数均可工作
"""

import numpy as np
import matplotlib.pyplot as plt
from core import AssetAllocationADP


def run_multi_asset_demo(n_assets, T, r, means, variances,
                         risk_aversion, p_init, W0=1.0, n_train=200):
    """运行一次多资产 ADP 求解并展示结果"""

    print(f"\n  n={n_assets}, T={T}, r={r}")
    print(f"  means     = {means}")
    print(f"  variances = {variances}")
    print(f"  p_init    = {list(np.round(p_init, 4))} "
          f"(cash={1 - sum(p_init):.4f})")
    print(f"  risk_aversion = {risk_aversion}")
    print()

    adp = AssetAllocationADP(
        n_assets=n_assets, T=T, r=r,
        means=means, variances=variances,
        risk_aversion=risk_aversion,
        max_adjust=0.10,
        n_mc=800
    )

    np.random.seed(42)
    adp.solve(n_train=n_train, W0=W0, p_init=p_init, verbose=True)

    # ---------- t=0 最优动作 ----------
    print(f"\n  [t=0, W={W0}] 最优调仓:")
    dp_opt, V_opt = adp.optimize_action(0, W0, p_init)
    new_p = p_init + dp_opt
    print(f"    Δp       = {np.round(dp_opt, 5)}")
    print(f"    new p    = {np.round(new_p, 5)}")
    print(f"    new cash = {1 - np.sum(new_p):.5f}")
    print(f"    |Δp|_sum = {np.sum(np.abs(dp_opt)):.5f}")
    print(f"    V*(s_0)  = {V_opt:.6f}")

    # ---------- 确定性轨迹 ----------
    print(f"\n  [确定性轨迹（用均值收益模拟）]")
    W = W0
    props = p_init.copy()
    trajectory_W = [W]
    trajectory_p = [props.copy()]
    trajectory_cash = [1.0 - np.sum(props)]

    for t in range(T):
        dp, V = adp.optimize_action(t, W, props)
        new_props = props + dp
        p0 = 1.0 - np.sum(new_props)

        asset_str = ", ".join(
            [f"p{k+1}={new_props[k]:.4f}" for k in range(n_assets)]
        )
        print(f"    t={t}: W={W:.4f}, {asset_str}, cash={p0:.4f}, "
              f"|Δp|={np.sum(np.abs(dp)):.4f}")

        # 用均值做确定性前推
        means_arr = np.array(means)
        growth = p0 * (1 + r) + np.sum(new_props * (1 + means_arr))
        W *= growth
        props = new_props * (1 + means_arr) / growth
        props = np.clip(props, 0, 1)

        trajectory_W.append(W)
        trajectory_p.append(props.copy())
        trajectory_cash.append(1.0 - np.sum(props))

    print(f"    终端: W_T = {W:.4f}, U(W_T) = {adp.utility(W):.6f}")

    return adp, trajectory_W, trajectory_p, trajectory_cash


def demo_2assets():
    print("=" * 65)
    print("  演示 1: n=2 资产, T=5")
    print("=" * 65)
    return run_multi_asset_demo(
        n_assets=2, T=5, r=0.02,
        means=[0.08, 0.12],
        variances=[0.04, 0.09],
        risk_aversion=2.0,
        p_init=np.array([0.3, 0.3]),
        n_train=200
    )


def demo_3assets():
    print("\n" + "=" * 65)
    print("  演示 2: n=3 资产, T=6")
    print("=" * 65)
    return run_multi_asset_demo(
        n_assets=3, T=6, r=0.03,
        means=[0.06, 0.10, 0.14],
        variances=[0.02, 0.05, 0.10],
        risk_aversion=1.5,
        p_init=np.array([0.25, 0.25, 0.25]),
        n_train=150
    )


def demo_4assets():
    print("\n" + "=" * 65)
    print("  演示 3: n=4 资产, T=8")
    print("=" * 65)
    return run_multi_asset_demo(
        n_assets=4, T=8, r=0.02,
        means=[0.05, 0.08, 0.11, 0.15],
        variances=[0.01, 0.04, 0.06, 0.12],
        risk_aversion=2.0,
        p_init=np.array([0.15, 0.20, 0.25, 0.20]),
        n_train=120
    )


def demo_custom(n_assets, T, r, means, variances, risk_aversion, p_init):
    """用户自定义参数"""
    print("\n" + "=" * 65)
    print(f"  自定义演示: n={n_assets}, T={T}")
    print("=" * 65)
    return run_multi_asset_demo(
        n_assets=n_assets, T=T, r=r,
        means=means, variances=variances,
        risk_aversion=risk_aversion,
        p_init=np.array(p_init),
        n_train=150
    )


def plot_all_results(all_results, filename='demo_multi.png'):
    """
    画所有演示的财富轨迹 + 比例堆叠图

    all_results: list of (label, adp, traj_W, traj_p, traj_cash)
    """
    n_demos = len(all_results)
    fig, axes = plt.subplots(2, n_demos, figsize=(6 * n_demos, 10))
    if n_demos == 1:
        axes = axes.reshape(-1, 1)

    for col, (label, adp, traj_W, traj_p, traj_cash) in enumerate(all_results):
        T_total = len(traj_W)
        t_range = range(T_total)
        n = adp.n

        # ---- 上图：财富轨迹 ----
        ax_top = axes[0, col]
        ax_top.plot(t_range, traj_W, 'b-o', linewidth=2, markersize=6)
        ax_top.set_xlabel('Time t')
        ax_top.set_ylabel('Wealth W')
        ax_top.set_title(f'{label}\nWealth Trajectory')
        ax_top.grid(True, alpha=0.3)

        # ---- 下图：比例堆叠柱状图 ----
        ax_bot = axes[1, col]
        traj_p_arr = np.array(traj_p)
        traj_cash_arr = np.array(traj_cash)

        colors = plt.cm.Set2(np.linspace(0, 1, n + 1))
        bottom = np.zeros(T_total)

        for k in range(n):
            ax_bot.bar(t_range, traj_p_arr[:, k], bottom=bottom,
                       color=colors[k], label=f'Asset {k+1}', alpha=0.8)
            bottom += traj_p_arr[:, k]

        ax_bot.bar(t_range, traj_cash_arr, bottom=bottom,
                   color=colors[n], label='Cash', alpha=0.8)

        ax_bot.set_xlabel('Time t')
        ax_bot.set_ylabel('Portfolio Proportion')
        ax_bot.set_title(f'{label}\nAllocation Over Time')
        ax_bot.legend(loc='upper right', fontsize=8)
        ax_bot.set_ylim(0, 1.05)
        ax_bot.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"\n  图已保存: {filename}")


if __name__ == "__main__":
    all_results = []

    adp2, tW2, tp2, tc2 = demo_2assets()
    all_results.append(("n=2, T=5", adp2, tW2, tp2, tc2))

    adp3, tW3, tp3, tc3 = demo_3assets()
    all_results.append(("n=3, T=6", adp3, tW3, tp3, tc3))

    adp4, tW4, tp4, tc4 = demo_4assets()
    all_results.append(("n=4, T=8", adp4, tW4, tp4, tc4))

    plot_all_results(all_results, filename='demo_multi.png')