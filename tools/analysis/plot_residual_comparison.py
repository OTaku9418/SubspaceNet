"""残差臂 A/B 对比图与 markdown 表（plan §5.4 / 决策 Q4）。

读同一场景下两条结构臂各一份结果 JSON —— `reproduce_array_mismatch.py --arch base`
与 `--arch res` 的产物 —— 画 SubspaceNet + Root-MUSIC 的 RMSPE–失配曲线，并给出
逐点数值、均值、相对提升。

为什么不复用 `reproduce_array_mismatch.plot_curve_figures`：那张图一个 JSON 一张
（文件名只含 scenario + metric），两条臂会互相覆盖；而且它按算法名上色，没有
"同一个算子、两种结构"这个概念。这里把 base 画成实线空心、res 画成虚线实心，
经典 Root-MUSIC 用灰点线做参考 —— 一眼能看出跳连是把曲线整体压下去还是只改了斜率。

图内文字用英文：与同目录的 fig9_*.png 保持一致，也避开 matplotlib 缺中文字形的坑。
表格用中文：最终是要贴进 REPRODUCE_ARRAY_MISMATCH.md 的。

产物（默认落在实验图目录，和 fig9_*.png 放一起）：
    <DATA_ROOT>/simulations/figures/residual_ab_<scenario>_deg.png
    <DATA_ROOT>/simulations/figures/residual_ab_<scenario>.md

用法：
    python tools/analysis/plot_residual_comparison.py \
        --base data/simulations/results/array_mismatch_gain_phase_xxx_base.json \
        --res  data/simulations/results/array_mismatch_gain_phase_yyy_res.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 只为了拿场景的 sweep_param / pair_scale / 横轴标题：JSON 的 config 里没存 pair_scale，
# 而横轴刻度上要标物理配对 (σ_g, σ_φ)，硬编码一份会跟场景配置漂移。
from reproduce_array_mismatch import (  # noqa: E402
    DATA_ROOT,
    PLOT_RCP,
    SCENARIOS,
    SWEEP_LABELS,
)

SUBNET_KEY = "SubNet+r-music"          # 两条臂都只比这一个算子（论文 Fig.9 的主曲线）
CLASSIC_KEY = "r-music"

COLORS = {"base": "tab:blue", "res": "tab:red", "classic": "gray"}
LABELS = {"base": "SubspaceNet (base)", "res": "SubspaceNet (residual)"}


def load_arm(path: Path, role: str) -> dict:
    """读一份结果 JSON 并核对它确实是 `role` 这条臂。

    角色认错了（把 base 当 res 传进来）不会报错，只会把图画反、表算反，所以这里
    用 config.arch 硬核对；真的传反了就自动换过来，同时打一行警告。
    """
    payload = json.loads(Path(path).read_text("utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        raise SystemExit(f"{path}: 没有 results 行")
    arch = payload.get("config", {}).get("arch")
    if arch is None:
        raise SystemExit(f"{path}: config.arch 缺失 —— 这是加 --arch 之前的旧结果，"
                         "不能拿来做 A/B")
    return {"path": Path(path), "role": role, "arch": arch, "payload": payload,
            "rows": rows}


def arm_role(arm: dict, other: dict) -> str:
    """按 JSON 里记录的 arch 决定这条臂在图上是 base 还是 res。"""
    arch = arm["arch"]
    if arch in LABELS:
        return arch
    raise SystemExit(f"{arm['path']}: 未知 arch={arch!r}，只认 {sorted(LABELS)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="残差臂 A/B 对比图与表")
    p.add_argument("--base", required=True, help="--arch base 跑出来的结果 JSON")
    p.add_argument("--res", required=True, help="--arch res 跑出来的结果 JSON")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="产物目录 (默认 <DATA_ROOT>/simulations/figures)")
    p.add_argument("--metric", choices=("deg", "ref"), default="deg",
                   help="deg=物理角度 (默认); ref=仓库 RMSPE 原始刻度 (含 deg/rad 约定)")
    args = p.parse_args(argv)

    a = load_arm(args.base, "base")
    b = load_arm(args.res, "res")
    # 允许 --base/--res 传反：按 JSON 里的 arch 覆盖角色，而不是按命令行位置。
    arms = {}
    for arm in (a, b):
        role = arm_role(arm, None)
        if role in arms:
            raise SystemExit(f"两份 JSON 都是 arch={role}，没有可比性")
        arms[role] = arm
    if arms["base"]["path"] == arms["res"]["path"]:
        raise SystemExit("两份 JSON 是同一个文件")
    if (a["arch"], b["arch"]) != ("base", "res"):
        print(f"[warn] 命令行顺序与 arch 不符，已按 JSON 自动纠正: "
              f"base={arms['base']['path'].name} res={arms['res']['path'].name}")

    base, res = arms["base"], arms["res"]
    cfg = base["payload"].get("config", {})
    res_cfg = res["payload"].get("config", {})
    scenario = base["payload"].get("scenario", "gain_phase")
    sweep = cfg.get("sweep_param")
    if not sweep:
        raise SystemExit("config.sweep_param 缺失")
    if res_cfg.get("sweep_param") != sweep:
        raise SystemExit(f"两条臂的横轴不同: {sweep} vs {res_cfg.get('sweep_param')}")
    # 同一次实验的硬约束：结构是唯一变量。种子/协议/超参不一致就不该放在一张图上。
    for key in ("init_seed", "epochs", "n_train", "n_test", "batch_size"):
        if cfg.get(key) != res_cfg.get(key):
            raise SystemExit(f"两条臂的 {key} 不一致: {cfg.get(key)} vs {res_cfg.get(key)}")
    for key in ("N", "M", "T", "snr", "tau"):
        if cfg.get(key) != res_cfg.get(key):
            raise SystemExit(f"两条臂的 {key} 不一致: {cfg.get(key)} vs {res_cfg.get(key)}")

    xb = [float(r[sweep]) for r in base["rows"]]
    xr = [float(r[sweep]) for r in res["rows"]]
    if len(xb) != len(xr) or any(abs(x - y) > 1e-12 for x, y in zip(xb, xr)):
        raise SystemExit(f"两条臂的评估网格不同:\n  base={xb}\n  res ={xr}")
    order = np.argsort(xb)
    x = np.asarray(xb, dtype=float)[order]

    def curve(arm, key):
        return np.asarray([r.get(f"{key}_{args.metric}", np.nan) for r in arm["rows"]],
                          dtype=float)[order]

    y_base = curve(base, SUBNET_KEY)
    y_res = curve(res, SUBNET_KEY)
    y_cls = curve(base, CLASSIC_KEY)
    y_cls_res = curve(res, CLASSIC_KEY)
    if np.any(np.isnan(y_base)) or np.any(np.isnan(y_res)):
        raise SystemExit("SubNet+r-music 曲线里有 NaN —— 检查算法名/指标名是否拼错")
    if not np.allclose(y_cls, y_cls_res, equal_nan=True):
        print("[warn] 两条臂的经典 Root-MUSIC 基线不同（理论上同测试集应当逐位一致）")

    improve = 100.0 * (y_base - y_res) / y_base
    mean_base, mean_res, mean_cls = y_base.mean(), y_res.mean(), y_cls.mean()
    mean_imp = 100.0 * (mean_base - mean_res) / mean_base

    # ---------------- markdown 表 ----------------
    pair_scale = (SCENARIOS.get(scenario) or {}).get("pair_scale")
    metric_unit = "°" if args.metric == "deg" else " (ref)"
    lines = [
        f"# 残差臂 A/B 对比：{scenario}（SubspaceNet + Root-MUSIC，RMSPE{metric_unit}）",
        "",
        f"- base: `{base['path'].name}`（arch={base['arch']}, "
        f"init_seed={cfg.get('init_seed')}, {cfg.get('n_train')} 训练 / "
        f"{cfg.get('epochs')} epoch）",
        f"- res:  `{res['path'].name}`（arch={res['arch']}, 其余超参相同）",
        f"- 提升 = (base - res) / base，正数表示残差臂更好；均值按网格点算术平均",
        "",
    ]
    head = "| $u$ | 配对 (增益, 相位) | base | res | 提升 | Root-MUSIC |" if pair_scale \
        else "| 失配水平 | base | res | 提升 | Root-MUSIC |"
    lines += [head, "|" + "---:|" * (6 if pair_scale else 5)]
    for i, u in enumerate(x):
        cells = [f"{u:g}"]
        if pair_scale:
            g, ph = pair_scale
            cells.append(f"{100.0 * g * u:.0f}%, {ph * u:.0f}°")
        cells += [f"{y_base[i]:.4f}", f"{y_res[i]:.4f}", f"{improve[i]:+.1f}%",
                  f"{y_cls[i]:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["**均值**"]
    if pair_scale:
        cells.append("")
    cells += [f"**{mean_base:.4f}**", f"**{mean_res:.4f}**", f"**{mean_imp:+.1f}%**",
              f"**{mean_cls:.4f}**"]
    lines += ["| " + " | ".join(cells) + " |", ""]

    fail_lines = []
    for role, arm in (("base", base), ("res", res)):
        fr = [r.get(f"{SUBNET_KEY}_fail", 0.0) for r in arm["rows"]]
        if any(v for v in fr):
            fail_lines.append(f"{role} fail rate: " + ", ".join(f"{v:.3f}" for v in fr))
    if fail_lines:
        lines += ["失败率（预测为 nan 的比例，非零说明曲线被「漏检」污染）：", ""]
        lines += [f"- {s}" for s in fail_lines] + [""]

    # ---------------- 图 ----------------
    fig_dir = Path(args.out_dir) if args.out_dir else DATA_ROOT / "simulations" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"residual_ab_{scenario}_{args.metric}"
    png, md = fig_dir / f"{tag}.png", fig_dir / f"{tag}.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    with plt.rc_context(PLOT_RCP):
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        ax.plot(x, y_base, "o-", color=COLORS["base"], mfc="none", ms=8, lw=1.8,
                label=f"{LABELS['base']}  mean {mean_base:.3f}{metric_unit}")
        ax.plot(x, y_res, "s--", color=COLORS["res"], mfc=COLORS["res"], ms=7, lw=1.8,
                label=f"{LABELS['res']}  mean {mean_res:.3f}{metric_unit}")
        ax.plot(x, y_cls, "^:", color=COLORS["classic"], mfc="none", ms=7, lw=1.4,
                label=f"Root-MUSIC (classic)  mean {mean_cls:.3f}{metric_unit}")
        if pair_scale:
            g, ph = pair_scale
            ax.set_xticks(x)
            ax.set_xticklabels([f"{u:g}\n({100.0 * g * u:.0f}%, {ph * u:.0f}°)"
                                for u in x], fontsize=8)
        else:
            ax.set_xticks(x)
        ax.set_xlabel(SWEEP_LABELS.get(sweep, f"array mismatch level ({sweep})"))
        ax.set_ylabel(f"RMSPE{' [deg]' if args.metric == 'deg' else ' (reference scale)'}")
        ax.set_title("Residual skip-connections: A/B under "
                     f"{'combined gain + phase' if scenario == 'gain_phase' else scenario} "
                     "mismatch\n"
                     f"SubspaceNet + Root-MUSIC | {cfg.get('n_train')} train / "
                     f"{cfg.get('epochs')} epochs | init_seed {cfg.get('init_seed')} | "
                     f"mean improvement {mean_imp:+.1f}%",
                     fontsize=11)
        ax.legend(fontsize=9)
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n".join(lines))
    print(f"[plot] {png}")
    print(f"[table] {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
