"""多种子版残差臂 A/B：均值±标准差曲线 + 配对差值表（plan §18 的后续）。

`plot_residual_comparison.py` 只吃一对 JSON（一个种子、两条臂）。这里吃多颗种子的
全部结果，回答两个它答不了的问题：

1. 逐点的臂间差是不是稳定地偏向某一侧（看每颗种子的符号，而不是只看均值）；
2. 这个差有多大，和种子之间的散布比是否显著。

图：左panel 是两条臂的均值曲线 + ±1 sd 带（经典 Root-MUSIC 作灰参考），右panel
是逐点的配对相对提升（每颗种子一条细线 + 均值粗线）。
表：markdown，逐点给出 base/res 的 mean±sd、每颗种子的提升、以及"res 更好的种子数"。

n=3 时不做显著性检验：三颗种子的符号与散布比 p 值诚实。sd 用样本标准差（ddof=1）。

产物（默认与 fig9 图放一起）：
    <DATA_ROOT>/simulations/figures/residual_seeds_<scenario>_deg.{png,md}

用法：
    python tools/analysis/plot_residual_seeds.py --epochs 40 --seeds 1234 2024 3407
"""
from __future__ import annotations

import argparse
import glob
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

from reproduce_array_mismatch import (  # noqa: E402
    DATA_ROOT,
    PLOT_RCP,
    SCENARIOS,
    SWEEP_LABELS,
)

SUBNET_KEY = "SubNet+r-music"
CLASSIC_KEY = "r-music"
ARCH_STYLE = {
    "base": dict(color="tab:blue", marker="o", ls="-", label="base"),
    "res": dict(color="tab:red", marker="s", ls="--", label="residual"),
}
# 同一次实验的硬约束：这些字段必须跨所有 run 一致，否则"多种子"实际混进了别的变量。
SAME_KEYS = ("N", "M", "T", "snr", "tau", "n_train", "n_test", "batch_size", "epochs",
             "sweep_param", "phase_gauge", "shared_array_mismatch")


def load_runs(pattern: str, epochs: int, seeds: list[int]) -> list[dict]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"没有匹配到文件: {pattern}")
    runs = []
    for p in paths:
        payload = json.loads(Path(p).read_text("utf-8"))
        cfg = payload.get("config", {})
        if cfg.get("arch") not in ARCH_STYLE:
            continue
        if cfg.get("epochs") != epochs or cfg.get("init_seed") not in seeds:
            continue
        runs.append({"path": Path(p), "payload": payload, "config": cfg,
                     "arch": cfg["arch"], "seed": int(cfg["init_seed"]),
                     "rows": payload.get("rows") or []})
    if not runs:
        raise SystemExit(f"匹配到 {len(paths)} 个文件，但没有 arch/epochs/seed 符合的")
    return runs


def dedupe(runs: list[dict]) -> list[dict]:
    """同一 (arch, seed) 有多份结果时只留最新的一份，其余打警告。

    重跑一次会再写一个 JSON；如果都留着，同一颗种子会被重复计数，sd 直接失真。
    """
    best: dict[tuple[str, int], dict] = {}
    for r in runs:
        key = (r["arch"], r["seed"])
        if key not in best or r["path"].stat().st_mtime > best[key]["path"].stat().st_mtime:
            if key in best:
                print(f"[warn] {r['arch']}/s{r['seed']} 有多份结果，用较新的 {r['path'].name}")
            best[key] = r
        else:
            print(f"[warn] {r['arch']}/s{r['seed']} 有多份结果，忽略较旧的 {r['path'].name}")
    return list(best.values())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="多种子残差臂 A/B 汇总")
    p.add_argument("--pattern", default="data/simulations/results/array_mismatch_gain_phase_*.json",
                   help="结果 JSON 的 glob (默认 gain_phase 全部)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 2024, 3407])
    p.add_argument("--out_dir", type=Path, default=None)
    args = p.parse_args(argv)

    runs = dedupe(load_runs(args.pattern, args.epochs, args.seeds))
    base = {r["seed"]: r for r in runs if r["arch"] == "base"}
    res = {r["seed"]: r for r in runs if r["arch"] == "res"}
    if set(base) != set(res):
        raise SystemExit(f"两条臂的种子不配对: base={sorted(base)} res={sorted(res)}")
    seeds = sorted(base)
    if len(seeds) < 2:
        raise SystemExit(f"只有 {len(seeds)} 颗种子，多种子汇总没有意义")

    ref = base[seeds[0]]
    cfg = ref["config"]
    scenario = ref["payload"].get("scenario", "gain_phase")
    sweep = cfg["sweep_param"]
    for other in runs:
        for key in SAME_KEYS:
            if other["config"].get(key) != cfg.get(key):
                raise SystemExit(f"{other['path'].name}: {key}={other['config'].get(key)} "
                                 f"与基准 {cfg.get(key)} 不一致，不是同一协议")
    x = np.asarray(sorted(float(r[sweep]) for r in ref["rows"]))
    for r in runs:
        xs = sorted(float(row[sweep]) for row in r["rows"])
        if len(xs) != len(x) or not np.allclose(xs, x, atol=1e-12):
            raise SystemExit(f"{r['path'].name}: 评估网格与基准不同")

    def values(run, key):
        rows = sorted(run["rows"], key=lambda row: float(row[sweep]))
        return np.asarray([row[f"{key}_deg"] for row in rows], dtype=float)

    vals = {(arch, seed): values(run, SUBNET_KEY) for arch, group in
            (("base", base), ("res", res)) for seed, run in group.items()}
    classic = {arch: values(group[seeds[0]], CLASSIC_KEY) for arch, group in
               (("base", base), ("res", res))}
    # 经典算法不依赖网络权重：同一份测试集上，所有 (arm, seed) 必须逐位相同。
    # 不同就说明测试集或评估代码变了，这条 A/B 不能直接比较。
    ref_cls = classic["base"]
    for r in runs:
        if not np.allclose(values(r, CLASSIC_KEY), ref_cls, equal_nan=True):
            raise SystemExit(f"{r['path'].name}: 经典 Root-MUSIC 与基准不一致，"
                             "说明测试集或评估路径不是同一份")

    base_arr = np.stack([vals[("base", s)] for s in seeds])      # (n_seed, n_point)
    res_arr = np.stack([vals[("res", s)] for s in seeds])
    base_mean, base_sd = base_arr.mean(0), base_arr.std(0, ddof=1)
    res_mean, res_sd = res_arr.mean(0), res_arr.std(0, ddof=1)
    delta = base_arr - res_arr                                   # >0 表示 res 更好
    delta_rel = 100.0 * delta / base_arr
    wins = (delta > 0).sum(0)                                    # 每个点上 res 更好的种子数

    mean_per_seed_base = base_arr.mean(1)
    mean_per_seed_res = res_arr.mean(1)
    mean_imp_per_seed = 100.0 * (mean_per_seed_base - mean_per_seed_res) / mean_per_seed_base

    # ---------------- markdown ----------------
    pair_scale = (SCENARIOS.get(scenario) or {}).get("pair_scale")
    lines = [
        f"# 多种子残差臂 A/B：{scenario}（{len(seeds)} 颗种子 × {cfg['epochs']} epoch）",
        "",
        f"- 种子: {', '.join(str(s) for s in seeds)}；两个臂各自同种子配对训练",
        f"- 协议: {cfg['n_train']} 训练 / {cfg['n_test']} 测试 / batch {cfg['batch_size']} / "
        f"RMSPE [deg]，指标为 SubspaceNet + Root-MUSIC",
        "- 提升 = (base - res)/base；sd 为样本标准差（ddof=1）；"
        "「胜」= res 更好的种子数",
        "",
    ]
    head = "| $u$ | 配对 (增益, 相位) | base | res | 提升 (逐种子) | 均值提升 | 胜 |" \
        if pair_scale else "| 失配水平 | base | res | 提升 (逐种子) | 均值提升 | 胜 |"
    ncol = 7 if pair_scale else 6
    lines += [head, "|" + "---:|" * ncol]
    for i, u in enumerate(x):
        cells = [f"{u:g}"]
        if pair_scale:
            g, ph = pair_scale
            cells.append(f"{100.0 * g * u:.0f}%, {ph * u:.0f}°")
        cells += [f"{base_mean[i]:.4f} ± {base_sd[i]:.4f}",
                  f"{res_mean[i]:.4f} ± {res_sd[i]:.4f}",
                  " / ".join(f"{v:+.1f}%" for v in delta_rel[:, i]),
                  f"{delta_rel[:, i].mean():+.1f}%",
                  f"{wins[i]}/{len(seeds)}"]
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["**均值**"]
    if pair_scale:
        cells.append("")
    cells += [f"**{base_mean.mean():.4f}**", f"**{res_mean.mean():.4f}**",
              " / ".join(f"{v:+.1f}%" for v in mean_imp_per_seed),
              f"**{mean_imp_per_seed.mean():+.1f}%**",
              f"**{(mean_imp_per_seed > 0).sum()}/{len(seeds)}**"]
    lines += ["| " + " | ".join(cells) + " |", ""]

    lines += [
        f"逐点胜场合计: res 在 {int(wins.sum())}/{wins.size * len(seeds)} 个 (点, 种子) 组合上更好。",
        f"整条曲线的均值: base {base_mean.mean():.4f} ± {base_sd.mean():.4f}，"
        f"res {res_mean.mean():.4f} ± {res_sd.mean():.4f}。",
        "",
    ]
    fail = []
    for arch, group in (("base", base), ("res", res)):
        for s in seeds:
            fr = [row.get(f"{SUBNET_KEY}_fail", 0.0) for row in group[s]["rows"]]
            if any(fr):
                fail.append(f"{arch}/s{s}: " + ", ".join(f"{v:.2f}" for v in fr))
    if fail:
        lines += ["失败率非零（曲线被漏检污染）：", ""] + [f"- {s}" for s in fail] + [""]

    fig_dir = Path(args.out_dir) if args.out_dir else DATA_ROOT / "simulations" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"residual_seeds_{scenario}_deg"
    png, md = fig_dir / f"{tag}.png", fig_dir / f"{tag}.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    # ---------------- 图 ----------------
    with plt.rc_context(PLOT_RCP):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.25, 1]})
        for arch, mean, sd in (("base", base_mean, base_sd), ("res", res_mean, res_sd)):
            st = ARCH_STYLE[arch]
            ax1.fill_between(x, mean - sd, mean + sd, color=st["color"], alpha=0.16, lw=0)
            ax1.plot(x, mean, color=st["color"], marker=st["marker"], ls=st["ls"],
                     mfc="none" if arch == "base" else st["color"], ms=7, lw=1.9,
                     label=f"{st['label']}  {mean.mean():.3f}°")
        cls = classic["base"]
        ax1.plot(x, cls, "^:", color="gray", mfc="none", ms=6, lw=1.3,
                 label=f"Root-MUSIC (classic)  {cls.mean():.3f}°")
        ax1.set_ylabel("RMSPE [deg]")
        ax1.set_title(f"{len(seeds)} seeds × {cfg['epochs']} epochs, SubspaceNet + Root-MUSIC\n"
                      "mean curve ±1 sd", fontsize=10.5)
        ax1.legend(fontsize=8.5)

        for j, s in enumerate(seeds):
            ax2.plot(x, delta_rel[j], "-", color="0.65", lw=1.0, marker=".",
                     label="per seed" if j == 0 else None)
        ax2.plot(x, delta_rel.mean(0), "o-", color="tab:red", lw=2.0, ms=6,
                 label=f"mean {delta_rel.mean():+.1f}%")
        ax2.axhline(0.0, color="0.3", lw=1.0, ls=":")
        ax2.set_ylabel("improvement of res  [%]")
        ax2.set_title("paired difference per point\n(base - res)/base, >0 = residual better",
                      fontsize=10.5)
        ax2.legend(fontsize=8.5)

        # 只有左 panel 标物理配对；右 panel 窄，再叠第二行会糊成一团，只留 u。
        ax1.set_xticks(x)
        ax2.set_xticks(x)
        if pair_scale:
            g, ph = pair_scale
            ax1.set_xticklabels([f"{u:g}\n({100.0 * g * u:.0f}%, {ph * u:.0f}°)"
                                 for u in x], fontsize=8)
        ax2.set_xticklabels([f"{u:g}" for u in x], fontsize=9)
        for ax in (ax1, ax2):
            ax.set_xlabel(SWEEP_LABELS.get(sweep, f"array mismatch level ({sweep})"))
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n".join(lines))
    print(f"[plot] {png}")
    print(f"[table] {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
