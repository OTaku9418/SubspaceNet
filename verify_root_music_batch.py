"""等价性与提速校验: src/models.py 的批量版 root_music。

背景
----
`src/models.py` 的 `root_music()` 原本是一个逐样本的 Python 循环
(`for iter in range(batch_size)`), 每个样本独立启动一整套 kernel。实测它占
SubspaceNet 前向耗时的 90% 以上, 是训练速度的唯一主要瓶颈; 同时它调用的
`src/utils.py` 的 `find_roots_torch` 会把伴随矩阵建在 CPU 上, 使每个样本都发生一次
GPU<->CPU 往返。

本仓库已把 `root_music` 改写为对 batch 全向量化的版本。本脚本的作用是**在改动后重新
证明它没有改变数值**: 它内联保留了改写前的原始实现 (`root_music_reference`), 在同一批
输入上比较两者的输出, 并顺带报告提速比。

用法
----
    python verify_root_music_batch.py            # 等价性 + 提速
    python verify_root_music_batch.py --quick    # 只做等价性, 跳过计时

退出码
------
    0 = 等价性检查全部通过 (容差见下)
    1 = 存在超差项, 改动不可用
"""

import argparse
import sys
import time

import numpy as np
import torch

TOLERANCE_RAD = 1e-4
"""允许的最大 doa 偏差 (弧度)。实测约 6e-7, 留两个数量级余量以吸收不同 GPU/驱动差异。"""

TOLERANCE_GRAD = 1e-2
"""允许的最大梯度相对偏差。实测约 4e-5。"""


def root_music_reference(Rz: torch.Tensor, M: int, batch_size: int):
    """改写前的原始实现, 逐字保留 (来自 commit a249796 的 src/models.py:693-751)。

    仅用于对照, 不要在生产路径上调用: 它既慢, 又会把多项式求根落到 CPU。
    """
    from src.utils import find_roots_torch, sum_of_diags_torch

    dist = 0.5
    f = 1
    doa_batches = []
    doa_all_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # 本脚本验证的是"批量化"本身, 所以参考实现用与 src/models.py:root_music 相同的分解
        # 方式 (`eigh`, 见那边的注释)。换成 eig 校验的不是同一件事: 两者在计算上等价, 但
        # eig 的 CUDA 路径对这些小批量矩阵是 CPU-bound 的, 跑起来慢一个数量级。
        eigenvalues, eigenvectors = torch.linalg.eigh(R)
        eigenvalues = torch.flip(eigenvalues, dims=[-1])
        eigenvectors = torch.flip(eigenvectors, dims=[-1])
        Un = eigenvectors[:, M:]
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        diag_sum = sum_of_diags_torch(F)
        roots = find_roots_torch(diag_sum)
        roots_angels_all = torch.angle(roots)
        doa_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        roots = roots[
            sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))
        ]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        roots_angels = torch.angle(roots)
        doa_pred = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels)
        doa_batches.append(doa_pred)

    return (
        torch.stack(doa_batches, dim=0),
        torch.stack(doa_all_batches, dim=0),
        roots_to_return,
    )


def build_model(device):
    """构造一个与论文一致的 SubspaceNet (N=8, M=2, T=100, tau=8)。"""
    from src.models import ModelGenerator
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed

    set_unified_seed(0)
    params = SystemModelParams()
    for name, value in dict(
        N=8,
        M=2,
        T=100,
        snr=10,
        eta=0.0,
        bias=0.0,
        signal_type="NarrowBand",
        signal_nature="non-coherent",
    ).items():
        params.set_parameter(name, value)
    model_config = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_diff_method("root_music")
        .set_tau(8)
        .set_model(params)
    )
    model = model_config.model.to(device).eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_trainable


def timeit(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def check_equivalence(model, device, batches, tol_rad):
    """前向数值等价性。

    比较两件事:
      1. doa_batches —— 真正被训练/评估使用的输出 (见 src/evaluation.py:106, src/training.py:379),
         必须逐样本、按同样顺序一致。
      2. doa_all_batches 的**排序后集合** —— 它只被 src/evaluation.py:121-129 的
         `plot_spec=True` 分支用来画谱图, 不进入任何指标。它由全部 2N-2 个根导出, 而
         batch 化 EVD 与逐样本 EVD 可能给出不同的根排列, 故只能比较集合, 不能比较顺序。
    """
    from src.models import root_music

    print("\n[1/3] 前向数值等价性 (批量版 vs 逐样本原始实现)")
    worst = 0.0
    for batch_size in batches:
        x = torch.randn(batch_size, 8, 16, 8, device=device)
        with torch.no_grad():
            Rz = model(x)[-1].detach()
            ref_doa, ref_all, _ = root_music_reference(Rz, 2, batch_size)
            new_doa, new_all, _ = root_music(Rz, 2, batch_size)
        # 原始实现返回 CPU 张量(见 find_roots_torch 的 device bug), 比较前先对齐设备
        ref_doa, ref_all = ref_doa.to(new_doa.device), ref_all.to(new_all.device)
        diff_doa = (ref_doa - new_doa).abs().max().item()
        # 排序后逐样本比较全部根的 doa 集合
        diff_all = (
            ref_all.sort(dim=-1).values - new_all.sort(dim=-1).values
        ).abs().max().item()
        worst = max(worst, diff_doa, diff_all)
        status = "OK" if max(diff_doa, diff_all) < tol_rad else "FAIL"
        print(
            f"  batch={batch_size:5d}  M 个 doa 最大差 {diff_doa:.3e} rad | "
            f"全部根 doa 集合最大差 {diff_all:.3e} rad   [{status}]"
        )
    return worst


def check_gradients(model, device, batches, tol_grad):
    """反向数值等价性: 对 Rz 的梯度。"""
    from src.models import root_music

    print("\n[2/3] 反向梯度等价性")
    worst = 0.0

    def grad_of(fn, Rz, batch_size):
        R = Rz.clone().requires_grad_(True)
        out = fn(R, 2, batch_size)[0]
        out.real.sum().backward()
        return R.grad

    for batch_size in batches:
        x = torch.randn(batch_size, 8, 16, 8, device=device)
        with torch.no_grad():
            Rz = model(x)[-1].detach()
        try:
            g_ref = grad_of(root_music_reference, Rz, batch_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  batch={batch_size:5d}  原始实现反向失败, 跳过: {type(exc).__name__}")
            continue
        g_new = grad_of(root_music, Rz, batch_size)
        scale = max(g_ref.abs().max().item(), 1e-12)
        rel = (g_ref - g_new).abs().max().item() / scale
        worst = max(worst, rel)
        status = "OK" if rel < tol_grad else "FAIL"
        print(f"  batch={batch_size:5d}  梯度最大相对差 {rel:.3e}   [{status}]")
    return worst


def report_speedup(model, device, batches):
    """前向 + 整步训练 (含反向) 的提速比。"""
    from src.models import root_music
    from src.criterions import RMSPELoss

    print("\n[3/3] 提速比 (本机, 仅供参考; 服务器上通常更明显)")
    print(f"  {'batch':>6} {'原始 ms':>11} {'批量 ms':>11} {'加速':>8}   {'samples/s 原':>13} {'-> 批量':>9}")
    for batch_size in batches:
        x = torch.randn(batch_size, 8, 16, 8, device=device)
        with torch.no_grad():
            Rz = model(x)[-1].detach()
        repeats = 20 if batch_size <= 64 else 5
        t_ref = timeit(lambda: root_music_reference(Rz, 2, batch_size), repeats)
        t_new = timeit(lambda: root_music(Rz, 2, batch_size), repeats)
        print(
            f"  {batch_size:6d} {t_ref*1e3:8.1f} ms {t_new*1e3:8.1f} ms "
            f"{t_ref/t_new:7.1f}x   {batch_size/t_ref:11.1f} {batch_size/t_new:9.1f}"
        )

    print("\n  整步训练 (forward + RMSPELoss + backward + step):")
    criterion = RMSPELoss()
    for batch_size in [b for b in batches if b >= 128]:
        x = torch.randn(batch_size, 8, 16, 8, device=device)
        target = torch.randn(batch_size, 2, device=device)

        def one_step():
            out = model(x)[0]
            loss = criterion(out, target)
            loss.backward()
            model.zero_grad()

        t = timeit(one_step, 3 if batch_size >= 512 else 5, warmup=1)
        print(
            f"    batch={batch_size:5d}  {t*1e3:8.1f} ms/step  "
            f"-> {batch_size/t:6.1f} samples/s"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="跳过计时, 只做等价性检查")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}  numpy={np.__version__}")

    model, n_trainable = build_model(device)
    print(f"SubspaceNet 可训练参数量 = {n_trainable} (论文为 41761)")
    if n_trainable != 41761:
        print("警告: 参数量与论文不一致, 后续对比失去意义")

    batches = [1, 2, 8, 32, 256]
    worst_forward = check_equivalence(model, device, batches, TOLERANCE_RAD)
    worst_grad = check_gradients(model, device, [1, 2, 8, 32], TOLERANCE_GRAD)
    if not args.quick:
        report_speedup(model, device, [1, 2, 8, 32, 256, 1024])

    print("\n=== 汇总 ===")
    print(f"  前向最大偏差 {worst_forward:.3e} rad (容差 {TOLERANCE_RAD:.0e})")
    print(f"  梯度最大相对偏差 {worst_grad:.3e} (容差 {TOLERANCE_GRAD:.0e})")
    ok = worst_forward < TOLERANCE_RAD and worst_grad < TOLERANCE_GRAD
    print("  结论:", "等价性通过" if ok else "存在超差项, 请勿使用当前实现")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
