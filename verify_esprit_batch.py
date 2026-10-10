"""校验 `src/models.py:esprit` 的批量实现与改写前的逐样本循环等价, 并给出提速比。

背景
----
`esprit` 原本是 `for iter in range(batch_size)` 的逐样本循环, 每个样本两次
`torch.linalg.eig`。subspace_method="esprit" 的 SubspaceNet（`SubspaceNetEsprit`）训练时
会穿过这个函数, 所以它和 `root_music` 一样是训练速度的直接瓶颈。改动包括:

  1. 第一次特征分解改用 `eigh`（替代协方差 Hermitian, `eig` 在 float32 下返回的特征向量
     并不正交 —— 见 `probe_forward_detail.py` 与文档 §16）;
  2. `phi = pinv(Us_upper) @ Us_lower` 的秩 2 左除改成批量 `pinv`;
  3. `M x M` 的 `phi` 特征值改成批量 `eigvals`;
  4. 去掉 Python 循环。

本脚本保留改写前的逐样本实现作为参考, 在同样的输入上比对 DoA;
容差 `TOLERANCE_RAD`。退出码 0 = 通过。

用法
----
    python verify_esprit_batch.py            # 等价性 + 提速
    python verify_esprit_batch.py --quick    # 跳过提速测量
"""

import argparse
import sys
import time

import numpy as np
import torch

TOLERANCE_RAD = 1e-3

# 本机参考值 (RTX 4060 Laptop, torch 2.0.1+cu118, 8 阵元 / M=2):
#   batch=  32   原始 ~ 8.0 ms   批量 ~ 0.4 ms
#   batch= 512   原始 ~130   ms   批量 ~ 0.6 ms
# 判读只看比例: 批量版应当随 batch 近乎持平, 而原始版线性增长。


def esprit_reference(Rz: torch.Tensor, M: int, batch_size: int):
    """改写前的原始实现, 逐字保留逐样本循环 (来自 commit 4e1afdb 的 src/models.py:825-867)。

    仅用于对照, 不要在生产路径上调用。
    """
    doa_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :M]
        Us_upper, Us_lower = (Us[0 : R.shape[0] - 1], Us[1 : R.shape[0]])
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        phi_eigenvalues, _ = torch.linalg.eig(phi)
        eigenvalues_angels = torch.angle(phi_eigenvalues)
        doa_predictions = -1 * torch.arcsin((1 / np.pi) * eigenvalues_angels)
        doa_batches.append(doa_predictions)
    return torch.stack(doa_batches, dim=0)


def timeit(fn, repeats=10, warmup=2, device=None):
    for _ in range(warmup):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def build_model(device):
    from src.models import ModelGenerator
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed

    set_unified_seed(0)
    params = SystemModelParams()
    for name, value in dict(
        N=8, M=2, T=100, snr=10, eta=0.0, bias=0.0,
        signal_type="NarrowBand", signal_nature="non-coherent",
    ).items():
        params.set_parameter(name, value)
    model = (
        ModelGenerator().set_model_type("SubspaceNet")
        .set_diff_method("esprit").set_tau(8).set_model(params)
    ).model
    return model.to(device).eval()


def check_equivalence(model, device, batches, tol_rad):
    from src.models import esprit

    worst = 0.0
    for batch_size in batches:
        x = torch.randn(batch_size, 8, 16, 8, device=device)
        with torch.no_grad():
            Rz = model(x)[-1].detach()
            ref = esprit_reference(Rz, 2, batch_size)
            new = esprit(Rz, 2, batch_size)
        # 逐样本比 M 个 doa。两个实现的 M 个预测都不排序, 顺序由特征值排序决定,
        # 退化特征值下顺序可能不同, 故比较"排序后的集合"。
        ref_sorted = torch.sort(ref, dim=-1).values
        new_sorted = torch.sort(new, dim=-1).values
        diff = (ref_sorted - new_sorted).abs().max().item()
        worst = max(worst, diff)
        flag = "OK" if diff < tol_rad else "FAIL"
        print(f"  batch={batch_size:5d}  M 个 doa 最大差 {diff:.3e} rad   [{flag}]")
    return worst


def check_gradients(model, device, batches, tol_grad):
    from src.models import esprit

    worst = 0.0
    for batch_size in batches:
        x = torch.randn(batch_size, 8, 16, 8, device=device)

        def grad_of(fn):
            inp = x.detach().clone().requires_grad_(True)
            out = fn(inp)
            return torch.autograd.grad(out.abs().sum(), inp)[0]

        g_ref = grad_of(lambda inp: esprit_reference(model(inp)[-1], 2, batch_size))
        g_new = grad_of(lambda inp: esprit(model(inp)[-1], 2, batch_size))
        rel = (g_ref - g_new).abs().norm().item() / (g_ref.abs().norm().item() + 1e-12)
        worst = max(worst, rel)
        flag = "OK" if rel < tol_grad else "FAIL"
        print(f"  batch={batch_size:5d}  梯度最大相对差 {rel:.3e}   [{flag}]")
    return worst


def report_speedup(model, device, batches):
    from src.models import esprit

    print("   batch       原始 ms       批量 ms       加速")
    for batch_size in batches:
        x = torch.randn(batch_size, 8, 16, 8, device=device)
        with torch.no_grad():
            Rz = model(x)[-1].detach()
            t_ref = timeit(lambda: esprit_reference(Rz, 2, batch_size), device=device)
            t_new = timeit(lambda: esprit(Rz, 2, batch_size), device=device)
        print(f"  {batch_size:5d}  {t_ref*1e3:12.2f}  {t_new*1e3:12.2f}  "
              f"{t_ref/t_new:9.1f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="跳过提速测量")
    args = parser.parse_args()

    from src.utils import device

    print(f"device={device}  torch={torch.__version__}  numpy={np.__version__}")
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SubspaceNet(esprit) 可训练参数量 = {n_params}\n")

    print("[1/3] 前向数值等价性 (批量版 vs 逐样本原始实现)")
    worst_forward = check_equivalence(model, device, [1, 2, 8, 32, 256, 512, 1024], TOLERANCE_RAD)

    print("\n[2/3] 反向梯度等价性")
    worst_grad = check_gradients(model, device, [1, 2, 8, 32], 1e-2)

    if not args.quick:
        print("\n[3/3] 提速比 (本机, 仅供参考; 服务器上通常更明显)")
        report_speedup(model, device, [1, 2, 8, 32, 256, 512])

    print("\n=== 汇总 ===")
    print(f"  前向最大偏差 {worst_forward:.3e} rad (容差 {TOLERANCE_RAD:.0e})")
    print(f"  梯度最大相对偏差 {worst_grad:.3e} (容差 1e-02)")
    ok = worst_forward < TOLERANCE_RAD and worst_grad < 1e-2
    print("  结论: " + ("等价性通过" if ok else "等价性未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


