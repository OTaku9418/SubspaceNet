"""Verify that the batched re-implementations match the original per-sample code paths.

The upstream repository writes several inference helpers as
`for iter in range(batch_size)` loops. On a fast GPU those loops dominate the runtime:
they launch hundreds of tiny kernels per step and, in `src/utils.py:147`, silently move
work to the CPU. They are therefore re-implemented in vectorized form, and this script is
the regression check that says the rewrite is numerically equivalent.

Covered:
  1. src/models.py  root_music()            (see verify_root_music_batch.py for the
                                             standalone deep comparison + speedup table)
  2. src/utils.py   gram_diagonal_overload()
  3. src/criterions.py RMSPELoss.forward()  (loss and gradient, against the original
                                             per-sample loop kept in this file)

Exit code 0 = every check passed.

Usage:
    python verify_batched_ops.py
    python verify_batched_ops.py --quick   # fewer batch sizes
"""

import argparse
import sys
import time

import numpy as np
import torch

from src.criterions import permute_prediction
from src.models import root_music, sum_of_diags_batched
from src.utils import device, gram_diagonal_overload, gram_diagonal_overload_reference
from verify_root_music_batch import root_music_reference

TOLERANCE_GRAM = 1e-4
TOLERANCE_RAD = 1e-4
TOLERANCE_GRAD = 1e-2


def timeit(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def check_gram(batches):
    """gram_diagonal_overload: batched vs the original per-sample loop."""
    print("[1/4] gram_diagonal_overload 等价性 (批量版 vs 逐样本原始实现)")
    worst = 0.0
    for batch_size in batches:
        Kx = (
            torch.randn(batch_size, 8, 8, dtype=torch.complex64, device=device)
            + 1j * torch.randn(batch_size, 8, 8, dtype=torch.complex64, device=device)
        )
        new = gram_diagonal_overload(Kx, 1.0, batch_size)
        ref = gram_diagonal_overload_reference(Kx, 1.0, batch_size)
        diff = (new - ref).abs().max().item()
        worst = max(worst, diff)
        status = "OK" if diff < TOLERANCE_GRAM else "FAIL"
        print(f"  batch={batch_size:5d}  最大差 {diff:.3e}   [OK]" if status == "OK"
              else f"  batch={batch_size:5d}  最大差 {diff:.3e}   [FAIL]")

    # 也要检查它确实给出了 Hermitian + 对角加 eps 的结果
    Kx = torch.randn(4, 8, 8, dtype=torch.complex64, device=device)
    out = gram_diagonal_overload(Kx, 1.0, 4)
    hermitian = (out - out.conj().transpose(-2, -1)).abs().max().item()
    print(f"  Hermitian 偏差 {hermitian:.3e}（应为 0）；"
          f"对角最小值 {out.diagonal(dim1=-2, dim2=-1).real.min().item():.4f}")
    return worst


def rmspe_loss_reference(predictions, targets):
    """The original per-sample RMSPELoss body, kept here verbatim for the equivalence check.

    `src/criterions.py:RMSPELoss.forward` no longer contains this loop; it is the reference this
    script compares against, so it must not be "un-vectorized" later by mistake.
    """
    rmspe = []
    for index in range(predictions.shape[0]):
        rmspe_list = []
        batch_predictions = predictions[index].to(device)
        sample_targets = targets[index].to(device)
        for prediction in permute_prediction(batch_predictions):
            error = (((prediction - sample_targets) + (np.pi / 2)) % np.pi) - np.pi / 2
            rmspe_val = (1 / np.sqrt(len(sample_targets))) * torch.linalg.norm(error)
            rmspe_list.append(rmspe_val)
        rmspe.append(torch.min(torch.stack(rmspe_list, dim=0)))
    return torch.sum(torch.stack(rmspe, dim=0))


def check_loss():
    """RMSPELoss: the vectorized forward vs the original per-sample loop, loss AND gradient."""
    from src.criterions import RMSPELoss

    print("[2/4] RMSPELoss 等价性 (批量版 vs 逐样本原始实现, 含梯度)")
    criterion = RMSPELoss()
    worst_loss, worst_grad = 0.0, 0.0
    for m in (2, 3):
        for batch_size in (1, 8, 512):
            predictions = torch.rand(batch_size, m, device=device) * np.pi - np.pi / 2
            targets = torch.rand(batch_size, m, device=device) * np.pi - np.pi / 2

            p_new = predictions.clone().requires_grad_(True)
            l_new = criterion(p_new, targets)
            l_new.backward()

            p_ref = predictions.clone().requires_grad_(True)
            l_ref = rmspe_loss_reference(p_ref, targets)
            l_ref.backward()

            d_loss = abs(l_new.item() - l_ref.item())
            scale = max(p_ref.grad.abs().max().item(), 1e-12)
            d_grad = (p_new.grad - p_ref.grad).abs().max().item() / scale
            worst_loss, worst_grad = max(worst_loss, d_loss), max(worst_grad, d_grad)
            flag = "OK" if (d_loss < 1e-4 and d_grad < 1e-5) else "FAIL"
            print(f"  M={m} batch={batch_size:5d}  loss 差 {d_loss:.3e}  "
                  f"梯度相对差 {d_grad:.3e}   [{flag}]")
    return worst_loss, worst_grad


def check_loss_device_pinning():
    """The criterion must follow the predictions, not the import-time `cuda:0` constant.

    Regression test for: running on a second card (``--device 1`` or ``CUDA_VISIBLE_DEVICES=1``)
    raised "Expected all tensors to be on the same device, but found at least two devices,
    cuda:0 and cuda:1". The mismatch is reproduced here by pointing the module constant at the
    wrong device on purpose; the criterion must ignore it.
    """
    import numpy as np

    import src.criterions as crit_mod
    from src.criterions import MSPELoss, RMSPELoss

    print("[3/4] 判据函数的设备跟随（回归：--device 1 时的 cuda:0/cuda:1 冲突）")
    original = crit_mod.device
    worst = 0.0
    try:
        for name, criterion in (("RMSPELoss", RMSPELoss()), ("MSPELoss", MSPELoss())):
            predictions = torch.rand(64, 2, device=device) * np.pi - np.pi / 2
            targets = torch.rand(64, 2, device=device) * np.pi - np.pi / 2

            crit_mod.device = device
            expected = criterion(predictions.clone(), targets).item()
            crit_mod.device = torch.device("cpu")   # wrong on purpose
            got = criterion(predictions.clone(), targets).item()
            crit_mod.device = original

            diff = abs(got - expected)
            worst = max(worst, diff)
            flag = "OK" if diff < 1e-6 else "FAIL"
            print(f"  {name}: 模块常量故意指错后 loss 差 {diff:.3e}  [{flag}]")
    finally:
        crit_mod.device = original
    return worst


def check_speed(batch_size=512, quick=False):
    """两处改动的提速比（仅供参考）。"""
    from src.models import find_roots_batched
    from src.utils import find_roots_torch

    print("\n[4/4] 提速比 (本机, 仅供参考)")
    Kx = (
        torch.randn(batch_size, 8, 8, dtype=torch.complex64, device=device)
        + 1j * torch.randn(batch_size, 8, 8, dtype=torch.complex64, device=device)
    )
    t_new = timeit(lambda: gram_diagonal_overload(Kx, 1.0, batch_size))
    t_ref = timeit(lambda: gram_diagonal_overload_reference(Kx, 1.0, batch_size), repeats=1,
                   warmup=0)
    print(f"  gram_diagonal_overload   batch={batch_size}")
    print(f"    原始逐样本 {t_ref*1e3:9.2f} ms  ->  批量 {t_new*1e3:8.3f} ms   "
          f"{t_ref/t_new:8.1f}x")

    if quick:
        return
    Rz = (
        torch.randn(batch_size, 8, 8, dtype=torch.complex64, device=device)
        + 1j * torch.randn(batch_size, 8, 8, dtype=torch.complex64, device=device)
    )
    t_batch = timeit(lambda: root_music(Rz, 2, batch_size))
    n_ref = min(batch_size, 16)
    t_one = timeit(lambda: root_music_reference(Rz[:n_ref], 2, n_ref), repeats=1, warmup=0)
    print(f"  root_music               batch={batch_size}")
    print(f"    原始逐样本 {t_one*1e3/max(n_ref,1)*batch_size:9.2f} ms (按 {n_ref} 样本外推)"
          f"  ->  批量 {t_batch*1e3:8.3f} ms   "
          f"{t_one/max(n_ref,1)*batch_size/t_batch:8.1f}x")
    print(f"  注: src/utils.py 的 find_roots_torch 仍把伴随矩阵建在 CPU 上; "
          f"find_roots_batched 已修, 两者不再相等（这正是要点）")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"device={device}  torch={torch.__version__}")
    batches = (1, 8) if args.quick else (1, 8, 512)

    worst = check_gram(batches)
    worst_loss, worst_grad = check_loss()
    worst_device = check_loss_device_pinning()
    check_speed(quick=args.quick)

    print("\n=== 汇总 ===")
    print(f"  gram 最大偏差 {worst:.3e} (容差 {TOLERANCE_GRAM})")
    print(f"  RMSPELoss loss 差 {worst_loss:.3e} / 梯度相对差 {worst_grad:.3e}")
    print(f"  判据函数的 device 跟随偏差 {worst_device:.3e} (容差 1e-6)")
    print(f"  root_music 的等价性与提速见 `python verify_root_music_batch.py`")
    if (worst >= TOLERANCE_GRAM or worst_loss >= 1e-4 or worst_grad >= 1e-5
            or worst_device >= 1e-6):
        print("  结论: 存在超差项, 请勿使用当前实现")
        sys.exit(1)
    print("  结论: 通过")


if __name__ == "__main__":
    main()
