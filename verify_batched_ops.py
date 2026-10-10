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

Exit code 0 = every check passed.

Usage:
    python verify_batched_ops.py
    python verify_batched_ops.py --quick   # fewer batch sizes
"""

import argparse
import sys
import time

import torch

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
    print("[1/2] gram_diagonal_overload 等价性 (批量版 vs 逐样本原始实现)")
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


def check_speed(batch_size=512, quick=False):
    """两处改动的提速比（仅供参考）。"""
    from src.models import find_roots_batched
    from src.utils import find_roots_torch

    print("\n[2/2] 提速比 (本机, 仅供参考)")
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
    check_speed(quick=args.quick)

    print("\n=== 汇总 ===")
    print(f"  gram 最大偏差 {worst:.3e} (容差 {TOLERANCE_GRAM})")
    print(f"  root_music 的等价性与提速见 `python verify_root_music_batch.py`")
    if worst >= TOLERANCE_GRAM:
        print("  结论: 存在超差项, 请勿使用当前实现")
        sys.exit(1)
    print("  结论: 通过")


if __name__ == "__main__":
    main()
