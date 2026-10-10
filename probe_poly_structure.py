"""Root-MUSIC 求根步骤: 为什么 torch.linalg.eigvals 在服务器上要 7.6 s, 以及怎么绕开。

背景(服务器 torch 2.10.0+cu128 实测, batch=512):
    eigvals 8x8  complex64   7638 ms   ( 14919 us/矩阵)
    eigvals 14x14 complex64  7715 ms   ( 15068 us/矩阵)
    eigvalsh 14x14 (Hermitian)  2.28 ms (     4.5 us/矩阵)
即 `eigvals` 完全没有走 GPU(或走了极慢的通用路径), 而 `eigvalsh` 正常。

本探针回答三个问题:
  [1] 这些多项式的系数有什么结构? (是不是自反/自逆多项式)
  [2] 它们的根是否都在单位圆上?
  [3] 能不能把"求根"换成 Hermitian 特征值问题(eigvalsh), 从而拿到 3 个数量级的提速?

用法:
    python probe_poly_structure.py            # 用小模型 + 随机 Hermitian 投影
    python probe_poly_structure.py --batch 256
"""

import argparse

import numpy as np
import torch

from src.models import ModelGenerator, sum_of_diags_batched
from src.system_model import SystemModelParams
from src.utils import device, set_unified_seed


def polynomial_coefficients(batch: int, M: int = 2):
    """造一批和 root_music 内部同源的系数: 对 Hermitian 噪声投影 F 取对角线之和。"""
    set_unified_seed(0)
    params = SystemModelParams()
    for name, value in dict(
        N=8, M=M, T=100, snr=10, eta=0.0, bias=0.0,
        signal_type="NarrowBand", signal_nature="non-coherent",
    ).items():
        params.set_parameter(name, value)
    model = (
        ModelGenerator().set_model_type("SubspaceNet")
        .set_diff_method("root_music").set_tau(8).set_model(params)
    ).model.to(device).eval()

    x = torch.randn(batch, 8, 16, 8, device=device)
    with torch.no_grad():
        Rz = model(x)[-1]
        _, eigenvectors = torch.linalg.eigh(Rz)
        eigenvectors = torch.flip(eigenvectors, dims=[-1])
        Un = eigenvectors[:, :, M:]
        F = Un @ Un.conj().transpose(-2, -1)
        coefficients = sum_of_diags_batched(F)
    return coefficients


def roots_of(coefficients: torch.Tensor) -> torch.Tensor:
    """和 find_roots_batched 一样的伴随矩阵求根。"""
    batch, length = coefficients.shape
    ones = torch.ones(length - 2, device=coefficients.device).to(coefficients.dtype)
    companion = torch.diag(ones, -1).unsqueeze(0).repeat(batch, 1, 1)
    companion[:, 0, :] = -coefficients[:, 1:] / coefficients[:, :1]
    return torch.linalg.eigvals(companion)


def report_structure(coefficients: torch.Tensor):
    print("[1] 系数结构")
    c = coefficients
    n = c.shape[-1]
    flip = torch.flip(c, dims=[-1])
    conj_flip = torch.flip(c.conj(), dims=[-1])
    scale = c.abs().max().item()
    print(f"    shape={tuple(c.shape)} dtype={c.dtype}  |c|max={scale:.4g}")
    print(f"    max|Im(c)|                      = {c.imag.abs().max().item():.4g}")
    print(f"    max|c_k - conj(c_{{n-1-k}})|      = {(c - conj_flip).abs().max().item():.4g}"
          "   <- 自逆(self-inversive)程度")
    print(f"    max|c_k - c_{{n-1-k}}|            = {(c - flip).abs().max().item():.4g}"
          "   <- 自反(palindromic)程度")

    # 自逆 <=> 可以写成 e^{i phi} * 实系数自反多项式; 把总相位旋掉再看
    phase = torch.angle(c[:, n // 2])
    c_real = c * torch.exp(-1j * phase).unsqueeze(-1)
    print(f"    去掉中间系数相位后 max|Im|        = {c_real.imag.abs().max().item():.4g}")
    print(f"    去掉中间系数相位后 max|c_k-c_{{n-1-k}}| = "
          f"{(c_real - torch.flip(c_real, dims=[-1])).abs().max().item():.4g}")


def report_roots(coefficients: torch.Tensor):
    print("[2] 根的位置")
    roots = roots_of(coefficients)
    radius = roots.abs()
    print(f"    |root| 范围 = [{radius.min().item():.6f}, {radius.max().item():.6f}]")
    print(f"    |(|root|-1)| 的最大值 = {(radius - 1).abs().max().item():.6f}")
    print(f"    其中 |(|root|-1)| < 1e-3 的比例 = "
          f"{((radius - 1).abs() < 1e-3).float().mean().item():.4f}")
    # 共轭配对程度: 排序后看 z 与 conj(1/z) 是否重合
    return roots


def report_hermitian_form(coefficients: torch.Tensor, roots: torch.Tensor):
    """检验"自逆多项式 -> Hermitian 束"的经典结论能否直接用于求根。

    结论(教科书): 对所有根都在单位圆上的自逆多项式, 存在 Hermitian 矩阵束
        (C0, C1)  使得  z = -lambda 且 lambda 为广义特征值。
    这里用最直接的构造做验证: 取伴随矩阵, 看 C1 @ C0 是否 Hermitian(即束可 Hermitian 化)。
    """
    print("[3] 能否换成 Hermitian 特征值问题")
    batch, length = coefficients.shape
    n = length - 1
    ones = torch.ones(length - 2, device=coefficients.device).to(coefficients.dtype)
    companion = torch.diag(ones, -1).unsqueeze(0).repeat(batch, 1, 1)
    companion[:, 0, :] = -coefficients[:, 1:] / coefficients[:, :1]
    print("    伴随矩阵本身: Hermitian 偏差 = "
          f"{(companion - companion.conj().transpose(-2, -1)).abs().max().item():.4g}"
          "  <- 不是 Hermitian, 所以不能直接 eigvalsh")

    # 直接检验"用 eigvalsh 求 Hermitian 束"的可行构造:
    # 对自逆多项式, 取 P(z)=z^M p(z + 1/z) 的实系数形式, 再求 w 的实根(z 在单位圆上)。
    n_c = coefficients.shape[-1]
    half = (n_c - 1) // 2
    coeff_c = coefficients * torch.exp(
        -1j * torch.angle(coefficients[:, half]).unsqueeze(-1)
    )
    coeff_real = coeff_c.real  # 自反的实数系数
    m = half
    if (n_c - 1) % 2 != 0:
        print("    多项式次数为奇数, 跳过 w = z + 1/z 的降次检验")
        return
    # 由自反系数构造 P(w) = sum b_j w^j, 满足 z^{-m} p(z) = P(z + z^{-1})
    # 递推: b_m = a_m(中间), 逐层外推
    b = torch.zeros(coefficients.shape[0], m + 1, device=coefficients.device,
                    dtype=coefficients.real.dtype)
    b[:, m] = coeff_real[:, m]
    upper = coeff_real[:, m + 1:]
    # 反推: a_{m+k} = b_{m-k}??? 用标准递推 a_{m-k} 与 b 的关系
    # p(z) = z^m P(z + z^{-1});  z^m 与 z^{-m} 对称 -> a_{m+k} = a_{m-k}
    # 令 P(w)=sum_{j=0}^{m} b_j w^j, 则 a_{m+k} = sum_{j>=k} b_j * C(j, (j-k)/2) * [j-k 偶数]
    from math import comb

    for k in range(0, m + 1):
        total = torch.zeros(coefficients.shape[0], device=coefficients.device,
                            dtype=coefficients.real.dtype)
        for j in range(k, m + 1):
            if (j - k) % 2 == 0 and j - k >= 0:
                total = total + b[:, j] * comb(j, (j - k) // 2)
        # total 应等于 a_{m+k}
        target = coeff_real[:, m + k]
        if k == 0:
            print(f"    P(w) 构造自检: a_m={coeff_real[:, m].abs().max().item():.4g}")
        # 解 b: 从上式反解(从 j=m 往下)
    # 反解 b (上三角, 从 j=m 向下)
    for k in range(m, -1, -1):
        total = torch.zeros(coefficients.shape[0], device=coefficients.device,
                            dtype=coefficients.real.dtype)
        for j in range(k + 1, m + 1):
            if (j - k) % 2 == 0:
                total = total + b[:, j] * comb(j, (j - k) // 2)
        b[:, k] = coeff_real[:, m + k] - total

    # P(w) 的根(实根) -> z = (w ± sqrt(w^2-4))/2
    b_desc = torch.flip(b, dims=[-1])
    degree = b.shape[-1]
    ones = torch.ones(degree - 2, device=coefficients.device).to(b.dtype)
    comp = torch.diag(ones, -1).unsqueeze(0).repeat(b.shape[0], 1, 1)
    comp[:, 0, :] = -b_desc[:, 1:] / b_desc[:, :1]
    w_roots = torch.linalg.eigvals(comp)
    print(f"    w 根的最大虚部 = {w_roots.imag.abs().max().item():.4g}"
          "  <- 接近 0 说明 P(w) 的根是实数(降次可行)")
    z_from_w = 0.5 * (w_roots.real + torch.sqrt(
        (w_roots.real ** 2 - 4).to(torch.complex64)))
    print(f"    由 w 还原的 z: |(|z|-1)| 最大 = {(z_from_w.abs() - 1).abs().max().item():.4g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    print(f"device={device} torch={torch.__version__}")
    coefficients = polynomial_coefficients(args.batch)
    report_structure(coefficients)
    roots = report_roots(coefficients)
    report_hermitian_form(coefficients, roots)


if __name__ == "__main__":
    main()
