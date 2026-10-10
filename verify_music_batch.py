"""回归: `src/methods.py` 里被向量化的 MUSIC 谱 / MVDR 响应曲线是否与逐角度循环等价.

背景: 评估阶段最贵的一步不是神经网络, 而是 MUSIC 在 18000 点网格上的
`spectrum_calculation` —— 它逐角度调用 `steering_vec` 再算一次二次型, 实测
157.7 ms/样本, 是经典 r-music(0.286) / esprit(0.140) 的几百倍.

本脚本把"改写前的逐角度循环"原样内联为参考实现, 与仓库当前实现逐点比较:
  [1] `SystemModel.steering_vec_batch` vs 逐角度 `steering_vec(nominal=True)`
  [2] `MUSIC.spectrum_calculation` vs 逐角度参考
  [3] `MVDR.narrowband` 的响应曲线 vs 逐角度参考
  [4] 耗时对照

用法:
    python verify_music_batch.py            # 含计时
    python verify_music_batch.py --quick    # 跳过计时
退出码 0 = 全部通过.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.methods import MVDR, MUSIC, RootMUSIC  # noqa: E402
from src.system_model import SystemModel, SystemModelParams  # noqa: E402

TOLERANCE = 1e-6


def build_case(N=8, M=2, T=100, signal_type="NarrowBand", seed=0):
    """构造一个良态的窄带测试用例.

    注意: 两个信源必须是**非相干**的 (各自独立的随机波形), 否则经验协方差退化成秩 1,
    MVDR 的 `inv(C)` 在零空间上被放大, 逐角度循环与批量写法会因浮点路径不同而放大出
    百分之几十的差异 —— 那是病态用例, 不是实现差异.
    """
    params = (
        SystemModelParams()
        .set_parameter("N", N)
        .set_parameter("M", M)
        .set_parameter("T", T)
        .set_parameter("signal_type", signal_type)
        .set_parameter("eta", 0.0)
        .set_parameter("bias", 0.0)
        .set_parameter("sv_noise_var", 0.0)
    )
    system_model = SystemModel(params)
    rng = np.random.default_rng(seed)
    angles = np.array([-0.4, 0.35])
    A = np.stack([system_model.steering_vec(theta=a, nominal=True) for a in angles]).T
    S = rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))   # 每行独立 -> 非相干
    V = (rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T))) * 0.1
    X = A @ S + V
    return system_model, X


def steering_reference(system_model, angels, f=1.0):
    return np.stack(
        [
            system_model.steering_vec(theta=a, f=f, array_form="ULA", nominal=True)
            for a in angels
        ]
    )


def spectrum_reference(music, Un, f=1.0):
    """改写前 `MUSIC.spectrum_calculation` 的逐角度循环, 原样保留."""
    core_equation = []
    for angle in music._angels:
        a = music.system_model.steering_vec(
            theta=angle, f=f, array_form="ULA", nominal=True
        )[: Un.shape[0]]
        core_equation.append(np.conj(a).T @ Un @ np.conj(Un).T @ a)
    core_equation = np.array(core_equation, dtype=complex)
    return 1 / core_equation, core_equation


def mvdr_reference(mvdr, X, eps=1.0):
    """改写前 `MVDR.narrowband` 的逐角度循环, 原样保留."""
    covariance_mat = mvdr.calculate_covariance(X=X, mode="sample")
    diagonal_loaded_covariance = covariance_mat + eps * np.trace(
        covariance_mat
    ) * np.identity(covariance_mat.shape[0])
    inv_covariance = np.linalg.inv(diagonal_loaded_covariance)
    f = mvdr.system_model.max_freq[mvdr.system_model.params.signal_type]
    response_curve = []
    for angle in mvdr._angels:
        a = mvdr.system_model.steering_vec(
            theta=angle, f=f, array_form="ULA", nominal=True
        ).reshape((mvdr.system_model.params.N, 1))
        optimal_weights = (inv_covariance @ a) / (
            np.conj(a).T @ inv_covariance @ a
        ).item()
        response_curve.append(
            (
                (
                    np.conj(optimal_weights).T
                    @ diagonal_loaded_covariance
                    @ optimal_weights
                ).reshape((1))
            ).item()
        )
    return np.asarray(response_curve), covariance_mat


def check_steering(quick=False):
    print("\n=== [1] steering_vec_batch vs 逐角度 steering_vec ===")
    system_model, _ = build_case()
    worst = 0.0
    for signal_type in ("NarrowBand", "Broadband"):
        system_model.params.signal_type = signal_type
        f = 1.0 if signal_type == "NarrowBand" else 0.02
        angels = np.linspace(-np.pi / 2, np.pi / 2, 18000, endpoint=False)
        ref = steering_reference(system_model, angels, f=f)
        got = system_model.steering_vec_batch(angels, f=f, nominal=True)
        diff = np.abs(ref - got).max()
        worst = max(worst, diff)
        print(f"  {signal_type:10s} shape ref={ref.shape} got={got.shape}  max|diff|={diff:.3e}")
        assert ref.shape == got.shape, f"shape mismatch: {ref.shape} vs {got.shape}"
    try:
        system_model.steering_vec_batch(np.array([0.1]), nominal=False)
    except ValueError:
        print("  nominal=False 被正确拒绝 (不会悄悄消耗随机数)")
    else:
        raise AssertionError("steering_vec_batch(nominal=False) 应当抛 ValueError")
    assert worst < TOLERANCE, f"steering_vec_batch 偏差 {worst:.3e} >= {TOLERANCE}"
    print(f"  => 通过 (max|diff| = {worst:.3e} < {TOLERANCE})")


def check_spectrum(quick=False):
    print("\n=== [2] MUSIC.spectrum_calculation 批量版 vs 逐角度参考 ===")
    system_model, X = build_case()
    music = MUSIC(system_model)
    Un, _ = music.subspace_separation(
        covariance_mat=music.calculate_covariance(X=X, mode="sample"),
        M=system_model.params.M,
    )
    ref_spec, ref_core = spectrum_reference(music, Un)
    got_spec, got_core = music.spectrum_calculation(Un)
    assert got_spec.shape == ref_spec.shape, f"{got_spec.shape} vs {ref_spec.shape}"
    d_core = np.abs(ref_core - got_core).max()
    d_spec = np.abs(ref_spec - got_spec).max()
    rel_core = d_core / max(np.abs(ref_core).max(), 1e-30)
    print(f"  core_equation max|diff| = {d_core:.3e} (rel {rel_core:.3e})")
    print(f"  spectrum      max|diff| = {d_spec:.3e}")
    print(f"  spectrum 形状 {got_spec.shape}, 峰值位置 {int(np.argmax(got_spec))} "
          f"(参考 {int(np.argmax(ref_spec))})")
    assert rel_core < TOLERANCE, f"core_equation 相对偏差 {rel_core:.3e} >= {TOLERANCE}"
    # 端到端: 预测角度必须逐位一致
    p_ref = music._angels[music.get_spectrum_peaks(ref_spec)][: system_model.params.M]
    p_got = music._angels[music.get_spectrum_peaks(got_spec)][: system_model.params.M]
    d_pred = np.abs(np.sort(p_ref) - np.sort(p_got)).max()
    print(f"  端到端预测角度 max|diff| = {d_pred:.3e} rad")
    assert d_pred < TOLERANCE, f"预测角度偏差 {d_pred:.3e}"

    # 与 Root-MUSIC 的预测互为对照 (两者应落在相近角度上)
    rm = RootMUSIC(system_model)
    rm_preds = rm.narrowband(X=X, mode="sample")[0]
    print(f"  参考: MUSIC 预测 {np.round(np.sort(p_got) * 180 / np.pi, 2)} deg | "
          f"r-music 预测 {np.round(np.sort(rm_preds), 2)} deg")
    print("  => 通过")


def check_mvdr(quick=False):
    print("\n=== [3] MVDR 响应曲线批量版 vs 逐角度参考 ===")
    system_model, X = build_case()
    mvdr = MVDR(system_model)
    ref_curve, _ = mvdr_reference(mvdr, X)
    got_curve = mvdr.narrowband(X=X, mode="sample")[1]
    got_curve = np.asarray(got_curve, dtype=float).ravel()
    d = np.abs(ref_curve - got_curve).max()
    rel = d / max(np.abs(ref_curve).max(), 1e-30)
    print(f"  shape ref={ref_curve.shape} got={got_curve.shape}  max|diff| = {d:.3e} (rel {rel:.3e})")
    assert ref_curve.shape == got_curve.shape, "shape mismatch"
    assert rel < TOLERANCE, f"MVDR 相对偏差 {rel:.3e} >= {TOLERANCE}"
    print(f"  峰值位置 {int(np.argmax(got_curve))} (参考 {int(np.argmax(ref_curve))})")
    print("  => 通过")


def check_speed(quick=False):
    print("\n=== [4] 耗时对照 (18000 点网格) ===")
    system_model, X = build_case()
    music = MUSIC(system_model)
    Un, _ = music.subspace_separation(
        covariance_mat=music.calculate_covariance(X=X, mode="sample"),
        M=system_model.params.M,
    )
    mvdr = MVDR(system_model)

    def timeit(fn, repeats=5):
        fn()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t0) / repeats

    t_ref = timeit(lambda: spectrum_reference(music, Un))
    t_got = timeit(lambda: music.spectrum_calculation(Un))
    print(f"  MUSIC 谱: 参考 {t_ref*1e3:9.2f} ms/样本 -> 批量 {t_got*1e3:7.3f} ms/样本 "
          f"({t_ref/t_got:.1f}x)")
    t_ref_m = timeit(lambda: mvdr_reference(mvdr, X))
    t_got_m = timeit(lambda: mvdr.narrowband(X=X, mode="sample"))
    print(f"  MVDR 响应: 参考 {t_ref_m*1e3:9.2f} ms/样本 -> 批量 {t_got_m*1e3:7.3f} ms/样本 "
          f"({t_ref_m/t_got_m:.1f}x)")
    # 经典两算法作为量级参照
    rm = RootMUSIC(system_model)
    t_rm = timeit(lambda: rm.narrowband(X=X, mode="sample"))
    print(f"  参照: Root-MUSIC {t_rm*1e3:.3f} ms/样本")
    print("  => 完成")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="跳过计时")
    args = parser.parse_args()

    print("=" * 72)
    print("MUSIC 谱 / MVDR 响应: 向量化实现的等价性回归")
    print("=" * 72)
    check_steering(args.quick)
    check_spectrum(args.quick)
    check_mvdr(args.quick)
    if not args.quick:
        check_speed(args.quick)
    print("\n" + "=" * 72)
    print("结论: 全部通过")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
