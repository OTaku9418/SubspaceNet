"""通道幅相误差 Γ 的回归测试。

对应 `docs/TASK_gain_phase_errors.md` §5 的 8 条用例，另加 3 条：
Γ 与 steering-vector 加噪的复合顺序、`nominal` 与 `gamma` 互斥、以及
`shared_array_mismatch=False` 时保持"逐信源各抽一套"的历史行为。

全部用例都不训练网络、不写磁盘，几秒跑完：``python -m pytest tests -q``
"""

from __future__ import annotations

import numpy as np
import pytest

from src.signal_creation import Samples
from src.system_model import (
    ArrayMismatchState,
    SystemModel,
    SystemModelParams,
    mismatch_suffix,
)

N = 8
M = 2


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def make_params(**overrides) -> SystemModelParams:
    """一套"没有别的失配"的参数，只让测试关心的那一项起作用。"""
    params = (
        SystemModelParams()
        .set_parameter("N", N)
        .set_parameter("M", M)
        .set_parameter("T", 100)
        .set_parameter("snr", 10)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0.0)
        .set_parameter("bias", 0.0)
        .set_parameter("sv_noise_var", 0.0)
    )
    for key, value in overrides.items():
        params.set_parameter(key, value)
    return params


def make_model(**overrides) -> SystemModel:
    return SystemModel(make_params(**overrides))


def rng_fingerprint() -> tuple:
    """把 ``np.random`` 的全局状态压成一个可比较的指纹。"""
    state = np.random.get_state()
    return (state[0], state[1].tobytes(), state[2], state[3], state[4])


def nominal_manifold(model: SystemModel, theta: float) -> np.ndarray:
    return model.steering_vec(np.array([theta]), nominal=True)


def expected_manifold(model: SystemModel, theta: float, state) -> np.ndarray:
    """按 `steering_vec` 的公式手算一遍（bias/eta/间距都从 state 取）。"""
    phase = -2 * np.pi * (state.bias + state.dist_offset + 0.5) * model.array * np.sin(theta)
    return np.exp(1j * phase) * state.gamma


# --------------------------------------------------------------------------- #
# 1. σ=0 时不得扰动随机流（§4.1 硬指标的前提）
# --------------------------------------------------------------------------- #
def test_zero_params_do_not_touch_rng():
    np.random.seed(2024)
    model = make_model()

    before = rng_fingerprint()
    gamma = model._draw_gamma()
    assert np.array_equal(gamma, np.ones(N, dtype=np.complex128))
    assert gamma.dtype == np.complex128
    assert rng_fingerprint() == before, "_draw_gamma 在 σ=0 时必须零消耗"

    before = rng_fingerprint()
    model.steering_vec(np.array([0.3]), nominal=True)
    assert rng_fingerprint() == before, "nominal 流形必须零消耗"

    before = rng_fingerprint()
    model.steering_vec_batch(np.linspace(-1.0, 1.0, 5), nominal=True)
    assert rng_fingerprint() == before, "批量 nominal 流形必须零消耗"


# --------------------------------------------------------------------------- #
# 2 + 3. 生成器统计量与取值范围
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sigma_g, sigma_phi_deg",
    [(0.05, 0.0), (0.0, 5.0), (0.1, 15.0), (0.05, 5.0)],
)
def test_generator_statistics_and_range(sigma_g, sigma_phi_deg):
    np.random.seed(1234)
    model = make_model(gain_err_std=sigma_g, phase_err_std_deg=sigma_phi_deg)

    draws = 20000
    gamma = np.stack([model._draw_gamma() for _ in range(draws)])  # (draws, N)
    gains = np.abs(gamma).ravel()
    phases_deg = np.rad2deg(np.angle(gamma)).ravel()

    if sigma_g > 0:
        # 均匀分布的标准差精确等于 σg，均值精确等于 1
        assert abs(gains.mean() - 1.0) < 0.005
        assert abs(gains.std() / sigma_g - 1.0) < 0.01
        # 硬边界：|g-1| <= sqrt(3)*σg
        assert np.abs(gains - 1.0).max() <= np.sqrt(3.0) * sigma_g + 1e-12
    else:
        # 相位扰动也会让模长有 ~1e-16 的舍入误差，不能直接 == 1.0
        assert np.allclose(gains, 1.0, rtol=0, atol=1e-15)

    if sigma_phi_deg > 0:
        assert abs(phases_deg.mean()) < 0.2
        assert abs(phases_deg.std() / sigma_phi_deg - 1.0) < 0.01
        assert np.abs(phases_deg).max() <= np.sqrt(3.0) * sigma_phi_deg + 1e-9
    else:
        assert np.all(phases_deg == 0.0)


# --------------------------------------------------------------------------- #
# 4 + 5. 同一 Γ 跨信源共享、且与来向无关（Step A）
# --------------------------------------------------------------------------- #
def test_gamma_is_shared_across_sources_and_angle_independent():
    model = Samples(
        make_params(gain_err_std=0.1, phase_err_std_deg=5.0)
    )
    angles = [-20.0, 30.0]
    model.set_doa(angles)
    _, _, A, _ = model.samples_creation()

    assert A.shape == (N, M)
    ratios = []
    for column, theta in enumerate(model.doa):
        ratios.append(A[:, column] / model.steering_vec(theta, nominal=True))

    # 两个信源除出来的 Γ 必须一样 —— 一副阵列只有一套通道响应
    assert np.allclose(ratios[0], ratios[1], rtol=1e-12, atol=1e-12)
    # 而且确实不是全 1（否则这个测试什么都没测）
    assert not np.allclose(ratios[0], np.ones(N))


# --------------------------------------------------------------------------- #
# 6. 缓存与权重文件名
# --------------------------------------------------------------------------- #
def test_filename_only_mentions_new_fields_when_non_default():
    from src.data_handler import set_dataset_filename
    from src.models import ModelGenerator
    from src.training import get_simulation_filename

    def names(**overrides):
        params = make_params(**overrides)
        generator = (
            ModelGenerator()
            .set_model_type("SubspaceNet")
            .set_diff_method("root_music")
            .set_tau(8)
            .set_model(params)
        )
        return (
            set_dataset_filename(params, 2000),
            get_simulation_filename(params, generator),
        )

    base = names()
    for name in base:
        assert "sg=" not in name and "sphi=" not in name and "share=" not in name
    assert base == names(), "同参数两次调用必须给出同一个名字"

    variants = [
        names(gain_err_std=0.05),
        names(gain_err_std=0.1),
        names(phase_err_std_deg=5.0),
        names(shared_array_mismatch=True),
    ]
    seen = {base}
    for variant in variants:
        assert variant not in seen, "不同的失配参数不能撞到同一个缓存文件"
        for name in variant:
            assert "_sg=" in name and "_sphi=" in name and "_share=" in name
        seen.add(variant)

    # 两个 sigma 都是 0 时 share 仍然必须进名字，否则 Step B 会读到 Step A 的缓存
    assert "share=1" in names(shared_array_mismatch=True)[0]
    assert "share=0" in names(gain_err_std=0.05)[0]


def test_mismatch_suffix_tagging():
    assert mismatch_suffix(make_params()) == ""
    assert mismatch_suffix(make_params(gain_err_std=0.05)) == "_sg=0p05_sphi=0p0_share=0"
    assert (
        mismatch_suffix(make_params(phase_err_std_deg=10.0, shared_array_mismatch=True))
        == "_sg=0p0_sphi=10p0_share=1"
    )


# --------------------------------------------------------------------------- #
# 7. 批量接口与两种 ValueError
# --------------------------------------------------------------------------- #
def test_batch_manifold_matches_per_angle():
    model = make_model(gain_err_std=0.1, phase_err_std_deg=5.0)
    np.random.seed(0)
    gamma = model._draw_gamma()

    thetas = np.linspace(-1.0, 1.0, 37)
    batch = model.steering_vec_batch(thetas, gamma=gamma)
    per_angle = np.stack([model.steering_vec(t, gamma=gamma) for t in thetas])

    assert batch.shape == (thetas.size, N)
    assert np.allclose(batch, per_angle, rtol=0, atol=1e-12)


def test_batch_manifold_rejects_ambiguous_calls():
    model = make_model(gain_err_std=0.1)
    thetas = np.linspace(-1.0, 1.0, 5)
    gamma = np.ones(N, dtype=np.complex128)

    with pytest.raises(ValueError):
        # nominal=False 但不给 gamma：谁来抽 Γ 是不明确的
        model.steering_vec_batch(thetas, nominal=False)
    with pytest.raises(ValueError):
        # nominal=True 又给 gamma：Γ 无处安放
        model.steering_vec_batch(thetas, nominal=True, gamma=gamma)


def test_nominal_forbids_gamma():
    model = make_model()
    with pytest.raises(ValueError):
        model.steering_vec(np.array([0.1]), nominal=True, gamma=np.ones(N, dtype=complex))


# --------------------------------------------------------------------------- #
# 8 + 9. Step B：位置误差是阵列级的，跨信源共享且零随机消耗
# --------------------------------------------------------------------------- #
def test_state_path_is_deterministic_and_zero_consumption():
    model = make_model(
        eta=0.05,
        bias=0.02,
        gain_err_std=0.1,
        phase_err_std_deg=5.0,
        shared_array_mismatch=True,
    )
    np.random.seed(7)
    state = model._draw_mismatch_state()
    assert isinstance(state, ArrayMismatchState)
    assert state.gamma.shape == (N,)

    before = rng_fingerprint()
    zeros = np.zeros(N)
    first = model.steering_vec(np.array([-0.3]), state=state, sv_noise=zeros)
    second = model.steering_vec(np.array([0.4]), state=state, sv_noise=zeros)
    assert rng_fingerprint() == before, "state 路径必须零随机消耗"

    for got, theta in ((first, -0.3), (second, 0.4)):
        assert np.allclose(got, expected_manifold(model, theta, state), rtol=0, atol=1e-12)


def test_samples_creation_shares_state_across_sources(monkeypatch):
    model = Samples(
        make_params(
            eta=0.05,
            gain_err_std=0.1,
            phase_err_std_deg=5.0,
            shared_array_mismatch=True,
        )
    )
    model.set_doa([-20.0, 30.0])

    gamma = np.exp(1j * np.linspace(0.0, 0.6, N))
    state = ArrayMismatchState(
        bias=0.01, dist_offset=np.linspace(-0.02, 0.02, N), gamma=gamma
    )
    monkeypatch.setattr(model, "_draw_mismatch_state", lambda: state)

    _, _, A, _ = model.samples_creation()
    assert A.shape == (N, M)
    for column, theta in enumerate(model.doa):
        assert np.allclose(
            A[:, column], expected_manifold(model, theta, state), rtol=0, atol=1e-12
        )


def test_step_a_still_draws_per_call():
    """`shared_array_mismatch=False` 时保持历史行为：每次调用各抽一套。"""
    model = make_model(eta=0.05)
    np.random.seed(11)
    first = model.steering_vec(np.array([0.2]))
    second = model.steering_vec(np.array([0.2]))
    assert not np.allclose(first, second)


# --------------------------------------------------------------------------- #
# 10. Γ 乘在 (导向矢量 + steering-vector 加噪) 外面
# --------------------------------------------------------------------------- #
def test_gamma_multiplies_the_whole_manifold_including_sv_noise():
    model = make_model()
    theta = np.array([0.25])
    gamma = np.exp(1j * np.linspace(0.0, 1.0, N))
    noise = np.linspace(-0.1, 0.1, N)

    state = ArrayMismatchState(bias=0.0, dist_offset=np.zeros(N), gamma=gamma)
    got = model.steering_vec(theta, state=state, sv_noise=noise)
    want = (model.steering_vec(theta, nominal=True) + noise) * gamma

    assert np.allclose(got, want, rtol=0, atol=1e-12)
