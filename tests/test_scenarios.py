"""组合误差场景（`gain_phase`）的接线：一个标量 u 同时驱动 σ_g 与 σ_φ。

这个场景的横轴是**虚拟**量 `severity_u`，`SystemModelParams` 里根本没有这个字段，
必须靠 `SCENARIOS["gain_phase"]["pair_scale"]` 换算成两个真实旋钮。换算一旦写错，
曲线不会报错，只会静悄悄地画出另一组失配 —— 所以这里逐点钉死。
"""
import pytest

import reproduce_array_mismatch as R
from src.system_model import mismatch_suffix


def test_pair_scale_matches_the_documented_diagonal():
    """u -> (σ_g, σ_φ) = u·(0.3, 40°)，其余旋钮全关。"""
    cfg = R.SCENARIOS["gain_phase"]
    assert cfg["pair_scale"] == (0.3, 40.0)
    for u in cfg["grid"]:
        params = R.params_for("gain_phase", u)
        assert params.gain_err_std == pytest.approx(0.3 * u, abs=1e-12)
        assert params.phase_err_std_deg == pytest.approx(40.0 * u, abs=1e-12)
        # 组合场景只开这两个旋钮；位置/加噪/偏置必须仍然是 0，否则曲线上会混进
        # 别的失配来源，"只增益 vs 只相位 vs 组合"三者的可比性就没了。
        assert params.eta == 0.0
        assert params.bias == 0.0
        assert params.sv_noise_var == 0.0
        assert params.shared_array_mismatch is True


def test_non_pair_scenarios_still_sweep_a_single_real_parameter():
    """原有四个场景不受影响：`pair_for` 返回 None，扫描的是真实参数。"""
    for scenario in ("spacing", "sv_noise", "gain", "phase"):
        assert R.pair_for(scenario, 0.05) is None
    assert R.params_for("spacing", 0.05).eta == pytest.approx(0.05)
    assert R.params_for("sv_noise", 0.25).sv_noise_var == pytest.approx(0.25)
    assert R.params_for("gain", 0.06).gain_err_std == pytest.approx(0.06)
    assert R.params_for("phase", 4.0).phase_err_std_deg == pytest.approx(4.0)


def test_zero_severity_is_the_same_mismatch_as_the_single_scenarios():
    """u=0 与 gain@0 / phase@0 是同一套失配（因此数据逐位相同）。"""
    zero = R.params_for("gain_phase", 0.0)
    for scenario in ("gain", "phase"):
        other = R.params_for(scenario, 0.0)
        assert zero.gain_err_std == other.gain_err_std == 0.0
        assert zero.phase_err_std_deg == other.phase_err_std_deg == 0.0
        assert zero.shared_array_mismatch == other.shared_array_mismatch is True
        assert mismatch_suffix(zero) == mismatch_suffix(other)


def test_every_grid_point_gets_a_distinct_cache_suffix():
    """`_sg=…_sphi=…_share=…` 必须逐点不同，否则后一个点会命中前一个点的缓存。"""
    grid = R.SCENARIOS["gain_phase"]["grid"]
    suffixes = [mismatch_suffix(R.params_for("gain_phase", u)) for u in grid]
    assert len(set(suffixes)) == len(grid)
    # 且每个点都带 share=1（组合场景走 per-sample 的 state 路径）
    assert all("_share=1" in s for s in suffixes)


def test_level_label_shows_the_physical_pair():
    """图/日志里的人话标签要同时给出 u 与它对应的物理配对。"""
    label = R.level_label("gain_phase", 0.5)
    assert "severity_u=0.5" in label
    assert "15.0%" in label          # σ_g = 0.15
    assert "20.0°" in label          # σ_φ = 20°
    # 非组合场景不加配对，保持原样
    assert R.level_label("spacing", 0.05) == "eta=0.05"
