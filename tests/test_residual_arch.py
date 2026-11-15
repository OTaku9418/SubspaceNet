"""残差臂 `SubspaceNetResidual` 的结构与等价性回归测试。

对应 `docs/plan_residual_network_2026-11-15.md` §3.5 / §4 的三条要求，另加一条
分派顺序检查（计划 §3.2 自己点出的坑：`SubspaceNetResidual` 若排在 `SubspaceNet`
之后会被前缀匹配吃掉）：

1. 参数量仍是 41761（两臂相同，论文 Sec. IV-A-2 同值）；
2. state_dict 键名与形状与基线逐字相同（基线权重可原位 `load_state_dict`）；
3. `skip_scale = 0.0` 时输出与基线**逐位**相同（不是 allclose）；
4. `ModelGenerator` 能把 `SubspaceNetResidual` 分派到正确的类。

两臂分处两个文件（`src/models.py` 与 `src/models_residual.py`），所以第 3 条是
**跨文件**对照：任何一侧的前向漂移都会让它失败。

全部用例不训练、不写盘，几秒跑完：``python -m pytest tests -q``
"""

from __future__ import annotations

import torch

from src.models import ModelGenerator, SubspaceNet
from src.models_residual import SubspaceNetResidual
from src.system_model import SystemModelParams
from src.utils import set_unified_seed

N = 8
M = 2
T = 100
TAU = 8
N_PARAMS = 41761
SEED = 1234


def _params() -> SystemModelParams:
    """一套"没有失配"的系统参数，只用来建模型（不生成数据）。"""
    return (
        SystemModelParams()
        .set_parameter("N", N)
        .set_parameter("M", M)
        .set_parameter("T", T)
        .set_parameter("snr", 10)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0.0)
        .set_parameter("bias", 0.0)
        .set_parameter("sv_noise_var", 0.0)
    )


def _pair() -> tuple[SubspaceNet, SubspaceNetResidual]:
    """同 seed 建两臂：继承同一 `__init__`、层形状不变 => 初始权重逐位相同。"""
    set_unified_seed(SEED)
    base = SubspaceNet(tau=TAU, M=M, diff_method="root_music")
    set_unified_seed(SEED)
    res = SubspaceNetResidual(tau=TAU, M=M, diff_method="root_music")
    return base.eval(), res.eval()


def _bits(t: torch.Tensor) -> torch.Tensor:
    """把浮点张量重解释成 int32，用来做真正的逐位比较。

    `torch.equal` 只比数值（-0.0 与 +0.0 算相等），这里要的是 IEEE-754 位模式。
    """
    t = t.detach().contiguous()
    if t.is_complex():
        t = torch.view_as_real(t)
    return t.view(torch.int32)


def test_parameter_count_is_unchanged():
    base, res = _pair()
    assert sum(p.numel() for p in base.parameters()) == N_PARAMS
    assert sum(p.numel() for p in res.parameters()) == N_PARAMS


def test_state_dict_keys_are_identical():
    base, res = _pair()
    sd_base, sd_res = base.state_dict(), res.state_dict()
    assert set(sd_base) == set(sd_res)
    for key in sd_base:
        assert sd_base[key].shape == sd_res[key].shape, key
    # 基线权重能严格灌进残差臂（将来做"从基线微调"的对照要用这条）
    res.load_state_dict(sd_base)
    assert "skip_scale" not in sd_res, "跳连系数不能进 state_dict，否则权重不再可互换"


def test_skip_scale_zero_matches_baseline_bitwise():
    base, res = _pair()
    res.load_state_dict(base.state_dict())
    res.skip_scale = 0.0

    set_unified_seed(7)
    rx_tau = torch.randn(4, TAU, 2 * N, N)          # (B, tau, 2N, N)，与数据集同形
    with torch.no_grad():
        out_base = base(rx_tau)
        out_res = res(rx_tau)

    assert out_base[0].shape == out_res[0].shape
    for name, a, b in (
        ("Rz", out_base[3], out_res[3]),
        ("doa_prediction", out_base[0], out_res[0]),
        ("doa_all_predictions", out_base[1], out_res[1]),
        ("roots", out_base[2], out_res[2]),
    ):
        assert a.shape == b.shape, f"{name} 形状不一致: {a.shape} vs {b.shape}"
        assert torch.equal(_bits(a), _bits(b)), f"{name} 在 skip_scale=0 时与基线不逐位相同"

    # 反向自检：默认 skip_scale=1.0 时输出必须真的不同，否则跳连等于没接上
    res.skip_scale = 1.0
    with torch.no_grad():
        out_default = res(rx_tau)
    assert not torch.equal(_bits(out_default[3]), _bits(out_base[3]))


def test_model_generator_dispatches_residual_class():
    gen = (
        ModelGenerator()
        .set_model_type("SubspaceNetResidual")
        .set_diff_method("root_music")
        .set_tau(TAU)
        .set_model(_params())
    )
    assert type(gen.model) is SubspaceNetResidual
    assert gen.model_type == "SubspaceNetResidual"
    # 基线不能被前缀匹配误伤
    gen_base = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_diff_method("root_music")
        .set_tau(TAU)
        .set_model(_params())
    )
    assert type(gen_base.model) is SubspaceNet
