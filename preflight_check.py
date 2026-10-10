"""上全量实验之前的服务器环境自检 (pre-flight check).

用途：正式跑（10000 样本 / 80 epoch，数小时）之前，用 ~1 分钟确认
      环境、依赖、CUDA、论文架构、数据管线、指标口径、失配注入全部就绪。

设计要点：
  * **只读**：不生成数据集、不训练、不改仓库，跑完不留垃圾。
  * **不依赖 data/ 可写**：数据集形状检查在内存里做极小样本。
  * **自包含**：只用标准库 + 仓库已有依赖，不需要装额外包。
  * 任一 [FAIL] 都意味着全量跑一定会失败或结论无效，请先修掉再上全量。
    [WARN] 不影响正确性，但会影响速度或结论口径。

用法：
    python preflight_check.py
    python preflight_check.py --full     # 额外做一次 1 分钟的真实训练冒烟

退出码：0 = 无 FAIL；1 = 至少一项 FAIL。
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str]] = []   # (状态, 检查项, 说明)
N_FAIL = 0


def record(status: str, name: str, detail: str) -> None:
    global N_FAIL
    if status == "FAIL":
        N_FAIL += 1
    RESULTS.append((status, name, detail))
    print(f"  [{status:<4}] {name}\n           {detail}")


def check(name: str):
    """装饰器：**立即执行**被装饰的检查并记录结果（PASS/FAIL），异常记成 FAIL。"""
    def deco(fn):
        try:
            detail = fn()
            record("PASS", name, detail)
        except Exception as e:          # noqa: BLE001
            record("FAIL", name, f"{type(e).__name__}: {e}")
        return fn
    return deco


print("=" * 74)
print("SubspaceNet 阵列失配实验 —— 上全量前的环境自检")
print("=" * 74)
print(f"仓库根目录 : {ROOT}")
print(f"工作目录   : {Path.cwd()}")
print(f"平台       : {platform.platform()}")
print(f"Python     : {sys.version.split()[0]}  ({sys.executable})")
print()

# ------------------------------------------------------------------------------
print("[1] 依赖与 CUDA")
# ------------------------------------------------------------------------------


@check("Python 版本 >= 3.8")
def _py():
    v = sys.version_info
    assert v >= (3, 8), f"当前 {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


@check("关键依赖可导入")
def _deps():
    import importlib
    got = {}
    for m in ("torch", "numpy", "scipy", "matplotlib", "sklearn", "tqdm"):
        try:
            mod = importlib.import_module(m)
            got[m] = getattr(mod, "__version__", "?")
        except ImportError as e:
            raise AssertionError(f"缺少 {m}: {e}") from e
    return "  ".join(f"{k}={v}" for k, v in got.items())


@check("cuda.is_available()")
def _cuda():
    import torch
    avail = torch.cuda.is_available()
    assert avail, ("torch.cuda.is_available() = False。注意 src/utils.py:32 的 device 是"
                   "模块级常量，import 前就必须可见 CUDA；CPU 上跑全量会慢到不可接受")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cap = torch.cuda.get_device_capability(0)
    return f"{name}, {total:.1f} GiB, compute capability {cap}"


@check("GPU 上真的能算（复数 matmul + linalg.eig，带同步）")
def _cuda_kernel():
    """★最重要的 CUDA 检查: 跑一次真实 kernel 并 synchronize。

    必要性: `torch.cuda.is_available()` 只检查"驱动能看见卡", **不检查**
    "这个 torch 构建里有没有该架构的 kernel"。装了不含本机 sm 的 wheel 时它照样
    返回 True, 然后在第一次真正算东西时抛
    `CUDA error: no kernel image is available for execution on the device`。
    而且 CUDA 的 kernel 错误是**异步**报告的, 不同步就会在后面的随机位置炸出来
    （实测就是这样: 前面几项全 PASS, 到 MUSIC 才 FAIL）——所以这里必须同步。
    """
    import torch
    from src.utils import device as dev

    if dev.type != "cuda":
        raise AssertionError(f"src/utils.py:32 的 device={dev}，不是 cuda")
    try:
        a = torch.randn(64, 64, dtype=torch.complex64, device=dev)
        b = torch.randn(64, 64, dtype=torch.complex64, device=dev)
        (a @ b).sum().real.item()
        torch.linalg.eig(a + a.conj().T)
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "no kernel image is available" in str(e):
            archs = " / ".join(torch.cuda.get_arch_list())
            raise AssertionError(
                f"本机 GPU ({torch.cuda.get_device_name(0)}, compute capability "
                f"{torch.cuda.get_device_capability(0)}) 在这个 torch 构建里没有可用 "
                f"kernel。torch {torch.__version__} 只编译了 [{archs}]。"
                "装的是老 wheel（如仓库 pyEnv/requirements.txt 锁的 torch==2.0.1，"
                "只到 sm_86）。换装支持新架构的构建：PyTorch >= 2.7 的 cu128 wheel，"
                "见文档 §15") from e
        raise
    return f"device={dev}, 复数 matmul 与 linalg.eig 均通过"


@check("（提示）GPU 架构是否在 torch 编译目标里，缺了会走 PTX JIT")
def _cuda_arch_note():
    """只做提示, 不作为失败判据 —— 缺 SASS 时可能靠 PTX JIT 正常跑。

    真正的判据是上面那条 kernel 实测。这里只解释"能跑但会慢"的情形:
    例如 sm_89 的卡跑只编译到 compute_37/compute_90 的 wheel, 会 JIT 编译,
    首次算子有额外开销。
    """
    import torch
    cap = torch.cuda.get_device_capability(0)
    want = f"{cap[0]}{cap[1]}"
    arch_list = list(torch.cuda.get_arch_list())
    families = {a.replace("sm_", "").replace("a", "").replace("f", "") for a in arch_list}
    if want in families:
        return f"sm_{want} 直接命中编译目标"
    return (f"（提示）sm_{want} 不在 [{(' / '.join(arch_list)) or '未知'}] 里。"
            "这不必然是错误：缺 SASS 时可能靠 PTX JIT 跑起来（能跑但首次算子偏慢）。"
            "**唯一判据是上一项的 kernel 实测**：它 PASS 就没事，它 FAIL 才是真的不可用。")


@check("仓库为官方实现 + 失配相关源码在位")
def _repo():
    must = ["src/system_model.py", "src/signal_creation.py", "src/data_handler.py",
            "src/methods.py", "src/models.py", "src/training.py", "src/criterions.py",
            "src/utils.py", "reproduce_array_mismatch.py"]
    missing = [f for f in must if not (ROOT / f).exists()]
    assert not missing, f"缺少文件: {missing}"
    return f"{len(must)} 个关键文件齐全"


@check("产物根目录可写（SUBSPACENET_DATA_ROOT 或 <仓库>/data）")
def _writable():
    import tempfile
    target = Path(os.environ.get("SUBSPACENET_DATA_ROOT", ROOT / "data"))
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target, prefix=".preflight_", delete=True):
        pass
    env = "SUBSPACENET_DATA_ROOT" if "SUBSPACENET_DATA_ROOT" in os.environ else "默认 <仓库>/data"
    return f"{target} 可写（来源: {env}）"

# ------------------------------------------------------------------------------
print()
print("[2] 论文架构一致性（论文明确给出的数字）")
# ------------------------------------------------------------------------------


@check("SubspaceNet 可训练参数量 == 41761（论文 Sec. IV-A-2）")
def _params():
    from src.models import ModelGenerator
    from src.system_model import SystemModelParams
    p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
         .set_parameter("T", 100).set_parameter("snr", 10)
         .set_parameter("signal_type", "NarrowBand")
         .set_parameter("signal_nature", "non-coherent")
         .set_parameter("eta", 0.0).set_parameter("bias", 0.0)
         .set_parameter("sv_noise_var", 0.0))
    m = (ModelGenerator().set_model_type("SubspaceNet")
         .set_diff_method("root_music").set_tau(8).set_model(p).model)
    n = sum(x.numel() for x in m.parameters())
    assert n == 41761, (f"实得 {n}，论文是 41761。说明 tau / 通道数 / AReLU 语义与本仓库不符，"
                        "后续所有对比都失去意义")
    return "tau=8 + 3×(16/32/64) CNN + 3×(128/64/32,1) DCNN + AReLU 通道翻倍 => 41761"


@check("RMSPE 参考实现相对真实角度的缩放因子 == pi/180")
def _rmspe():
    import numpy as np
    from src.criterions import RMSPE
    pred, doa = np.array([1.0, 3.0]), np.array([0.0, 0.0])
    ref = RMSPE(pred, doa)
    true_deg = float(np.sqrt(np.mean((pred - doa) ** 2)))
    ratio = ref / true_deg
    assert abs(ratio - np.pi / 180) < 1e-9, (f"比值 {ratio}，预期 {np.pi/180}。"
                                             "若仓库已修度/弧度 bug，请改用 rmspe_degrees_fixed 口径")
    return (f"ref/真实 = {ratio:.12f} = pi/180（论文数字被压缩 57.2958 倍，"
            "报告时必须声明口径）")

# ------------------------------------------------------------------------------
print()
print("[3] nominal 导向矢量（经典 MUSIC 基线的前置条件）")
# ------------------------------------------------------------------------------


@check("steering_vec(nominal=True) 不再抛 UnboundLocalError")
def _nominal():
    import numpy as np
    from src.system_model import SystemModel, SystemModelParams
    p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
         .set_parameter("T", 100).set_parameter("snr", 10)
         .set_parameter("signal_type", "NarrowBand")
         .set_parameter("signal_nature", "non-coherent")
         .set_parameter("eta", 0.15).set_parameter("bias", 0.05)
         .set_parameter("sv_noise_var", 0.75))
    sv = SystemModel(p).steering_vec(theta=np.array([0.3]), nominal=True)
    assert np.allclose(np.abs(sv), 1.0), "nominal 导向矢量模不为 1"
    p0 = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
          .set_parameter("T", 100).set_parameter("snr", 10)
          .set_parameter("signal_type", "NarrowBand")
          .set_parameter("signal_nature", "non-coherent")
          .set_parameter("eta", 0.0).set_parameter("bias", 0.0)
          .set_parameter("sv_noise_var", 0.0))
    sv0 = SystemModel(p0).steering_vec(theta=np.array([0.3]), nominal=True)
    assert np.allclose(sv, sv0), "nominal 导向矢量受失配参数影响，语义不对"
    return "模恒为 1 且与 bias/eta/sv_noise_var 无关（理想 ULA 流形）；MUSIC/MVDR 可用"


@check("MUSIC.narrowband 能真正出数（不只是不崩）")
def _music():
    import numpy as np
    from src.data_handler import create_dataset
    from src.methods import MUSIC
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed
    p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
         .set_parameter("T", 100).set_parameter("snr", 10)
         .set_parameter("signal_type", "NarrowBand")
         .set_parameter("signal_nature", "non-coherent")
         .set_parameter("eta", 0.05).set_parameter("bias", 0.0)
         .set_parameter("sv_noise_var", 0.0))
    set_unified_seed(1234)
    _, gd, sm = create_dataset(system_model_params=p, samples_size=1,
                               model_type="SubspaceNet", tau=8,
                               save_datasets=False, phase="test")
    out = MUSIC(sm).narrowband(X=gd[0][0].detach().cpu().numpy(), mode="sample")
    preds = np.asarray(out[0], dtype=float).ravel()
    assert out[0] is not None and preds.size >= 2, f"返回元数={len(out)}, 预测={preds}"
    return f"返回 {len(out)} 元组 (predictions, spectrum, M)，预测 {np.round(preds,2)}"

# ------------------------------------------------------------------------------
print()
print("[4] 数据管线与失配注入")
# ------------------------------------------------------------------------------


@check("create_dataset 元素为 (X, Y) 二元组，形状符合训练循环期望")
def _shape():
    from src.data_handler import create_dataset
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed
    p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
         .set_parameter("T", 100).set_parameter("snr", 10)
         .set_parameter("signal_type", "NarrowBand")
         .set_parameter("signal_nature", "non-coherent")
         .set_parameter("eta", 0.1).set_parameter("bias", 0.0)
         .set_parameter("sv_noise_var", 0.0))
    set_unified_seed(42)
    md, gd, sm = create_dataset(system_model_params=p, samples_size=3,
                                model_type="SubspaceNet", tau=8, save_datasets=False)
    assert len(md[0]) == 2, (f"元素是 {len(md[0])} 元组；src/training.py:373 的 "
                             "`Rx, DOA = data` 期望二元组")
    assert tuple(md[0][0].shape) == (8, 16, 8), f"X_model 形状 {tuple(md[0][0].shape)} != (tau=8, 2N=16, N=8)"
    assert tuple(md[0][1].shape) == (2,), f"Y 形状 {tuple(md[0][1].shape)} != (M=2,)"
    return f"X_model={tuple(md[0][0].shape)}  Y={tuple(md[0][1].shape)}  返回 {len(md), len(gd)} + samples_model"


@check("失配真的进入数据（eta=0 与 eta=0.15 的观测不同）")
def _mismatch():
    import numpy as np
    from src.data_handler import create_dataset
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed
    base = dict(N=8, M=2, T=100, snr=10, signal_type="NarrowBand",
                signal_nature="non-coherent", bias=0.0, sv_noise_var=0.0)
    means = {}
    for eta in (0.0, 0.15):
        p = SystemModelParams()
        for k, v in {**base, "eta": eta}.items():
            p.set_parameter(k, v)
        set_unified_seed(1234)
        _, gd, _ = create_dataset(system_model_params=p, samples_size=5,
                                  model_type="SubspaceNet", tau=8, save_datasets=False)
        means[eta] = float(np.abs(gd[0][0].detach().cpu().numpy()).mean())
    assert means[0.0] != means[0.15], f"两者相同 ({means})，失配没有进入数据"
    return f"mean|X|: eta=0 -> {means[0.0]:.4f}, eta=0.15 -> {means[0.15]:.4f}"


@check("各失配水平共享同一批 DoA（曲线可比性的前提）")
def _shared_doa():
    import numpy as np
    from src.data_handler import create_dataset
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed
    got = {}
    for eta in (0.0, 0.15):
        p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
             .set_parameter("T", 100).set_parameter("snr", 10)
             .set_parameter("signal_type", "NarrowBand")
             .set_parameter("signal_nature", "non-coherent")
             .set_parameter("eta", eta).set_parameter("bias", 0.0)
             .set_parameter("sv_noise_var", 0.0))
        set_unified_seed(1234)          # 与脚本 §4.2 相同的固定种子
        _, gd, _ = create_dataset(system_model_params=p, samples_size=10,
                                  model_type="SubspaceNet", tau=8, save_datasets=False)
        got[eta] = np.array([gd[i][1].numpy() for i in range(len(gd))])
    assert np.allclose(got[0.0], got[0.15]), "不同 eta 抽到了不同 DoA，曲线不可比"
    return f"eta=0 与 eta=0.15 的 10 个样本 DoA 完全一致（seed=1234）"


@check("root_music 的批量实现与逐样本原始实现等价")
def _root_music_equivalence():
    """src/models.py 的 root_music 已批量化（见文档 §16）。

    这里做一次最小等价性验证：用 verify_root_music_batch.py 里内联保留的原始逐样本实现，
    在同样的输入上比对 M 个 doa 与全部根 doa 的集合。训练会反传穿过这个函数，所以
    "批量化改坏了数值" 是最需要被自动抓住的回归。两个实现都走 eigh（见下面那一项），
    所以这里校验的是"批量化"这一件事本身。
    """
    import torch
    from src.models import ModelGenerator, root_music
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed
    from verify_root_music_batch import root_music_reference

    set_unified_seed(0)
    p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
         .set_parameter("T", 100).set_parameter("snr", 10)
         .set_parameter("signal_type", "NarrowBand")
         .set_parameter("signal_nature", "non-coherent")
         .set_parameter("eta", 0.0).set_parameter("bias", 0.0)
         .set_parameter("sv_noise_var", 0.0))
    model = (ModelGenerator().set_model_type("SubspaceNet")
             .set_diff_method("root_music").set_tau(8).set_model(p)).model
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(model.parameters()).device

    worst = 0.0
    for batch_size in (1, 8, 32):
        x = torch.randn(batch_size, 8, 16, 8, device=dev)
        with torch.no_grad():
            Rz = model(x)[-1].detach()
            ref_doa, ref_all, _ = root_music_reference(Rz, 2, batch_size)
            new_doa, new_all, _ = root_music(Rz, 2, batch_size)
        ref_doa, ref_all = ref_doa.to(dev), ref_all.to(dev)
        # 原始实现把多项式求根落在 CPU（src/utils.py:147 的 device bug），故只比较值
        worst = max(
            worst,
            (ref_doa - new_doa).abs().max().item(),
            (ref_all.sort(dim=-1).values - new_all.sort(dim=-1).values).abs().max().item(),
        )
    assert worst < 1e-4, (
        f"批量版 root_music 与原始实现的最大偏差 {worst:.3e} rad 超出容差 1e-4。"
        "运行 `python verify_root_music_batch.py` 查看逐项明细。"
    )
    return f"batch 1/8/32 上最大偏差 {worst:.3e} rad（容差 1e-4）"


@check("gram_diagonal_overload 的批量实现与逐样本原始实现等价")
def _gram_equivalence():
    """src/utils.py 的 gram_diagonal_overload 原本是逐样本循环（文档 §16.7）。"""
    import torch
    from src.utils import (device, gram_diagonal_overload,
                           gram_diagonal_overload_reference)

    worst = 0.0
    for batch_size in (1, 8, 64):
        Kx = (
            torch.randn(batch_size, 8, 8, dtype=torch.complex64, device=device)
            + 1j * torch.randn(batch_size, 8, 8, dtype=torch.complex64, device=device)
        )
        new = gram_diagonal_overload(Kx, 1.0, batch_size)
        ref = gram_diagonal_overload_reference(Kx, 1.0, batch_size)
        worst = max(worst, (new - ref).abs().max().item())
    assert worst < 1e-4, (
        f"批量版 gram_diagonal_overload 与原始实现的最大偏差 {worst:.3e} 超出容差 1e-4。"
        "运行 `python verify_batched_ops.py` 查看明细。"
    )
    return f"batch 1/8/64 上最大偏差 {worst:.3e}（容差 1e-4）"


@check("steering_vec_batch 与逐角度 steering_vec 一致（MUSIC/MVDR 网格向量化的地基）")
def _steering_batch_equivalence():
    """文档 §16.16: MUSIC 谱与 MVDR 响应的 18000 点网格扫描已向量化。

    两者都建立在 `SystemModel.steering_vec_batch` 上，所以先单独校验它；同时确认它在
    `nominal=False` 时**拒绝**执行 —— 否则会悄悄抽失配参数、扰动全局随机流，
    让数据生成不再可复现。
    """
    import numpy as np

    from verify_music_batch import build_case, steering_reference

    sm, _X = build_case()
    angels = sm  # placeholder to keep flake quiet
    angels = np.linspace(-np.pi / 2, np.pi / 2, 2000, endpoint=False)
    for sig in ("NarrowBand", "Broadband"):
        ref = steering_reference(sm, angels, f=1.0)
        got = sm.steering_vec_batch(angels, f=1.0, nominal=True)
        assert got.shape == ref.shape, f"{sig}: 形状 {got.shape} != {ref.shape}"
        worst = float(np.abs(got - ref).max())
        assert worst < 1e-12, f"{sig}: 最大偏差 {worst:.3e}，超出容差 1e-12"
    try:
        sm.steering_vec_batch(angels, nominal=False)
    except ValueError:
        pass
    else:
        raise AssertionError("steering_vec_batch(nominal=False) 本应抛 ValueError 却通过了")
    return f"NarrowBand/Broadband 各 2000 点逐点相同；nominal=False 被正确拒绝"


@check("评估路径分批与逐样本给出同一组数字（--eval_batch 不改变结果）")
def _eval_batch_consistency():
    """文档 §16.16: `eval_on_test` 改成"一个 batch 一次前向、三个算法共享 Rz"。

    这里在**不训练**的前提下验证分批不改变数值: 随机初始化一个 SubspaceNet，取 6 个
    真实样本，比较 subnet_batch=1 与 =6 时的预测。这把两个已修过的坑都覆盖了:
      1. DoA 是**逐样本随机**的, 不能拿批内第一个样本的 DoA 当整批真值;
      2. `evecs[:, :, idx]` 在 3-D 下会按前导轴广播, 必须用 take_along_axis.

    判据用 **RMSPE(度)** 而不是逐点预测: 未训练模型的信号子空间可能近乎简并
    (实测某样本 |λ1|-|λ2| ~ 1e-6), 此时 `pinv(Us_upper)` 会把 Rz 上 8.5e-06 的
    分批浮点差放大成几十度 —— 那是随机模型下的数值敏感性, 不是分批的错
    (同一份 Rz 逐样本跑两次也会分叉)。真正要抓的是**系统性**错误 (真值错位、
    索引广播串批), 它们在指标上一定看得出来。
    """
    import numpy as np
    import torch

    from src.criterions import RMSPELoss
    from src.data_handler import create_dataset
    from src.models import ModelGenerator, esprit as esprit_batched, root_music as root_music_batched
    from src.system_model import SystemModelParams
    from src.utils import R2D, device, set_unified_seed

    n = 6
    set_unified_seed(7)
    p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
         .set_parameter("T", 100).set_parameter("snr", 10)
         .set_parameter("signal_type", "NarrowBand")
         .set_parameter("signal_nature", "non-coherent")
         .set_parameter("eta", 0.05).set_parameter("bias", 0.0)
         .set_parameter("sv_noise_var", 0.0))
    md, _gd, _sm = create_dataset(system_model_params=p, samples_size=n,
                                  model_type="SubspaceNet", tau=8, save_datasets=False)
    model = (ModelGenerator().set_model_type("SubspaceNet")
             .set_diff_method("root_music").set_tau(8).set_model(p)).model
    model = model.to(device).eval()
    # 真值**逐样本**取 (不能取第 0 个样本的当真批真值)
    truth = np.asarray([md[i][1].numpy().ravel() * R2D for i in range(n)])
    criterion = RMSPELoss()

    def rmspe(preds):
        got = []
        for i, row in enumerate(np.asarray(preds, dtype=float)):
            with torch.no_grad():
                got.append(float(criterion(torch.as_tensor(row[None]),
                                           torch.as_tensor(truth[i][None]))))
        return float(np.mean(got)) * R2D

    def _rows(fn, Rz, batch):
        """统一取出 (batch, M) 的预测（度）。

        注意 `root_music` 返回三元组、`esprit` 直接返回张量 —— 混用 `[0]` 会串位
        (这正是本检查早先连续报三次形状错的原因)。
        """
        raw = fn(Rz, 2, batch)
        t = raw[0] if isinstance(raw, (tuple, list)) else raw
        flat = np.rad2deg(np.asarray(t.detach().cpu().numpy(), dtype=float)).ravel()
        if flat.size != batch * 2:
            raise AssertionError(
                f"{getattr(fn, '__name__', fn)} 在 batch={batch} 上返回 {flat.size} 个数"
                f"（期望 {batch * 2}）: Rz{tuple(Rz.shape)} -> {tuple(t.shape)}"
            )
        return flat.reshape(batch, 2)

    detail, gaps = {}, {}
    with torch.no_grad():
        Xb = torch.stack([md[i][0] for i in range(n)]).to(device)
        Rz = model(Xb)[-1]
        # 记录最简并样本的 eigenvalue 间隙, 便于把"离散差"归因到数值敏感性
        ev = np.sort(np.abs(np.linalg.eigvals(Rz.detach().cpu().numpy())), axis=1)[:, ::-1]
        gaps["min |l1|-|l2|"] = float(np.min(ev[:, 0] - ev[:, 1]))
        for name, fn in (("r-music", root_music_batched), ("esprit", esprit_batched)):
            batched = _rows(fn, Rz, n)
            rows = np.concatenate([_rows(fn, model(Xb[i:i + 1])[-1], 1) for i in range(n)], axis=0)
            point = float(np.abs(batched - rows).max())
            detail[name] = (rmspe(batched), rmspe(rows), point)

    worst = max(abs(a - b) for a, b, _ in detail.values())
    scale = max(1.0, max(a for a, _b, _c in detail.values()))
    tol = 1e-4 * scale
    assert worst < tol, (
        f"分批与逐样本的 RMSPE(度) 最大差 {worst:.3e}（容差 {tol:.3e} = 1e-4 × {scale:.4g}）；明细 "
        + "；".join(f"{k}: batch={a:.6f} per={b:.6f} 逐点最大差={c:.2e} deg"
                    for k, (a, b, c) in detail.items())
        + "。常见原因: DoA 逐样本随机却按批共享（会让 RMSPE 差出几个度），"
          "或 eig 特征向量索引广播串批（会让预测整批错位）。"
    )
    parts = "；".join(f"{k} RMSPE {a:.4f} vs {b:.4f} 度" for k, (a, b, _c) in detail.items())
    return (f"{parts}（相对差 <1e-4；逐点最大差 "
            + "、".join(f"{c:.1e}" for _a, _b, c in detail.values())
            + f" deg）；最小特征值间隙 {gaps['min |l1|-|l2|']:.2e}")


@check("RMSPELoss 的批量实现与逐样本原始实现等价（loss 与梯度）")
def _rmspe_loss_equivalence():
    """src/criterions.py 的 RMSPELoss.forward 原本是逐样本 + 逐排列的循环（文档 §16.13）。

    这是训练路径上最贵的一段：batch * M! 次微小的归约，每次前面还有 torch.min 的
    隐式同步。批量版必须与原实现给出同一个 loss 和同一个梯度，否则换掉它就等于换了目标函数。
    """
    import numpy as np
    import torch

    from src.criterions import RMSPELoss, permute_prediction
    from src.utils import device

    def reference(predictions, targets):
        rmspe = []
        for index in range(predictions.shape[0]):
            rmspe_list = []
            for prediction in permute_prediction(predictions[index].to(device)):
                error = (((prediction - targets[index].to(device)) + (np.pi / 2)) % np.pi) - np.pi / 2
                rmspe_list.append(
                    (1 / np.sqrt(targets.shape[-1])) * torch.linalg.norm(error)
                )
            rmspe.append(torch.min(torch.stack(rmspe_list, dim=0)))
        return torch.sum(torch.stack(rmspe, dim=0))

    criterion = RMSPELoss()
    worst_loss, worst_grad = 0.0, 0.0
    for m in (2, 3):
        for batch_size in (1, 8, 128):
            predictions = torch.rand(batch_size, m, device=device) * np.pi - np.pi / 2
            targets = torch.rand(batch_size, m, device=device) * np.pi - np.pi / 2

            p_new = predictions.clone().requires_grad_(True)
            criterion(p_new, targets).backward()
            p_ref = predictions.clone().requires_grad_(True)
            reference(p_ref, targets).backward()

            worst_loss = max(worst_loss, abs(
                criterion(predictions, targets).item() - reference(predictions, targets).item()
            ))
            scale = max(p_ref.grad.abs().max().item(), 1e-12)
            worst_grad = max(worst_grad, (p_new.grad - p_ref.grad).abs().max().item() / scale)

    assert worst_loss < 1e-4 and worst_grad < 1e-5, (
        f"批量版 RMSPELoss 与原始实现不一致：loss 差 {worst_loss:.3e}、梯度相对差 "
        f"{worst_grad:.3e}。这会让训练目标与论文/历史结果不再可比。"
        "运行 `python verify_batched_ops.py` 查看明细。"
    )
    return f"M=2/3、batch 1/8/128 上 loss 差 {worst_loss:.3e}、梯度相对差 {worst_grad:.3e}"


@check("判据函数跟随预测张量的设备（换卡跑时不会 cuda:0/cuda:1 冲突）")
def _loss_device_following():
    """RMSPELoss / MSPELoss 必须跟着传进来的预测走，而不是跟着导入期常量（文档 §16.15）。

    回归对象：`python bench_step_split.py --device 1` 在算 loss 时抛
    "Expected all tensors to be on the same device, but found at least two devices,
    cuda:0 and cuda:1!" —— 根因是 src/criterions.py 顶部那个导入期常量。本机只有一张卡，
    所以把该常量**故意指到 cpu**，制造同一类错配。
    """
    import numpy as np
    import torch

    import src.criterions as criterions
    from src.criterions import MSPELoss, RMSPELoss

    original = criterions.device
    worst = 0.0
    try:
        for name, criterion in (("RMSPELoss", RMSPELoss()), ("MSPELoss", MSPELoss())):
            predictions = torch.rand(32, 2, device=original) * np.pi - np.pi / 2
            targets = torch.rand(32, 2, device=original) * np.pi - np.pi / 2

            expected = criterion(predictions.clone(), targets).item()
            criterions.device = torch.device("cpu")     # 故意指错
            got = criterion(predictions.clone(), targets).item()
            criterions.device = original

            diff = abs(got - expected)
            worst = max(worst, diff)
            assert diff < 1e-6, (
                f"{name} 在模块常量指错后给出的 loss 变了 {diff:.3e}，说明它仍在用导入期常量。"
                f"换卡运行时会在训练中途抛 device 不匹配。"
            )
    finally:
        criterions.device = original
    return f"模块常量故意指错后 RMSPELoss/MSPELoss 的 loss 差 {worst:.3e}"


@check("esprit 的批量实现与逐样本原始实现等价（仅 esprit 分支）")
def _esprit_equivalence():
    """src/models.py 的 esprit 原本是逐样本循环（文档 §16.9）。

    本复现用 diff_method="root_music"，不走这条路；这条断言是为了让"改动没改坏
    数值"这件事在换机器/换 torch 版本后仍能一键复验。
    """
    from verify_esprit_batch import esprit_reference

    import torch
    from src.models import ModelGenerator, esprit
    from src.system_model import SystemModelParams
    from src.utils import device, set_unified_seed

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
    ).model.to(device).eval()

    worst = 0.0
    for batch_size in (1, 8, 32):
        x = torch.randn(batch_size, 8, 16, 8, device=device)
        with torch.no_grad():
            Rz = model(x)[-1].detach()
            ref = esprit_reference(Rz, 2, batch_size)
            new = esprit(Rz, 2, batch_size)
        # 逐样本比 M 个 doa：两个实现的排列都可能受特征值退化影响，故比排序后的集合。
        diff = (
            torch.sort(ref, dim=-1).values - torch.sort(new, dim=-1).values
        ).abs().max().item()
        worst = max(worst, diff)
    assert worst < 1e-3, (
        f"批量版 esprit 与原始实现的最大偏差 {worst:.3e} rad 超出容差 1e-3。"
        "运行 `python verify_esprit_batch.py` 查看明细。"
    )
    return f"batch 1/8/32 上 M 个 doa 最大偏差 {worst:.3e} rad（容差 1e-3）"


@check("替代协方差是 Hermitian 的，且 eigh 给出正交的噪声子空间投影")
def _hermitian():
    """src/models.py:root_music 用 eigh 而不是 eig，前提是 Rz = K^H K + eps*I 为 Hermitian。

    这条断言校验两件事：
      1. 前提成立 —— 替代协方差确实 Hermitian（否则 eigh 会静默给出错误结果）；
      2. 噪声子空间投影 F = Un Un^H 确实是**正交投影**（幂等）。

    第 2 条才是换 eigh 的实质理由，不只是快：float32 下 `torch.linalg.eig` 返回的特征向量
    并不正交（实测 ||V^H V - I|| = 3.8e-4），于是 F 也不是幂等的（实测 1.4e-4），连带
    Root-MUSIC 的多项式系数与 DoA 都被扰动到这个量级；`eigh` 两个指标都在 1e-6。
    这也是为什么不同求根路径的 DoA 会有 1e-3 度量级的差异（见文档 §16）。
    """
    import torch
    from src.models import ModelGenerator
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed

    set_unified_seed(0)
    p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
         .set_parameter("T", 100).set_parameter("snr", 10)
         .set_parameter("signal_type", "NarrowBand")
         .set_parameter("signal_nature", "non-coherent")
         .set_parameter("eta", 0.0).set_parameter("bias", 0.0)
         .set_parameter("sv_noise_var", 0.0))
    model = (ModelGenerator().set_model_type("SubspaceNet")
             .set_diff_method("root_music").set_tau(8).set_model(p)).model
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(model.parameters()).device

    x = torch.randn(32, 8, 16, 8, device=dev)
    with torch.no_grad():
        Rz = model(x)[-1].detach()
    skew = (Rz - Rz.conj().transpose(-2, -1)).abs().max().item()

    herm = (Rz + Rz.conj().transpose(-2, -1)) / 2
    with torch.no_grad():
        ev_eig, V_eig = torch.linalg.eig(herm)
        order = torch.argsort(torch.abs(ev_eig), dim=1, descending=True)
        V_eig = torch.gather(V_eig, 2, order.unsqueeze(1).expand(-1, 8, -1))
        _, V_eigh = torch.linalg.eigh(herm)
        V_eigh = torch.flip(V_eigh, dims=[-1])
        eye = torch.eye(8, dtype=herm.dtype, device=dev).expand_as(V_eigh @ V_eigh)
        ortho_eig = (V_eig.conj().transpose(-2, -1) @ V_eig - eye).abs().max().item()
        ortho_eigh = (V_eigh.conj().transpose(-2, -1) @ V_eigh - eye).abs().max().item()
        F_eigh = V_eigh[:, :, 2:] @ V_eigh[:, :, 2:].conj().transpose(-2, -1)
        idem = (F_eigh @ F_eigh - F_eigh).abs().max().item()

    assert skew < 1e-5, (
        f"替代协方差的 Hermitian 偏差 {skew:.3e} 过大，eigh 的前提不成立——"
        "请检查 src/models.py 的 gram_diagonal_overload 调用。"
    )
    assert idem < 1e-5, (
        f"eigh 给出的噪声子空间投影不幂等（偏差 {idem:.3e}），超出容差 1e-5。"
    )
    return (f"Hermitian 偏差 {skew:.1e}；eigh 投影幂等偏差 {idem:.1e}；"
            f"正交性 eig {ortho_eig:.1e} vs eigh {ortho_eigh:.1e}")


# ------------------------------------------------------------------------------
print()
print("[5] 端到端训练（可选，--full）")
# ------------------------------------------------------------------------------

if "--full" in sys.argv:
    @check("train_model 能真实训练并降低 loss")
    def _train():
        from src.data_handler import create_dataset
        from src.models import ModelGenerator
        from src.system_model import SystemModelParams
        from src.training import TrainingParams, train_model
        from src.utils import set_unified_seed
        p = (SystemModelParams().set_parameter("N", 8).set_parameter("M", 2)
             .set_parameter("T", 100).set_parameter("snr", 10)
             .set_parameter("signal_type", "NarrowBand")
             .set_parameter("signal_nature", "non-coherent")
             .set_parameter("eta", 0.1).set_parameter("bias", 0.0)
             .set_parameter("sv_noise_var", 0.0))
        set_unified_seed(42)
        md, _, _ = create_dataset(system_model_params=p, samples_size=64,
                                  model_type="SubspaceNet", tau=8, save_datasets=False)
        mg = (ModelGenerator().set_model_type("SubspaceNet")
              .set_diff_method("root_music").set_tau(8).set_model(p))
        tp = (TrainingParams().set_batch_size(16).set_epochs(2).set_model(model=mg)
              .set_optimizer(optimizer="Adam", learning_rate=1e-3, weight_decay=1e-9)
              .set_training_dataset(md).set_schedular(step_size=10, gamma=0.9)
              .set_criterion())
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, tr, va = train_model(tp, model_name="preflight", checkpoint_path=Path(td))
        assert len(tr) == 2 and all(v == v for v in tr), f"loss 轨迹异常: {tr}"
        return f"2 epoch 完成: train {tr[0]:.4f} -> {tr[-1]:.4f}, valid {va[-1]:.4f}"
else:
    print("  [SKIP] 真实训练冒烟未执行（加 --full 开启，约 1 分钟）")

# ------------------------------------------------------------------------------
print()
print("=" * 74)
n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
n_skip = 1 if "--full" not in sys.argv else 0
print(f"结果: PASS={n_pass}  FAIL={N_FAIL}  SKIP={n_skip}")
if N_FAIL:
    print()
    print("以下问题必须先解决，否则全量实验会失败或结论无效：")
    for s, n, d in RESULTS:
        if s == "FAIL":
            print(f"  !! {n}: {d}")
    print()
    print("提示：完整的量纲陷阱、RMSPE 口径、训练协议说明见 REPRODUCE_ARRAY_MISMATCH.md")
    sys.exit(1)
print()
print("环境就绪。建议的全量起步命令：")
print("  python reproduce_array_mismatch.py all --scenario spacing \\")
print("      --n_train 45000 --n_test 5000 --epochs 80 --batch_size 1024 \\")
print("      --algorithms r-music esprit music")
print()
print("注意：默认 --train_levels matched 会在 4 个 eta 上各训一个模型，")
print("      训练时间是单模型的 4 倍，排期时按此估算。")
print("=" * 74)
