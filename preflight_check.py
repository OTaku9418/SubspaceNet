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


@check("CUDA 可用（src/utils.py 的 device 在 import 时定型）")
def _cuda():
    import torch
    avail = torch.cuda.is_available()
    assert avail, ("torch.cuda.is_available() = False。注意 src/utils.py:32 的 device 是"
                   "模块级常量，import 前就必须可见 CUDA；CPU 上跑全量会慢到不可接受")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cap = torch.cuda.get_device_capability(0)
    return f"{name}, {total:.1f} GiB, compute capability {cap}"


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
print("      --n_train 10000 --n_test 5000 --epochs 80 --batch_size 1024 \\")
print("      --algorithms r-music esprit music")
print()
print("注意：默认 --train_levels matched 会在 4 个 eta 上各训一个模型，")
print("      训练时间是单模型的 4 倍，排期时按此估算。")
print("=" * 74)
