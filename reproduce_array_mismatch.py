"""SubspaceNet 论文「阵列失配 (Array Miscalibration)」实验复现脚本
================================================================================
对应论文: D. H. Shmuel et al., "SubspaceNet: Deep Learning-Aided Subspace Methods
          for DoA Estimation", Section IV-B-4 与 Fig. 9。

复现的两个失配场景 (论文原文):
  (1) 阵元间距失配 (spacing): 相邻阵元间距 = 标称半波长 d 加上 δ_m ~ U(-eta, eta),
      即 [a(theta)]_m = exp(-2j*pi*(d+δ_m)*(m-1)*sin(theta)/c)  —— 论文式 (21)。
      论文扫描 eta ∈ [0.025, 0.15] (Fig. 9(a) 标注为 0.025/0.05/0.10/0.15)。
  (2) 导向矢量加噪 (sv_noise): 每个阵元导向矢量分量加零均值复高斯噪声,
      方差 sigma_sv^2 ∈ [0, 0.75] (Fig. 9(b) 标注为 0/0.25/0.5/0.75)。

共同设置 (论文 Sec. IV-A / IV-B-4):
  N=8 阵元 ULA, M=2 非相干窄带信源, DoA 均匀取自 [-pi/2, pi/2],
  SNR = 10*log10(sigma_S^2 / sigma_V^2), T 个快拍, 结果按 5000 次 Monte Carlo 平均,
  评估指标为 RMSPE (式 (16))。

--------------------------------------------------------------------------------
复现关键点 (详见 REPRODUCE_ARRAY_MISMATCH.md):
  A. 论文文字「η 是从标称间距偏移的百分比」与参考代码不一致:
     代码中 mis_distance ~ U(-eta, eta) 是与 d=0.5 同量纲的绝对值,
     即实际间距 = 0.5*(1 + 2*eta)。eta=0.15 -> 最大偏离标称 30%。
  B. src/criterions.py 的 RMSPE() 有「度/弧度」缩放 bug, 论文表格里的数字
     由参考代码产生, 需乘 180/pi 才是有物理意义的「角度」。本脚本两者都算。
  C. bias 在参考代码中是一个「所有阵元共享」的随机位置偏置 (size=1),
     默认 bias=0.05, 相当于随机阵列相位中心偏移, 请务必显式置 0。

用法:
  python reproduce_array_mismatch.py --help
  python reproduce_array_mismatch.py all      --scenario spacing  --smoke
  python reproduce_array_mismatch.py all      --scenario spacing
  python reproduce_array_mismatch.py all      --scenario sv_noise
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import warnings
from datetime import datetime
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # 服务器无显示器, 必须在 pyplot 之前设定
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scipy.signal
import torch

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent
# 产物根目录: 默认 <仓库>/data, 可用环境变量 SUBSPACENET_DATA_ROOT 覆盖
# (服务器上想把数据集放到大盘, 或本机 data/ 不可写时, 都靠它)
DATA_ROOT = Path(os.environ.get("SUBSPACENET_DATA_ROOT", ROOT / "data"))
sys.path.insert(0, str(ROOT))


def _early_device():
    """在任何人 import `src.*` 之前解析 `--device` 并改写 `src.utils.device`。

    `src.utils.device`（`src/utils.py:32`）是导入期常量, `src.data_handler` / `src.models`
    等模块在 import 时各自绑定一份, 所以 import 之后再改就晚了 ---- 必须在 `from src...`
    之前把 `sys.argv` 里的这一个参数取出来。

    多卡机器上两种做法等价, 任选其一:
      * `CUDA_VISIBLE_DEVICES=1 python ...`        ---- 零改动, 可见卡会重新编号成 cuda:0
      * `python ... --device 1`                    ---- 本函数, 不改环境变量
    两者同时用也没问题: `--device` 的索引是在"可见卡"里数, 不是物理号。
    """
    spec = None
    argv = sys.argv[1:]
    for position, token in enumerate(argv):
        if token == "--device":
            if position + 1 < len(argv):
                spec = argv[position + 1]
            break
        if token.startswith("--device="):
            spec = token.split("=", 1)[1]
            break

    import src.utils as _utils

    if not spec or spec == "default":
        return _utils.device
    chosen = _utils.resolve_device(spec)
    _utils.device = chosen
    if chosen.type == "cuda" and chosen.index:
        torch.cuda.set_device(chosen.index)

    # `src.training` / `src.evaluation` / `src.models` 用的是 `from src.utils import device`
    # (或 import *), 那是**值绑定**: 它们各自持有一份拷贝, 只改 `src.utils.device` 不够。
    # 这里在它们被 import 之前把要用的那几个模块先导进来并逐份改写。
    import src.criterions
    import src.data_handler
    import src.evaluation
    import src.models
    import src.training

    for module in (src.criterions, src.data_handler, src.evaluation, src.models, src.training):
        if hasattr(module, "device"):
            module.device = chosen
    return chosen


DEVICE = _early_device()

from src.data_handler import create_dataset, read_data  # noqa: E402
from src.methods import Esprit, MUSIC, RootMUSIC  # noqa: E402
from src.models import ModelGenerator, esprit as esprit_batched, root_music as root_music_batched  # noqa: E402
from src.plotting import initialize_figures  # noqa: E402
from src.system_model import SystemModel, SystemModelParams  # noqa: E402
from src.training import TrainingParams, train_model  # noqa: E402
from src.utils import R2D, set_unified_seed  # noqa: E402

device = DEVICE          # 本模块其余代码沿用它, 与改写后的 src.utils.device 保持一致

# ------------------------------------------------------------------------------
# 实验配置: 论文里的固定量都在这里
# ------------------------------------------------------------------------------
N_SENSORS = 8          # 阵元数 N
M_SOURCES = 2          # 信源数 M
T_SNAPSHOTS = 100      # 快拍数 T (论文未在 IV-B-4 明写, 取满足 AS4 的充裕值)
SNR_DB = 10            # SNR
TAU_LAGS = 8           # SubspaceNet 自相关最大 lag
# 注: 式 (15) 的对角加载 eps = 1 由仓库硬编码在 src/models.py:387-389 的
#     gram_diagonal_overload(Kx=Kx_tag, eps=1, batch_size=...) 里, 脚本无需(也无法)传参。
N_TRAIN = 45000        # 训练样本数 (论文 Sec. IV-A-3 为 45000; 显存不够可下调, 见 --n_train)
N_TEST = 5000          # 测试样本数 = 论文的 5000 次 Monte Carlo
N_VALID = 500          # 训练内验证集 (训练集额外留出)
BATCH_SIZE = 1024      # 训练 batch (显存不够就往下降, 见 --batch_size)
EPOCHS = 80
LR = 1e-3              # 论文 Sec. IV-A: Adam, mu = 0.001
WEIGHT_DECAY = 1e-9
SCHED_STEP = 80
SCHED_GAMMA = 0.2
BIAS = 0.0             # 重要: 显式关闭参考代码默认的 0.05 共享位置偏置

SCENARIOS = {
    # 场景 1: 阵元间距失配, 训练/评估用的 eta 网格
    "spacing": {
        "sweep_param": "eta",
        "grid": [0.025, 0.05, 0.10, 0.15],
        "train_at": 0.10,   # 论文未说明训练时 eta; 取网格中偏上值作主模型
        "folder": "array_mismatch_spacing",
    },
    # 场景 2: 导向矢量加噪
    "sv_noise": {
        "sweep_param": "sv_noise_var",
        "grid": [0.0, 0.25, 0.5, 0.75],
        "train_at": 0.5,
        "folder": "array_mismatch_sv_noise",
    },
}


# ------------------------------------------------------------------------------
# 指标: 复刻 src/criterions.RMSPE 的置零/补齐约定, 但同时给出有物理意义的版本
# ------------------------------------------------------------------------------
def rmspe_reference(predictions: np.ndarray, doa: np.ndarray) -> float:
    """完全复刻 src/criterions.py:190 的 RMSPE(度->rad 缩放 bug 一并保留)。"""
    pred = np.asarray(predictions, dtype=float).ravel()
    pred = pred[pred != 0.0]                      # 论文式 (16): M_hat < M 时补零
    doa = np.asarray(doa, dtype=float).ravel()
    if pred.size < doa.size:                      # 估计数不足 -> 补随机猜测 (随机下界)
        pred = np.concatenate(
            [np.round(np.random.rand(doa.size - pred.size) * 180, 2) - 90.0, pred]
        )
    best = np.inf
    for p in permutations(pred, len(pred)):
        p = np.asarray(p)
        err = (((p - doa) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        best = min(best, (1 / np.sqrt(len(p))) * float(np.linalg.norm(err)))
    return best


def rmspe_degrees(predictions: np.ndarray, doa: np.ndarray) -> float:
    """有物理意义的 RMSPE, 单位为度; 与上式恒差因子 180/pi。"""
    return rmspe_reference(predictions, doa) * R2D


def rmspe_degrees_fixed(predictions: np.ndarray, doa: np.ndarray) -> float:
    """修正写法: 全程在度域做 mod-180 环绕, 返回值就是角度误差 (单位: 度)。

    参考代码的 bug 是「(p-d)*pi/180」(把度当弧度转) 却用度为单位的 ±pi/2 做环绕,
    导致输出被压缩了 180/pi 倍。这里改为 mod 180, 语义与论文式 (16) 一致。
    """
    pred = np.asarray(predictions, dtype=float).ravel()
    pred = pred[pred != 0.0]
    doa = np.asarray(doa, dtype=float).ravel()
    if pred.size < doa.size:
        pred = np.concatenate(
            [np.round(np.random.rand(doa.size - pred.size) * 180, 2) - 90.0, pred]
        )
    best = np.inf
    for p in permutations(pred, len(pred)):
        p = np.asarray(p)
        err = (((p - doa) + 90.0) % 180.0) - 90.0
        best = min(best, (1 / np.sqrt(len(p))) * float(np.linalg.norm(err)))
    return best


def as_pred_array(predictions) -> np.ndarray:
    """把算法返回的预测安全转成 float 数组。

    必要性: `MVDR.narrowband` 的 predictions 是 `None` (src/methods.py:657-658),
    而 numpy 1.x 下 `np.asarray(None, dtype=float)` 会静默变成 `array([nan])`
    —— 指标会算成 nan 而不报错; numpy 2.x 则直接抛 TypeError。
    统一在这里显式处理。
    """
    if predictions is None:
        return np.empty(0, dtype=float)
    return np.asarray(predictions, dtype=float).ravel()


# ------------------------------------------------------------------------------
# 系统模型构造
# ------------------------------------------------------------------------------
def make_params(eta: float = 0.0, sv_noise_var: float = 0.0,
                m: int = M_SOURCES, t: int = T_SNAPSHOTS) -> SystemModelParams:
    """按复现要求构造 SystemModelParams (bias 显式置 0)。"""
    return (
        SystemModelParams()
        .set_parameter("N", N_SENSORS)
        .set_parameter("M", m)
        .set_parameter("T", t)
        .set_parameter("snr", SNR_DB)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", float(eta))
        .set_parameter("bias", BIAS)
        .set_parameter("sv_noise_var", float(sv_noise_var))
    )


def datasets_dir(scenario: str) -> Path:
    return DATA_ROOT / "datasets" / SCENARIOS[scenario]["folder"]


# ------------------------------------------------------------------------------
# 出图 (全部 PNG)
# ------------------------------------------------------------------------------
PLOT_RCP = {
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.autolayout": True,
    "savefig.dpi": 150,
}
# Fig.9 复现图的配色 (与结果 JSON 里的键一致)
CURVE_COLORS = {
    "SubNet+r-music": "tab:blue", "r-music": "tab:cyan",
    "SubNet+esprit": "tab:red", "esprit": "tab:orange",
    "SubNet+music": "tab:green", "music": "tab:olive",
}
CURVE_LABELS = {
    "SubNet+r-music": "SubspaceNet + Root-MUSIC", "r-music": "Root-MUSIC",
    "SubNet+esprit": "SubspaceNet + ESPRIT", "esprit": "ESPRIT",
    "SubNet+music": "SubspaceNet + MUSIC", "music": "MUSIC",
}
# 传给仓库 `plot_spectrum` 的 algorithm 字符串只用来做分派, 但它会先判 "music"
# (src/plotting.py:62-69)。因此 Root-MUSIC 这一路必须用一个**不含** "music" 子串
# 又含 "r-music" 的名字, 否则会被当成 MUSIC 谱去取 figures["music"] 而崩掉。
RMUS_DISPATCH_NAME = "SubNet+ROOT-R-MUSIC"


class _savefig_redirect:
    """把 plt.savefig 的落盘路径临时改写掉 (上下文管理器).

    必要性: 仓库的 `src/plotting.py:142 plot_root_music_spectrum` 没有"输出路径"
    参数, 而是**硬编码**往当前工作目录的 `data/spectrums/{algorithm}_spectrum.pdf`
    写 PDF (见 `src\plotting.py:174`)。本脚本要求全部出 PNG 且路径可控, 因此只
    拦截 savefig 这一下, 不改动仓库源码。
    """

    def __init__(self, target: Path):
        self.target = Path(target)
        self.original = None

    def __enter__(self):
        self.original = plt.savefig

        def _redirected(*args, **kwargs):
            self.target.parent.mkdir(parents=True, exist_ok=True)
            # 丢掉调用方自己的 format/bbox_inches, 否则会与下面的显式关键字重复
            kwargs.pop("format", None)
            kwargs.pop("bbox_inches", None)
            return self.original(str(self.target), format="png", bbox_inches="tight")

        plt.savefig = _redirected
        return self

    def __exit__(self, *exc):
        plt.savefig = self.original
        return False


class _polar_degree_shim:
    """把 `ax.plot` 的**角度参数从度换算成弧度**的极坐标 Axes 外壳.

    必要性: 仓库的 `src/plotting.py:142 plot_root_music_spectrum` 用
    `ax.plot([angle * np.pi / 180], [r])` 给极坐标轴喂角度 —— 多乘了一次 pi/180。
    在 matplotlib 的极坐标投影里 `Theta` 本来就以**弧度**解释, 于是
    °(真实角度)→刻度 的换算变成 (θ*π/180 rad)→(θ*π/180)*180/π = θ 度刻度,
    结果恰好等于原始角度值: **预测值** (本来就按度传进去) 反而画对了, 而
    `true_DOA` 那行传的是 `doa * np.pi / 180` (°→rad), 会被再放大 180/π 倍,
    真值标记因此几乎总贴在 0° 附近 —— 见指导文档 §10.6。
    这里只在**用户态**纠正它, 不改仓库源码: 角度一律按度传入, 由外壳换算。
    """

    def __init__(self, ax):
        self._ax = ax

    def __getattr__(self, name):
        return getattr(self._ax, name)

    def plot(self, theta, r, **kwargs):
        theta = np.asarray(theta, dtype=float).ravel()
        return self._ax.plot(theta * np.pi / 180.0, r, **kwargs)


class _PolarFigShim:
    """把 `fig.add_subplot(...)` 的返回值包成 `_polar_degree_shim`。"""

    def __init__(self, fig):
        self._fig = fig

    def __getattr__(self, name):
        return getattr(self._fig, name)

    def add_subplot(self, *args, **kwargs):
        return _polar_degree_shim(self._fig.add_subplot(*args, **kwargs))


class _PltShim:
    """只在调用仓库 `plot_root_music_spectrum` 期间替换 `src.plotting.plt`。"""

    def __getattr__(self, name):
        return getattr(plt, name)

    def figure(self, *args, **kwargs):
        return _PolarFigShim(plt.figure(*args, **kwargs))


def _eigs(covariance: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(np.asarray(covariance, dtype=complex)))[::-1]


def _music_preds_from_spectrum(spectrum: np.ndarray, angels: np.ndarray, M: int) -> np.ndarray:
    """把一条 MUSIC 谱还原成预测角度, 逐字复刻 `MUSIC.narrowband` 的后处理.

    必要性: 批量化之后不能再用 `MUSIC(...).narrowband(...)` 拿预测, 但必须保持
    单样本数值完全一致. 原路径 (`src/methods.py:405-409`) 是:
        peaks = list(scipy.signal.find_peaks(spectrum)[0])
        peaks.sort(key=lambda x: spectrum[x], reverse=True)
        predictions = self._angels[peaks] * R2D
        predictions = predictions[:M][::-1]
    这里逐行照搬, 保证 `--subnet_batch 1` 与旧行为逐位相同.
    """
    peaks = list(scipy.signal.find_peaks(spectrum)[0])
    peaks.sort(key=lambda x: spectrum[x], reverse=True)
    predictions = angels[peaks] * R2D
    return predictions[:M][::-1]


def _capture_path(capture_dir: Path, scenario: str, tag: str) -> Path:
    return Path(capture_dir) / f"spectrum_{scenario}_{tag}.npz"


def _level_tag(sweep_name: str, value) -> str:
    """失配水平 -> 文件名片段, 例 eta=0.025 -> eta0p025 (与训练集文件名同风格)。

    加 p 前缀是为了避免 `eta0.1.png` 这种双扩展名, 也让所有水平长度一致。
    """
    return f"{sweep_name}{str(value).replace('.', 'p')}"


def _capture_params(data) -> dict:
    """从 capture dict 的 `params` 里取出模型参数字典。

    兼容两种历史存法: 新格式是 JSON 字符串 (np.array(json.dumps(...)));
    早期版本直接存了 dict, `.item()` 拿到的是 Python dict 的 repr,
    不是合法 JSON (单引号), 此时 `ast.literal_eval` 兜底。
    """
    raw = data["params"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, dict):
        return raw
    text = str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _load_capture(cap: Path):
    """读 `--capture` 存下的 npz, 返回普通 dict。

    注意: `np.load` 是**惰性**的 —— 错误要到真正取某个数组时才抛, 所以必须
    在这里就把所有数组取出来 (否则 try/except 抓不到)。
    早期版本把 params 存成 dict 的 npz 会抛
    `ValueError: Object arrays cannot be loaded when allow_pickle=False`,
    此时给出提示并回退到 `allow_pickle=True` (数据是自己生成的, 可接受)。
    """
    cap = Path(cap)

    def _read(allow_pickle: bool) -> dict:
        with np.load(cap, allow_pickle=allow_pickle) as z:
            return {k: z[k] for k in z.files}

    try:
        return _read(False)
    except ValueError as e:
        if "allow_pickle" not in str(e):
            raise
        print(f"[plot ] {cap.name} 是旧格式 (params 存成 dict), 回退 allow_pickle=True")
        return _read(True)


def plot_spectrum_figures(cap: Path, out_dir: Path, model=None,
                          algorithms=("r-music", "esprit", "music")):
    """按 `--capture` 存下的 npz 出「谱图 / 特征值分离图」(全部 PNG).

    MUSIC 谱图复用仓库的 `src/plotting.py:39 plot_spectrum`;
    Root-MUSIC 根图用仓库的 `src/plotting.py:142 plot_root_music_spectrum`
    (输出路径被 `_savefig_redirect` 改写);
    ESPRIT 是求根类算法、仓库没有对应谱函数, 因此**不画**(只出现在曲线图里);
    特征值分离图由本脚本自己画 (仓库没有这个图, 对应论文 Fig. 10 的体检思路).
    """
    from src.plotting import initialize_figures, plot_spectrum  # noqa: PLC0415
    from src.system_model import SystemModelParams  # noqa: PLC0415

    cap = Path(cap)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_capture(cap)
    made = []
    scenario = str(data["scenario"].item())
    sweep_name = str(data["sweep_name"].item())
    sweep_value = float(data["sweep_value"].item())
    level_tag = _level_tag(sweep_name, sweep_value)

    def _save(fig, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(path)
        print(f"[plot ] {path}")
        return path

    # ---- 1) MUSIC 谱 (仓库 plot_spectrum → plot_music_spectrum) ----
    if "music" in algorithms and "mus_spectrum" in data:
        try:
            params = SystemModelParams()
            for k, v in _capture_params(data).items():
                params.set_parameter(k, v)
            sm = SystemModel(params)
            figures = initialize_figures()
            spectrum = data["mus_spectrum"]
            figures["music"]["norm factor"] = np.max(spectrum)   # 官方写法
            with plt.rc_context(PLOT_RCP):
                plot_spectrum(
                    predictions=data["mus_preds"], true_DOA=data["true_doa"],
                    system_model=sm, spectrum=spectrum,
                    algorithm="SubNet+MUSIC", figures=figures,
                )
                ax = figures["music"]["ax"]
                ax.set_title(f"MUSIC spectrum ({scenario}, "
                             f"{sweep_name}={sweep_value:g})")
                _save(ax.figure, out_dir / f"spectrum_{scenario}_{level_tag}_music.png")
        except Exception as e:                                   # noqa: BLE001
            print(f"[plot ] MUSIC 谱图跳过: {type(e).__name__}: {e}")

    # ---- 2) Root-MUSIC 根图 (仓库 plot_root_music_spectrum, 输出重定向为 PNG) ----
    if "rmus_preds" in data:
        try:
            import src.plotting as _plotting  # noqa: PLC0415

            target = out_dir / f"spectrum_{scenario}_{level_tag}_r-music.png"
            with plt.rc_context(PLOT_RCP), _savefig_redirect(target):
                _orig_plt = _plotting.plt
                _plotting.plt = _PltShim()
                try:
                    plot_spectrum(
                        predictions=data["rmus_preds"], true_DOA=data["true_doa"],
                        roots=data["rmus_mags"], algorithm=RMUS_DISPATCH_NAME,
                    )
                finally:
                    _plotting.plt = _orig_plt
            plt.close("all")
            if target.exists():
                made.append(target)
                print(f"[plot ] {target}")
            else:
                print("[plot ] Root-MUSIC 根图跳过: 仓库函数未产出文件")
        except Exception as e:                                   # noqa: BLE001
            print(f"[plot ] Root-MUSIC 根图跳过: {type(e).__name__}: {e}")

    # ---- 3) 特征值分离图 (SubspaceNet 预测协方差 vs 经验协方差) ----
    if "eig_subnet" in data and "eig_sample" in data:
        try:
            n = len(data["eig_sample"])
            m = int(data["M"])
            x = np.arange(1, n + 1)
            with plt.rc_context(PLOT_RCP):
                fig, ax = plt.subplots(figsize=(7.5, 5))
                ax.semilogy(x, data["eig_sample"], "o--", color="tab:cyan",
                            label="empirical covariance (sample)")
                ax.semilogy(x, data["eig_subnet"], "s-", color="tab:blue",
                            label="SubspaceNet predicted covariance")
                ax.axvline(m + 0.5, color="gray", ls=":", lw=1)
                ax.text(m + 0.6, data["eig_subnet"][0], f"  signal(M={m}) | noise",
                        va="top", fontsize=9, color="gray")
                ax.set_xticks(x)
                ax.set_xlabel("eigenvalue index")
                ax.set_ylabel("eigenvalue (log scale)")
                ax.set_title(f"Eigenvalue separation ({scenario}, "
                             f"{sweep_name}={sweep_value:g})")
                ax.legend()
                _save(fig, out_dir / f"eigsep_{scenario}_{level_tag}.png")
        except Exception as e:                                   # noqa: BLE001
            print(f"[plot ] 特征值分离图跳过: {type(e).__name__}: {e}")
    return made


def plot_curve_figures(json_paths, out_dir: Path, metrics=("deg", "ref"),
                       algorithms=("r-music", "esprit", "music")) -> list:
    """把结果 JSON 画成「Fig.9 复现曲线图」(PNG).

    横轴是失配水平 (eta 或 sigma_sv^2); 实线/圆点 = SubspaceNet 增强, 虚线/方块 = 经典。
    需要仓库的绘图模块, 只用 matplotlib。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    metric_info = {
        # 这里刻意用英文: matplotlib 默认字体没有中文字形, 写成中文会变成方框
        "deg": ("RMSPE [deg]", "(reference RMSPE rescaled by 180/pi; physical angle)"),
        "ref": ("RMSPE (reference scale)",
                "(raw output of src/criterions.py:190 RMSPE, keeps the deg/rad bug)"),
    }
    for jp in json_paths:
        jp = Path(jp)
        payload = json.loads(jp.read_text("utf-8"))
        scenario = payload.get("scenario", jp.stem)
        cfg = SCENARIOS.get(scenario, {})
        sweep_name = cfg.get("sweep_param", "level")
        rows = payload.get("rows", [])
        if not rows:
            print(f"[plot ] {jp.name} 没有 results 行, 跳过")
            continue
        levels = [r[sweep_name] for r in rows]
        for metric in metrics:
            if metric not in metric_info:
                continue
            ylabel, note = metric_info[metric]
            with plt.rc_context(PLOT_RCP):
                fig, ax = plt.subplots(figsize=(7.5, 5))
                for algo in algorithms:
                    for enhanced in (True, False):
                        key = f"SubNet+{algo}" if enhanced else algo
                        ys = [r.get(f"{key}_{metric}") for r in rows]
                        if any(y is None for y in ys):
                            continue
                        ax.plot(levels, ys,
                                marker="o" if enhanced else "s",
                                ls="-" if enhanced else "--",
                                color=CURVE_COLORS.get(key, "gray"),
                                label=CURVE_LABELS.get(key, key))
                ax.set_xlabel(f"array mismatch level ({sweep_name})")
                ax.set_ylabel(ylabel)
                ax.set_title(f"Fig. 9 reproduction - {scenario} mismatch\n"
                             f"train protocol: {payload.get('train_levels', '?')} | {note}",
                             fontsize=11)
                ax.legend(fontsize=9)
                path = out_dir / f"fig9_{scenario}_{metric}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                made.append(path)
                print(f"[plot ] {path}")
    return made


# ------------------------------------------------------------------------------
# nominal 导向矢量补丁 (仓库 bug 的用户态兜底; 已改源码时可自动跳过)
# ------------------------------------------------------------------------------


def enable_nominal_steering_without_repo_edit():
    """让 MUSIC / MVDR 能在"未改动仓库源码"的前提下跑起来.

    背景 (真实 bug, 已在 Python 3.10 + torch 2.0.1 实测复现):
      `src/system_model.py:161-188` 的 `steering_vec(..., nominal=True)` 分支
      (:175-176) 只给 `mis_distance` / `mis_geometry_noise` 赋了 0, **漏了
      `uniform_bias`**; 而 :184 无条件使用 `(uniform_bias + mis_distance + ...)`,
      于是 nominal=True 时抛
        UnboundLocalError: local variable 'uniform_bias' referenced before assignment

    受影响的调用点: `src/methods.py:270` (MUSIC 的 spectrum_calculation) 与
    `src/methods.py:641` (MVDR) —— 两者都传 nominal=True。因此**本仓库的经典
    MUSIC / MVDR 基线开箱即崩**。Root-MUSIC 与 ESPRIT 走求根路径、不调用
    nominal 网格, 所以不受影响 (默认算法列表已因此只含这两个)。

    这里用一个只补 `uniform_bias = 0` 的包装器就地替换掉该方法: 语义与
    "nominal 阵列 = 无失配" 完全一致 (η/σ²_sv 也同样是 0), 是论文式(21)中
    δ_m=0 的理想导向矢量。若你选择直接改仓库源码 (推荐, 见指导文档 §12),
    可把 `--algorithms` 里加上 music, 本包装器会自动发现源码已修好而跳过。
    """
    import inspect

    from src.system_model import SystemModel

    original = SystemModel.steering_vec
    try:
        src = inspect.getsource(original)
    except OSError:
        src = ""
    if "mis_distance, mis_geometry_noise = 0, 0" in src and "uniform_bias = 0" in src:
        return False  # 源码已修好

    def patched(self, theta, f=1, array_form="ULA", nominal=False):
        if not nominal:
            return original(self, theta, f=f, array_form=array_form, nominal=False)
        return np.exp(
            -2j * np.pi * {"NarrowBand": 1, "Broadband": f}[self.params.signal_type]
            * self.dist[self.params.signal_type] * self.array * np.sin(theta)
        )

    SystemModel.steering_vec = patched
    print("[fix  ] 已启用 uniform_bias 用户态补丁 (源码 nominal 分支漏赋值), "
          "MUSIC/MVDR 可用; 详见指导文档 §12")
    return True


def load_or_create_train(scenario: str, value: float, samples: int, force: bool = False):
    """生成/加载某个失配水平下的训练集 (SubspaceNet 输入为自相关张量)。"""
    cfg = SCENARIOS[scenario]
    path = datasets_dir(scenario) / "train"
    path.mkdir(parents=True, exist_ok=True)
    params = make_params(**{cfg["sweep_param"]: value})
    tag = f"{cfg['sweep_param']}={value}".replace(".", "p")
    fname = f"SubspaceNet_DataSet_{scenario}_{tag}_n{samples}.pt"
    if not force and (path / fname).exists():
        print(f"[train] 复用已存在训练集 {fname}")
        return read_data(path / fname)

    print(f"[train] 生成训练集 {scenario} {cfg['sweep_param']}={value}, n={samples}")
    set_unified_seed(42)                          # 固定 seed, 保证可复现
    model_dataset, _, _ = create_dataset(
        system_model_params=params,
        samples_size=samples,
        model_type="SubspaceNet",
        tau=TAU_LAGS,
        save_datasets=False,
    )
    torch.save(model_dataset, path / fname)       # 下次直接复用, 省十几分钟
    print(f"[train] 已保存 {path / fname}")
    return model_dataset


def load_or_create_test(scenario: str, value: float, samples: int, force: bool = False):
    """生成/加载某个失配水平下的测试集。

    关键: 每个失配水平使用相同的 set_unified_seed, 因此各条曲线面对的是
    「同一批 DoA / 同一批信号与噪声」, 唯一变化的是阵列流形 —— 这才是公平对比。
    """
    cfg = SCENARIOS[scenario]
    path = datasets_dir(scenario) / "test"
    path.mkdir(parents=True, exist_ok=True)
    params = make_params(**{cfg["sweep_param"]: value})
    tag = f"{cfg['sweep_param']}={value}".replace(".", "p")
    model_f = f"SubspaceNet_DataSet_{scenario}_{tag}_n{samples}.pt"
    generic_f = f"Generic_DataSet_{scenario}_{tag}_n{samples}.pt"
    if not force and (path / model_f).exists() and (path / generic_f).exists():
        print(f"[test ] 复用已存在测试集 {tag}")
        return read_data(path / model_f), read_data(path / generic_f)

    print(f"[test ] 生成测试集 {scenario} {cfg['sweep_param']}={value}, n={samples}")
    set_unified_seed(1234)                        # 所有失配水平共用同一 seed
    model_dataset, generic_dataset, samples_model = create_dataset(
        system_model_params=params,
        samples_size=samples,
        model_type="SubspaceNet",
        tau=TAU_LAGS,
        save_datasets=False,
    )
    torch.save(model_dataset, path / model_f)
    torch.save(generic_dataset, path / generic_f)
    torch.save(samples_model, path / "samples_model.pt")  # 评估要用它算导向矢量
    return model_dataset, generic_dataset


# ------------------------------------------------------------------------------
# 训练
# ------------------------------------------------------------------------------
def train_one(scenario: str, value: float, train_ds, epochs: int, batch_size: int,
              out_dir: Path, smoke: bool = False):
    cfg = SCENARIOS[scenario]
    params = make_params(**{cfg["sweep_param"]: value})
    model_config = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_diff_method("root_music")     # 论文: 训练用可微的 Root-MUSIC
        .set_tau(TAU_LAGS)
        .set_model(params)
    )
    tag = f"{scenario}_{cfg['sweep_param']}={value}".replace(".", "p")
    ckpt = out_dir / f"{tag}.pt"
    if ckpt.exists() and not smoke:
        print(f"[model] 权重已存在, 跳过训练: {ckpt.name}")
        return ckpt

    # 训练集/验证集切分.
    # 注意: N_VALID 是"正式跑"的留出量(500); 冒烟/小数据时必须按比例缩小,
    # 否则 train_only 会变成空集, 训练直接崩在 train_test_split 里
    # (ValueError: With n_samples=0, test_size=0.1 ... train set will be empty).
    n_total = len(train_ds)
    valid_count = max(1, min(N_VALID, n_total // 10)) if n_total > 1 else 0
    rng = np.random.default_rng(0)
    idx = rng.permutation(n_total)
    valid_ds = [train_ds[i] for i in idx[:valid_count]]
    train_only = [train_ds[i] for i in idx[valid_count:]]
    print(f"[model] 样本 {n_total} -> 训练 {len(train_only)} / 验证 {len(valid_ds)}")

    tparams = (
        TrainingParams()
        .set_batch_size(batch_size)
        .set_epochs(epochs)
        .set_model(model=model_config)
        .set_optimizer(optimizer="Adam", learning_rate=LR, weight_decay=WEIGHT_DECAY)
        .set_training_dataset(train_only)
        .set_schedular(step_size=SCHED_STEP, gamma=SCHED_GAMMA)
        .set_criterion()
    )
    # 让训练循环用我们切出来的验证集
    # batch 与训练集保持一致: 验证 loss 按样本数归一化(src/evaluation.py:117-118),
    # 与 batch 无关, 故提速不影响任何报告数值。
    tparams.valid_dataset = torch.utils.data.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tr_loss, va_loss = train_model(
        tparams, model_name=tag, checkpoint_path=out_dir
    )
    torch.save(model.state_dict(), ckpt)
    print(f"[model] {tag}: 训练完成 {(time.time()-t0)/60:.1f} min, "
          f"final train loss={tr_loss[-1]:.4f}, valid loss={va_loss[-1]:.4f}")
    return ckpt


# ------------------------------------------------------------------------------
# 评估: 复刻 evaluate_augmented_model / evaluate_model_based 的调用路径,
#       但用我们自己的 RMSPE(同时输出参考值与角度值), 并在同一批测试样本上跑。
# ------------------------------------------------------------------------------
def eval_on_test(scenario: str, value: float, model_test_ds, generic_test_ds,
                 samples_model, ckpt: Path, algorithms: list[str], limit: int = 0,
                 capture_dir: Path | None = None, capture_index: int = 0,
                 subnet_batch: int = 1):
    cfg = SCENARIOS[scenario]
    params = make_params(**{cfg["sweep_param"]: value})
    # 评估时必须用测试集自己的系统模型 (决定导向矢量/频率)
    model_config = (
        ModelGenerator().set_model_type("SubspaceNet")
        .set_diff_method("root_music").set_tau(TAU_LAGS).set_model(params)
    )
    model = model_config.model
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    # 官方约定: src/training.py:371 与 src/evaluation.py:80 都显式把模型/样本搬到 utils.device
    model = model.to(device)
    model.eval()

    # 为了让 evaluate_* 内部把样本解释成 (1,N,T), 按 main.py 的方式包一层.
    # SubNet 分支现在按 `subnet_batch` 直接索引 model_test_ds (一个 batch 只前向一次),
    # 所以只有传统算法还需要这个 batch=1 的 DataLoader;
    # 传统算法本身是逐样本 numpy, 批不起来.
    generic_dl = torch.utils.data.DataLoader(generic_test_ds, batch_size=1, shuffle=False)
    n_eval = limit if limit > 0 else len(model_test_ds)

    # 经典算法对象是只读的 (narrowband 不改自身状态), 建一次复用, 省掉每样本的构造开销
    music_classic = MUSIC(samples_model)
    rmus_classic = RootMUSIC(samples_model)
    esprit_classic = Esprit(samples_model)

    capture_sample = capture_index if capture_index >= 0 else n_eval - 1
    cap_store: dict = {}

    def _capture(X_model, X_sample, DOA, algo):
        """存下第 capture_sample 个样本的谱/根/协方差, 供 --capture 之后出图用。"""
        gt = DOA.detach().cpu().numpy().ravel()
        mu = MUSIC(samples_model).narrowband(X=X_model, mode="SubspaceNet", model=model)
        rm = RootMUSIC(samples_model).narrowband(X=X_model, mode="SubspaceNet",
                                                 model=model)
        with torch.no_grad():
            cov_sub = model(X_model)[-1].detach().cpu().squeeze().numpy()
        cov_smp = np.cov(X_sample.detach().cpu().numpy())
        tag = f"{cfg['sweep_param']}={value}".replace(".", "p")
        cap_path = _capture_path(capture_dir, scenario, tag)
        cap_path.parent.mkdir(parents=True, exist_ok=True)
        # 系统模型参数单独存成 JSON 字符串: np.savez 存 dict 会变成 object 数组,
        # 而 allow_pickle=False 下读不回来 (ValueError: Object arrays cannot be loaded)
        sm_params = {k: getattr(params, k) for k in
                     ("N", "M", "T", "snr", "signal_type", "signal_nature",
                      "eta", "bias", "sv_noise_var")}
        store = {
            "scenario": scenario, "sweep_name": cfg["sweep_param"],
            "sweep_value": value, "M": M_SOURCES, "algo": algo,
            "true_doa": np.asarray(gt, dtype=float),
            "mus_spectrum": np.asarray(mu[1], dtype=float).ravel(),
            "mus_preds": np.asarray(mu[0], dtype=float).ravel(),
            # 注意: `RootMUSIC.narrowband` 返回的是「已按到单位圆的距离排序」的**全部**根
            # (src/methods.py:475-486), 而 doa_predictions 只取单位圆内的前 M 个
            # (roots_inside = [root for root in roots if ((abs(root)-1) < 0)][:M])。
            # 出图要的是与预测一一对应的那 M 个半径, 所以这里按同样规则再筛一次。
            "rmus_preds": np.asarray(rm[0], dtype=float).ravel(),
            "rmus_mags": np.asarray(
                [abs(r) for r in
                 [r for r in rm[1] if (abs(r) - 1) < 0][:len(np.asarray(rm[0]).ravel())]],
                dtype=float),
            "eig_subnet": _eigs(cov_sub), "eig_sample": _eigs(cov_smp),
            "params": np.array(json.dumps(sm_params, ensure_ascii=False)),
        }
        np.savez(cap_path, **store)
        print(f"[cap  ] 已存谱/根/协方差数据 -> {cap_path.name}")

    results = {}

    # --- SubspaceNet 增强: 所有算法在同一个 batch 上共享一次前向 ---
    # 旧写法的结构是 `for algo in algorithms:` 包住整个测试集循环, 于是每个算法都把整份
    # 测试集前向重算一遍, 而 Rz 与用哪个算法无关 (3 个算法 => 3x 冗余前向).
    # 现在改为外层走 batch, 一次前向喂给全部算法. subnet_batch=1 时算法调用与旧写法逐字相同,
    # 数值逐位一致; subnet_batch>1 时走 src.models 的批量化 root_music/esprit.
    sub_agg = {a: {"ref": 0.0, "deg": 0.0, "fixed": 0.0} for a in algorithms}
    sub_fail = {a: 0 for a in algorithms}
    n_scored = 0
    n_done = 0

    def _score(algo: str, preds, M: int, gt) -> None:
        """累计一个样本 (或一对 (样本, 算法)) 的三套 RMSPE 与失败计数."""
        preds = as_pred_array(preds)
        if preds.size < M:
            sub_fail[algo] += 1
        agg = sub_agg[algo]
        agg["ref"] += rmspe_reference(preds, gt)
        agg["deg"] += rmspe_degrees(preds, gt)
        agg["fixed"] += rmspe_degrees_fixed(preds, gt)

    with torch.no_grad():
        while n_done < n_eval:
            want = min(subnet_batch, n_eval - n_done)
            Xs, DOAs = [], []
            for i in range(n_done, n_done + want):
                xi, di = model_test_ds[i]
                Xs.append(xi)
                DOAs.append(di)
            Xb = torch.stack(Xs).to(device)
            # 注意: DoA 是**逐样本随机**的 (signal_creation.set_doa 每个样本重抽),
            # 不能拿批内第一个样本的 DoA 当整批的真值 —— 那样会把 RMSPE 抬高几个度
            # (实测把逐样本的 25.61 变成 28.78). 批内共享的只有"同一失配水平"这件事.
            gt_b = [d.detach().cpu().numpy().ravel() * R2D for d in DOAs]
            if subnet_batch == 1:
                X = Xb
                for algo in algorithms:
                    if algo == "esprit":
                        preds, M = esprit_classic.narrowband(
                            X=X, mode="SubspaceNet", model=model
                        )
                    elif algo == "music":
                        preds, _spectrum, M = music_classic.narrowband(
                            X=X, mode="SubspaceNet", model=model
                        )
                    else:
                        preds, _roots, _all, _rang, M = rmus_classic.narrowband(
                            X=X, mode="SubspaceNet", model=model
                        )
                    _score(algo, preds, M, gt_b[0])
            else:
                Rz = model(Xb)[-1]           # 一次前向, 供下面全部算法复用
                if "esprit" in algorithms:
                    doa = esprit_batched(Rz, M_SOURCES, Rz.shape[0])
                    rows = np.rad2deg(doa.detach().cpu().numpy())
                    for b in range(rows.shape[0]):
                        _score("esprit", rows[b], M_SOURCES, gt_b[b])
                if "r-music" in algorithms:
                    doa, _all, _roots = root_music_batched(Rz, M_SOURCES, Rz.shape[0])
                    rows = np.rad2deg(doa.detach().cpu().numpy())
                    for b in range(rows.shape[0]):
                        _score("r-music", rows[b], M_SOURCES, gt_b[b])
                if "music" in algorithms:
                    # `eig` (不是 eigh) 是为了与 `subspace_separation` 逐位对齐:
                    # src/methods.py:200-203 用 np.linalg.eig, 且相同 eigenvalue 差
                    # (eigh 会在简并子空间内旋转特征向量, 实测 |λ1|-|λ2| 可小到 4.6e-5).
                    # 注意: `evecs[:, :, idx]` 在 3-D 下是错的 —— (B,N) 的整数索引数组会
                    # 按**前导轴**广播, 于是把 batch 打乱. 必须用 take_along_axis 沿最后一维取.
                    evals, evecs = np.linalg.eig(Rz.detach().cpu().numpy())
                    order = np.argsort(evals, axis=1)[:, ::-1]
                    evecs = np.take_along_axis(evecs, order[:, None, :], axis=2)
                    if evecs.shape[1] != Rz.shape[-1]:
                        raise RuntimeError(
                            f"eig 返回的特征向量形状异常: {evecs.shape} (期望 (B,{Rz.shape[-1]},N))"
                        )
                    m_obj = MUSIC(samples_model)
                    for b in range(Rz.shape[0]):
                        Un = evecs[b][:, M_SOURCES:]
                        spectrum, _core = m_obj.spectrum_calculation(Un, f=1)
                        _score("music",
                               _music_preds_from_spectrum(spectrum, m_obj._angels, M_SOURCES),
                               M_SOURCES, gt_b[b])
            n_done += want
            n_scored += want
            if (capture_dir is not None and n_done - want <= capture_sample < n_done
                    and capture_sample < len(generic_test_ds)):
                for algo in algorithms:
                    if algo in cap_store:
                        continue
                    cap_store[algo] = True
                    off = capture_sample - (n_done - want)
                    _capture(Xb[off:off + 1], generic_test_ds[capture_sample][0],
                             DOAs[off], algo)
    for algo in algorithms:
        results[f"SubNet+{algo}"] = dict(
            rmspe_ref=sub_agg[algo]["ref"] / n_scored,
            rmspe_deg=sub_agg[algo]["deg"] / n_scored,
            rmspe_fixed=sub_agg[algo]["fixed"] / n_scored,
            fail_rate=sub_fail[algo] / n_scored,
        )

    for algo in algorithms:
        # --- 传统经验协方差 (经典算法本身逐样本, 批不起来) ---
        agg = {"ref": 0.0, "deg": 0.0, "fixed": 0.0}
        n_fail, n_used_c = 0, 0
        for i, (X, DOA) in enumerate(generic_dl):
            if i >= n_eval:
                break
            X = X[0].detach().cpu().numpy()
            gt = DOA.detach().cpu().numpy().ravel() * R2D
            if algo == "esprit":
                preds, M = esprit_classic.narrowband(X=X, mode="sample")
            elif algo == "music":
                preds, _, M = music_classic.narrowband(X=X, mode="sample")
            else:
                preds, _roots, _all, _rang, M = rmus_classic.narrowband(
                    X=X, mode="sample"
                )
            preds = np.asarray(preds, dtype=float).ravel()
            if preds.size < M:
                n_fail += 1
            agg["ref"] += rmspe_reference(preds, gt)
            agg["deg"] += rmspe_degrees(preds, gt)
            agg["fixed"] += rmspe_degrees_fixed(preds, gt)
            n_used_c += 1
        results[algo] = dict(
            rmspe_ref=agg["ref"] / n_used_c, rmspe_deg=agg["deg"] / n_used_c,
            rmspe_fixed=agg["fixed"] / n_used_c, fail_rate=n_fail / n_used_c,
        )
    return results


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def run(scenario: str, args):
    cfg = SCENARIOS[scenario]
    if "music" in args.algorithms:
        enable_nominal_steering_without_repo_edit()
    out_dir = DATA_ROOT / "weights" / "array_mismatch"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train, n_test = args.n_train, args.n_test
    epochs, batch_size = args.epochs, args.batch_size
    if args.smoke:
        n_train, n_test, epochs, batch_size = 2000, 200, 3, 512
        print("*** SMOKE 模式: 小数据/少轮次, 只验证流程能否跑通 ***")

    # 0) 设备: 显式打印出来, 免得"以为在卡 1 上跑"结果落在别人占用的卡 0 上.
    #    (device 已在模块导入前由 _early_device() 按 --device 定好)
    if device.type == "cuda":
        print(f"[设备] {device}  {torch.cuda.get_device_name(device)}  "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')})")
    else:
        print(f"[设备] {device}  (全量训练不建议用 CPU)")

    # 1) 用哪个失配水平训练 —— 这是 Fig.9 复现里最关键的协议选择.
    #
    #    论文原文只说明被比较的方法"均在同一批数据上训练", 没有把"训练时失配水平"
    #    写成独立变量; 结合 Sec.IV-B 的组织方式(每种失配程度单独比较), 忠实复现是
    #    **逐点匹配**: 网格上每个 η / σ²_sv 都用同一水平训练出来的模型去评估,
    #    这样曲线反映的是"该失配程度下能达到的精度", 与论文 Fig.9 的读法一致.
    #
    #    --train_levels single 则只训一个模型(默认取 cfg['train_at'])并在整个网格上评估,
    #    得到的是"训练/评估失配不匹配时的鲁棒性", 属于额外消融(E3), 不要与论文曲线混画.
    grid = cfg["grid"] if not args.grid else [float(v) for v in args.grid.split(",")]
    if args.smoke and not args.grid and len(grid) > 2:
        full = grid
        keep = np.unique(np.linspace(0, len(grid) - 1, 2).round().astype(int))
        grid = [float(grid[i]) for i in keep]
        print(f"*** SMOKE: 失配网格缩到 {grid} (完整网格 {full}) ***")
    sweep = cfg["sweep_param"]
    matched = args.train_levels == "matched"
    if matched:
        print(f"[协议] 逐点匹配: 网格上每个 {sweep} 各训练一个模型 (忠实复现 Fig.9)")
    else:
        fixed_at = args.train_at if args.train_at is not None else cfg["train_at"]
        print(f"[协议] 单一模型: 仅在 {sweep}={fixed_at} 训练, 在整个网格上评估 (鲁棒性消融)")

    # 产物路径
    cap_dir = Path(args.capture_dir) if args.capture_dir else DATA_ROOT / "simulations" / "spectra"
    fig_dir = Path(args.fig_dir) if args.fig_dir else DATA_ROOT / "simulations" / "figures"
    json_dir = DATA_ROOT / "simulations" / "results"
    made_figs: list = []

    # 2) 生成数据 + 训练 + 逐水平评估
    rows = []
    for value in grid:
        train_at = value if matched else fixed_at
        train_ds = load_or_create_train(scenario, train_at, n_train, force=args.force_data)
        ckpt = train_one(
            scenario, train_at, train_ds, epochs, batch_size, out_dir, smoke=args.smoke
        )
        model_test_ds, generic_test_ds = load_or_create_test(
            scenario, value, n_test, force=args.force_data
        )
        samples_model = SystemModel(make_params(**{sweep: value}))
        res = eval_on_test(
            scenario, value, model_test_ds, generic_test_ds, samples_model,
            ckpt, args.algorithms, limit=args.limit,
            capture_dir=cap_dir if args.capture else None,
            capture_index=args.capture_index,
            subnet_batch=max(1, args.eval_batch),
        )
        row = {sweep: value,
               "train_at": train_at,
               "matched": bool(matched or train_at == value),
               "method_used": ckpt.name}
        for k, v in res.items():
            row[f"{k}_ref"] = round(v["rmspe_ref"], 4)
            row[f"{k}_deg"] = round(v["rmspe_deg"], 4)
            row[f"{k}_fail"] = round(v["fail_rate"], 4)
        rows.append(row)
        print(f"  {sweep}={value} (train_at={train_at}): " + " | ".join(
            f"{k}={v['rmspe_deg']:.3f}°" for k, v in res.items()))

        # 每个失配水平评估完立刻出「谱图 / 特征值分离图」(PNG)
        if args.capture:
            try:
                tag = f"{sweep}={value}".replace(".", "p")
                cap_file = _capture_path(cap_dir, scenario, tag)
                if cap_file.exists():
                    made_figs += plot_spectrum_figures(
                        cap_file, fig_dir, algorithms=tuple(args.algorithms))
            except Exception as e:                               # noqa: BLE001
                print(f"[plot ] 谱图生成失败(不影响结果): {type(e).__name__}: {e}")

    # 3) 落盘
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = json_dir / f"array_mismatch_{scenario}_{stamp}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"scenario": scenario, "config": {
            "N": N_SENSORS, "M": M_SOURCES, "T": T_SNAPSHOTS, "snr": SNR_DB,
            "tau": TAU_LAGS, "n_train": n_train, "n_test": n_test,
            "batch_size": batch_size, "epochs": epochs, "lr": LR, "bias": BIAS},
         "train_levels": args.train_levels, "rows": rows},
        indent=2, ensure_ascii=False), "utf-8")
    print(f"\n结果已写入 {out_json}")

    # 4) 出「Fig.9 复现曲线图」(PNG). 除非显式 --no_plot, 否则 always 出图
    if not args.no_plot:
        try:
            made_figs += plot_curve_figures([out_json], fig_dir,
                                            algorithms=tuple(args.algorithms))
        except Exception as e:                                   # noqa: BLE001
            print(f"[plot ] 曲线图生成失败(不影响结果): {type(e).__name__}: {e}")
    print(f"\n共生成 {len(made_figs)} 张 PNG -> {fig_dir}")
    return rows


def plot_only(args):
    """`plot` 子命令: 只从已有结果 JSON 出「Fig.9 复现曲线图」(PNG), 不跑实验。"""
    if args.json:
        paths = [Path(p) for p in args.json]
    else:
        paths = sorted((DATA_ROOT / "simulations" / "results").glob("array_mismatch_*.json"))
    if not paths:
        print("未找到结果 JSON; 请先跑 all 或显式给 --json <file>")
        return []
    fig_dir = Path(args.fig_dir) if args.fig_dir else DATA_ROOT / "simulations" / "figures"
    made = plot_curve_figures(paths, fig_dir, algorithms=tuple(args.algorithms))
    print(f"\n共 {len(made)} 张 PNG -> {fig_dir}")
    return made


def main():
    p = argparse.ArgumentParser(
        description="SubspaceNet 阵列失配实验复现 (论文 Sec. IV-B-4 / Fig. 9)")
    p.add_argument("mode", choices=["data", "train", "eval", "all", "plot"],
                   help="只生成数据 / 只训练 / 只评估 / 全流程 / 只用已有 JSON 出曲线图")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="spacing")
    p.add_argument("--n_train", type=int, default=N_TRAIN)
    p.add_argument("--n_test", type=int, default=N_TEST)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--train_levels", choices=["matched", "single"], default="matched",
                   help="matched: 网格上每个失配水平各训一个模型 (忠实复现 Fig.9, 默认); "
                        "single: 只训一个模型在整个网格评估 (鲁棒性消融, 见 --train_at)")
    p.add_argument("--train_at", type=float, default=None,
                   help="--train_levels single 时训练所用的失配水平 (默认取场景配置)")
    p.add_argument("--grid", type=str, default=None,
                   help="评估失配网格, 逗号分隔, 如 0.025,0.05,0.1,0.15")
    p.add_argument("--algorithms", nargs="+", default=["r-music", "esprit", "music"],
                   help="被增强/对比的算法: 三个都给才能出齐谱图 (r-music esprit music)")
    p.add_argument("--limit", type=int, default=0, help="评估样本上限, 0=全部")
    p.add_argument("--eval_batch", type=int, default=1,
                   help="SubNet 评估的分批大小: 1=逐样本 (默认, 与旧结果逐位一致); "
                        ">1 时一个 batch 只前向一次, 三个算法共享 (见文档 §16.16)")
    p.add_argument("--force_data", action="store_true", help="强制重新生成数据集")
    p.add_argument("--device", type=str, default=None,
                   help="用哪块卡跑: default|cpu|cuda|cuda:2|2 "
                        "(默认 default = 按 CUDA_VISIBLE_DEVICES 重编号后的 cuda:0)")
    p.add_argument("--smoke", action="store_true",
                   help="小规模冒烟测试 (网格自动缩到 2 个点)")
    # --- 出图 (全部 PNG) ---
    p.add_argument("--capture", action="store_true",
                   help="评估时顺带保存第 --capture_index 个样本的谱/根/协方差, "
                        "并立刻出谱图与特征值分离图 (PNG)")
    p.add_argument("--capture_index", type=int, default=0,
                   help="采样第几个测试样本存谱, -1=最后一个 (默认 0)")
    p.add_argument("--capture_dir", type=str, default=None,
                   help="谱数据 npz 目录 (默认 <DATA_ROOT>/simulations/spectra)")
    p.add_argument("--fig_dir", type=str, default=None,
                   help="PNG 输出目录 (默认 <DATA_ROOT>/simulations/figures)")
    p.add_argument("--no_plot", action="store_true", help="不出 Fig.9 曲线图")
    p.add_argument("--json", nargs="+", default=None,
                   help="plot 子命令: 指定结果 JSON (默认自动找最新的)")
    args = p.parse_args()

    if args.mode in ("train", "all"):
        # 提前提示 batch 过大可能 OOM
        if args.batch_size > 1024 and not args.smoke:
            print("提示: batch_size > 1024 在 8GB 显存上可能 OOM, OOM 就调小。")
    if args.mode == "plot":
        plot_only(args)
    else:
        run(args.scenario, args)


if __name__ == "__main__":
    main()
