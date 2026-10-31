"""Subspace-Net 
Details
----------
Name: system_model.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the SystemModel class for defining the settings of the DoA estimation system model.
"""

# Imports
import numpy as np
from dataclasses import dataclass


@dataclass
class SystemModelParams:
    """Class for setting parameters of a system model.
    Initialize the SystemModelParams object.

    Parameters:
        None

    Attributes:
        M (int): Number of sources.
        N (int): Number of sensors.
        T (int): Number of observations.
        signal_type (str): Signal type ("NarrowBand" or "Broadband").
        freq_values (list): Frequency values for Broadband signal.
        signal_nature (str): Signal nature ("non-coherent" or "coherent").
        snr (float): Signal-to-noise ratio.
        eta (float): Level of deviation from sensor location.
        bias (float): Sensors locations bias deviation.
        sv_noise_var (float): Steering vector added noise variance.
        gain_err_std (float): 通道增益误差的标准差（线性比值，不是 dB）。默认 0。
            0.05 表示 5%。按式 g_m = 1 + sqrt(12)*sigma_g*U[-1/2,1/2] 抽样，
            故 g_m 的均值恰为 1、标准差恰为 sigma_g、取值落在 1 ± sqrt(3)*sigma_g。
        phase_err_std_deg (float): 通道相位误差的标准差，单位**度**。默认 0。
            名字带 `_deg` 是为了防止按弧度传值；进生成器前会 np.deg2rad 换算。
            按式 phi_m = sqrt(12)*sigma_phi*U[-1/2,1/2] 抽样（弧度），零均值。
        shared_array_mismatch (bool): 位置/幅相失配是否"每个样本抽一次、多信源共享"。
            默认 False —— 这一路径与历史随机流逐位一致。置 True 才走
            `ArrayMismatchState`（见 `SystemModel._draw_mismatch_state`），
            此时同一样本内所有信源看到同一个物理阵列（物理上更正确），
            但与历史数据集不再逐位可比。

    Returns:
        None
    """

    M = None  # Number of sources
    N = None  # Number of sensors
    T = None  # Number of observations
    signal_type = "NarrowBand"  # Signal type ("NarrowBand" or "Broadband")
    freq_values = [0, 500]  # Frequency values for Broadband signal
    signal_nature = "non-coherent"  # Signal nature ("non-coherent" or "coherent")
    snr = 10  # Signal-to-noise ratio
    eta = 0  # Sensor location deviation
    bias = 0  # Sensor bias deviation
    sv_noise_var = 0  # Steering vector added noise variance
    gain_err_std = 0.0  # 通道增益误差的 1σ（线性比值, 0.05 = 5%）
    phase_err_std_deg = 0.0  # 通道相位误差的 1σ（度, 内部转弧度）
    shared_array_mismatch = False  # True: 位置/幅相每样本抽一次, 多信源共享

    # --- 占位: 互耦矩阵 C（下一步的任务, 本次不实现） ---------------------------
    # mc_coeff = None  # 互耦矩阵首行（长度 N 的复向量）, 将来据此生成带状 Toeplitz C

    def set_parameter(self, name: str, value):
        """
        Set the value of the desired system model parameter.

        Args:
            name(str): the name of the SystemModelParams attribute.
            value (int, float, optional): the desired value to assign.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.__setattr__(name, value)
        return self


@dataclass
class ArrayMismatchState:
    """一个样本抽一次的**阵列级**失配量。

    为什么需要它：`samples_creation` 里对每个信源各调一次 `steering_vec`。若让
    `steering_vec` 自己抽失配，同一个快拍内 M 个信源就会看到 M 个不同的物理阵列
    —— 物理上不成立，而且相当于对失配做了 M 次平均、**低估**失配的严重程度。
    把一次抽样结果装进本对象再逐个信源复用，就得到"一个阵列、多个来向"。

    只装阵列自身的物理量，**不含** steering-vector 加噪：后者是"某个来向上的流形
    扰动"，属于信源级，由调用方按信源各自抽出后经 `steering_vec(..., sv_noise=)`
    显式传入。这样 `steering_vec(state=...)` 才能真正做到零随机消耗。

    Attributes:
        bias (float): 全阵共享的间距偏置（λ 计）。
        dist_offset (np.ndarray): 长度 N 的阵元间距偏移（λ 计），U(-eta, eta)。
        gamma (np.ndarray): 长度 N 的复增益（通道幅相误差），见
            `SystemModel._draw_gamma`。
    """

    bias: float
    dist_offset: np.ndarray
    gamma: np.ndarray


def _num_tag(value: float) -> str:
    """把数值变成文件名片段：`0.05 -> 0p05`、`5.0 -> 5p0`。

    文件名里不放小数点（沿用本仓库 `spacing_eta=0p025.pt` 的既有写法）；整数也补
    一位小数，这样 `5` 与 `50` 不会因为 `g` 格式的差别而看起来不统一。
    """
    text = f"{float(value):g}"
    if "." not in text:
        text += ".0"
    return text.replace(".", "p")


def mismatch_suffix(params: "SystemModelParams") -> str:
    """缓存/权重文件名里新增的三个失配字段。

    `src/data_handler.set_dataset_filename`（数据集）与
    `src/training.get_simulation_filename`（权重）**共用**这一个函数，保证两处写法
    永远不会分叉 —— 这两个文件名的历史格式本来就不一致（`SNR={x}` vs
    `SNR_{x}`、`sv_noise_var{}` vs `sv_noise={}`），只有新增字段才要求严格一致。

    规则：三者全为默认（`0 / 0 / False`）时返回空串。此时新代码产生的数据与改动前
    **逐位相同**（见 `probe_sigma0_baseline.py`），让旧缓存与旧权重继续命中既正确、
    又省掉一次毫无意义的重训。任一非默认时把三个字段**全部**写上，避免同一个实验
    出现"一半带字段、一半不带"。

    注意 `shared_array_mismatch=True` **必须**参与命名，即使 eta=0：该路径改变了
    随机数消耗次数（每样本一次而非每信源一次），数据会与 False 时不同。

    Args:
        params (SystemModelParams): 系统模型参数。

    Returns:
        str: 形如 `_sg=0p05_sphi=5p0_share=0` 的后缀；全默认时为空串。
    """
    sigma_g = float(getattr(params, "gain_err_std", 0.0) or 0.0)
    sigma_phi = float(getattr(params, "phase_err_std_deg", 0.0) or 0.0)
    shared = bool(getattr(params, "shared_array_mismatch", False))
    if sigma_g == 0.0 and sigma_phi == 0.0 and not shared:
        return ""
    return (
        f"_sg={_num_tag(sigma_g)}_sphi={_num_tag(sigma_phi)}"
        f"_share={1 if shared else 0}"
    )


class SystemModel(object):
    def __init__(self, system_model_params: SystemModelParams):
        """Class used for defining the settings of the system model.

        Attributes:
        -----------
            signal_type (str): Signals type. Options: "NarrowBand", "Broadband".
            N (int): Number of sensors.
            M (int): Number of sources.
            freq_values (list, optional): Frequency range for broadband signals. Defaults to None.
            min_freq (dict): Minimal frequency value for different scenarios.
            max_freq (dict): Maximal frequency value for different scenarios.
            f_rng (dict): Frequency range of interest for different scenarios.
            f_sampling (dict): Sampling rate for different scenarios.
            time_axis (dict): Time axis for different scenarios.
            dist (dict): Distance between array elements for different scenarios.
            array (np.ndarray): Array of sensor locations.

        Methods:
        --------
            define_scenario_params(freq_values: list): Defines the signal_type parameters.
            create_array(): Creates the array of sensor locations.
            steering_vec(theta: np.ndarray, f: float = 1, array_form: str = "ULA",
                eta: float = 0, geo_noise_var: float = 0) -> np.ndarray: Computes the steering vector.

        """
        self.params = system_model_params
        # Assign signal type parameters
        self.define_scenario_params()
        # Define array indices
        self.create_array()

    def define_scenario_params(self):
        """Defines the signal type parameters based on the specified frequency values."""
        freq_values = self.params.freq_values
        # Define minimal frequency value
        self.min_freq = {"NarrowBand": None, "Broadband": freq_values[0]}
        # Define maximal frequency value
        self.max_freq = {"NarrowBand": None, "Broadband": freq_values[1]}
        # Frequency range of interest
        self.f_rng = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                start=self.min_freq["Broadband"],
                stop=self.max_freq["Broadband"],
                num=self.max_freq["Broadband"] - self.min_freq["Broadband"],
                endpoint=False,
            ),
        }
        # Define sampling rate as twice the maximal frequency
        self.f_sampling = {
            "NarrowBand": None,
            "Broadband": 2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"]),
        }
        # Define time axis
        self.time_axis = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                0, 1, self.f_sampling["Broadband"], endpoint=False
            ),
        }
        # distance between array elements
        self.dist = {
            "NarrowBand": 1 / 2,
            "Broadband": 1
            / (2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"])),
        }

    def create_array(self):
        """create an array of sensors locations"""
        self.array = np.linspace(0, self.params.N, self.params.N, endpoint=False)

    def _draw_gamma(self) -> np.ndarray:
        """抽一次通道幅相误差 Γ 的对角线（长度 N）。

        定义（Guo 等的标准生成式）：

            g_m   = 1 + sqrt(12) * sigma_g * xi_m          xi_m  ~ U[-1/2, 1/2]
            phi_m =     sqrt(12) * sigma_phi * zeta_m      zeta_m ~ U[-1/2, 1/2]
            gamma_m = g_m * exp(j * phi_m)

        `U[-1/2,1/2]` 的方差是 1/12，乘 sqrt(12) 后方差为 1，所以 g_m 的标准差
        **恰好**等于 sigma_g、phi_m 的标准差恰好等于 sigma_phi；取值范围分别是
        `1 ± sqrt(3)*sigma_g` 与 `± sqrt(3)*sigma_phi`。**不要**改成不缩放的
        `U[-sigma, sigma]`：两种写法的有效离散度差 sqrt(3) 倍。

        两个 sigma 都为 0 时**直接返回全 1、不调用任何随机函数**——这是硬要求：
        零幅相误差必须与历史随机流逐位一致，而 np.random.uniform(0, 0, size=N)
        是**会**消耗随机数的（零宽度不是 no-op）。

        Returns:
            np.ndarray: shape (N,) 的 complex128 复增益。
        """
        sigma_g = float(self.params.gain_err_std)
        sigma_phi = np.deg2rad(float(self.params.phase_err_std_deg))
        if sigma_g == 0.0 and sigma_phi == 0.0:
            return np.ones(self.params.N, dtype=np.complex128)
        xi = np.random.uniform(low=-0.5, high=0.5, size=self.params.N)
        zeta = np.random.uniform(low=-0.5, high=0.5, size=self.params.N)
        gain = 1.0 + np.sqrt(12.0) * sigma_g * xi
        phase = np.sqrt(12.0) * sigma_phi * zeta
        return (gain * np.exp(1j * phase)).astype(np.complex128)

    def _draw_mismatch_state(self) -> "ArrayMismatchState":
        """抽一次阵列级失配（位置 + 幅相），装进 `ArrayMismatchState`。

        仅供 `shared_array_mismatch=True` 的路径使用：每样本调一次，再让所有
        信源共用。抽出的随机数个数为 `1 + N + (2N or 0)`，与旧路径的
        `M * (1 + N + N)` 不同，因此该路径的数据与历史数据集不可逐位对照。

        Returns:
            ArrayMismatchState: 本次样本的阵列级失配量。
        """
        bias = float(
            np.random.uniform(
                low=-1 * self.params.bias, high=self.params.bias, size=1
            )[0]
        )
        dist_offset = np.random.uniform(
            low=-1 * self.params.eta, high=self.params.eta, size=self.params.N
        )
        return ArrayMismatchState(
            bias=bias, dist_offset=dist_offset, gamma=self._draw_gamma()
        )

    def steering_vec(
        self,
        theta: np.ndarray,
        f: float = 1,
        array_form="ULA",
        nominal=False,
        gamma=None,
        state=None,
        sv_noise=None,
    ):
        """Computes the steering vector based on the specified parameters.

        Args:
        -----
            theta (np.ndarray): Array of angles.
            f (float, optional): Frequency. Defaults to 1.
            array_form (str, optional): Array form. Defaults to "ULA".
            nominal (bool): flag for creating sv without array mismatches.
            gamma (np.ndarray, optional): 长度 N 的复增益（通道幅相误差 Γ 的对角线）。
                给了就乘到流形上；不给则按 `params.gain_err_std` /
                `params.phase_err_std_deg` 自抽（两者为 0 时返回全 1 且不消耗随机数）。
                `nominal=True` 时**禁止**传 gamma（标称流形按定义不含失配）。
            state (ArrayMismatchState, optional): 若给出，位置与幅相**全部**取自该
                状态，本函数一个随机数都不抽（Step B 专用）。
            sv_noise (np.ndarray, optional): 已按 `sv_noise_var` 缩放好的
                steering-vector 加噪向量（长度 N，实数）。仅在 `state` 路径下使用：
                它是**信源级**的流形扰动，由调用方逐个信源抽取后传进来。

        Returns:
        --------
            np.ndarray: Computed steering vector.

        """
        f_sv = {"NarrowBand": 1, "Broadband": f}
        if not array_form.startswith("ULA"):
            raise Exception(
                f"SystemModel.steering_vec: array form {array_form} is not defined"
            )
        if nominal:
            # 标称流形: 既没有失配, 也不乘 Γ (否则"标称"就没意义了)
            if gamma is not None:
                raise ValueError(
                    "SystemModel.steering_vec: nominal=True 时不能传 gamma"
                    " (标称流形按定义不含通道幅相误差); 要带 Γ 的流形请用"
                    " gamma=... 且 nominal=False."
                )
            return np.exp(
                -2
                * 1j
                * np.pi
                * f_sv[self.params.signal_type]
                * self.dist[self.params.signal_type]
                * self.array
                * np.sin(theta)
            )
        # --- 非标称: 两种落法, 都不能改变 σ=0 时的随机流 ----------------------
        if state is not None:
            # Step B: 位置与幅相全部由外部给, 本函数零随机消耗
            uniform_bias = state.bias
            mis_distance = state.dist_offset
            gamma_vec = state.gamma
            mis_geometry_noise = 0.0 if sv_noise is None else np.asarray(sv_noise)
        else:
            # 历史路径: 抽样顺序/次数一律照旧
            # Calculate uniform bias for sensors locations
            uniform_bias = np.random.uniform(
                low=-1 * self.params.bias, high=self.params.bias, size=1
            )
            # Calculate non-uniform bias for each pair of sensors
            mis_distance = np.random.uniform(
                low=-1 * self.params.eta, high=self.params.eta, size=self.params.N
            )
            # Calculate additional steering vector noise
            mis_geometry_noise = np.sqrt(self.params.sv_noise_var) * (
                np.random.randn(self.params.N)
            )
            gamma_vec = self._draw_gamma() if gamma is None else np.asarray(gamma)
        return (
            np.exp(
                -2
                * 1j
                * np.pi
                * f_sv[self.params.signal_type]
                * (uniform_bias + mis_distance + self.dist[self.params.signal_type])
                * self.array
                * np.sin(theta)
            )
            + mis_geometry_noise
        ) * gamma_vec

    def steering_vec_batch(
        self, theta: np.ndarray, f: float = 1, array_form="ULA", nominal=False, gamma=None
    ):
        """Steering vectors for *many* angles in one shot.

        数学上等价于对每个角度各调用一次 `steering_vec(..., nominal=True)` 再把结果
        摞起来, 但只用一次广播 —— 18000 点的 MUSIC 网格因此从毫秒级 Python 循环
        变成一次矩阵运算。

        两种用法二选一(都必须**不消耗随机数**):
          1. `nominal=True, gamma=None`: 标称流形。`spectrum_calculation` / MVDR
             求响应曲线用的就是这一种。
          2. `nominal=False, gamma=g`: 把已知的通道幅相误差 g 乘到标称流形上。
             这是给评估阶段"固定一个 Γ 看 MUSIC 谱"用的 —— Γ 由外部传入,
             不在本函数里抽。

        `nominal=False` 且 `gamma=None` 时**拒绝执行**: 会去抽失配参数, 而批量的失配
        矩阵不是单个流形, 没有意义; 同时抽随机数会改变全局随机状态, 破坏数据生成的
        可复现性。

        Args:
        -----
            theta (np.ndarray): 角度数组, shape (G,) (也可以传 (G, 1)).
            f (float, optional): 频率. Defaults to 1.
            array_form (str, optional): 阵列形式. Defaults to "ULA".
            nominal (bool): 标称流形开关.
            gamma (np.ndarray, optional): 长度 N 的复增益; 给了就乘到每行上.

        Returns:
        --------
            np.ndarray: shape (G, N) 的复数矩阵, 第 g 行是角度 theta[g] 的导向矢量.
        """
        if nominal and gamma is not None:
            raise ValueError(
                "steering_vec_batch: nominal=True 与 gamma 互斥"
                " (标称流形不含通道幅相误差)."
            )
        if not nominal and gamma is None:
            raise ValueError(
                "steering_vec_batch 只接受 nominal=True 或显式传入 gamma"
                " (见 docstring)."
            )
        if not array_form.startswith("ULA"):
            raise Exception(
                f"SystemModel.steering_vec_batch: array form {array_form} is not defined"
            )
        f_sv = {"NarrowBand": 1, "Broadband": f}
        theta = np.asarray(theta, dtype=float).reshape(-1)
        phase = (
            -2
            * np.pi
            * f_sv[self.params.signal_type]
            * self.dist[self.params.signal_type]
            * self.array[None, :]
            * np.sin(theta)[:, None]
        )
        vectors = np.exp(1j * phase)
        if gamma is not None:
            vectors = vectors * np.asarray(gamma).reshape(1, -1)
        return vectors

    def __str__(self):
        """Returns a string representation of the SystemModel object.
        ...

        """
        print("System Model Summery:")
        for key, value in self.__dict__.items():
            print(key, " = ", value)
        return "End of Model"
