# SubspaceNet 阵列失配实验复现指导

> 论文: D. H. Shmuel, J. P. Merkofer, G. Revach, R. J. G. van Sloun, N. Shlezinger,
> *"SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation"*
> 对应章节: **Section IV-B-4 "Array Miscalibration" + Fig. 9**（另见 Sec. IV-A 实验设置）
>
> 目标仓库: `D:\SubspaceNet`（shmueldo/SubspaceNet 官方实现 + 本地 CUDA 适配）
> 复现位置: **不本机执行**，本文档提供规格与脚本，实际实验在 SSH 专用服务器上跑。
>
> **文档导航**：§1–§3 是"必须先搞清的规格与陷阱"（读这三节就能避开 90% 的坑）；§4–§6 是数据/训练/资源的具体配置；§7–§9 是可直接执行的脚本与部署命令；**§10.5 说明本文档为何不依赖 Appendix，以及 Sec. IV-C 为什么必读**；§11–§12 是检查清单与总结。

---

## 1. 实验规格：论文到底做了什么

### 1.1 共同设置（Sec. IV-A + IV-B-4）

| 项目 | 取值 | 出处 |
|---|---|---|
| 阵元数 N | 8（半波长均匀线阵 ULA） | Sec. IV-A-1 |
| 信源数 M | **2**（失配实验专用） | Sec. IV-B-4 首段 |
| 信源性质 | **non-coherent**（非相干） | Sec. IV-B-4 首段 |
| 信号类型 | NarrowBand（窄带，式 (1)） | Sec. IV-A-1 |
| DoA 生成 | 在 $[-\pi/2,\pi/2]$ 上均匀随机 | Sec. IV-A-1 |
| SNR | $10\log_{10}\sigma_S^2/\sigma_V^2$，失配实验为 **10 dB** | Sec. IV-A-1/2 |
| 快拍数 T | 论文失配节未明写（推荐 T=100） | 推断 |
| 评估指标 | RMSPE（式 (16)） | Sec. IV-A-3 |
| Monte Carlo | **5000 次平均** | Sec. IV-B 首段 |
| $\hat M$ | 评估时**固定为真实 M**（对消"数源不准"的影响） | Sec. IV-B 首段 |
| 每个失配水平训练一个模型 | 是——"trained ... for a given array" | Sec. III-E |

### 1.2 场景一：阵元间距失配（Fig. 9(a)）

相邻阵元间距 = 标称半波长 $d$ + 随机扰动 $\delta_m \sim U(-\eta,\eta)$，导向矢量变为论文式 (21)：

$$[\boldsymbol a(\theta)]_m = e^{-2\pi j\frac{(d+\delta_m)(m-1)}{c}\sin\theta},\quad m\in\{1,\dots,N\}$$

- **扫描变量**：$\eta$，Fig. 9(a) 横轴为 **0.025 / 0.05 / 0.10 / 0.15**
- **论文原话**："for different values of $\eta\in[0.025d,0.15d]$, i.e., a maximum deviation of 30% from the calibrated spacing"
- 结论：经典子空间方法随 $\eta$ 增大而明显劣化，SubspaceNet 增强后曲线基本平坦

### 1.3 场景二：导向矢量加噪（Fig. 9(b)）

不扰动几何，直接给每个阵元导向矢量分量叠加零均值复高斯噪声，方差 $\sigma_{sv}^2$：

$$[\boldsymbol a(\theta)]_m \leftarrow [\boldsymbol a(\theta)]_m + \mathcal{CN}(0,\sigma_{sv}^2)$$

- **扫描变量**：$\sigma_{sv}^2$，Fig. 9(b) 横轴为 **0 / 0.25 / 0.5 / 0.75**
- **论文原话**："for both $\sigma_{sv}^2=0.75$ and $\eta=0.075d$"（说明两个场景的"最难点"分别是这两个值）
- 结论：经典方法对导向矢量污染**极其敏感**，随方差增大迅速崩坏；SubspaceNet 差距随污染程度拉大

### 1.4 论文明确写了的量 vs 必须自己定的量

**论文写了的（Sec. IV-A-2 末段，原文）**：

> "All data-driven DoA estimators are trained on the same data, comprised of $J = 45000$ samples corresponding to SNR of 10 dB (unless stated otherwise). When evaluating the data-driven estimators with varying number of sources, the data is comprised of all considered values of $M$, and training is followed by **adaption to the considered value of $M$ using 8000 samples with $M$ sources**."

即：**训练集 45000 样本、SNR=10 dB**，并且**对每个 M 各有一个 8000 样本的适配集**。所以"每个失配水平/每个 M 训练一个模型"是有论文依据的，不是我们的臆测。

**论文没写的（必须自己定）**：

| 未明确项 | 本文建议 | 理由 |
|---|---|---|
| 失配实验训练时用多大的失配 | **逐点匹配**（每个失配水平用它自己的失配训练一个模型，脚本默认 `--train_levels matched`），另补 `--train_levels single --train_at 0.0` 与 `--train_at 0.15` 两组消融（见 §8） | `IV-B-4` 完全没写训练协议；论文其它实验凡是"训练条件≠评估条件"都明说了（宽带、低 SNR），失配这一节什么都没说，故取"训练=评估"作为主口径；论文 Sec. III-E 只说 "trained in a supervised manner with sufficient data **for a given array**" |
| 快拍数 T | 100 | 论文失配节未写；取充裕快拍以隔离失配这一单一变量（失配实验不考察 AS4） |
| 训练样本量 | 45000（脚本默认，与论文一致） | 论文 Sec. IV-A-3 原文 $J=45000$；显存/时间不够时再下调，见 §6 与 §16 |

> ⚠️ **"逐点匹配"的代价**：4 个失配水平就是 4 个模型、4 份训练数据，训练时间是单模型的 4 倍。这是忠实复现 Fig. 9 的必要代价——用单一失配训练的模型去评估其它失配水平，衡量的是另一件事（那正是 §8 的消融 E3）。

> ⚠️ **代码不支持"混合 M 训练"**：`src/system_model.py:43` 的 `SystemModelParams.M` 是**单个整数**（`set_parameter("M", 2)`），数据集与模型都绑定单一 M。因此论文那套"用所有 M 的混合数据预训练 + 每个 M 用 8000 样本适配"的流程**在本仓库里无法直接照搬**。
> 好消息：失配实验固定 M=2，本来就是单 M 场景，**直接用一个 M=2 的数据集训练即可**，不构成阻塞。只是不要误以为可以复用 Table I 那套混合 M 的数据集。

---

## 2. ⚠️ 第一号陷阱：论文的 $\eta$ 与代码的 `eta` 不是同一个东西

`src/system_model.py:161-188` 里失配是这么加的：

```python
uniform_bias = np.random.uniform(-bias, bias, size=1)          # 所有阵元共享 (!)
mis_distance = np.random.uniform(-eta, eta, size=N)            # 每个阵元独立
mis_geometry_noise = np.sqrt(sv_noise_var) * np.random.randn(N)
return np.exp(-2j*np.pi*f_sv*(uniform_bias + mis_distance + self.dist[...])*self.array*np.sin(theta)) + mis_geometry_noise
```

其中 `self.dist["NarrowBand"] = 1/2`，`self.array = [0,1,...,N-1]`。于是：

$$\text{实际间距} = \underbrace{0.5}_{\text{半波长}} + \underbrace{\delta_m}_{\sim U(-\eta,\eta)} \quad\Longrightarrow\quad \text{相对偏离} = \frac{\eta}{0.5} = 2\eta$$

| 代码 `eta` | 实际间距范围 | 相对标称间距的最大偏离 |
|---|---|---|
| 0.025 | 0.475 ~ 0.525 | 5% |
| 0.05 | 0.45 ~ 0.55 | 10% |
| 0.10 | 0.40 ~ 0.60 | 20% |
| **0.15** | **0.35 ~ 0.65** | **30%** ← 论文说的就是这一行 |

**结论与操作**：

1. **直接把 0.025/0.05/0.10/0.15 填进 `eta` 即可**，与论文数值一致，不要再去乘 0.5 或除 0.5。
2. **论文自身就自相矛盾，而且矛盾的一方正好证明了本节的映射**。原文两句紧挨着的话：
   - 式(21) 之前："$\delta_m\sim U(-\eta,\eta)$) where **$\eta$ is the percentage of deviation from the nominal spacing**"（说 η 是百分比）
   - 式(21) 之后："for different values of $\eta\in[0.025d,0.15d]$, i.e., **a maximum deviation of 30% from the calibrated spacing**"（说 η 是长度量，带单位 $d$，且 0.15 = 30%）

   后一句只有在 $\eta$ 是**绝对偏移量**（相对 $d=0.5$ 归一化后 $0.15/0.5=30\%$）时才自洽。所以"η 是百分比"是论文措辞失误，**以式(21) 的记号和数值陈述为准**。写作时用"$\eta$ 为相对半波长的绝对偏移量，最大偏离标称间距 30%"来表述最安全。
2b. **代码的 docstring 也是这个意思**：`src/system_model.py:35` 把 `eta` 注释为 `Level of deviation from sensor location`（"传感器位置的偏离**量**"），而不是"偏离百分比"。三处证据（式(21) 记号、"30%"那句、代码 docstring）互相印证。
3. `bias` 是**所有阵元共用的一个**位置偏置（`size=1`），物理上是随机阵列相位中心平移，与论文描述的失配机制无关（代码 docstring 亦区分：`eta` = "Sensor location deviation"，`bias` = "Sensor bias deviation"，见 `src/system_model.py:35-36`）。**必须显式 `bias=0`**，否则会额外引入最多 $0.05\times2\pi\times7=1.26\pi$ 的相位误差（0.05 是 `main.py` 里的历史默认值），把失配曲线整体压低、结论失真。
4. `sv_noise_var` 就是论文的 $\sigma_{sv}^2$，**语义完全一致**，可直接填 0/0.25/0.5/0.75。
5. 注意式 (21) 的**累加**语义：论文写 $(d+\delta_m)(m-1)$，代码写 $(\text{bias}+\delta_m+d)\cdot m_{\text{index}}$，二者都等价于"第 $m$ 段间距独立扰动"，一致。

---

## 3. ⚠️ 第二号陷阱：RMSPE 的度/弧度缩放 bug（决定你能否对齐论文数字）

`src/criterions.py:190-214` 的最终评估指标：

```python
error = (((p - doa) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
rmspe_val = (1 / np.sqrt(len(p))) * np.linalg.norm(error)
```

`doa` 与 `p` 都是**度**（调用处 `evaluate_model_based` 传的是 `doa * R2D`、`predictions` 也是度）。代码想把度转弧度，于是乘了 `np.pi/180`，但**环绕区间仍然用的 `np.pi/2`（度为量纲）**。结果是：

- 数值上等价于"把角度误差先除以 $180/\pi$，再无意义地 mod 一次"
- **输出值 = 真实角度误差(度) × π/180 ≈ 真实值 ÷ 57.2958**

我实测验证过（`pred=[1,3], doa=[0,0]`）：

```
RMSPE 参考实现输出 = 0.03902674850578186
真实 RMSE (度)     = 2.23606797749979
比值               = 0.0174532925199433 = π/180
```

**这意味着**：论文 Table I/II/III 与 Fig. 9 纵轴的**绝对数字都被压缩了 57.3 倍**，它们不是"度"。例如 Table II 里 `Root-MUSIC = 26.3560`，换算成真实角度误差是约 $26.36\times57.2958\approx1510°$ —— 这显然不是角度，而是"完全失效"的数值表现。

### 3.1 好消息：归一化方式其实是对的（顺手排除一个嫌疑点）

论文式(16) 为 $l = \min_{\boldsymbol P}\big(\frac{1}{M}\|\mathrm{mod}_\pi(\boldsymbol\theta - \boldsymbol P\hat{\boldsymbol\theta})\|^2\big)^{1/2}$，即 $1/\sqrt{M}$ 倍的误差范数。代码 `rmspe_val = (1/np.sqrt(len(p))) * np.linalg.norm(error)` 中 `p` 是**过滤掉 0 之后的预测数组**，在 $\hat M = M$ 时 $1/\sqrt{M}$ 与论文的 $1/\sqrt M$ **完全等价**（$\frac{1}{\sqrt M}\|\boldsymbol e\|_2 = \sqrt{\frac1M\|\boldsymbol e\|_2^2}$）。

- **失配实验按论文固定 $\hat M = M = 2$，这一项完全对齐**，不用改 ✅
- 仅在 $\hat M < M$ 时有细微差别：论文恒除以 $M$，代码除以 $\hat M$（相差 $\sqrt{M/\hat M}$）。做 §8 的"估计 $\hat M$"补充实验时需要注意这一点
- 论文 reference `[41]` 确实是 RMSPE 的出处（Routtenberg & Tabrikian, *Bayesian parameter estimation using periodic cost functions*, IEEE TSP 2012，论文 .md 第 544 行），与之相符

也就是说：**式(16) 的归一化和置换不变性在代码里都实现对了，出问题的只有度/弧度这一个点** —— 这反而让 57.3 倍的结论更干净（不是多个 bug 叠加出来的）。

**你应该怎么做（二选一，并在文中说明）**：

| 目的 | 做法 |
|---|---|
| **对齐论文曲线形状与数字** | 复用 `src/criterions.RMSPE` 原样，纵轴标注"参考实现 RMSPE（含 π/180 缩放）"。所有方法受到同一缩放，**相对比较与曲线形状完全有效** |
| 得到**有物理意义**的角度误差 | 用下面这段修正版（全程度域 mod 180），或把现有输出乘 $180/\pi$ 作为"角度 RMSPE [°]" |

```python
def rmspe_degrees(predictions, doa):
    """物理意义正确的 RMSPE，单位: 度。与原实现恒差 180/pi。"""
    pred = np.asarray(predictions, float).ravel()
    pred = pred[pred != 0.0]                      # 式 (16): M_hat < M 时补零
    doa = np.asarray(doa, float).ravel()
    best = np.inf
    for p in permutations(pred, len(pred)):
        err = (((np.asarray(p) - doa) + 90.0) % 180.0) - 90.0   # 度域 mod 180
        best = min(best, np.linalg.norm(err) / np.sqrt(len(pred)))
    return best
```

> 推荐做法：**两套都算并都报**（`reproduce_array_mismatch.py` 已同时输出 `rmspe_ref` / `rmspe_deg` / `rmspe_fixed` 三个口径），
> 对齐论文用前者，讨论物理精度用后者，避免审稿人质疑。

---

## 4. 数据管线：怎么把失配灌进数据里

### 4.1 关键事实

- 失配在 `Samples.samples_creation()` → `self.steering_vec(theta)` 生成观测矩阵 $X=A(\theta)S+V$（`src/signal_creation.py:125`）时注入
- `create_dataset()`（`src/data_handler.py:101-110`）在 **10000 次循环里每次重新调用** `samples_creation`，所以**每个样本的阵列几何都是独立新抽的**，模型看到的是失配的分布而不是某个固定阵列 ✅ 这正是复现需要的
- 算法一侧（MUSIC / Root-MUSIC / MVDR）在做谱搜索时用 `nominal=True` 拿**标称理想**导向矢量（`src/methods.py:270`、`src/methods.py:641`），与数据生成端不一致 —— 这个不一致**就是失配**，也是经典方法性能下降的根源 ✅ 逻辑正确

### 4.2 测试集构建要求

| 要求 | 做法 | 原因 |
|---|---|---|
| 每个失配水平一个测试集 | 独立目录 `data/datasets/array_mismatch_{scenario}/test/` | 文件名只含 `eta/sv_noise_var/bias`，同名会互相覆盖 |
| 各水平**共享同一批 DoA 与噪声** | 生成每个测试集前调用同一个 `set_unified_seed(1234)` | 保证曲线差异只来自失配，而非不同的随机角度；否则 5000 次 MC 也压不住组间差异 |
| 测试样本数 = 5000 | `n_test = 5000` | 对齐论文的 5000 次 Monte Carlo |
| 评估用该系统模型对象 | `SystemModel(make_params(eta=该水平))` | Root-MUSIC/ESPRIT 的 `steering_vec` 与频率依赖它 |

> ⚠️ `set_unified_seed()` 会同时重置 `random`/`numpy`/`torch`，因此**在两次 `create_dataset` 之间不要插入其它随机操作**（含 `model.eval()` 之外的任何采样），否则共享样本的保证会被破坏。

### 4.3 直接改 `main.py` 的最简路径（想快速出单点结果时用）

`main.py:78-89` 就是失配参数的注入口，改这三行即可：

```python
system_model_params = (
    SystemModelParams()
    .set_parameter("N", 8).set_parameter("M", 2)      # 2 个数源
    .set_parameter("T", 100).set_parameter("snr", 10)
    .set_parameter("signal_type", "NarrowBand")
    .set_parameter("signal_nature", "non-coherent")
    .set_parameter("eta", 0.15)          # ← 场景一扫描这个
    .set_parameter("bias", 0.0)          # ← 必须置 0
    .set_parameter("sv_noise_var", 0.0)  # ← 场景二扫描这个
)
```

并将 `scenario_data_path = "array_mismatch_spacing"`（`main.py:46`）以免和已有数据混在一起。
**缺点**：一条命令只能跑一个失配水平，扫描要多跑十几次；建议用 §7 的脚本。

---

## 5. 训练配置：怎么对齐论文

论文 Sec. IV-A-2 给出的**架构与优化器**（必须照抄的部分）：

| 项目 | 论文值 | 代码位置 |
|---|---|---|
| 编码器 3 层 CNN 输出通道 | 16, 32, 64 | `src/models.py:296-298` ✅ 已一致 |
| 解码器 3 层 DCNN 输出通道 | 32, 16, 1 | `src/models.py:299-301` ✅ 已一致 |
| 卷积核 | 2×2（所有层） | 同上 ✅ |
| 激活函数 | AReLU（正负部分拼接） | `src/models.py:328-340` ✅ |
| 后处理 | $\hat R = KK^H+\epsilon I_N$，$\epsilon=1$ | `src/models.py:387-389` |
| 可训练参数量 | 41,761 | 可直接 `sum(p.numel() for p in model.parameters())` 验证 |
| 优化器 / 学习率 | Adam，$\mu=0.001$ | `main.py:166` 当前是 `1e-5` ⚠️ **建议改回 1e-3** |
| 训练用可微子空间方法 | Root-MUSIC | `set_diff_method("root_music")` ✅ |
| 每 epoch | 打乱后分 B 个 batch | `src/training.py:281-283` |

**需要注意的代码行为**：

1. `TrainingParams.set_training_dataset()`（`src/training.py:262-287`）内部会再做一次 `train_test_split(test_size=0.1)`，而训练时**只用切出来的 90%**，剩下的 10% 仅用于验证。所以 `n_train=10000` 实际只有 9000 参与梯度更新。
2. `train_model` 用 `batch_size=1` 跑验证集、逐样本求 loss（`src/training.py:284-286`）非常慢；脚本里可把验证集降到 500 条。
3. `train()` 里的 `tensorboard`-style 逐 epoch checkpoint 没必要，直接调 `train_model` 更干净（`reproduce_array_mismatch.py` 就是这么做的）。
4. 损失 `RMSPELoss`（`src/criterions.py:87-125`）是**累加求和**而非求平均（`result = torch.sum(...)`），所以打印出来的训练 loss 会随 batch 内容波动很大，**看趋势不看绝对值**。

---

## 6. 服务器资源与耗时估算

**先看结论**：数据生成在自动相关张量向量化之后**已经不是瓶颈**（实测提速约 180 倍），真正的瓶颈是训练，而训练的热点又集中在 `root_music`（已批量向量化，见 §16）。所以排期时按"训练时间 × 失配水平数"估算即可。

本机已有实测基准：`10000` 样本、`T=200`、`tau=8` 生成耗时约 5 分钟，落盘 47 MB（`SubspaceNet_DataSet_*_10000_*.h5` = 47.41 MB）。注意这是**向量化前**的数字。

| 阶段 | 规模 | 磁盘 | 时间（参考） |
|---|---|---|---|
| 单个训练集生成 | 45000 样本, T=100 | ~105 MB | 向量化后约 1–3 min（取决于 CPU/IO） |
| 单个测试集生成 | 5000 样本, T=100 | ~12 MB + Generic ~30 MB | 同上，约 1 min |
| 训练（单模型） | 40500 训练样本 × 80 epoch，batch 1024 | 权重 0.17 MB | 见 §16 的吞吐实测换算 |
| 评估（单失配水平） | 5000 样本 × (1 增强 + 2 基线) | — | ~2–5 min |

**场景一总预算**（4 个失配水平，逐点匹配）：数据 ~10 min + 训练 4 个模型 + 评估 20 min。
**场景二总预算**：同上。

**显存与 batch**（重要）：

- 输入张量 $[B, \tau, 2N, N]$ 是主要占用：$B=2048,\tau=8,N=8$ 时 float32 约 16 MB，不是瓶颈
- 真正吃显存的是对 $B$ 个 $8\times8$ 复矩阵做 `torch.linalg.eig` 并**反传**（`src/models.py:744` `root_music`）
- **建议从 `batch_size=1024` 起测**；OOM 就降到 512/256。实测 batch 1024 时峰值显存仅约 244 MiB（RTX 4060 Laptop 8 GiB），所以显存不是限制项
- 训练集用论文的 45000 样本时，单个文件约 105 MB，数据集目录预留 1–2 GB

---

## 7. 直接可用的脚本

仓库里已放好 `reproduce_array_mismatch.py`，把 §1–§5 的所有规格固化成命令行参数：

```bash
# 0) 先冒烟测试（极小数据量，1~2 分钟，只为确认链路通）
#    默认 --smoke 是 2000 样本/3 epoch，仍要几分钟；想更快就直接给小数
python reproduce_array_mismatch.py all --scenario spacing \
    --n_train 40 --n_test 20 --epochs 1 --batch_size 8 --limit 20

# 1) 场景一：间距失配。默认 --train_levels matched，网格上每个 eta 各训一个模型
#    （--n_train 默认已是 45000，与论文 Sec. IV-A-3 一致，这里显式写出以便看清）
python reproduce_array_mismatch.py all --scenario spacing \
    --n_train 45000 --n_test 5000 --epochs 80 --batch_size 1024 \
    --algorithms r-music esprit music

# 2) 场景二：导向矢量加噪
python reproduce_array_mismatch.py all --scenario sv_noise \
    --algorithms r-music esprit music

# 3) ⚠️ data / train / eval / all 四个模式跑的是**同一套流程**（同一处 run()），
#    "只生成数据 / 只训练 / 只评估"只是历史命名。真正的"只评估"靠缓存命中：
#    第二次跑同一条命令会跳过已存在的数据集与权重；唯一只读的模式是 plot（只出图）。
python reproduce_array_mismatch.py all --scenario spacing   # 想继续上一次就跑同一条命令
python reproduce_array_mismatch.py plot                     # 只读已有 JSON 出图

# 4) 鲁棒性消融（E3）：只训一个模型，在整个网格上评估
#    注意这与论文 Fig.9 的曲线口径不同，别混画
python reproduce_array_mismatch.py all --scenario spacing \
    --train_levels single --train_at 0.0

# 5) 评估侧先用少量样本试跑（--limit 只截断评估样本数，不影响生成与训练）
python reproduce_array_mismatch.py all --scenario spacing --limit 200
```

脚本参数一览：`--scenario spacing|sv_noise`、`--n_train`、`--n_test`、`--epochs`、`--batch_size`、
`--train_levels matched|single`（**默认 matched = 逐点匹配训练，忠实复现 Fig.9**；single 见上）、
`--train_at`（仅 `--train_levels single` 时生效）、`--grid`（评估网格，逗号分隔）、
`--algorithms r-music esprit music`（默认三个都给）、`--limit`（评估样本上限，0=全部）、
`--force_data`（强制重新生成数据集）、`--smoke`（2000 样本/3 epoch 的小规模冒烟；且未显式给 `--grid` 时只跑首尾两个失配水平）。

出图相关参数（详见 **§14**）：子命令 `plot`（从已有 JSON 重画曲线图，不重跑实验）、
`--json`（`plot` 用，可给多个结果 JSON）、`--metrics deg ref`、`--fig_dir`（PNG 输出目录）、
`--capture`（评估时顺带抓 MUSIC 谱 / Root-MUSIC 根 / 两套协方差特征值）、
`--capture_index`（用第几个测试样本出体检图，默认 0，-1=最后一个）、
`--capture_dir`（npz 落盘目录）、`--no_plot`（只要数字不要图）。

**产物根目录**：默认 `<仓库>/data`，可用环境变量覆盖（服务器想放数据集到大盘、或 `data/` 不可写时用）：

```bash
SUBSPACENET_DATA_ROOT=/data/subspacenet python reproduce_array_mismatch.py all --scenario spacing
```

### 7.1 上全量之前先跑环境自检

同目录下的 `preflight_check.py` 用 **~1 分钟、只读、不留垃圾**的方式验证 17 项前置条件，
任何 `[FAIL]` 都意味着全量跑（数小时）会失败或结论无效：

```bash
python preflight_check.py          # 17 项静态+轻量检查
python preflight_check.py --full   # 额外做一次真实训练冒烟（约 1 分钟）
```

覆盖内容：Python/依赖版本、**CUDA 可用性**（`src/utils.py:32` 的 `device` 在 import 时定型，
import 前就必须可见 CUDA）、**GPU 上真的能算**（跑一次复数 `matmul` + `linalg.eig` 并 `synchronize`，
这是唯一能识破"装了不含本机 sm 的 wheel"的检查，详见 §15）、关键文件在位、产物目录可写、
**参数量 == 41761**、**RMSPE 缩放因子 == π/180**、**`nominal=True` 不再崩且 MUSIC 真能出数**、
`create_dataset` 的二元组结构与张量形状、**失配真的进入数据**、**各失配水平共享同一批 DoA**、
**`root_music` 的批量实现与逐样本原始实现等价**（回归保护，见 §16.6）。
退出码 0 = 全部通过。自检也顺手充当了"论文数字 vs 本仓库行为"的回归测试——
如果你改了源码导致参数量或 RMSPE 口径变了，它会立刻报出来。

输出：`$DATA_ROOT/simulations/results/array_mismatch_{scenario}_{时间戳}.json`，每行同时给出三个 RMSPE 口径
（`rmspe_ref` 对齐论文 / `rmspe_deg` = ref×57.2958 / `rmspe_fixed` 物理正确）与**估计失败率** `fail_rate`
（Root-MUSIC 返回根数不足 M 而需要补随机猜测的比例——论文曲线在高失配处的"地板"就来自这里）。
每行还记录了 `train_at` 与 `matched`，便于回头检查这条曲线是不是按论文口径跑的。

---

## 8. 实验矩阵与消融设计（给出可信结论的推荐组合）

| 编号 | 场景 | 训练用失配 | 评估网格 | 回答的问题 |
|---|---|---|---|---|
| E1 | spacing | **逐点匹配**（每个水平各训一个，脚本默认 `--train_levels matched`） | 0.025/0.05/0.10/0.15 | 主结果，对齐 Fig. 9(a) |
| E2 | sv_noise | **逐点匹配** | 0/0.25/0.5/0.75 | 主结果，对齐 Fig. 9(b) |
| E3 | spacing | $\eta=0$（理想阵列），`--train_levels single --train_at 0.0` | 0.025/0.05/0.10/0.15 | **关键消融**：不做失配数据增强能否泛化？论文强调"learn from data to handle miscalibration"，E3 是其直接证据 |
| E4 | spacing | $\eta=0.15$（最差情形），`--train_at 0.15` | 同上 | 训练在最坏情形的代价（轻度失配下是否反而变差） |
| E5 | sv_noise | $\sigma_{sv}^2=0.75$，`--train_at 0.75` | 同上 | 同上 |

> E3–E5 都属于 `--train_levels single`，衡量的是"用**一个**失配水平训练的模型能否泛化到整个网格"，与 E1/E2 的口径**不同**，报告里必须分开画、并标注 `matched=false`（结果 JSON 的每行都带这个字段）。

另外三个**低成本必做项**：

1. **`bias=0` 校验**：故意跑一次 `bias=0.05`，看曲线整体是否被压低。若压低明显，说明你的主实验必须坚持 `bias=0`，并在文档中写明"已关闭参考实现的历史默认偏置"。
2. **`M_hat` vs `M`**：评估阶段按论文固定 $\hat M=M$；再补一组"用特征值间隙估 $\hat M$"的结果，可呼应 Sec. IV-C-1 的可解释性结论（本文档不展开）。
3. **参数对照（已离线验证过，可直接当自检用）**：按论文 Sec. IV-A-2 的架构逐层手算，在 $\tau=8$、AReLU 使通道数翻倍的前提下：

   | 层 | 输入通道 | 参数量 |
   |---|---|---|
   | conv1 | $\tau=8$ | $(8\cdot4+1)\times16 = 528$ |
   | conv2 | $16\times2=32$（AReLU 翻倍） | $(32\cdot4+1)\times32 = 4128$ |
   | conv3 | $32\times2=64$ | $(64\cdot4+1)\times64 = 16448$ |
   | deconv2 | 64 | $(64\cdot4+1)\times32 = 16416$ |
   | deconv3 | $32\times2=64$ | $(64\cdot4+1)\times16 = 4112$ |
   | deconv4 | $16\times2=32$ | $(32\cdot4+1)\times1 = 129$ |
   | **合计** | | **41761** ✅ |

   > 层名以代码为准（`src/models.py:296-301`）：SubspaceNet 用的是 `conv1/conv2/conv3` + `deconv2/deconv3/deconv4`，**没有 `deconv1`**（论文正文说的"3 层 DCNN"对应代码里的 deconv2/3/4）。注意别和同文件里另一个模型 `DeepRootMUSIC`（`src/models.py:196-201`，有 `deconv1`、且 `conv2` 是 `Conv2d(16,32,2)`）混淆，两者行号相邻但层定义不同。

   **手算结果与论文的 41,761 完全一致**（已核对）。这条自检很有价值，因为它一次性验证了三件事：
   ① 通道数 16/32/64 → 32/16/1 与 2×2 核的描述正确；② AReLU 的通道翻倍语义正确；③ **$\tau=8$ 确实是论文所用的滞后阶数**。
   若你在服务器上 `sum(p.numel() for p in model.parameters())` 得到的不是 41761，说明 `tau` 或通道数被改动过（常见错误是忘掉 AReLU 的翻倍），**先修再跑**。

   > **顺带解决一个论文明的空白**：论文 Sec. IV-A-2 只说"the lag $\tau$ ... **is a hyper-parameter (set via manual tuning in our experiments)**"，**全篇没有给出 $\tau$ 的具体数值**。但既然只有 $\tau=8$ 能复现出 41,761 这个数，论文所用的就是 $\tau=8$ —— 这也与本仓库所有历史权重文件名（`..._tau=8_...`）一致。所以 `--tau 8` / `main.py` 里的 `tau=8` 是有据可依的，不是我们猜的。

---

## 9. 远端服务器部署与运行（本机不执行实验）

### 9.1 同步代码（本机 PowerShell）

```powershell
# 首次：整仓上传（排除本地数据与虚拟环境）
scp -r D:\SubspaceNet user@server:/home/user/SubspaceNet

# 之后：只同步改动过的源码与脚本
scp D:\SubspaceNet\reproduce_array_mismatch.py user@server:/home/user/SubspaceNet/
scp -r D:\SubspaceNet\src user@server:/home/user/SubspaceNet/
```

> 注意：本仓库现有的 `data/datasets/*`（含 130 MB 的 Generic 数据集）与 `data/weights/*` 都是**本机历史产物**，不要上传，服务器会自己生成。

### 9.2 服务器环境（Linux，conda 或 uv）

```bash
# 方案 A：conda（推荐，仓库自带 pyEnv/SubspaceNetEnv.yaml）
conda env create -f pyEnv/SubspaceNetEnv.yaml && conda activate SubspaceNetEnv

# 方案 B：uv
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r pyEnv/requirements.txt
# GPU 版 torch（仓库 requirements.txt 锁定的是 CPU/旧版，按服务器 CUDA 版本装）
uv pip install torch --index-url https://download.pytorch.org/whl/cu121   # 按实际 CUDA 调整
```

验证（务必先跑，`src/utils.py:32` 的 `device` 是模块级常量，**import 时就定型**）：

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

必须打印 `True`。若为 `False`，说明装成了 CPU 版 torch，训练会慢 50–100 倍。

### 9.3 运行（建议 tmux，全程几小时）

```bash
cd /home/user/SubspaceNet
tmux new -s subn
python reproduce_array_mismatch.py all --scenario spacing --smoke   # 先冒烟
python reproduce_array_mismatch.py all --scenario spacing > log_spacing.txt 2>&1
# Ctrl-B D 脱离；tmux attach -t subn 回来
```

### 9.4 取回结果

```powershell
scp -r user@server:/home/user/SubspaceNet/data/simulations/results/array_mismatch_*.json D:\SubspaceNet\data\simulations\results\
```

---

## 10. 画图：对齐 Fig. 9 的形式

> **本节讲"图应该长什么样"；具体怎么一键出图（PNG）见 §14。**

- **横轴**：场景一为 $\eta$（0.025→0.15），场景二为 $\sigma_{sv}^2$（0→0.75），线性坐标
- **纵轴**：RMSPE，**建议线性坐标**（Fig. 9 不像是对数轴）。若要压缩动态范围可用 dB：$10\log_{10}(\text{MSPE})$
- **曲线**：每个失配水平上画两条 —— 经典方法（经验协方差）与 SubspaceNet 增强，同色不同 marker
- **图例**：SubspaceNet 用论文的缩写 `SubNet`（Sec. IV-A-2 明说）
- **理想做法**：同时对 `r-music` 与 `esprit` 画两张子图；`music` 网格 0.01° 需要 18000 次导向矢量计算（`src/methods.py:246`），在 5000 样本上非常慢，**先别把它放进主网格扫描**
- 复现预期形状（供你判断是否跑通）：
  - 场景一：经典方法从 $\eta=0.025$ 起就开始劣化，$\eta=0.15$ 时误差约为理想情形的数倍；SubNet 曲线基本平坦
  - 场景二：经典方法随 $\sigma_{sv}^2$ 增大**急剧**崩坏，SubspaceNet 的**相对优势随污染程度单调扩大**（论文 Sec. IV-B-4 结尾的原话）

---

### 10.5 关于"Appendix"：这篇论文没有 Appendix，但 Sec. IV-C 是必读的

#### 10.5.1 先澄清一件事

**2025 年的 IEEE TSP 版本没有 Appendix**。我核对了从 PDF 提取的全文，章节结构是：

```
I. INTRODUCTION
II. SYSTEM MODEL AND PRELIMINARIES   (A/B/C)
III. SUBSPACENET                      (A. High-Level Rationale / B. Architecture
                                       / C. Training / D. Inference / E. Discussion)
IV. NUMERICAL EVALUATION              (A. Experimental Setup / B. DoA Recovery Performance
                                       / C. Interpretability)
V. CONCLUSION
REFERENCES                            ← 直接结束，无附录
```

即所有超参数（$\mu=0.001$、$\epsilon=1$、41761 参数、45,000 样本等）都写在 **Sec. IV-A-2** 正文里，没有一个"Appendix 补充材料"可以查。如果你手头某个版本提到 Appendix，那多半是 arXiv 早期版本或会议版的编号差异——**复现时以本文档 §1/§5 汇总的 Sec. IV-A-2 数值为准**。

#### 10.5.2 真正应该替代"找 Appendix"去读的：Sec. IV-C Interpretability

失配实验只给了 RMSPE 曲线，而**RMSPE 曲线本身无法证明 SubspaceNet 学到了正确的子空间**——它可能只是过拟合出一个"碰巧算出接近角度"的矩阵。Sec. IV-C 提供的正是这个**内部正确性判据**，强烈建议复现时一并做，成本很低：

| Sec. IV-C 小节 | 图 | 复现做法 | 判据（做对了应该看到） |
|---|---|---|---|
| 1) Subspace Separation | Fig. 10 | 对同一批观测分别算：经验协方差、SubspaceNet 协方差的**归一化特征值** | SubspaceNet 的 $M$ 个主特征值与噪声特征值**清晰分离**；经验协方差在相干源下只能分出 1 个、SPS 只能分出 2 个 |
| 2) Spectrum Presentation | Fig. 11–13 | 画 MUSIC 谱 + Root-MUSIC 单位圆极点图 | SubspaceNet+Root-MUSIC 的**非 DoA 根被推离单位圆**（论文举例：某非 DoA 根被推到 $|\text{root}|>8$），从而可分辨 |
| 3) Beamforming Pattern | Fig. 14 | 用 SubspaceNet 协方差跑 MVDR 波束图 | 主瓣对准真实 DoA、旁瓣被压低；宽带场景下经验协方差的波束图会严重偏离真实 DoA |

论文里给出的**固定 DoA 测试点**（可直接借来画谱，比随机 DoA 更利于和论文图对照）：

- Fig. 11：$M=3$ 窄带相干源，$\theta=[-12.34°, 34.56°, 65.78°]$，$T=100$，SNR=10 dB
- Fig. 12：$M=2$ 窄带相干源，$\theta=[23.45°, 56.78°]$（与 Table II 同设置，少快拍+中等 SNR）
- Fig. 13：$M=3$ 宽带相干源，$\theta=[-45.67°, -23.45°]$，$T=50$

> 说明一下这些点的正确用法：它们是**相干源**场景（违反 AS2），不是失配场景。所以**不要**拿它们当失配实验的数据点；正确的用法是——在你的失配模型训练好之后，把它们喂进去当"体检"：
> **如果 SubspaceNet+Root-MUSIC 在这些相干源上仍能把非 DoA 根推离单位圆，说明模型学到的是真正的子空间结构；如果谱完全混乱，那即使 §7 的 RMSPE 曲线"看起来像 Fig. 9"，也应视为复现失败。**

#### 10.5.3 与本仓库的对应关系

- 特征值分离：`SubspaceMethod.calculate_covariance(X, mode, model)` 已支持三种协方差来源（`sample` / `spatial_smoothing` / `SubspaceNet`，见 `src/methods.py`），**同一份数据可以一次算出三者对比**，这就是 Fig. 10 的现成实现路径
- MUSIC 谱：`src/methods.py:246` 的 18000 点网格本来就会返回整条谱（Root-MUSIC 则返回 `roots` 与 `roots_angels_all` 两个中间量），画谱不需要额外改动
- MVDR：`src/methods.py:641` 附近的 MVDR 实现同样支持 `model=` 参数，可直接换成 SubspaceNet 协方差

---

## 11. 复现检查清单

**跑之前**

- [ ] `bias` 显式设为 `0`（不是留空依赖默认值 0.05）
- [ ] `M=2`、`signal_nature="non-coherent"`、`N=8`、`SNR=10`
- [ ] `set_diff_method("root_music")`，`tau=8`
- [ ] `sv_noise_var` 与 `eta` **不要同时非零**（否则两个失配源耦合，无法归因）
- [ ] 每个失配水平独立数据目录，避免文件名冲撞
- [ ] `torch.cuda.is_available()` 为 `True`

**跑之后**

- [ ] RMSPE 口径已声明（参考实现 / 乘 180/π / 度域修正），曲线与论文形状一致
- [ ] 报告了每个点的估计失败率（根数不足 M 的比例）
- [ ] 训练/测试失配水平记录在案（文件名与 JSON 都有）
- [ ] 参数总量 41,761 复核通过
- [ ] 附上"训练用失配水平"的消融（E3 最关键）
- [ ] **做了 §10.5.2 的特征值分离体检**（见 §12 第 4 条；这是"曲线对"和"模型对"的分水岭）
- [ ] 确认没有把论文的 45,000 样本 / 混合 M 流程误当成 SubspaceNet 的配置（§1.4）
- [ ] 若要用 MUSIC 基线，确认已打上 §13.1 的 `uniform_bias` 补丁（否则一跑就崩；**本仓库已修**）
- [ ] 训练/评估协议已声明：是逐点匹配（默认，对齐论文）还是单一模型（E3 消融），见 §13.3

**环境自检（1–2 分钟，建议正式跑之前先过一遍）**

- [ ] 用极小数据跑通一次并得到四行结果：`--n_train 40 --n_test 20 --epochs 1 --batch_size 8 --limit 20`（§13.5 有预期形状）
- [ ] `python preflight_check.py` 全绿（13 项：依赖/CUDA/参数量/RMSPE 口径/nominal 导向矢量/MUSIC 出数/数据管线/失配注入/DoA 可比性）
- [ ] 需要连训练路径一起验时加 `--full`（约 1 分钟）

**出图（PNG，见 §14）**

- [ ] 曲线图两种口径都出：`plot --metrics deg ref`（`ref` 对齐论文数字，`deg` 是物理角度）
- [ ] 体检图已出齐：每个失配水平一张 `eigsep_*.png`（对应 Fig. 10）、一张 MUSIC 谱、一张 Root-MUSIC 根图
- [ ] 已**目视检查**体检图：SubspaceNet 预测协方差的噪声特征值不应比经验协方差高出一两个数量级（否则模型没学到噪声子空间，见 §14.4）
- [ ] 谱图**不止看一张**（8 阵元 ULA 在 θ≈0 附近分辨力最差，单张谱有偶然性），用换 `--capture_index` 多采几个样本
- [ ] 记住两个仓库绘图缺陷：真值红 × 只画一个（`src/plotting.py:137`）、半径 >1.2 的根被裁掉（`set_ylim`）——别把它当成"根没被推离单位圆"的证据

---

## 12. 一句话总结

复现这个实验的真正难点**不在模型，而在三个隐性约定**：
(1) 代码里的 `eta` 是"相对半波长的绝对偏移"，论文正文说成百分比，最大偏离标称间距 30%；
(2) `bias` 是历史遗留的共享位置偏置，必须置 0；
(3) 评估用的 `RMSPE` 有度/弧度缩放 bug，论文所有数字被压缩了 57.3 倍。
把这三点固定下来，再用 §4.2 的"各失配水平共享同一随机种子"保证曲线可比，Fig. 9 的形状是可以复现出来的。

**还有三条容易被忽略的**：

4. **别只看 RMSPE 曲线就宣布成功**。论文没有 Appendix，但 Sec. IV-C 的**特征值分离**（Fig. 10）才是"模型真的学到了子空间"的证据，成本很低，务必做（§10.5）；脚本已能一键出这套体检图（**§14**）。
5. **训练配置别照搬论文那句 45,000 样本**：那句话描述的是**所有数据驱动方法（含 CNN、DA-MUSIC）共用的数据集**，且代码 `SystemModelParams.M` 是单值、不支持混合 M 训练；失配实验固定 M=2，用一个 M=2 的数据集直接训即可（§1.4）。
6. **式(16) 本身是对的**——$1/\sqrt M$ 归一化与置换不变性代码都实现正确，唯一的错是度/弧度。所以 57.3 倍这个结论是干净的，不是多个 bug 的叠加结果（§3.1）。

**最后，两个会影响结论的工程决定**（§13）：

7. **仓库的经典 MUSIC 基线原本是坏的**：`src/system_model.py` 的 `nominal=True` 分支漏给 `uniform_bias` 赋值，导致 MUSIC/MVDR 一调用就抛 `UnboundLocalError`。**已修**（一行）。不修的话，Fig. 9 里那条 MUSIC 曲线根本产不出来。
8. **"训练时用多大失配"是个必须显式选择的协议**：论文没把它写成独立变量。忠实复现 Fig. 9 要用**逐点匹配**（每个 η 各训一个模型，脚本默认），代价是训练时间成倍；"训一个再跨网格评估"是鲁棒性消融，可以报告，但不能和论文曲线混画。这个选择对最终曲线的解释**影响极大**，务必在报告里写清楚。

---

## 13. 实现层面：本仓库的一个真实 bug 与已做的本地修改

这一节记录**实测发现的代码缺陷**以及为跑通实验对仓库做的改动。它与失配机制本身无关，但会直接决定"经典基线能不能跑"，所以务必知悉。

### 13.1 `steering_vec(nominal=True)` 崩溃（已修）

`src/system_model.py:141-193` 的 `steering_vec(..., nominal=True)` 用于取"理想（无失配）导向矢量"。`nominal=True` 时走 `:175-176` 分支，原代码只赋值了两个变量：

```python
else:
    mis_distance, mis_geometry_noise = 0, 0      # ← 漏了 uniform_bias
```

而 `:184` 无条件使用三者之和：

```python
* (uniform_bias + mis_distance + self.dist[self.params.signal_type])
```

于是 `nominal=True` 必然抛：

```
UnboundLocalError: local variable 'uniform_bias' referenced before assignment
```

**受影响范围**（实测确认）：

| 调用点 | 结果 |
|---|---|
| `src/methods.py:270`（MUSIC 的 `spectrum_calculation`，传 `nominal=True`） | ❌ 崩 |
| `src/methods.py:641`（MVDR，传 `nominal=True`） | ❌ 崩 |
| Root-MUSIC / ESPRIT（走求根路径，不调用 nominal 网格） | ✅ 正常 |

也就是说：**在修之前，经典 MUSIC 基线开箱即崩**。而论文 Fig. 9 的经典基线里就有 MUSIC，所以不修就跑不出对照曲线。

**已做的修改**（用户确认后应用，一行）：

```python
# src/system_model.py:175-176
else:
    uniform_bias, mis_distance, mis_geometry_noise = 0, 0, 0
```

修改后实测：`nominal=True` 的导向矢量恒为理想 ULA 形式（模全为 1），且**与 `bias`/`eta`/`sv_noise_var` 取值无关**——正是"标称阵列流形"的定义，语义正确。

> 若你不想改仓库源码：`reproduce_array_mismatch.py` 内的 `enable_nominal_steering_without_repo_edit()` 提供了等价的用户态补丁，且会自动检测源码是否已修好而跳过。两种方式选其一即可。
> **当前状态：仓库源码已经修好**（§13.5），所以脚本运行时会直接跳过补丁——日志里出现 `[fix]` 那一行说明源码还没修。

### 13.2 MVDR 不是 DoA 估计器（无需修改，但别误用）

`src/methods.py:582-658` 的 `MVDR` 继承自 `MUSIC`，其 `narrowband` 末尾是：

```python
predictions = None
return predictions, response_curve     # src/methods.py:657-658
```

即它**只返回波束形成响应曲线，不返回 DoA 估计**。论文用 MVDR 只是为了画 Fig. 14 的波束图（§10.5.2），它**不能**作为 Fig. 9 的 RMSPE 基线。看到的 `nan` 是"拿了 `None` 去算指标"造成的，不是数值 bug：numpy 1.x 下 `np.asarray(None, dtype=float)` 会**静默**变成 `array([nan])`，numpy 2.x 则抛 `TypeError`，两种都不易察觉，脚本里用 `as_pred_array()` 统一显式处理。

### 13.3 训练/评估协议：逐点匹配（已改脚本默认）

论文没有把"训练时用多大失配"写成独立变量，只说被比较的方法训练数据相同。结合 Sec. IV-B 的组织方式，忠实复现 Fig. 9 应采用**逐点匹配**：网格上每个 $\eta$（或 $\sigma^2_{\mathrm{sv}}$）都用**同一水平训练**的模型去评估。

因此脚本默认 `--train_levels matched`：在 `{0.025, 0.05, 0.10, 0.15}` 上各训一个模型。这会**成倍增加训练时间**（4 个网格点 = 4 次训练），但它是唯一与论文曲线读法一致的口径。

之前"只训一个模型（如 $\eta=0.10$）再在整个网格上评估"的做法，衡量的是**训练/评估失配不匹配时的鲁棒性**，属于额外消融（§8 的 E3）。它仍然有价值，但**不要与论文曲线混画**，脚本用 `--train_levels single --train_at <值>` 触发，并在输出 JSON 里用 `matched` 字段标记每条曲线的口径。

### 13.4 其他实测结论（用于排除嫌疑，不必再查）

- **参数量 41,761 与论文完全一致**（`ModelGenerator` + `SubspaceNet(tau=8)` 实测），间接证实 $\tau=8$ 与论文架构逐层吻合（§5）
- **各失配水平共享同一批 DoA** 已验证成立（固定 `set_unified_seed(1234)` 后，$\eta=0$ 与 $\eta=0.15$ 抽到完全相同的 DoA），曲线可比性成立
- **失配确实进入观测数据**（$\eta=0$ 与 $\eta=0.15$ 的 `mean|X|` 不同）
- **`train_model` 在真实 API 下可正常训练**（2 epoch 实测 loss 下降），训练/验证切分、scheduler、checkpoint 均工作
- 训练/评估的**设备处理**沿用官方约定：`src/training.py:371` 与 `src/evaluation.py:80` 都显式 `.to(device)`。自行写评估循环时必须同样处理，否则会撞 `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`
- `create_dataset` 返回 `(model_dataset, generic_dataset, samples_model)`；前两者元素为 `(X, Y)` **二元组**（与 `src/training.py:373` 的 `Rx, DOA = data` 一致）。若你自行组装数据集，别把三元组塞进 DataLoader
- `numpy 2.x` 下 `np.asarray(None, dtype=float)` 会抛 `TypeError`（1.x 静默变 `nan`）。脚本用 `as_pred_array()` 统一处理，自行写指标代码时注意同样问题

### 13.5 端到端验证结果（已完成，可作为环境自检基线）

上机实测跑通了完整链路（极小数据量配置：`--n_train 40 --n_test 20 --epochs 1 --batch_size 8 --limit 20`，逐点匹配，4 个 η 各训一个模型）。下表取自结果 JSON 的 `*_deg` 字段（单位：度）：

| η | SubNet+r-music | r-music | SubNet+esprit | esprit | SubNet+music | music |
|---|---|---|---|---|---|---|
| 0.025 | 25.28 | **4.07** | 25.34 | 5.02 | 26.47 | 8.55 |
| 0.05 | 26.64 | 13.18 | 26.44 | 10.47 | 27.95 | 12.26 |
| 0.10 | 33.08 | 21.92 | 30.14 | 25.31 | 31.47 | 24.58 |
| 0.15 | 39.68 | 26.48 | 38.22 | 28.32 | 38.59 | 29.17 |

四个失配水平上 `fail_rate` 全为 **0.0**（Root-MUSIC 每次都取到 M 个根），说明这些数字不是"补随机猜测"造成的地板值。

**怎么读这张表**：它证明的是**链路通**，不是复现成功。经典基线随 η 单调劣化（4.07° → 26.48°）正是论文描述的物理趋势；而 SubspaceNet 在这里反而更差，纯粹因为这是 40 样本 / 1 epoch 的冒烟配置——模型根本没学到东西。**不要把这个结果当作复现结论**，正式跑必须回到 `--n_train 45000 --epochs 80` 起步。

顺带确认了三件实现细节：`SubspaceNet.forward(batch=1)` 输出 `Rz` 形状 `(1, 8, 8)`、`dtype=torch.complex64`；三种算法的返回元数分别为 RootMUSIC=5、Esprit=2、MUSIC=3（`MUSIC.narrowband` 返回 `(predictions, spectrum, M)`）；`_deg` 与 `_ref` 两列的比值在每行都是 57.2958（例：η=0.15 的 r-music：`0.4622 × 57.2958 = 26.48`），再次印证 §3 的口径结论。

### 13.6 出图链路也已实测跑通（§14 的实现验证）

同一批冒烟数据上加 `--capture --grid 0.025,0.15`，exit 0，产出 **14 张 PNG**（4 个失配水平 × {MUSIC 谱, Root-MUSIC 根图, 特征值分离图} + 2 张曲线图），落在 `data/simulations/figures/`，逐水平的谱/根/特征值原始数据落在 `data/simulations/spectra/*.npz`。

出图**没有扰动数值**：加图后各水平的 `*_deg` 与上表逐格一致。目视核验了四类图（`fig9_spacing_deg.png` / `spectrum_*_music.png` / `spectrum_*_r-music.png` / `eigsep_*.png`），确认：曲线图 6 条线、线型与图例正确、标题无中文方框（matplotlib 默认字体没有中文字形，所以图内文字一律用英文）；MUSIC 谱峰值归一化到 1.0、横轴为度；Root-MUSIC 根图在 `_polar_degree_shim` 生效后预测/真值标记位置正确；特征值分离图为 semilogy、signal|noise 分界线在 `M+0.5` 处。

另外两个在加图过程中修掉的真实缺陷，与失配机制无关但会影响出图，记录在此：
- `--capture` 存 `params` 时若直接 `np.savez(dict)`，读回会撞 `ValueError: Object arrays cannot be loaded when allow_pickle=False`。**注意 `np.load` 是惰性的**——错误要到你真正取那个数组时才抛，所以 `try/except` 包住 `np.load` 本身没用，必须把数组全部取出来。现在存的是 JSON 字符串，并且 `_load_capture` 会为旧 npz 回退到 `allow_pickle=True`。
- `RootMUSIC.narrowband` 返回的 `roots`（`src/methods.py:475-486`）是**全部 N 个根**（按到单位圆的距离排序），而 `doa_predictions` 只取单位圆内的前 M 个（`roots_inside = [root for root in roots if ((abs(root)-1) < 0)][:M]`）。画根图若按 `roots[:len(predictions)]` 取，会拿到半径 >1 的根。

> 自检小抄：`python preflight_check.py` 全绿（13 项，见 §7.1），或 `python reproduce_array_mismatch.py all --scenario spacing --n_train 40 --n_test 20 --epochs 1 --batch_size 8 --limit 20` 能在 1–2 分钟内跑出上面这种形状的四行输出，说明**环境、依赖、CUDA、数据管线、三算法、指标口径**全部就绪，可以放心上全量了。

---

## 14. 出图（PNG）：曲线图 + 体检图

脚本能出四类图，**全部是 PNG**，默认落在 `<产物根目录>/simulations/figures/`（`--fig_dir` 可改）。

### 14.1 四类图分别是什么

| 图 | 文件名 | 对应论文 | 怎么产生 |
|---|---|---|---|
| **RMSPE 曲线图** | `fig9_{scenario}_{metric}.png` | Fig. 9(a)/(b) | `plot` 子命令，读结果 JSON |
| **MUSIC 谱图** | `spectrum_{scenario}_{level}_music.png` | Fig. 11–13（谱那一半） | 评估时 `--capture` 抓数据 → 自动出图 |
| **Root-MUSIC 根图** | `spectrum_{scenario}_{level}_r-music.png` | Fig. 11–13（单位圆根那一半） | 同上 |
| **特征值分离图** | `eigsep_{scenario}_{level}.png` | **Fig. 10** | 同上 |

`{level}` 是失配水平标签（`eta=0.025` → `eta0p025`，与训练集文件名同风格）。
曲线图给两种口径（`--metrics` 可改）：`deg`（×180/π 的物理角度）与 `ref`（仓库原始 RMSPE，保留 §3 的度/弧度 bug）。

### 14.2 命令

```bash
# 只要曲线图（从已有的结果 JSON 重画，不重跑实验）
python reproduce_array_mismatch.py plot \
    --json data/simulations/results/array_mismatch_spacing_20261009_195849.json \
    --metrics deg ref

# 评估时顺带抓谱/根/协方差，并出体检图 + 曲线图
python reproduce_array_mismatch.py all --scenario spacing \
    --capture --capture_index 0 \
    --algorithms r-music esprit music

# 只要数字、不要图
python reproduce_array_mismatch.py all --scenario spacing --no_plot
```

`--capture_index` 决定"用测试集里第几个样本出体检图"（默认 0，`-1` 表示最后一个）。想多看几个样本就换值重跑 `plot`。

### 14.3 两个必须知道的坑（都已在本脚本里绕开）

**(1) 仓库的 Root-MUSIC 根图把角度当弧度用。** `src/plotting.py:164-171` 写的是：

```python
ax.plot([0, angle * np.pi / 180], [0, r])       # 预测值：传度、自己转弧度 -> 投影再×180/π -> 落回度，正确
ax.plot([doa * np.pi / 180], [1], marker='x')   # 真值：也自己转了弧度 -> 被投影再乘一次 180/π -> 刻度变成 θ·π/180
```

matplotlib 极坐标投影的 `Theta` **本身就按弧度解释**，所以第二行多转了一次。实测：真值 `[-0.969°, 0.384°]` 在图上分别落在 **-0.97°** 和 **约 0°**——**两个真实 DoA 都塌在 0° 附近，图上位置不可信**。

脚本**不改仓库源码**，而是用 `_polar_degree_shim` 在用户态包一层 Axes，把角度统一按度传入再换算。修后实测：预测 `[-0.9775, 23.40]` 的橙点落在约 23°、真值红 × 落在近 0°，位置正确。

附带两个仓库缺陷（未修，看图时注意）：
- `src/plotting.py:137` 的 `for doa in true_DOA[0]` 对一维数组只取第一个元素，所以**真值红 × 只画一个**；
- `set_ylim([0,1.2])` + `set_yticks([0,1])` 会把半径 >1.2 的根裁掉——而论文 Fig. 11–13 想展示的恰恰是"非 DoA 根被推离单位圆"（论文举过 `|root| > 8` 的例子），画面上看不到。

**(2) 仓库绘图函数把输出路径硬编码成 PDF。** `src/plotting.py:174` 写死 `data/spectrums/{algorithm}_spectrum.pdf`。脚本用 `_savefig_redirect` 上下文管理器临时替换 `plt.savefig` 来改写到 `--fig_dir` 的 PNG（注意它必须 `pop` 掉 `format` 与 `bbox_inches`，否则会撞重复关键字）。

### 14.4 怎么用这些图判断"是不是真复现了"

`fig9_*.png` 只能说明**角度误差**这个结果，说明不了模型学到了子空间。判据在体检图里（对应 §10.5.2）：

- **特征值分离图**：SubspaceNet 预测协方差的噪声特征值应当**贴近经验协方差**的量级。反例（本机 40 样本 / 1 epoch 冒烟）：经验协方差 `[1054, 428, 1.34, 1.05, 0.97, 0.95, 0.72, 0.58]`（M=2 分离干净），SubspaceNet 预测 `[6340, 677, 191, 103, 26, 13.5, 3.0, 1.16]`——**噪声特征值高出一两个数量级**，说明模型没学到噪声子空间。这种模型即使 RMSPE 看着还行，也是错的。
- **MUSIC 谱 / Root-MUSIC 根图**：谱峰应落在真实 DoA 上、非 DoA 根应被推离单位圆。注意**单张谱有偶然性**：8 阵元半波长 ULA 在 θ≈0 附近分辨力最差，实测 `eta=0.15` 时谱上只剩一个宽钝峰。报告里请多采几个样本（换 `--capture_index`），并用谱峰/根的位置做判据，而不是只看一张图。

### 14.5 已实测跑通（本机冒烟基线）

```bash
python reproduce_array_mismatch.py all --scenario spacing \
    --n_train 40 --n_test 20 --epochs 1 --batch_size 8 --limit 20 \
    --capture --grid 0.025,0.15
```

exit 0，产出 14 张 PNG（4 个失配水平 × 3 类体检图 + 2 张曲线图），且各水平的 `*_deg` 数值与 §13.5 表完全一致——说明加图**没有改动数值链路**。

---

## 15. 服务器 GPU 是 Blackwell（sm_120）时的 torch 版本问题

### 15.1 症状与根因

在 RTX PRO 6000 Blackwell（compute capability 12.0）服务器上跑 `preflight_check.py`，会看到这样的组合：

```
  [PASS] CUDA 可用
           NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GiB, compute capability (12, 0)
  [FAIL] MUSIC.narrowband 能真正出数（不只是不崩）
           RuntimeError: CUDA error: no kernel image is available for execution on the device
```

同时前面还有一条容易被划过去的警告：

```
NVIDIA RTX PRO 6000 Blackwell Server Edition with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
```

**根因**：仓库 `pyEnv/requirements.txt:23` 锁的是 `torch==2.0.1`。这个版本的官方 wheel **只编译到 sm_86**（`torch.cuda.get_arch_list()` 里最高是 `sm_86`/`compute_37`），而 Blackwell 是 **sm_120**。wheel 里没有 sm_120 的 SASS，且 CUDA 12.8 起已不再为 sm_120 做老 PTX 的 JIT 兼容，于是**任何真实计算**都抛 `no kernel image is available`。

**为什么前面 6 项还能 PASS**：`torch.cuda.is_available()` 只问"驱动看得见卡吗"，**它不检查这个构建里有没有对应架构的 kernel**。所以 `is_available()` 返回 `True`、`get_device_name()` 也能拿到名字，但第一次真正算东西就炸。

**为什么错误看起来"位置随机"**：CUDA 的 kernel 错误是**异步**上报的。真正失败的可能是好几行之前的某个算子，直到某次同步才抛出来。所以你在 `[3]` MUSIC 那里看到的报错，不代表 MUSIC 有问题——**它只是第一个触发同步的地方**。

> 实测旁证：这条报错本身与环境无关。本机 RTX 4060 Laptop（sm_89）跑 `torch 2.0.1+cu118`（同样没有 sm_89 SASS）却一切正常，因为它能靠 wheel 里的 **PTX JIT** 回退编译。Blackwell 上这条回退路径断了，于是暴露出来。这也说明"能不能跑"必须实测，不能只看 arch 列表。

### 15.2 解决：换装带 sm_120 的 torch（PyTorch ≥ 2.7 的 cu128 wheel）

Blackwell 支持从 **PyTorch 2.7** 起进入官方稳定 wheel（cu128 构建）。推荐 **2.7.x / 2.8.x**（不要用太新的版本，见 §15.5 的兼容性提醒）。

**先确认驱动够新**（cu128 wheel 需要驱动支持 CUDA 12.8，即 Linux ≥ 570）：

```bash
nvidia-smi        # 看右上角 "CUDA Version: 12.x"，>= 12.8 即可
```

**方案 A（推荐，不改动现有环境，风险最低）**——新建一个环境：

```bash
conda create -n subn_sm120 python=3.10 -y
conda activate subn_sm120

# 1) torch 和它的 CUDA 运行时一起装（会自动带 nvidia-* 依赖，别单独装 cu117 那套）
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 2) 其余依赖（注意 requirements.txt 里锁了 torch==2.0.1，**不要直接 pip install -r**，
#    否则会把刚装好的 torch 降级回去）
pip install numpy==1.24.3 scipy==1.10.1 matplotlib==3.7.1 scikit-learn==1.2.2 \
            tqdm==4.65.0 h5py pandas seaborn
```

**方案 B（就地升级，省空间，但要先记录当前版本以便回滚）**：

```bash
conda activate SubspaceNet_original
pip freeze > ~/pip_freeze_before_sm120.txt          # ★回滚依据，务必先做
pip install --upgrade torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

> 若 pip 因为"已安装 torch 2.0.1"而不肯换，用
> `pip install --force-reinstall --no-deps torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128`
> 再补齐 `nvidia-*` 依赖（它们会随 torch 一起装，`--no-deps` 会漏掉，所以更推荐不加 `--no-deps`）。

**方案 C（内网/无法直连 pypi 的场景）**：在能联网的机器上
`pip download torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128 -d ./wheels`
（大约 800 MB~1 GB，含 `nvidia-*` 依赖），scp 上去后 `pip install --no-index --find-links ./wheels torch==2.7.0`。

**不要尝试的三条路**（都会浪费时间）：

1. 在服务器上装 `cudatoolkit=11.7/11.8`——问题不在 CUDA 工具链，在 wheel 里缺 sm_120 kernel；
2. 用 `CUDA_LAUNCH_BLOCKING=1` 让它"跑起来"——它只是让报错同步，不产生缺失的 kernel；
3. 改 `src/utils.py:32` 强制 CPU——全量 10000 样本 / 80 epoch 在 CPU 上是不可接受的。

### 15.3 验证（换完必须做，两条命令）

```bash
# ① 一行确认 kernel 真的能跑（关键是 synchronize，否则错误会异步溜走）
python -c "import torch; d='cuda'; a=torch.randn(64,64,dtype=torch.complex64,device=d); \
b=torch.randn(64,64,dtype=torch.complex64,device=d); (a@b).sum().real.item(); \
torch.linalg.eig(a+a.conj().T); torch.cuda.synchronize(); \
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"

# ② 跑完整自检（这一步会真的训 2 个 epoch，是最终判据）
python preflight_check.py --full
```

`①` 应当打印出 `2.7.0+cu128` 和一张含 `sm_120` 的 arch 列表；`②` 应当 `FAIL=0` 且 exit 0。
**只要 `②` 里有任何一项 FAIL，就先别上全量**——正式跑是几小时，自检是 1 分钟。

### 15.4 preflight_check.py 已增强（本机 v2）

原先的自检漏掉了这个问题（它只查 `is_available()`），现在 `[1] 依赖与 CUDA` 里改成三项、并且**第一个真正的 kernel 测试放在最前面**：

| 项 | 判据 | 失败时会说什么 |
|---|---|---|
| `cuda.is_available()` | 驱动能看见卡 | 提示 `src/utils.py:32` 的 `device` 在 import 时定型 |
| **`GPU 上真的能算（复数 matmul + linalg.eig，带同步）`** | **跑真实 kernel + `torch.cuda.synchronize()`** | 直接说出"本机是 sm_xxx，这个构建只编译了 [...]，请换 cu128 wheel"，并指向本节 |
| `（提示）GPU 架构是否在 torch 编译目标里` | 仅提示，不算失败 | 说明"缺 SASS 可能靠 PTX JIT 跑起来"，并指明唯一判据是上一项 |

这么改的用意：把这类错误**从"跑到第 3 项才炸、且报错位置随机"提前到第 1 组、并且直接给出结论**。第 1 组还会先做复数 `matmul` 与 `linalg.eig`——这正是 SubspaceNet 推理最依赖、也最容易撞 sm 问题的两个算子。

### 15.5 换 torch 版本的兼容性提醒（本项目特有）

torch 从 2.0.1 跳到 2.7/2.8 跨了 7 个 minor，跑之前留意这几点：

- **`torch.load` 的默认值变了**：torch 2.6 起 `weights_only` 默认为 `True`。仓库用 `state_dict` 级别的保存/加载不需要改；但如果你自己 `torch.save` 过**自定义对象**（不是 `state_dict`），加载会报 `UnpicklingError`，那时显式传 `weights_only=False`。
- **`root_music(Rz, M, batch_size)` 是显存/算力热点**（`src/models.py:744`，其两个批量辅助函数在 `src/models.py:693` 与 `src/models.py:720`）：它对 batch 内每个 8×8 复矩阵做 `torch.linalg.eig` 并反传。换版本后如果遇到 `linalg.eig` 的 backward 报错或变慢，先确认 `--batch_size` 不是罪魁；论文设置里评估是逐样本（batch=1），训练时才用大 batch。该函数已做批量向量化，详见 §16。
- **复现口径不受影响**：参数量 41761、RMSPE 的 π/180 缩放、`nominal=True` 补丁、`bias=0` 这些都与 torch 版本无关。`preflight_check.py` 里的第 [2][3] 组会在换版本后**重新替你验一遍**，所以"换了 torch 会不会偏离论文"这个问题不用自己推理——跑一次自检即可。
- **不要顺手升 numpy/python**：仓库锁 `numpy==1.24.3`，脚本里 `as_pred_array()` 就是为了兼容 numpy 1.x 对 `None` 的静默行为（见 §13.4）。换 torch 时保持其余版本不动，能把变量控制到最少。

---

## 16. 训练速度：热点定位与三处逐样本循环的消除

这一节只讲**性能**，不改任何复现口径。如果你觉得训练太慢（例如一个 epoch 要几十分钟），先读这里再动手。
**最快的自查方式**是在服务器上跑 `python bench_train.py`，它会逐段计时并与本机参考值并列，一眼看出是哪一段慢了几倍（§16.6）。

### 16.1 热点在 `root_music`，占了前向的 90% 以上

把 `SubspaceNet.forward` 拆成三段实测（T=100、N=8、τ=8，RTX 4060 Laptop）：

| batch | backbone | gram | `root_music` | `root_music` 占比 |
|---|---|---|---|---|
| 1 | 0.81 ms | 0.16 ms | 2.71 ms | 73.6% |
| 32 | 0.77 ms | 3.81 ms | 43.6 ms | 90.5% |
| 1024 | 5.28 ms | 123.9 ms | 1565.7 ms | **92.4%** |

**CNN 主干只占 0.3%**——所以换更大网络、调 batch、加 `num_workers` 都不会有实质改善，唯一的杠杆是 `root_music` 自己。

原因：`root_music` 原本是**逐样本的 Python 循环**（`for iter in range(batch_size)`），每个样本都独立启动一整套 kernel（EVD → argsort → gather → 矩阵乘 → 15 次对角线求和 → 多项式求根 → 排序取根）。batch=1024 就是 1024 轮。单样本的算子合计只有 0.889 ms，而整段实测 1.44 ms/sample——**多出来的 0.55 ms 全是解释器与 kernel 启动开销**。

### 16.2 顺带发现的 device 陷阱：多项式求根被静默丢到 CPU

`src/utils.py:147` 的 `find_roots_torch`：

```python
A = torch.diag(torch.ones(len(coefficients) - 2, dtype=coefficients.dtype), -1)   # ← 没给 device
```

伴随矩阵建在 CPU 上。往这个 CPU 矩阵里赋值 CUDA 张量时 PyTorch **不报错、而是静默拷回 CPU**，于是 `torch.linalg.eigvals` 每个样本都在 CPU 上跑一次，并返回 CPU 张量——**每个样本一次 GPU↔CPU 往返**。

这里有个反直觉的结论，我实测过：

| batch | 原样（求根落 CPU） | 只把 device 补上（求根上 GPU） |
|---|---|---|
| 256 | 1255 ms/step | **2072 ms/step**（更慢） |
| 1024 | 5177 ms/step | **8475 ms/step**（更慢） |

因为 14×14 矩阵的 `eigvals` 在 CPU 上本来就快，搬到 GPU 反而多了 kernel 启动与每步同步。**所以千万不要单独给那一行加 `.to(device)`**——那会让训练更慢。正确做法是连同循环一起批量化（下面），批量化之后求根需要多快有多快（一次算 256 个样本的 8×8 特征分解只要 0.369 ms）。

这个陷阱也是"服务器比我本机还慢"的原因：本机 GPU 慢，CPU 往返被计算掩住了；高端卡上 GPU 早早算完，只能干等 CPU 往返，比例反而放大。

### 16.3 改动一：`root_music` 批量向量化（`src/models.py:698-800`）

三个函数：

| 函数 | 位置 | 作用 |
|---|---|---|
| `sum_of_diags_batched(matrix)` | `src/models.py:698` | `(B,N,N) → (B,2N-1)`，一次算完所有对角线（原版是 15 次独立 kernel） |
| `find_roots_batched(coefficients)` | `src/models.py:725` | `(B,L) → (B,L-1)`，伴随矩阵**显式建在系数所在设备**上 |
| `root_music(Rz, M, batch_size)` | `src/models.py:749` | 全 batch 向量化：一次 EVD + gather + 噪声子空间投影 + **根的选择也用张量完成** |

**为什么结果不变**：`F = U_n U_n^H` 对 `U_n` 的每一列尺度不变，所以批量 EVD 与逐样本 EVD 在特征向量归一化上的差异不影响 `F`；列顺序由 `argsort` 保证一致。返回值三元组的语义也保持一致（第三个仍是"最后一个样本的根"，与原实现的既有行为相同）。

**根的选择**（最后一步）是最容易漏掉的一处：原实现是 `for i in range(batch_size)` 逐样本切片，看起来"不涉及算子所以不要紧"，实际上它每样本一次 `.item()`/索引操作，在 batch 1024 时让这一步独占 **172 ms**。现在用 `torch.where(outside, _SORT_SENTINEL, distance_to_circle)` 把所有"圆外的根"推到排序末尾，再 `argsort` + 切片，整段无 Python 循环。

### 16.4 改动二：验证集 DataLoader 的 batch（`src/training.py:284-286`）

原代码把验证集 batch 写死为 1，4500 个验证样本就是 4500 次逐样本前向。改为 `batch_size=self.batch_size`。**数值不变**：`src/evaluation.py:117-118` 是 `overall_loss += eval_loss.item()` 再除以按样本数累加的 `test_length`，与 batch 无关。
`reproduce_array_mismatch.py:672-675` 里也有同样的一处（脚本会覆盖 `tparams.valid_dataset`），已一并改为 `BATCH_SIZE`。

### 16.5 实测提速（RTX 4060 Laptop，T=100/N=8/τ=8）

`root_music` 本身：

| batch | 原始 ms | 批量 ms | 加速 | samples/s |
|---|---|---|---|---|
| 8 | 11.5 | 2.9 | 4.0× | 693 → 2739 |
| 32 | 41.4 | 7.4 | 5.6× | 774 → 4318 |
| 256 | 337.3 | 54.2 | 6.2× | 759 → 4726 |
| 1024 | 1485.1 | 223.0 | 6.7× | 690 → 4592 |

上面的"批量 ms"是**只做循环批量化**的结果；再把最后那个逐样本的根选择也向量化（§16.3 末段）之后：

| batch | 原始 ms | 批量+根选择向量化 ms | 加速 |
|---|---|---|---|
| 32 | 52.4 | 3.2 | 16.5× |
| 256 | 373.3 | 14.0 | 26.7× |
| 1024 | 1559.7 | 49.3 | **31.7×** |

整步训练（forward + RMSPELoss + backward + Adam）：

| batch | 原始 | 批量 | 再向量化根选择 |
|---|---|---|---|
| 256 | 1255 ms/step（204 samples/s） | 497 ms/step（516 samples/s） | **353 ms/step（725 samples/s）** |
| 1024 | 5177 ms/step（198 samples/s） | 1900 ms/step（539 samples/s） | **1287 ms/step（796 samples/s）** |

峰值显存不变（batch 256 时 75 MiB、batch 1024 时 244 MiB）。换算到论文规模（40500 训练 + 4500 验证），**每 epoch 约 61–64 秒**（batch 512 与 1024 差别很小，见 §16.6），80 epoch 单模型约 **1.4 小时**。

**batch 怎么选**：从 2 到 1024 的实测看，batch ≥ 512 之后每样本成本已经平台化（0.205 → 0.202 ms/样本），batch=2 则要 6.3 ms/样本（**31 倍**）。所以**不要用小 batch 训练**——`--smoke` 的 512 是合适的，全量用 1024。

### 16.6 怎么验证它没改坏数值 / 怎么定位服务器上的慢

仓库根目录的 `verify_root_music_batch.py` 内联保留了改写前的**逐样本原始实现**，在同一批输入上逐项对比：

```bash
python verify_root_music_batch.py           # 等价性 + 提速
python verify_root_music_batch.py --quick   # 只做等价性
```

它检查三件事并给出退出码（0 = 通过）：

- **前向**：M 个 doa 逐样本一致、全部根的 doa **集合**一致（容差 1e-4 rad）。实测最大偏差 **1.6e-6 rad**。
  （只比较集合是因为 `doa_all_batches` 由全部 2N-2 个根导出，批量 EVD 与逐样本 EVD 的**根排列**可能不同；它只被 `src/evaluation.py:121-129` 的 `plot_spec=True` 分支用于画谱图，不进入任何指标。）
- **反向**：对 `Rz` 的梯度相对偏差（容差 1e-2）。实测 **1.3e-5**。
- **提速**：上面两张表。

**定位"训练慢"该跑哪个**：用 `bench_train.py`，它逐段计时并给出本机参考值，一眼就能看出是哪一段慢了几倍：

```bash
python bench_train.py                       # --smoke 的配置 (batch=512)
python bench_train.py --batch 2 256 1024    # 对比不同 batch 的每样本成本
python bench_train.py --device cpu          # 排除"其实跑在 CPU 上"
```

本机（4060）参考输出：

```
数据生成  2000 样本 2.02 s  (1.008 ms/样本)
batch=2     forward   3.8 ms/step   step   12.6 ms/step   吞吐 158.5 样本/s
batch=512   forward 105.1 ms/step   step  772.3 ms/step   吞吐 662.8 样本/s
batch=1024  forward 206.6 ms/step   step 1468.9 ms/step   吞吐 696.9 样本/s
```

判读方法：如果 `forward ÷ step`、或者每样本成本与上表的比例关系明显不对（例如 batch=512 的 step 超过 3 秒，或 batch=2 与 batch=512 的每样本成本差不多），说明瓶颈**不在** `root_music`，而在别处（数据搬运、验证集、或落到了 CPU）——把这三种 batch 的完整输出发回来即可定位。

改动梯度路径意味着**最终 RMSPE 会有微小偏移**，所以换用这个版本后请重跑一次 `--smoke --force_data`，对照 §13.5 的表确认量级没变（η=0.025 时 SubNet+r-music ≈ 25.28°、r-music ≈ 4.07°）。已生成的数据集与权重格式不变，可以继续复用。

### 16.7 改动三：`gram_diagonal_overload` 也是逐样本循环（`src/utils.py:247-279`）

这一处最容易被忽略：它**不在** `src/models.py` 里，而在 `src/utils.py`，从 `SubspaceNet.forward` 的最后一步调用。原实现：

```python
for iter in range(batch_size):
    K = bs_kx[iter]
    Kx_garm = torch.matmul(torch.t(torch.conj(K)), K).to(device)   # 每样本一次 matmul
    eps_addition = (eps * torch.diag(torch.ones(Kx_garm.shape[0]))).to(device)
    Rz = Kx_garm + eps_addition
    Kx_list.append(Rz)
```

batch=512 就是 512 次 `conj`+`transpose`+`matmul`+`diag`+`matmul`+`stack`，每次都 `.to(device)`。现在：

```python
Kx_gram = torch.matmul(Kx.conj().transpose(-2, -1), Kx).to(device)
eye = torch.eye(Kx_gram.shape[-1], device=device, dtype=Kx_gram.dtype)
return Kx_gram + eps * eye
```

**注意这里是 `K^H K`（共轭转置在左）**，不是 `K K^H`，改写时别顺手"纠正"。实测：batch 512 时 **57.21 ms → 0.093 ms（612 倍）**，完整 forward **96.5 ms → 28.9 ms（3.3 倍）**。等价性见 §16.6 的 `verify_batched_ops.py`（batch 1/8/512 最大差 3.8e-6）。

**教训（值得记）**：这个仓库的"逐样本 Python 循环"不止一处，而且分布在不同文件里。批量化的顺序应该是**从 profiler 的头号算子往下查**，而不是从自己以为的热点开始——`root_music` 看起来最可疑（又是求根又是 EVD），但 `gram_diagonal_overload` 才是第一个该动的（它只占 2 个 matmul，容易被当成"已经很便宜了"）。用 `python profile_forward.py` 一次就能看到。

**顺带修掉的一个死代码 bug**：`gram_diagonal_overload` 的签名要求 `batch_size`，但 `DeepRootMUSIC.forward`（`src/models.py:255`，改写前）调用它时**没传**，且同一个 `forward` 还在读一个从未赋值的 `self.M`（`src/models.py:258`）。也就是说 **`DeepRootMUSIC` 这个模型从来没能跑过一次**——两次调用都会抛异常。已修：`batch_size` 改为可选（默认从 `Kx.shape[0]` 推断），`DeepRootMUSIC.__init__` 增加 `M: int = 2` 参数。它与本复现无关（`ModelGenerator` 构造不出这个模型），但既然在同一行上，就一并修掉并记在这里。

### 16.8 还没做的项（按性价比排序）

1. **`src/utils.py:128` `find_roots_torch`**：伴随矩阵建在 CPU 上（§16.2 的陷阱）。现在只有 `esprit` 分支（`src/models.py:818`）还在用它，`root_music` 已经改用 `find_roots_batched`。若你要用 `--diff_method esprit` 训练，需要同样处理。
2. **`torch.linalg.eig` → `torch.linalg.eigh`**：`F = U_n U_n^H` 一定是 Hermitian，而实测 `eigh` 比 `eig` 快 **33 倍**（batch 512：7.95 ms → 0.24 ms）。但没有直接替换，原因写在下面。
3. 两个评估用 DataLoader 加 `num_workers=4, pin_memory=True`（`reproduce_array_mismatch.py:710-711`）。注意数据是**内存里的张量列表**，不是磁盘 I/O，这项收益有限。
4. `src/training.py:437` 每个 best epoch 都会 `copy.deepcopy(model.state_dict())`；改成存盘 + 结束回读可省一份拷贝峰值。模型只有 0.17 MB，收益很小。

> **为什么没有把 `eig` 换成 `eigh`**：`root_music` 现在用 `torch.linalg.eig(Rz)` 对**非 Hermitian** 的预测协方差做特征分解，再按 `|λ|` 排序取噪声子空间。换成 `eigh` 需要先把 `Rz` 对称化，而 `eigh` 在**特征值简并**时会返回简并子空间内的任意正交基——此时 `U_n` 会整体旋转，`F` 随之改变（随机复矩阵上实测 `F` 差了 1.6）。理论上只要简并集合**整个**落在噪声子空间、且不与信号特征值简并，`U_n` 张成的子空间就唯一、`F` 不变，DoA 也就相同；但实测 `|λ₁|-|λ₂|` 的最小值只有 4.6e-5（即确实会碰到近似简并），所以这不是纯理论问题。要换需要先用真实模型输出验证 DoA 逐样本一致，**目前没做，也没有必要**——`eig` 只占 forward 的 4.2 ms / 28.9 ms。

### 16.9 完整的提速账（RTX 4060 Laptop，batch 512）

| 阶段 | 原始 | 现在 | 倍数 |
|---|---|---|---|
| `gram_diagonal_overload` | 57.21 ms | 0.093 ms | 612× |
| `root_music` | ≈ 1670 ms（按 batch 512 外推） | 24.1 ms | ≈ 69× |
| 完整 forward | 96.5 ms | 28.9 ms | 3.3× |
| **整步训练** | 1112 ms/step | **535.6 ms/step** | **2.1×** |
| 吞吐 | 663 样本/s | **956 样本/s** | 1.4× |
| 推算论文规模每 epoch | ≈ 64 s | **≈ 44 s** | 1.5× |

（`root_music` 的"原始"列取自 `verify_batched_ops.py` 里按 16 样本外推的值；`gram` 与整步来自本机实测。）

到这一步，**剩下的大头是反向传播本身**：forward 只占 29 ms，而整步要 535 ms。所以再想提速得换思路（混合精度、或者减少 `root_music` 反向的代价），不是继续找循环了。


