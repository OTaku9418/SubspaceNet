"""SubspaceNet 的残差臂：两条加法恒等跳连（严格 ResNet），零新增参数。

为什么单独一个文件
------------------
基线 `src/models.py: SubspaceNet.forward` 一个字都不动，本文件独立实现同一张图的
"带跳连"版本。两个前向各自成文，`skip_scale=0` 的逐位等价测试
（`tests/test_residual_arch.py`）于是变成**跨文件**对照：任何一侧漂移都会让它失败。
如果两臂共用同一段尾段代码，这条测试就只能证明"自己等于自己"。

结构约定（改动前先读）
----------------------
* `__init__` 完全继承 `SubspaceNet`：层形状不变 => state_dict 键与基类逐字相同
  （12 键）、可训练参数量仍是 41761（tau=8, M=2, N=8，论文 Sec. IV-A-2 同值）。
* 只加两处 `+ x_skip`，位置恰在两个同形接缝：
      anti_rectifier(deconv2) 与 anti_rectifier(conv2) 都是 (B, 64, 14, 6)
      anti_rectifier(deconv3) 与 anti_rectifier(conv1) 都是 (B, 32, 15, 7)
* `skip_scale` 是**不参与训练**的普通浮点属性（不在 state_dict 里）：默认 1.0；
  置 0.0 时本类输出必须与 `SubspaceNet` 逐位相同，用于等价性测试与 preflight 自检。
* 尾段（view -> Kx_tag -> Rz -> diff_method）是**故意重复**的，不要改成 import 共用：
  见上面"为什么单独一个文件"。

`ModelGenerator.set_model` 里对 `SubspaceNetResidual` 的分支是懒加载（函数内 import），
避免 `src.models` 与 `src.models_residual` 循环导入。
"""

from __future__ import annotations

import torch

from src.models import SubspaceNet
from src.utils import gram_diagonal_overload


class SubspaceNetResidual(SubspaceNet):
    """SubspaceNet + 两条加法恒等跳连（严格 ResNet，零新增参数）。

    `y = F(x) + x`，其中 F 是两个同形接缝之间的那段反卷积块；
    跳连系数 `skip_scale` 默认 1.0，置 0.0 时输出与 `SubspaceNet` 逐位相同。
    """

    skip_scale = 1.0

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the residual SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        s = float(self.skip_scale)
        ## Architecture flow (与 SubspaceNet.forward 逐字相同, 只多了两处 + s * x_skip) ##
        # CNN block #1
        x1 = self.anti_rectifier(self.conv1(Rx_tau))            # (B, 32, 15, 7)
        # CNN block #2
        x2 = self.anti_rectifier(self.conv2(x1))                # (B, 64, 14, 6)
        # CNN block #3
        x3 = self.anti_rectifier(self.conv3(x2))                # (B, 128, 13, 5)
        # DCNN block #1 + skip #1: 与 conv2 的输出同形
        y2 = self.anti_rectifier(self.deconv2(x3)) + s * x2     # (B, 64, 14, 6)
        # DCNN block #2 + skip #2: 与 conv1 的输出同形
        y1 = self.anti_rectifier(self.deconv3(y2)) + s * x1     # (B, 32, 15, 7)
        # DCNN block #3
        Rx = self.deconv4(self.DropOut(y1))                     # (B, 1, 16, 8)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to the differentiable subspace algorithm
        method_output = self.diff_method(Rz, self.M, self.batch_size)
        if isinstance(method_output, tuple):
            # Root MUSIC output
            doa_prediction, doa_all_predictions, roots = method_output
        else:
            # Esprit output
            doa_prediction = method_output
            doa_all_predictions, roots = None, None
        return doa_prediction, doa_all_predictions, roots, Rz
