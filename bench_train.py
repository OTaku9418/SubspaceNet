"""训练性能拆解: 逐段计时, 并与本仓库在 RTX 4060 Laptop 上的参考值对照。

用途
----
判断"训练慢"到底慢在哪。它不做等价性检查(那是 `verify_root_music_batch.py` 的事),
只回答一个问题: **哪一段、比参考值慢多少倍**。

分段方式与 `train_model` 的真实流程一致:
    step      = 取一个 batch -> forward -> RMSPELoss -> backward -> Adam -> zero_grad
    validate  = 对验证集做一次 no_grad 前向 + 损失累加
    epoch     = 若干 step + 一次 validate + 数据取样/搬运

用法
----
    python bench_train.py                       # --smoke 的配置 (batch=512)
    python bench_train.py --batch 2             # 换成小 batch (检查每样本开销是否爆炸)
    python bench_train.py --batch 256 512 1024  # 多个 batch 一次跑完
    python bench_train.py --device cpu          # 对照 CPU

参考值 (RTX 4060 Laptop, torch 2.0.1+cu118, T=100, N=8, tau=8):
    数据生成    0.920 ms/样本
    batch=2     forward  2.3 ms/step    step    9.6 ms/step   吞吐  206.9 样本/s
    batch=512   forward 21.7 ms/step    step  488.8 ms/step   吞吐 1047.0 样本/s
    batch=1024  forward 41.0 ms/step    step  968.8 ms/step   吞吐 1056.5 样本/s
    推算论文规模 (40500 训练 + 4500 验证) 约 40.7 s/epoch

注意这是 **eigh + 批量 root_music + 批量 gram** 之后的数字；这些优化之前
batch=512 是 forward 96.5 ms / step 1112 ms。所以**先确认服务器上的代码包含
`git log --oneline -5` 里的那批性能提交**，再拿这里的数字对照。

如果某项比参考值慢 5 倍以上, 基本可以断定不是"卡慢", 而是那一段落到了 CPU、
或者被 JIT/同步拖住 —— 请把完整输出发回来。慢在 forward 里的话再跑
`python profile_forward.py` 与 `python probe_forward_detail.py`,
它们会指出具体是哪个模块/哪个 linalg 调用。
"""

import argparse
import platform
import sys
import time

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_TRAIN_DEFAULT = 2000
N_VALID_DEFAULT = 200


def describe_environment(device):
    print("=== 环境 ===")
    print(f"  python      {sys.version.split()[0]} ({platform.platform()})")
    print(f"  torch       {torch.__version__}")
    print(f"  cuda        runtime={torch.version.cuda} 可用={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  gpu         {props.name}, {props.total_memory / 1024**3:.1f} GiB, "
              f"compute capability {torch.cuda.get_device_capability(0)}")
        print(f"  arch list   {' / '.join(torch.cuda.get_arch_list())}")
    print(f"  bench 用的 device = {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("  !! 指定了 cuda 但不可用")
        sys.exit(1)
    print()


def build_model(device):
    from src.models import ModelGenerator
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed

    set_unified_seed(0)
    params = SystemModelParams()
    for name, value in dict(
        N=8, M=2, T=100, snr=10, eta=0.0, bias=0.0,
        signal_type="NarrowBand", signal_nature="non-coherent",
    ).items():
        params.set_parameter(name, value)
    model = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_diff_method("root_music")
        .set_tau(8)
        .set_model(params)
    ).model
    return model.to(device)


def measure(fn, repeats, warmup=1):
    for _ in range(warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def probe_data_generation(n_samples):
    """数据生成速度 (src/data_handler.py)。"""
    from src.data_handler import create_dataset
    from src.system_model import SystemModelParams
    from src.utils import set_unified_seed

    params = SystemModelParams()
    for name, value in dict(
        N=8, M=2, T=100, snr=10, eta=0.1, bias=0.0, sv_noise_var=0.0,
        signal_type="NarrowBand", signal_nature="non-coherent",
    ).items():
        params.set_parameter(name, value)
    set_unified_seed(42)
    start = time.perf_counter()
    dataset, _, _ = create_dataset(
        system_model_params=params, samples_size=n_samples,
        model_type="SubspaceNet", tau=8, save_datasets=False,
    )
    elapsed = time.perf_counter() - start
    per_sample = elapsed / max(len(dataset), 1) * 1e3
    print("=== 数据生成 ===")
    print(f"  {len(dataset)} 样本 {elapsed:.2f} s  ({per_sample:.3f} ms/样本)   "
          f"[参考: 0.9 ms/样本]")
    return dataset


def probe_train_step(model, batch_size, device):
    """一个完整训练步, 并单独报 forward 的耗时。"""
    from src.criterions import RMSPELoss

    criterion = RMSPELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, 8, 16, 8, device=device)
    target = torch.randn(batch_size, 2, device=device) * 20

    model.eval()
    with torch.no_grad():
        t_forward = measure(lambda: model(x), 5, warmup=2)

    model.train()

    def step():
        loss = criterion(model(x)[0], target)
        loss.backward()
        optimizer.step()
        model.zero_grad()

    t_step = measure(step, 5, warmup=2)

    # 纯数据搬运 (DataLoader -> device), 与计算解耦
    host = torch.randn(batch_size, 8, 16, 8)
    t_copy = measure(lambda: host.to(device), 20, warmup=3)

    print(f"=== batch={batch_size} ===")
    scale = 1.0
    print(f"  forward        {t_forward*1e3:9.1f} ms/step  ({t_forward/batch_size*1e3:7.3f} ms/样本)")
    print(f"  train step     {t_step*1e3:9.1f} ms/step  ({t_step/batch_size*1e3:7.3f} ms/样本)"
          f"   [含 fwd+bwd+Adam]")
    print(f"  数据搬运       {t_copy*1e3:9.1f} ms/batch ({t_copy/batch_size*1e3:7.3f} ms/样本)")
    print(f"  => 吞吐        {batch_size/ (t_step + t_copy):9.1f} 样本/s")
    return t_step, t_copy, scale


def probe_validation(model, batch_size, device, n_valid):
    """验证集一轮 (evaluate_dnn_model 的等价物)。"""
    from src.criterions import RMSPELoss

    criterion = RMSPELoss()
    model.eval()

    def validate():
        with torch.no_grad():
            for i in range(0, n_valid, batch_size):
                size = min(i + batch_size, n_valid) - i
                xx = torch.randn(size, 8, 16, 8, device=device)
                dd = torch.randn(size, 2, device=device) * 20
                criterion(model(xx)[0], dd)

    elapsed = measure(validate, 3, warmup=1)
    print(f"  验证 {n_valid} 样本 {elapsed*1e3:9.1f} ms/epoch  "
          f"({(1 if n_valid else 0)*elapsed:6.3f} s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, nargs="+", default=[512],
                        help="训练 batch 大小, 可给多个 (默认 512, 与 --smoke 一致)")
    parser.add_argument("--n-train", type=int, default=N_TRAIN_DEFAULT)
    parser.add_argument("--n-valid", type=int, default=N_VALID_DEFAULT)
    parser.add_argument("--device", default=None, help="默认沿用 src/utils.py:32 的 device")
    parser.add_argument("--skip-data", action="store_true", help="跳过数据生成计时")
    args = parser.parse_args()

    from src.utils import device as repo_device

    device = torch.device(args.device) if args.device else repo_device
    describe_environment(device)

    data_elapsed = 0.0
    if not args.skip_data:
        start = time.perf_counter()
        probe_data_generation(min(args.n_train, 2000))
        data_elapsed = time.perf_counter() - start
        print()

    model = build_model(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SubspaceNet 可训练参数量 = {n_trainable} (论文 41761)")
    print()

    for batch_size in args.batch:
        t_step, t_copy, _ = probe_train_step(model, batch_size, device)
        t_valid = probe_validation(model, batch_size, device, args.n_valid)
        n_steps = max(1, args.n_train // batch_size)
        epoch = n_steps * (t_step + t_copy) + t_valid
        print(f"  => 每 epoch 合计 (n_train={args.n_train}, "
              f"{n_steps} steps + 验证): {epoch:7.2f} s")
        print(f"     推算论文规模 (40500 训练 + 4500 验证, 同 batch): "
              f"{40500*(t_step+t_copy)/batch_size + 4500*t_valid/args.n_valid:7.1f} s/epoch")
        print()


if __name__ == "__main__":
    main()
