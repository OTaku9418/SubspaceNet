"""定位 SubspaceNet 前向里到底哪个算子慢。

用途
----
`bench_train.py` 已经确认: 服务器上 forward 的**每样本成本与 batch 无关**
(batch=2 时 49.4 ms/样本, batch=512 时 43.4 ms/样本) —— 这是"逐样本串行"或
"某个算子整体异常慢"的典型特征, 而不是显存/算力不足。

本脚本把 `SubspaceNet.forward` 按模块拆开计时, 再用 torch.profiler 列出最慢的
CUDA 算子, 直接指出是哪一步。

用法
----
    python profile_forward.py                # 默认 batch 512
    python profile_forward.py --batch 2 512  # 对比小/大 batch 的每样本成本

判读
----
每一行都会打印"本机 4060 参考值", 直接看倍数。重点看:
  * backbone 慢  -> 卷积问题 (cuDNN 缺失/fallback 到慢算法/张量很大)
  * root_music 慢 -> n/a
  * 每个模块的**每样本成本**是否随 batch 增大而下降; 若不下降, 说明该模块内有
    逐样本串行的东西
"""

import argparse
import platform
import sys
import time

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 本机 (RTX 4060 Laptop) 参考值: (模块, batch, 每样本 ms)
LOCAL_REFERENCE = {
    "backbone":   {512: 0.0046},
    "gram":       {512: 0.0003},
    "root_music": {512: 0.0471},
}


def describe():
    print("=== 环境 ===")
    print(f"  python      {sys.version.split()[0]} ({platform.platform()})")
    print(f"  torch       {torch.__version__}")
    print(f"  cuda        runtime={torch.version.cuda} 可用={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  gpu         {props.name}, {props.total_memory / 1024**3:.1f} GiB, "
              f"cc {torch.cuda.get_device_capability(0)}, SMs={props.multi_processor_count}")
    print(f"  cudnn       可用={torch.backends.cudnn.is_available()} "
          f"启用={torch.backends.cudnn.enabled} 版本={torch.backends.cudnn.version()}")
    try:
        print(f"  cudnn 基准   {torch.backends.cudnn.benchmark}")
    except Exception:
        pass
    print()


def build_model():
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
        ModelGenerator().set_model_type("SubspaceNet")
        .set_diff_method("root_music").set_tau(8).set_model(params)
    ).model
    print(f"参数量 = {sum(p.numel() for p in model.parameters() if p.requires_grad)} "
          f"(论文 41761)")
    return model.to(DEVICE).eval()


def timeit(fn, repeats=3, warmup=1):
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


def breakdown(model, batch_size):
    import numpy as np

    from src.models import find_roots_batched, root_music, sum_of_diags_batched
    from src.utils import gram_diagonal_overload

    x = torch.randn(batch_size, 8, 16, 8, device=DEVICE)
    with torch.no_grad():
        # 完整前向
        t_full = timeit(lambda: model(x))

        # 逐模块。用 forward 里同样的中间量, 但分开计时
        conv1, conv2, conv3 = model.conv1, model.conv2, model.conv3
        deconv2, deconv3, deconv4 = model.deconv2, model.deconv3, model.deconv4
        arelu = model.anti_rectifier

        def backbone():
            h = arelu(conv1(x))
            h = arelu(conv2(h))
            h = arelu(conv3(h))
            h = arelu(deconv2(h))
            h = arelu(deconv3(h))
            h = model.DropOut(h)
            return deconv4(h)

        t_backbone = timeit(backbone)

        out = backbone()
        kx = out.view(batch_size, 2 * 8, 8)

        def gram():
            return gram_diagonal_overload(Kx=kx, eps=1, batch_size=batch_size)

        t_gram = timeit(gram)
        Rz = gram()

        t_root = timeit(lambda: root_music(Rz.detach(), 2, batch_size))

        # root_music 内部
        eigenvalues, eigenvectors = torch.linalg.eig(Rz.detach())
        t_eig = timeit(lambda: torch.linalg.eig(Rz.detach()))
        order = torch.argsort(torch.abs(eigenvalues), dim=1, descending=True)
        Un = torch.gather(eigenvectors, 2, order[:, 2:].unsqueeze(1).expand(-1, 8, -1))
        F = Un @ Un.conj().transpose(-2, -1)
        diag_sum = sum_of_diags_batched(F)
        t_diag = timeit(lambda: sum_of_diags_batched(F))
        t_roots = timeit(lambda: find_roots_batched(diag_sum))

    print(f"=== batch={batch_size} ===")
    rows = [
        ("完整 forward", t_full),
        ("  backbone(3 conv + 3 deconv + AReLU)", t_backbone),
        ("  gram_diagonal_overload", t_gram),
        ("  root_music 合计", t_root),
        ("    linalg.eig (8x8 complex)", t_eig),
        ("    sum_of_diags_batched", t_diag),
        ("    find_roots_batched (14x14 eigvals)", t_roots),
    ]
    for name, seconds in rows:
        per_sample = seconds / batch_size * 1e3
        print(f"  {name:38s} {seconds*1e3:10.2f} ms   {per_sample:9.4f} ms/样本")
    print()

    # 与参考值的倍数 (只对能对上号的模块)
    print("  与 4060 参考值的倍数 (越大越不正常):")
    for key, name in (("backbone", "backbone"), ("gram", "gram"),
                      ("root_music", "root_music 合计")):
        ref = LOCAL_REFERENCE[key].get(batch_size)
        if ref is None:
            continue
        got = {"backbone": t_backbone, "gram": t_gram, "root_music": t_root}[key]
        got_per = got / batch_size * 1e3
        print(f"    {name:22s} 本机 {ref:.4f} -> 服务器 {got_per:.4f} ms/样本  "
              f"= {got_per/ref:8.1f}x")
    print()


def profile_ops(model, batch_size, top=15):
    from torch.profiler import ProfilerActivity, profile

    x = torch.randn(batch_size, 8, 16, 8, device=DEVICE)
    with torch.no_grad():
        for _ in range(2):
            model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                model(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

    print(f"=== 最慢的算子 (batch={batch_size}, 3 次前向合计) ===")
    key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"
    events = sorted(prof.key_averages(), key=lambda e: getattr(e, key), reverse=True)[:top]
    for event in events:
        cuda = getattr(event, "cuda_time_total", 0.0)
        cpu = getattr(event, "cpu_time_total", 0.0)
        print(f"  {event.key[:48]:50s} cuda {cuda/1e3:10.2f} ms  cpu {cpu/1e3:9.2f} ms  "
              f"calls={event.count}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, nargs="+", default=[512])
    parser.add_argument("--no-profile", action="store_true", help="跳过算子级 profile")
    args = parser.parse_args()

    describe()
    model = build_model()
    for batch_size in args.batch:
        breakdown(model, batch_size)
    if not args.no_profile:
        profile_ops(model, args.batch[-1])


if __name__ == "__main__":
    main()
