"""定位 SubspaceNet 前向里到底哪个算子慢 —— 把 forward 拆到单个算子级别。

用途
----
当 `bench_train.py` 报告"每样本成本与 batch 无关"时（逐样本串行的特征），
用本脚本找出**具体是哪一行**。它会:

  1. 打印每个中间张量的 device/dtype/shape —— 抓"某一步被静默挪到 CPU";
  2. 把 root_music 里的每一步单独计时 —— 抓"批量算子在某个版本/架构上退化";
  3. 对 eigh / eig / svd 做微基准 —— 对比同一个矩阵规模下本来该多快;
  4. 用 torch.profiler 列出最慢的 CUDA 算子（前向 + 反向各一次）;
  5. 可选 `--cpu` 把 eig/eigvals 强制放到 CPU 上跑一次做对照。

本机 (RTX 4060 Laptop, torch 2.0.1+cu118) 参考值, batch=512:
    backbone            2.3 ms
    gram                0.1 ms
    linalg.eig (8x8)    4.2 ms
    sum_of_diags        0.4 ms
    find_roots_batched 22.4 ms      <- 里面是 linalg.eigvals(14x14)
    root_music 合计    24.1 ms
    完整 forward       28.9 ms

用法
----
    python probe_forward_detail.py                    # 默认 batch 512
    python probe_forward_detail.py --batch 2 512      # 对比
    python probe_forward_detail.py --cpu-comparison   # 额外做 CPU 对照
"""

import argparse
import platform
import sys
import time

import torch


def _early_device():
    """Resolve ``--device`` before anything imports ``src.utils.device``.

    ``src.utils.device`` is a module-level constant (`cuda:0` whenever CUDA is visible) and
    ``src.models`` binds it at import time, so a script cannot switch the device afterwards.
    Parsing the single flag from ``sys.argv`` here and patching the constant right away keeps
    every project module on the requested device without touching repository code.

    ``CUDA_VISIBLE_DEVICES=3`` is the alternative and needs no flag at all: it renumbers the
    visible GPUs, so the hard-coded `cuda:0` lands on physical GPU 3.
    """

    import src.utils as project_utils

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
    if not spec or spec == "default":
        return project_utils.device
    chosen = project_utils.resolve_device(spec)
    project_utils.device = chosen
    if chosen.type == "cuda" and chosen.index:
        torch.cuda.set_device(chosen.index)
    return chosen


DEVICE = _early_device()


def timeit(fn, repeats=5, warmup=2):
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


def describe():
    print("=== 环境 ===")
    print(f"  python      {sys.version.split()[0]} ({platform.platform()})")
    print(f"  torch       {torch.__version__}")
    print(f"  cuda        runtime={torch.version.cuda} 可用={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  gpu         {props.name}, {props.total_memory / 1024**3:.1f} GiB, "
              f"cc {torch.cuda.get_device_capability(0)}, SMs={props.multi_processor_count}")
    try:
        print(f"  threads     torch.get_num_threads()={torch.get_num_threads()}")
    except Exception:
        pass
    print(f"  device      {DEVICE}")
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


def trace_devices(model, batch_size):
    """逐步执行一次 forward, 打印每个中间张量的 device —— 抓静默的 CPU 回退。"""
    from src.models import find_roots_batched, sum_of_diags_batched
    from src.utils import gram_diagonal_overload

    print("=== 中间张量的 device / shape ===")
    print("  (生产路径用 eigh —— 替代协方差 K^H K + eps*I 是 Hermitian 的)")
    x = torch.randn(batch_size, 8, 16, 8, device=DEVICE)
    with torch.no_grad():
        rows = []
        h = model.anti_rectifier(model.conv1(x))
        rows.append(("conv1 + AReLU", h))
        h = model.anti_rectifier(model.conv2(h))
        rows.append(("conv2 + AReLU", h))
        h = model.anti_rectifier(model.conv3(h))
        rows.append(("conv3 + AReLU", h))
        h = model.anti_rectifier(model.deconv2(h))
        rows.append(("deconv2 + AReLU", h))
        h = model.anti_rectifier(model.deconv3(h))
        rows.append(("deconv3 + AReLU", h))
        h = model.DropOut(h)
        h = model.deconv4(h)
        rows.append(("deconv4", h))
        kx = h.view(batch_size, 16, 8)
        rows.append(("Kx view", kx))
        Rz = gram_diagonal_overload(kx, eps=1, batch_size=batch_size)
        rows.append(("Rz (gram)", Rz))
        eigenvalues, eigenvectors = torch.linalg.eigh(Rz)
        rows.append(("eigenvalues", eigenvalues))
        rows.append(("eigenvectors", eigenvectors))
        order = torch.argsort(eigenvalues, dim=1, descending=True)
        Un = torch.gather(eigenvectors, 2, order[:, 2:].unsqueeze(1).expand(-1, 8, -1))
        F = Un @ Un.conj().transpose(-2, -1)
        rows.append(("F = Un Un^H", F))
        diag_sum = sum_of_diags_batched(F)
        rows.append(("diag_sum", diag_sum))
        roots = find_roots_batched(diag_sum)
        rows.append(("roots", roots))

    for name, tensor in rows:
        flag = ""
        if isinstance(tensor, torch.Tensor) and tensor.device.type != DEVICE.type:
            flag = "   <<< 不在目标 device 上!"
        shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None
        dev = tensor.device if isinstance(tensor, torch.Tensor) else None
        dtype = tensor.dtype if isinstance(tensor, torch.Tensor) else None
        print(f"  {name:22s} {str(dev):10s} {str(dtype):16s} {str(shape):20s}{flag}")
    print()


def breakdown(model, batch_size, cpu_comparison=False, repeats=10):
    from src.models import find_roots_batched, root_music, sum_of_diags_batched
    from src.utils import gram_diagonal_overload

    def t(fn):
        return timeit(fn, repeats=repeats)

    x = torch.randn(batch_size, 8, 16, 8, device=DEVICE)
    with torch.no_grad():
        t_full = t(lambda: model(x))

        def backbone():
            h = model.anti_rectifier(model.conv1(x))
            h = model.anti_rectifier(model.conv2(h))
            h = model.anti_rectifier(model.conv3(h))
            h = model.anti_rectifier(model.deconv2(h))
            h = model.anti_rectifier(model.deconv3(h))
            h = model.DropOut(h)
            return model.deconv4(h)

        t_backbone = t(backbone)
        kx = backbone().view(batch_size, 16, 8)
        t_gram = t(lambda: gram_diagonal_overload(kx, eps=1, batch_size=batch_size))
        Rz = gram_diagonal_overload(kx, eps=1, batch_size=batch_size)
        Rz = Rz.detach()

        t_root = t(lambda: root_music(Rz, 2, batch_size))

        eigenvalues, eigenvectors = torch.linalg.eig(Rz)
        t_eig = t(lambda: torch.linalg.eig(Rz))
        t_eigh = t(lambda: torch.linalg.eigh(Rz))
        order = torch.argsort(torch.abs(eigenvalues), dim=1, descending=True)
        Un = torch.gather(eigenvectors, 2, order[:, 2:].unsqueeze(1).expand(-1, 8, -1))
        F = Un @ Un.conj().transpose(-2, -1)
        t_diag = t(lambda: sum_of_diags_batched(F))
        diag_sum = sum_of_diags_batched(F)
        t_roots = t(lambda: find_roots_batched(diag_sum))

        # 根的选择那一步（已向量化, 应当很便宜）
        roots = find_roots_batched(diag_sum)

        def select():
            distance = torch.abs(torch.abs(roots) - 1)
            outside = (torch.abs(roots) - 1) >= 0
            keys = torch.where(outside, torch.tensor(1e30, device=roots.device), distance)
            idx = torch.argsort(keys, dim=1)
            return torch.gather(roots, 1, idx)[:, :2]

        t_select = t(select)

    print(f"=== batch={batch_size} 逐段计时 (repeats={repeats}) ===")
    for name, seconds in [
        ("完整 forward", t_full),
        ("  backbone", t_backbone),
        ("  gram_diagonal_overload", t_gram),
        ("  root_music 合计", t_root),
        ("    linalg.eig (8x8)", t_eig),
        ("    linalg.eigh (8x8) 对照", t_eigh),
        ("    sum_of_diags_batched", t_diag),
        ("    find_roots_batched", t_roots),
        ("    根的选择(向量化后)", t_select),
    ]:
        print(f"  {name:28s} {seconds*1e3:10.2f} ms   {seconds/batch_size*1e6:9.2f} us/样本")
    print()

    # 大矩阵微基准: 同规模下 CUDA 该有多快
    print("  linalg 微基准 (batch=512):")
    for size, kind in ((8, "complex64"), (14, "complex64")):
        a = torch.randn(512, size, size, dtype=torch.complex64, device=DEVICE)
        t = timeit(lambda: torch.linalg.eigvals(a), repeats=3)
        print(f"    eigvals {size}x{size} {kind:10s} {t*1e3:9.2f} ms  "
              f"({t/512*1e6:8.1f} us/矩阵)")
        if size == 14:
            h = a + a.conj().transpose(-2, -1)
            t2 = timeit(lambda: torch.linalg.eigvalsh(h), repeats=3)
            print(f"    eigvalsh {size}x{size} (Hermitian) {t2*1e3:9.2f} ms  "
                  f"({t2/512*1e6:8.1f} us/矩阵)")
    print()

    if cpu_comparison:
        print("  CPU 对照 (同样 512 个 14x14, 放 CPU 上算):")
        a = torch.randn(512, 14, 14, dtype=torch.complex64, device="cpu")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.linalg.eigvals(a)
        t_cpu = time.perf_counter() - t0
        print(f"    CPU eigvals 14x14 x512: {t_cpu*1e3:9.2f} ms  ({t_cpu/512*1e6:8.1f} us/矩阵)")
        print()


def reference_pipeline(model, batch_size, repeats=10):
    """同一批数据上跑三档: 只修 gram / +eigh / 完整当前实现 —— 直接读出每一档的收益。

    前两档是"历史快照", 用当前代码 + 局部替换来复现, 只为了给出对比数字;
    真实的生产路径只有最后一档。
    """
    from src.models import find_roots_batched, sum_of_diags_batched, root_music
    from src.utils import gram_diagonal_overload

    x = torch.randn(batch_size, 8, 16, 8, device=DEVICE)
    with torch.no_grad():
        h = model.anti_rectifier(model.conv1(x))
        h = model.anti_rectifier(model.conv2(h))
        h = model.anti_rectifier(model.conv3(h))
        h = model.anti_rectifier(model.deconv2(h))
        h = model.anti_rectifier(model.deconv3(h))
        h = model.DropOut(h)
        kx = model.deconv4(h).view(batch_size, 16, 8)
        Rz = gram_diagonal_overload(kx, eps=1, batch_size=batch_size).detach()

        def eig_path():
            ev, evec = torch.linalg.eig(Rz)
            order = torch.argsort(torch.abs(ev), dim=1, descending=True)
            Un = torch.gather(evec, 2, order[:, 2:].unsqueeze(1).expand(-1, 8, -1))
            F = Un @ Un.conj().transpose(-2, -1)
            return find_roots_batched(sum_of_diags_batched(F))

        def eigh_path():
            ev, evec = torch.linalg.eigh(Rz)
            evec = torch.flip(evec, dims=[-1])
            Un = evec[:, :, 2:]
            F = Un @ Un.conj().transpose(-2, -1)
            return find_roots_batched(sum_of_diags_batched(F))

        t_eig = timeit(eig_path, repeats=repeats)
        t_eigh = timeit(eigh_path, repeats=repeats)
        t_now = timeit(lambda: root_music(Rz, 2, batch_size), repeats=repeats)

    print(f"=== batch={batch_size} root_music 三档对照 (repeats={repeats}) ===")
    print(f"  eig  路径 (改动前的分解方式)  {t_eig*1e3:8.2f} ms  "
          f"({t_eig/batch_size*1e6:7.2f} us/样本)")
    print(f"  eigh 路径                      {t_eigh*1e3:8.2f} ms  "
          f"({t_eigh/batch_size*1e6:7.2f} us/样本)   {t_eig/t_eigh:.1f}x")
    print(f"  当前生产实现 root_music        {t_now*1e3:8.2f} ms  "
          f"({t_now/batch_size*1e6:7.2f} us/样本)")
    print()


def profile_ops(model, batch_size, backward=False, top=18):
    from torch.profiler import ProfilerActivity, profile

    x = torch.randn(batch_size, 8, 16, 8, device=DEVICE)
    label = "前向 + 反向" if backward else "仅前向"
    print(f"=== 最慢算子 ({label}, batch={batch_size}, 3 次) ===")

    if backward:
        from src.criterions import RMSPELoss
        model.train()
        criterion = RMSPELoss()
        target = torch.randn(batch_size, 2, device=DEVICE) * 20

        def once():
            loss = criterion(model(x)[0], target)
            loss.backward()
            model.zero_grad()

    else:
        def once():
            with torch.no_grad():
                model(x)

    for _ in range(2):
        once()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            once()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    # torch 2.9+ 的 FunctionEventAvg 去掉了 cuda_time_total/device_time_total，只留
    # cpu_time_total 与 self_*_time_total；逐个候选名探测，避免整节输不出来。
    def time_of(event, device_side=True):
        names = ("device_time_total", "cuda_time_total", "self_device_time_total",
                 "self_cuda_time_total") if device_side else ("cpu_time_total",)
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                return value
        return 0.0

    events = sorted(prof.key_averages(), key=lambda e: time_of(e, DEVICE.type == "cuda"),
                    reverse=True)[:top]
    for event in events:
        cuda = time_of(event, True)
        cpu = getattr(event, "cpu_time_total", 0.0)
        print(f"  {event.key[:44]:46s} cuda {cuda/1e3:10.2f} ms  cpu {cpu/1e3:9.2f} ms  "
              f"calls={event.count}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, nargs="+", default=[512])
    parser.add_argument("--cpu-comparison", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=10,
                        help="每项计时的重复次数（默认 10；数值会随此处变化, 看比例不看绝对值）")
    parser.add_argument("--device", default=None,
                        help="default|cpu|cuda|cuda:2|2。在导入 src 之前生效，等价于设 "
                             "CUDA_VISIBLE_DEVICES，但不需要改环境。")
    args = parser.parse_args()

    describe()
    model = build_model()
    trace_devices(model, args.batch[-1])
    for batch_size in args.batch:
        breakdown(model, batch_size, cpu_comparison=args.cpu_comparison,
                  repeats=args.repeats)
        reference_pipeline(model, batch_size, repeats=args.repeats)
    if not args.no_profile:
        profile_ops(model, args.batch[-1], backward=False)
        profile_ops(model, args.batch[-1], backward=True)


if __name__ == "__main__":
    main()
