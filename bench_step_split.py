"""Timing harness for the training step: forward vs backward vs the Root-MUSIC root finder.

Why this exists
---------------
`bench_train.py` reports "forward" and "train step" (forward + backward + Adam), but not the
split between them explicitly, and it does not say how much of the forward is the root finder.
On a server whose `torch.linalg.eigvals` is very slow (see REPRODUCE_ARRAY_MISMATCH.md section
16.8) the interesting question is "how much would removing the root finder actually save?", and
that needs the four numbers this script prints:

    forward (no grad)   forward (with graph)   backward only   full step

Usage
-----
    python bench_step_split.py                     # batch 512, root_music
    python bench_step_split.py --batch 2 512 1024
    python bench_step_split.py --diff-method esprit
    python bench_step_split.py --device cpu

How to read it
--------------
*   If `forward (no grad)` is most of `full step`, the forward pass is the bottleneck and the
    per-module table below says where.
*   If `backward only` is most of `full step`, the bottleneck has moved to autograd and no
    amount of forward optimisation will help much. That is the expected situation after the
    batched Root-MUSIC work: on an RTX 4060 Laptop with batch 512 the forward is ~22 ms while
    the backward is ~550 ms.
*   The `root_music` sub-timings show the share of the forward that is the polynomial root
    finder (`torch.linalg.eigvals` on the 14x14 companion matrix). That single call is
    CPU-bound in cuSolver: it costs the same on CPU and on GPU, and the same for 8x8 and 14x14,
    so it cannot be fixed by batching, dtype or device. See section 16.8 of the document.
"""

import argparse
import sys
import time

import torch

from src.criterions import RMSPELoss
from src.models import ModelGenerator
from src.system_model import SystemModelParams
import src.utils as _project_utils
from src.utils import resolve_device


def _early_device():
    """Resolve ``--device`` before anything imports ``src.utils.device``.

    ``src.utils.device`` is a module-level constant (`cuda:0` whenever CUDA is visible) and
    ``src.models`` binds it at import time, so a script cannot change the device after the
    fact. Parsing the one flag from ``sys.argv`` here and patching the constant immediately
    keeps every downstream module on the requested device without touching repository code.

    Note that ``src.training`` / ``src.evaluation`` / ``src.models`` do
    ``from src.utils import device`` (a *value* binding), so they each hold their own copy:
    patching ``src.utils.device`` alone is not enough, every copy has to be rebound.
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
    if not spec or spec == "default":
        return _project_utils.device
    chosen = resolve_device(spec)
    _project_utils.device = chosen
    if chosen.type == "cuda" and chosen.index:
        torch.cuda.set_device(chosen.index)

    import src.criterions
    import src.evaluation
    import src.models
    import src.training

    for module in (src.criterions, src.evaluation, src.models, src.training):
        if hasattr(module, "device"):
            module.device = chosen
    return chosen


device = _early_device()

LOCAL_REFERENCE = {
    # ms per training step, RTX 4060 Laptop 8 GiB, torch 2.0.1+cu118, T=100 tau=8 N=8 M=2
    "root_music": {2: 13.7, 512: 514.4, 1024: 1011.2},
}



def build_model(n=8, m=2, t=100, tau=8, diff_method="root_music"):
    params = SystemModelParams()
    for key, value in dict(N=n, M=m, T=t, snr=10, eta=0.0, bias=0.0,
                           signal_type="NarrowBand",
                           signal_nature="non-coherent").items():
        params.set_parameter(key, value)
    model = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_diff_method(diff_method)
        .set_tau(tau)
        .set_model(params)
    ).model.to(device)
    return model, params


def timeit(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e3


def describe():
    print("=== 环境 ===")
    print(f"  python       {__import__('sys').version.split()[0]}")
    print(f"  torch        {torch.__version__}")
    print(f"  device       {device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"  gpu          {prop.name}, {prop.total_memory / 2**30:.1f} GiB, "
              f"cc {prop.major}.{prop.minor}")
        try:
            print(f"  arch list    {' / '.join(torch.cuda.get_arch_list())}")
        except Exception:  # noqa: BLE001
            pass
    print(f"  threads      torch.get_num_threads() = {torch.get_num_threads()}")
    print()


def split_timings(model, batch_size, tau=8, repeats=5):
    """forward(no_grad) / forward(+graph) / backward / full step."""
    model.train()
    criterion = RMSPELoss()
    x = torch.randn(batch_size, tau, 2 * 8, 8, device=device)
    target = torch.randn(batch_size, 2, device=device) * 20

    def forward_no_grad():
        with torch.no_grad():
            model(x)

    def forward_with_graph():
        model(x)

    def backward_only():
        out = model(x)
        loss = criterion(out[0], target)
        loss.backward()
        model.zero_grad()

    def full_step():
        out = model(x)
        loss = criterion(out[0], target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_fwd = timeit(forward_no_grad, repeats)
    t_graph = timeit(forward_with_graph, repeats)
    t_bwd = timeit(backward_only, repeats)
    t_full = timeit(full_step, repeats)

    print(f"=== batch={batch_size}  (repeats={repeats}, 稳态值) ===")
    print(f"  forward (no_grad)      {t_fwd:9.2f} ms/step  ({t_fwd / batch_size * 1000:8.3f} us/样本)")
    print(f"  forward (带计算图)     {t_graph:9.2f} ms/step  ({t_graph / batch_size * 1000:8.3f} us/样本)")
    print(f"  backward only          {t_bwd - t_graph:9.2f} ms/step")
    print(f"  完整 step (含 Adam)    {t_full:9.2f} ms/step  ({t_full / batch_size * 1000:8.3f} us/样本)")
    print(f"  => 吞吐 {batch_size / (t_full / 1000):8.1f} 样本/s")
    share = (t_fwd / t_full * 100) if t_full else 0.0
    print(f"  => 前向占完整 step 的 {share:.1f}%  (反向+Adam 占 {100 - share:.1f}%)")
    steps = 40500 / batch_size
    print(f"  => 推算论文规模 (40500 训练样本): {steps * t_full / 1000:.1f} s/epoch"
          f"  + 验证 (4500 样本 {t_fwd:.1f} ms/batch): {4500 / batch_size * t_fwd / 1000:.1f} s")
    print()
    return t_fwd, t_full


def root_finder_split(batch_size, tau=8, repeats=5):
    """把 root_music 拆开, 单独计时那条 eigvals。"""
    from src.models import find_roots_batched, sum_of_diags_batched

    model, _ = build_model(tau=tau)
    x = torch.randn(batch_size, tau, 2 * 8, 8, device=device)
    with torch.no_grad():
        rz = model(x)[-1]

    n = rz.shape[-1]
    _, eigvec = torch.linalg.eigh(rz)
    eigvec = torch.flip(eigvec, dims=[-1])
    un = eigvec[:, :, 2:]
    f = un @ un.conj().transpose(-2, -1)
    # [Batch, 2N-1]: mind the shape. `find_roots_batched` takes the 2N-1 coefficients and
    # builds a (2N-2)x(2N-2) companion matrix from them, so the full row is what goes in.
    diag_sum = sum_of_diags_batched(f).contiguous()
    roots = find_roots_batched(diag_sum)
    ones = torch.ones(diag_sum.shape[1] - 2, device=device).to(diag_sum.dtype)
    companion = torch.diag(ones, -1).unsqueeze(0).repeat(batch_size, 1, 1)
    companion[:, 0, :] = -diag_sum[:, 1:] / diag_sum[:, :1]
    companion = companion.contiguous()

    print(f"=== root_music 内部拆分 (batch={batch_size}) ===")
    print(f"  sum_of_diags_batched          {timeit(lambda: sum_of_diags_batched(f), repeats):9.3f} ms")
    print(f"  find_roots_batched 合计       {timeit(lambda: find_roots_batched(diag_sum), repeats):9.3f} ms")
    print(f"    - 构造伴随矩阵              {timeit(lambda: torch.diag(ones, -1).unsqueeze(0).repeat(batch_size, 1, 1), repeats):9.3f} ms")
    print(f"    - linalg.eigvals({n}x{n}) complex64 {timeit(lambda: torch.linalg.eigvals(companion), repeats):9.3f} ms"
          f"   ({timeit(lambda: torch.linalg.eigvals(companion), repeats) / batch_size * 1000:.2f} us/矩阵)")
    print(f"    - linalg.eigh({n}x{n}) 对照        {timeit(lambda: torch.linalg.eigh(rz), repeats):9.3f} ms")
    print(f"  roots shape/dtype             {tuple(roots.shape)} {roots.dtype}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, nargs="+", default=[512])
    parser.add_argument("--diff-method", default="root_music",
                        choices=["root_music", "esprit", "esprit_root_music"])
    parser.add_argument("--repeats", type=int, default=5,
                        help="每项计时的重复次数（默认 5；看比例不看绝对值）")
    parser.add_argument("--device", default=None,
                        help="default/cpu/cuda/cuda:2/2。等价于先设 CUDA_VISIBLE_DEVICES，"
                             "但不需要改环境。默认跟随 src.utils.device（cuda:0）。")
    args = parser.parse_args()

    describe()
    model, _ = build_model(diff_method=args.diff_method)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"diff_method = {args.diff_method}   可训练参数量 = {params} (论文 41761)")
    print()

    for batch_size in args.batch:
        split_timings(model, batch_size, repeats=args.repeats)

    if args.diff_method == "root_music":
        root_finder_split(args.batch[-1], repeats=args.repeats)

    reference = LOCAL_REFERENCE.get(args.diff_method, {})
    if reference:
        print("=== 与 RTX 4060 Laptop 参考值对比 (ms/step) ===")
        for batch_size, local in reference.items():
            print(f"  batch={batch_size:<5d} 本机参考 {local:8.1f} ms")


if __name__ == "__main__":
    main()
