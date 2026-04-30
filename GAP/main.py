import time
import torch
import sys
import os
import argparse

sys.path.append(os.path.abspath("src"))

from slicegpt import hf_utils, layernorm_fusion, rotate, data_utils, gpu_utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler
from slicegpt.adaptive_scheduler import AdaptiveSlicingScheduler


def _adjust_dims_to_target(layer_dims: list[int], target_total: int, hidden_size: int, round_interval: int) -> list[int]:
    dims = [int(max(round_interval, min(hidden_size, d - (d % round_interval)))) for d in layer_dims]
    delta = target_total - sum(dims)
    step = round_interval
    idx = 0
    while delta != 0:
        if delta > 0:
            if dims[idx] + step <= hidden_size:
                dims[idx] += step
                delta -= step
        else:
            if dims[idx] - step >= round_interval:
                dims[idx] -= step
                delta += step
        idx = (idx + 1) % len(dims)
    return dims


def _allocate_dims_grad_knapsack(
    layer_eig_vals: list[torch.Tensor],
    grad_sensitivities: list[torch.Tensor],
    hidden_size: int,
    total_budget: int,
    round_interval: int,
    min_layer_sparsity: float,
    max_layer_sparsity: float,
    layer0_full: bool = True,
) -> list[int]:
    import heapq
    import math

    n_layers = len(layer_eig_vals)
    d = hidden_size
    r = round_interval

    lo = int((1 - min_layer_sparsity) * d) // r * r
    hi = int((1 - max_layer_sparsity) * d) // r * r
    lo = max(lo, r)
    hi = min(hi, d)

    g_raw = []
    for g in grad_sensitivities:
        gv = float(g.float().mean().item())
        g_raw.append(max(gv, 1e-30))

    log_g = [math.log(g) for g in g_raw]
    min_log_g = min(log_g)
    max_log_g = max(log_g)

    g_bar = []
    for lg in log_g:
        if max_log_g > min_log_g:
            weight = 1.0 + 9.0 * (lg - min_log_g) / (max_log_g - min_log_g)
        else:
            weight = 1.0
        g_bar.append(weight)
    print(f"  Grad Knapsack: 梯度权重范围 [{min(g_bar):.3f} ~ {max(g_bar):.3f}]")

    eigs = []
    for ev in layer_eig_vals:
        ev_f = ev.float().tolist()
        total_var = sum(ev_f) + 1e-12
        ev_f = [v / total_var for v in ev_f]
        if len(ev_f) < d:
            ev_f = ev_f + [0.0] * (d - len(ev_f))
        eigs.append(ev_f)

    def marginal_gain(layer_idx: int, cur_dim: int) -> float:
        if cur_dim + r > d:
            return 0.0
        return g_bar[layer_idx] * sum(eigs[layer_idx][k] for k in range(cur_dim, min(cur_dim + r, d)))

    dims = [hi] * n_layers
    if layer0_full:
        dims[0] = d

    current_total = sum(dims)
    remaining = total_budget - current_total

    if remaining < 0:
        for l in range(1 if layer0_full else 0, n_layers):
            if current_total <= total_budget:
                break
            excess = current_total - total_budget
            cut = min(excess, dims[l] - hi)
            cut -= cut % r
            dims[l] -= cut
            current_total -= cut
    elif remaining > 0:
        heap = []
        for l in range(0 if not layer0_full else 1, n_layers):
            if dims[l] < lo:
                heapq.heappush(heap, (-marginal_gain(l, dims[l]), l))

        steps = remaining // r
        for _ in range(int(steps)):
            if not heap:
                break
            _, l = heapq.heappop(heap)
            if dims[l] + r > lo:
                continue
            dims[l] += r
            if dims[l] + r <= lo:
                heapq.heappush(heap, (-marginal_gain(l, dims[l]), l))

    for l in range(0 if not layer0_full else 1, n_layers):
        dims[l] = max(hi, min(lo, dims[l]))
        dims[l] = (dims[l] // r) * r
        dims[l] = max(r, dims[l])

    if layer0_full:
        dims[0] = d

    return _adjust_dims_to_target(dims, total_budget, d, r)


def _build_model_and_loader(model_name: str, args):
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
        model_name,
        model_path=getattr(args, "model_path", None),
        token=None,
        dtype=config.dtype,
    )
    dataset = data_utils.get_dataset(args.cal_dataset)
    train_loader = data_utils.prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=args.max_seqlen,
        batch_size=1,
        nsamples=args.cal_nsamples,
    )
    layernorm_fusion.replace_layers(model_adapter)
    layernorm_fusion.fuse_modules(model_adapter)

    original_params = sum(p.numel() for p in model_adapter.model.parameters())
    model_adapter.original_params = original_params
    print(f"原始模型参数量: {original_params:,}")

    return model_adapter, tokenizer, dataset, train_loader


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to load")
parser.add_argument("--model-path", type=str, default=None, help="Local path to model weights (optional)")
parser.add_argument("--sparsity", type=float, default=0.25, help="Sparsity level")
parser.add_argument("--cal-dataset", type=str, default="wikitext2", help="Calibration dataset")
parser.add_argument("--cal-nsamples", type=int, default=128, help="Number of calibration samples")
parser.add_argument("--max-seqlen", type=int, default=2048, help="Max sequence length for calibration")
parser.add_argument(
    "--orientation",
    type=str,
    default="pca",
    choices=["pca", "gap"],
    help="Rotation matrix calculation method",
)
parser.add_argument("--round-interval", type=int, default=8, help="Dimension rounding interval")
parser.add_argument("--min-layer-sparsity", type=float, default=0.0, help="Minimum per-layer sparsity")
parser.add_argument("--max-layer-sparsity", type=float, default=None, help="Maximum per-layer sparsity")
args = parser.parse_args()

start_time = time.time()
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

model_name = args.model
sparsity = args.sparsity
if args.max_layer_sparsity is None:
    args.max_layer_sparsity = min(0.9, sparsity + 0.1)
if args.max_layer_sparsity < args.min_layer_sparsity:
    args.max_layer_sparsity = args.min_layer_sparsity
device = "cuda" if torch.cuda.is_available() else "cpu"
config.device = torch.device(device)
config.dtype = torch.float16

print(f"正在 {device} 上加载 {model_name} ...")
print(f"使用旋转矩阵计算方法: {args.orientation}")
print(f"max_layer_sparsity 使用值: {args.max_layer_sparsity:.3f}")
print(f"正在加载校准数据 {args.cal_dataset}...")
model_adapter, tokenizer, dataset, train_loader = _build_model_and_loader(model_name, args)

if args.orientation == "gap":
    print("步骤1: 收集各层特征值谱（原始模型）...")
    layer_eig_vals = rotate.collect_layer_pca_eigenvalues(model_adapter, train_loader)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("步骤2: 逐层收集梯度敏感度（显存高效，无需全模型上 GPU）...")
    grad_sens = rotate.collect_gradient_sensitivity_layerwise(model_adapter, train_loader, n_batches=8)
    mean_g = [float(g.mean().item()) for g in grad_sens]
    print(f"  各层均值梯度敏感度: {[f'{v:.3e}' for v in mean_g]}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("步骤3: 梯度加权贪心背包分配...")
    n_layers = len(model_adapter.get_layers())
    full_dim = model_adapter.hidden_size
    total_budget = int((1 - sparsity) * full_dim) * n_layers
    total_budget -= total_budget % args.round_interval

    
    layer_dims = _allocate_dims_grad_knapsack(
        layer_eig_vals=layer_eig_vals,
        grad_sensitivities=grad_sens,
        hidden_size=full_dim,
        total_budget=total_budget,
        round_interval=args.round_interval,
        min_layer_sparsity=args.min_layer_sparsity,
        max_layer_sparsity=args.max_layer_sparsity,
        layer0_full=True,
    )

    print(f"层0: {layer_dims[0]} 维 (完整); 其余: {min(layer_dims[1:])}--{max(layer_dims[1:])}")
    for i, m in enumerate(layer_dims):
        pct = 1 - m / full_dim
        print(f"  层 {i:2d}: {m:4d} 维 ({pct:5.1%} 剪枝)")

    print("步骤4: 标准 PCA 旋转 + 切片...")
    scheduler = AdaptiveSlicingScheduler(layer_dims)
    rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation="pca")
else:
    original_dim = model_adapter.hidden_size
    new_embedding_dimension = int((1 - sparsity) * original_dim)
    new_embedding_dimension -= new_embedding_dimension % 8
    print(f"从 {original_dim} 切片到 -> {new_embedding_dimension}")
    scheduler = ConstSlicingScheduler(new_embedding_dimension)
    print("正在使用 pca 方法旋转和切片...")
    rotate.rotate_and_slice(
        model_adapter,
        train_loader,
        scheduler,
        final_orientation="pca",
    )

print("成功！模型已完成切片。")
output_path = f"sliced_{model_name.split('/')[-1]}_{args.orientation}.pt"
torch.save(model_adapter.model.state_dict(), output_path)
print(f"模型已保存至 {output_path}")

print("正在评估困惑度...")
test_key = "test" if "test" in dataset else "validation"
test_loader = data_utils.prepare_test_dataloader(
    dataset=dataset[test_key],
    tokenizer=tokenizer,
    batch_size=8,
)
ppl = gpu_utils.evaluate_ppl(model_adapter.model, model_adapter.model.config.pad_token_id, test_loader)
print(f"切片后 {args.cal_dataset} 困惑度: {ppl}")

end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\n总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.1f}秒")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
