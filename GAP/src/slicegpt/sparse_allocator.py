"""
跨层 ℓ0 稀疏维度分配器

基于联合重要性 s_{l,j} = λ_{l,j} · ‖W_l q_j‖² 和全局阈值 μ，
自动为每层分配不同的保留维度数。
"""

import torch


class SparseAllocator:
    """
    数学模型：
        min_{α} Σ_l Σ_j (1 - α_{l,j})² s_{l,j} + μ ‖α_l‖₀
        s.t.  α_{l,j} ∈ {0, 1}

    闭式解：
        α_{l,j} = 𝟙[s_{l,j} > μ]

    通过二分搜索 μ 使总稀疏率达到目标值。
    """

    def __init__(self, round_interval: int = 8):
        self.round_interval = round_interval

    def allocate(
        self,
        layer_info: list[dict],
        target_sparsity: float,
        hidden_size: int,
        min_dim: int | None = None,
    ) -> list[int]:
        """
        分配每层保留的维度数。

        Args:
            layer_info: rotate() 返回的列表，每项含 'eig_val' 和 'w_col_norms_sq'
            target_sparsity: 目标总稀疏率，如 0.25
            hidden_size: 模型隐藏维度 d
            min_dim: 每层最少保留维度数

        Returns:
            每层保留的维度数列表 [m_0, m_1, ..., m_{L-1}]
        """
        if not layer_info:
            raise ValueError("layer_info is empty")

        if min_dim is None:
            min_dim = max(self.round_interval, int(0.1 * hidden_size))

        all_scores: list[torch.Tensor] = []
        layers_num = len(layer_info)
        for layer_idx, info in enumerate(layer_info):
            eig_val = info["eig_val"][:hidden_size].float()
            w_norms = info["w_col_norms_sq"][:hidden_size].float()
            distance_weight = float(layers_num - layer_idx)
            s = distance_weight * eig_val * w_norms
            all_scores.append(s)

        all_s_flat = torch.cat(all_scores)
        left, right = 0.0, all_s_flat.max().item() * 2.0
        mu = right

        for _ in range(200):
            mu = (left + right) / 2.0
            total_kept = sum((s > mu).sum().item() for s in all_scores)
            total_dims = len(all_scores) * hidden_size
            current_sparsity = 1.0 - total_kept / total_dims

            if abs(current_sparsity - target_sparsity) < 1e-4:
                break
            if current_sparsity < target_sparsity:
                left = mu
            else:
                right = mu

        layer_dims = []
        for s in all_scores:
            m = int((s > mu).sum().item())
            m = max(min_dim, m)
            if m % self.round_interval != 0:
                m = m - (m % self.round_interval)
            m = max(self.round_interval, min(m, hidden_size))
            layer_dims.append(m)

        return layer_dims

    def print_allocation(self, layer_dims: list[int], hidden_size: int, target_sparsity: float) -> None:
        """打印分配结果"""
        uniform_dim = int((1 - target_sparsity) * hidden_size)
        uniform_dim -= uniform_dim % self.round_interval

        print(f"\n{'=' * 50}")
        print(f"跨层稀疏分配结果 (目标稀疏率: {target_sparsity:.0%})")
        print(f"{'=' * 50}")
        print(f"均匀维度 (原版SliceGPT): {uniform_dim}")
        print(f"{'─' * 50}")
        for l, m in enumerate(layer_dims):
            bar = '█' * int(m / hidden_size * 30)
            print(f"  层 {l:2d}: {m:4d} 维 ({1 - m / hidden_size:5.1%} 剪枝) {bar}")
        print(f"{'─' * 50}")
        actual = 1.0 - sum(layer_dims) / (len(layer_dims) * hidden_size)
        print(f"实际总稀疏率: {actual:.1%}")
        print(f"{'=' * 50}\n")
