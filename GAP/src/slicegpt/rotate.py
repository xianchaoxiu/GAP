# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors


# ==============================================================================
# 基础 Helper Functions (保持不变)
# ==============================================================================

def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension
    layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])

def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    W = layer_adapter.get_attention_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    cleanup_memory()

def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]

def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension

# ==============================================================================
# 核心计算逻辑 (PCA, Fisher, Random)
# ==============================================================================

def random_orthogonal_upper_left(total_dim, upper_block_dim):
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)

@torch.no_grad()

def collect_fisher_diagonal(
    model_adapter: ModelAdapter,
    train_loader: torch.utils.data.DataLoader[torch.Tensor],
    n_batches: int = 8,
) -> list[torch.Tensor]:
    """
    Compute Fisher information diagonal for all layers.
    
    This is MUCH faster than full Fisher (O(d) vs O(d²)).
    
    Returns:
        List of (d,) tensors, Fisher diagonal per layer
    """
    from .model_utils import get_layer0_inputs
    import torch.nn as nn
    
    # Collect inputs from first n_batches batches
    inps, args, kwargs, ignore_masks = [], [], [], []
    batch_count = 0
    for batch in train_loader:
        if batch_count >= n_batches:
            break
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if batch.get("attention_mask") is not None:
            ignore_masks.append(batch["attention_mask"])
        batch_count += 1
    
    all_layers = model_adapter.get_layers()
    n_layers = len(all_layers)
    fisher_diagonals = []
    
    print(f"\n=== Computing Fisher Diagonal ({n_layers} layers) ===")
    
    for layer_idx in range(n_layers):
        print(f"Layer {layer_idx+1}/{n_layers}...", flush=True)
        
        # Compute Fisher diagonal for this layer
        fisher_diag = _fisher_diagonal_calc_real(
            model_adapter,
            layer_idx,
            inps,
            args,
            kwargs,
            train_loader,
            ignore_masks
        )
        
        fisher_diagonals.append(fisher_diag)
    
    print(f"Done! Fisher diagonal computed for {n_layers} layers.\n")
    return fisher_diagonals


def _fisher_diagonal_calc_real(
    model_adapter: ModelAdapter,
    layer_idx: int,
    inps: list[torch.Tensor],
    args: list[tuple],
    kwargs: list[dict],
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    ignore_masks: list[torch.Tensor] | None = None
) -> torch.Tensor:
    """
    Compute Fisher information diagonal using REAL gradients.
    
    Returns:
        (d,) tensor, Fisher diagonal
    """
    cleanup_memory()
    
    all_layers = model_adapter.get_layers()
    subsequent_layers = all_layers[layer_idx:]
    pre_head_ln = model_adapter.get_pre_head_layernorm()
    lm_head = model_adapter.get_lm_head()
    
    # Move layers to device
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to(config.device)
    if pre_head_ln is not None:
        pre_head_ln.to(config.device)
    lm_head.to(config.device)
    
    data_iter = iter(dataloader)
    fisher_diag = None
    
    for i, X in enumerate(inps):
        model_adapter.model.zero_grad(set_to_none=True)
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        
        labels = batch.get("labels", batch.get("input_ids"))
        if labels is None: continue
        labels = labels.to(config.device)
        
        X = X.detach().clone().to(config.device).requires_grad_(True)
        batch_args = map_tensors(args[i], device=config.device)
        batch_kwargs = map_tensors(kwargs[i], device=config.device)
        
        current_hidden = X
        
        def run_tail(hidden_state):
            h = hidden_state
            for layer_adapter in subsequent_layers:
                layer_args_updated = layer_adapter.get_updated_args(h, batch_args)
                out = layer_adapter.layer(*layer_args_updated, **batch_kwargs)
                h = out[layer_adapter.hidden_states_output_position] if isinstance(out, tuple) else out
            if pre_head_ln is not None:
                h = pre_head_ln(h)
            return lm_head(h)
        
        target_grad, target_input = [], []
        hook_handles = []
        
        with torch.enable_grad():
            if len(subsequent_layers) > 0:
                target_module = subsequent_layers[0].layer
                # Register hook to capture input and gradient
                def forward_hook(m, i, o):
                    if i and i[0] is not None:
                        target_input.append(i[0].detach())
                
                def backward_hook(m, gi, go):
                    if gi and gi[0] is not None:
                        target_grad.append(gi[0].detach())
                
                hook_handles.append(target_module.register_forward_hook(forward_hook))
                hook_handles.append(target_module.register_full_backward_hook(backward_hook))
            
            logits = run_tail(current_hidden)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]
            
            loss = nn.CrossEntropyLoss()(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            loss.backward()
        
        for h in hook_handles:
            h.remove()
        
        if target_grad and target_input:
            # Compute Fisher diagonal: E[(dL/dx)^2]
            g = target_grad[0].double()
            x = target_input[0].double()
            
            # Reshape to (batch*seq, hidden)
            g_reshaped = g.reshape(-1, g.shape[-1])
            
            # Fisher diagonal = mean of squared gradients
            fisher_diag_batch = torch.mean(g_reshaped ** 2, dim=0)
            
            if fisher_diag is None:
                fisher_diag = fisher_diag_batch
            else:
                fisher_diag += fisher_diag_batch
    
    if fisher_diag is None:
        raise ValueError("Fisher diagonal is None! No batches processed?")
    
    # Normalize
    fisher_diag = fisher_diag / len(inps)
    
    # Add small damping for numerical stability
    fisher_diag = fisher_diag + 0.01 * fisher_diag.mean()
    
    fisher_diag_cpu = fisher_diag.to('cpu')
    del fisher_diag
    cleanup_memory()
    
    # Move layers back to CPU
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to('cpu')
    if pre_head_ln is not None:
        pre_head_ln.to('cpu')
    lm_head.to('cpu')
    
    return fisher_diag_cpu


def pca_calc(X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    cleanup_memory()
    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec


def _build_H(X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None) -> torch.Tensor:
    """Build covariance matrix H from list of activation tensors."""
    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H + H_batch
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    return H


@torch.no_grad()

@torch.no_grad()
def admm_l21_pca_calc(
    X: list[torch.Tensor],
    ignore_masks: list[torch.Tensor] | None,
    m: int,
    lambda_: float = 0.01,
    rho: float = 1.0,
    max_outer: int = 100,
    n_inner: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ADMM on Stiefel + L2,1 regularization.

    Solves:  min_{W in St(d,m)} -Tr(W^T H W) + lambda * ||W||_{2,1}
    via ADMM splitting W = Z:
      W-step: Riemannian gradient descent + QR retraction
      Z-step: row-wise soft thresholding (closed-form prox of L_{2,1})
      U-step: dual ascent
    Then extracts support S, does restricted PCA on H_S, QR-extends to d x d.
    """
    from .sparse_pca import admm_sparse_pca
    cleanup_memory()
    H = _build_H(X, ignore_masks)
    return admm_sparse_pca(H, m, lambda_=lambda_, rho=rho,
                           max_outer=max_outer, n_inner=n_inner)

def l21_pca_calc(
    X: list[torch.Tensor],
    ignore_masks: list[torch.Tensor] | None,
    m: int,
    lambda_: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2,1-inspired importance-WEIGHTED PCA (safe version).
    Builds H then calls sparse_pca(H, m, lambda_) which reweights the
    covariance by row importances without discarding any input dimensions.
    """
    from .sparse_pca import sparse_pca
    cleanup_memory()
    H = _build_H(X, ignore_masks)
    return sparse_pca(H, m, lambda_=lambda_)


@torch.no_grad()
def shrinkage_pca_calc(
    X: list[torch.Tensor],
    ignore_masks: list[torch.Tensor] | None,
    m: int,
    alpha: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ledoit-Wolf shrinkage PCA for noisy covariance.
    Regularises H towards the scaled identity before computing PCA.
    """
    from .sparse_pca import shrinkage_pca
    cleanup_memory()
    H = _build_H(X, ignore_masks)
    return shrinkage_pca(H, m, alpha=alpha)


@torch.no_grad()
def corr_pca_calc(
    X: list[torch.Tensor],
    ignore_masks: list[torch.Tensor] | None,
    m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale-invariant (correlation-based) PCA.
    Normalises H to correlation matrix before computing PCA.
    """
    from .sparse_pca import corr_pca
    cleanup_memory()
    H = _build_H(X, ignore_masks)
    return corr_pca(H, m)


@torch.no_grad()
def reorder_q_by_joint_importance(eig_val: torch.Tensor, Q: torch.Tensor, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder PCA directions by joint score s_j = lambda_j * ||W q_j||^2 (descending)."""
    eig_val = eig_val.float()
    Q = Q.float()
    W = W.float()
    W = W.to(device=Q.device)
    if W.shape[1] == Q.shape[0]:
        WQ = torch.matmul(W, Q)
    elif W.shape[0] == Q.shape[0]:
        WQ = torch.matmul(W.T, Q)
    else:
        raise ValueError(f"Incompatible shapes for joint importance: W={tuple(W.shape)} Q={tuple(Q.shape)}")
    col_norms_sq = (WQ ** 2).sum(dim=0)
    score = eig_val * col_norms_sq
    order = torch.argsort(score, descending=True)
    return eig_val[order], Q[:, order]

@torch.no_grad()
def fisher_calc_real(
    model_adapter: ModelAdapter,
    layer_idx: int,
    inps: list[torch.Tensor],
    args: list[tuple],
    kwargs: list[dict],
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    target_norm_getter=None,
    ignore_masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Fisher information matrix using REAL gradients.
    """
    cleanup_memory()
    H = None
    all_layers = model_adapter.get_layers()
    subsequent_layers = all_layers[layer_idx:]
    pre_head_ln = model_adapter.get_pre_head_layernorm()
    lm_head = model_adapter.get_lm_head()
    
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to(config.device)
    if pre_head_ln is not None:
        pre_head_ln.to(config.device)
    lm_head.to(config.device)
    
    data_iter = iter(dataloader)
    
    for i, X in enumerate(inps):
        model_adapter.model.zero_grad(set_to_none=True)
        try:
            batch = next(data_iter)
        except StopIteration:
            break
            
        labels = batch.get("labels", batch.get("input_ids"))
        if labels is None: continue
        labels = labels.to(config.device)
        
        X = X.detach().clone().to(config.device).requires_grad_(True)
        batch_args = map_tensors(args[i], device=config.device)
        batch_kwargs = map_tensors(kwargs[i], device=config.device)
        
        current_hidden = X
        
        def run_tail(hidden_state):
            h = hidden_state
            for layer_adapter in subsequent_layers:
                layer_args_updated = layer_adapter.get_updated_args(h, batch_args)
                out = layer_adapter.layer(*layer_args_updated, **batch_kwargs)
                h = out[layer_adapter.hidden_states_output_position] if isinstance(out, tuple) else out
            if pre_head_ln is not None:
                h = pre_head_ln(h)
            return lm_head(h)

        target_grad, target_input, hook_handles = [], [], []
        
        with torch.enable_grad():
            if target_norm_getter is not None and len(subsequent_layers) > 0:
                target_module = target_norm_getter(subsequent_layers[0])
                hook_handles.append(target_module.register_full_backward_hook(
                    lambda m, gi, go: target_grad.append(gi[0].detach()) if gi and gi[0] is not None else None
                ))
                hook_handles.append(target_module.register_forward_hook(
                    lambda m, i, o: target_input.append(i[0].detach()) if i and i[0] is not None else None
                ))
            
            logits = run_tail(current_hidden)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Some models/batches may produce slight sequence-length mismatch between logits and labels.
            # Align by truncating to the common minimum length before flattening.
            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss = nn.CrossEntropyLoss()(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            loss.backward()
        
        for h in hook_handles: h.remove()
        
        if target_grad and target_input:
            g = target_grad[0].double().reshape(-1, target_grad[0].shape[-1])
            x = target_input[0].double().reshape(-1, target_input[0].shape[-1])
            w = torch.norm(g, dim=1, keepdim=True)
            w = w / (w.mean() + 1e-8) if w.mean() > 1e-8 else w
            x_weighted = x * torch.sqrt(w)
            H_batch = x_weighted.T @ x_weighted
        else:
            x = target_input[0] if target_input else X
            x = x.reshape(-1, x.shape[-1]).double().to(config.device)
            H_batch = x.T @ x
            
        H = H_batch if H is None else H + H_batch

    if H is None: raise ValueError("H is None! No batches processed?")
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp

    H_cpu = H.to('cpu')
    del H
    cleanup_memory()

    eig_val, eig_vec = torch.linalg.eigh(H_cpu)
    Q_fish = eig_vec.flip(1)
    eig_val = eig_val.flip(0)
    del H_cpu
    cleanup_memory()
    
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to('cpu')
    if pre_head_ln is not None:
        pre_head_ln.to('cpu')
    lm_head.to('cpu')
    return eig_val, Q_fish


@torch.no_grad()
def entropy_calc_proxy(
    model_adapter: ModelAdapter,
    layer_idx: int,
    inps: list[torch.Tensor],
    args: list[tuple],
    kwargs: list[dict],
    ignore_masks: list[torch.Tensor] | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entropy-weighted covariance without backward pass.
    Weight per token: w_t = H(p_t) where p_t = softmax(logits_t).
    """
    cleanup_memory()
    H = None
    all_layers = model_adapter.get_layers()
    subsequent_layers = all_layers[layer_idx:]
    pre_head_ln = model_adapter.get_pre_head_layernorm()
    lm_head = model_adapter.get_lm_head()

    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to(config.device)
    if pre_head_ln is not None:
        pre_head_ln.to(config.device)
    lm_head.to(config.device)

    for i, X in enumerate(inps):
        X = X.to(config.device)
        batch_args = map_tensors(args[i], device=config.device)
        batch_kwargs = map_tensors(kwargs[i], device=config.device)

        h = X
        for layer_adapter in subsequent_layers:
            layer_args_updated = layer_adapter.get_updated_args(h, batch_args)
            out = layer_adapter.layer(*layer_args_updated, **batch_kwargs)
            h = out[layer_adapter.hidden_states_output_position] if isinstance(out, tuple) else out

        if pre_head_ln is not None:
            h = pre_head_ln(h)

        logits = lm_head(h).float()
        shift_logits = logits[:, :-1, :]
        probs = torch.softmax(shift_logits, dim=-1)
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)

        min_len = min(entropy.size(1), X.size(1))
        x_use = X[:, :min_len, :].double()
        w_use = entropy[:, :min_len].double()

        if ignore_masks is not None:
            mask = ignore_masks[i].to(config.device)
            mask = mask[:, 1:1 + min_len] if mask.size(1) >= min_len + 1 else mask[:, :min_len]
            w_use = w_use * mask.double()

        x_flat = x_use.reshape(-1, x_use.shape[-1])
        w_flat = w_use.reshape(-1, 1)
        if w_flat.numel() == 0:
            continue

        w_mean = w_flat.mean()
        if w_mean > 1e-8:
            w_flat = w_flat / (w_mean + 1e-8)

        x_weighted = x_flat * torch.sqrt(torch.clamp_min(w_flat, 0.0))
        H_batch = x_weighted.T @ x_weighted
        H = H_batch if H is None else H + H_batch

    if H is None:
        raise ValueError("H is None! No valid entropy batches processed?")

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp

    H_cpu = H.to('cpu')
    del H
    cleanup_memory()

    eig_val, eig_vec = torch.linalg.eigh(H_cpu)
    eig_val = eig_val.flip(0)
    eig_vec = eig_vec.flip(1)

    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to('cpu')
    if pre_head_ln is not None:
        pre_head_ln.to('cpu')
    lm_head.to('cpu')

    return eig_val, eig_vec

# ==============================================================================
# 主要逻辑：Rotate and Slice
# ==============================================================================

def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
    eig_val_collector: dict | None = None,
    importance_collector: list[dict[str, torch.Tensor]] | None = None,
    l21_lambda: float = 0.01,
    grad_sensitivities: list | None = None,
    fisher_diagonals: list | None = None,
    saliency_diagonals: list | None = None,
) -> None:
    """
    Rotate and slice a model.
    Args:
        final_orientation: 'pca', 'random', 'fisher', 'pca_joint', or 'l21_pca'.
        l21_lambda: regularization strength for L2,1 sparse PCA.
    """
    if model_adapter.parallel_blocks:
        rotate_and_slice_parallel(
            model_adapter,
            dataloader,
            slicing_scheduler,
            apply_mask,
            final_orientation,
            eig_val_collector=eig_val_collector,
            importance_collector=importance_collector,
            l21_lambda=l21_lambda,
        )
    else:
        rotate_and_slice_sequential(
            model_adapter,
            dataloader,
            slicing_scheduler,
            apply_mask,
            final_orientation,
            eig_val_collector=eig_val_collector,
            importance_collector=importance_collector,
            l21_lambda=l21_lambda,
            grad_sensitivities=grad_sensitivities,
            fisher_diagonals=fisher_diagonals,
            saliency_diagonals=saliency_diagonals,
        )


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
    eig_val_collector: dict | None = None,
    importance_collector: list[dict[str, torch.Tensor]] | None = None,
    l21_lambda: float = 0.01,
    grad_sensitivities: list | None = None,
    fisher_diagonals: list | None = None,
    saliency_diagonals: list | None = None,
) -> None:
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # --- 统一的旋转矩阵计算 Helper ---
    def compute_Q(signals, layer_idx=None, target_getter=None, is_input=False, target_m=None):
        if final_orientation == 'fisher':
            # Fisher 需要具体的 layer_idx 和后续梯度
            target_idx = layer_idx if layer_idx is not None else 0
            if is_input and layer_idx is not None:
                # 如果是计算下一层的输入（output rotation），我们需要看 next layer
                target_idx = layer_idx + 1

            eig_val, eig_vec = fisher_calc_real(
                model_adapter,
                target_idx,
                signals,
                args,
                kwargs,
                dataloader,
                target_getter,
                ignore_masks,
            )
            if eig_val_collector is not None and layer_idx is not None:
                eig_val_collector.setdefault(layer_idx, []).append(eig_val.detach().cpu())
            return eig_val, eig_vec
        elif final_orientation == 'fisher_saliency':
            # Fisher Saliency: 使用预收集的 Fisher + Saliency 对角线加权 PCA
            f_diag = fisher_diagonals[layer_idx] if fisher_diagonals is not None else None
            s_diag = saliency_diagonals[layer_idx] if saliency_diagonals is not None else None
            return fisher_saliency_pca_calc(signals, ignore_masks, fisher_diagonal=f_diag, saliency_diagonal=s_diag)
        elif final_orientation == 'fisher_fast':
            # Fisher Fast: based on pre-collected Fisher diagonal
            f_diag = fisher_diagonals[layer_idx] if fisher_diagonals is not None else None
            return fisher_fast_pca_calc(signals, ignore_masks, fisher_diagonal=f_diag)
        elif final_orientation == 'entropy':
            if not is_input:
                return pca_calc(signals, ignore_masks)

            target_idx = layer_idx if layer_idx is not None else 0
            if is_input and layer_idx is not None:
                target_idx = layer_idx + 1

            eig_val, eig_vec = entropy_calc_proxy(
                model_adapter,
                target_idx,
                signals,
                args,
                kwargs,
                ignore_masks,
            )
            if eig_val_collector is not None and layer_idx is not None:
                eig_val_collector.setdefault(layer_idx, []).append(eig_val.detach().cpu())
            return eig_val, eig_vec
        elif final_orientation == 'l21_pca':
            m = target_m if target_m is not None else model_adapter.hidden_size
            return l21_pca_calc(signals, ignore_masks, m=m, lambda_=l21_lambda)
        elif final_orientation == 'grad_pca':
            # Gradient-weighted PCA: O(d) Riemannian optimisation with loss-gradient objective
            m = target_m if target_m is not None else model_adapter.hidden_size
            g_sens = grad_sensitivities[layer_idx] if grad_sensitivities is not None else None
            if g_sens is None:
                return pca_calc(signals, ignore_masks)
            return gradient_pca_calc(signals, ignore_masks, grad_sensitivity=g_sens)
        elif final_orientation == 'oblique_pca':
            # Oblique projection: biorthogonal (U_decoder, V_encoder) pair
            g_sens = grad_sensitivities[layer_idx] if grad_sensitivities is not None else None
            if g_sens is None:
                eig_v, Q_tmp = pca_calc(signals, ignore_masks)
                return eig_v, Q_tmp, Q_tmp  # fallback: 3-tuple with U=V=Q
            return oblique_pca_calc(signals, ignore_masks, grad_sensitivity=g_sens)
        elif final_orientation == 'admm_l21_pca':
            m = target_m if target_m is not None else model_adapter.hidden_size
            return admm_l21_pca_calc(signals, ignore_masks, m=m,
                                     lambda_=l21_lambda, rho=l21_lambda,
                                     max_outer=100, n_inner=5)
        elif final_orientation == 'shrinkage_pca':
            m = target_m if target_m is not None else model_adapter.hidden_size
            return shrinkage_pca_calc(signals, ignore_masks, m=m, alpha=l21_lambda)
        elif final_orientation == 'corr_pca':
            m = target_m if target_m is not None else model_adapter.hidden_size
            return corr_pca_calc(signals, ignore_masks, m=m)
        elif final_orientation == 'pure_fisher_weighted':
            # Pure Fisher-Weighted PCA: 用 Fisher 对角线替代梯度敏感度对协方差加权
            f_diag = fisher_diagonals[layer_idx] if fisher_diagonals is not None else None
            if f_diag is None:
                return pca_calc(signals, ignore_masks)
            return gradient_pca_calc(signals, ignore_masks, grad_sensitivity=f_diag)
        elif final_orientation == 'reconstruction_optimized':
            # Reconstruction-Optimized Rotation: PCA 即最小化重构误差的最优旋转
            return pca_calc(signals, ignore_masks)
        elif final_orientation == 'grad_sorted_pca':
            # Grad-Sorted PCA: 先做 PCA，再按梯度敏感度分数对特征向量重排序
            g_sens = grad_sensitivities[layer_idx] if grad_sensitivities is not None else None
            if g_sens is None:
                return pca_calc(signals, ignore_masks)
            return fisher_fast_pca_calc(signals, ignore_masks, fisher_diagonal=g_sens)
        else:
            # PCA 和 Random 基于信号协方差
            return pca_calc(signals, ignore_masks)
    # -------------------------------

    # 1. Embeddings
    _emb_m = slicing_scheduler.get_embedding_dimensions()[0] if final_orientation in ('l21_pca', 'admm_l21_pca', 'grad_pca') else None
    _cq_result = compute_Q(inps, layer_idx=0, target_m=_emb_m)
    if len(_cq_result) == 3:
        eig_val, Q, _Q_enc_emb = _cq_result
    else:
        eig_val, Q = _cq_result
    Q = Q.to(device=config.device)
    
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
        
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # Rotate/Slice Attention Input
        rotate_attention_inputs(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # Rotate Internal (Attn -> MLP)
        rotated_inps = []
        for i, inp in enumerate(inps):
            rot_inp = torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                        :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                    ].cpu()
            args[i] = layer_adapter.get_updated_args(rot_inp, args[i])
            if final_orientation == 'fisher':
                rotated_inps.append(rot_inp) # Fisher需要旋转后的输入

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        
        # 计算内部旋转矩阵 Q
        calc_inputs = rotated_inps if final_orientation == 'fisher' else mlp_ln_inputs
        _attn_m = slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False) if final_orientation == 'l21_pca' else None
        _cq_result = compute_Q(calc_inputs, layer_idx=idx, target_getter=lambda la: la.get_second_layernorm(), target_m=_attn_m)
        if len(_cq_result) == 3:
            eig_val, Q, Q_enc = _cq_result
        else:
            eig_val, Q = _cq_result
            Q_enc = Q

        if final_orientation == 'pca_joint':
            W_attn_out = layer_adapter.get_attention_output().weight.data
            eig_val, Q = reorder_q_by_joint_importance(eig_val, Q, W_attn_out)
            Q_enc = Q  # pca_joint: encoder = decoder

        Q = Q.to(device=config.device, dtype=torch.float64)
        Q_enc = Q_enc.to(device=config.device, dtype=torch.float64)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False))
            Q = Q @ R.to(Q.device)
            Q_enc = Q

        # Apply internal rotation
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q.to(device=config.device),
                Q_enc.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q_enc)
        slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False))

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))
        cleanup_memory()

        # Compute Output / Next Layer Input
        _, inps = get_signals(layer_adapter, args, kwargs)
        
        # 计算输出旋转矩阵 Q
        _mlp_m = slicing_scheduler.get_mlp_output_dimension(idx) if final_orientation == 'l21_pca' else None
        _cq_result = compute_Q(inps, layer_idx=idx, is_input=True, target_m=_mlp_m)
        if len(_cq_result) == 3:
            eig_val, Q, Q_enc_out = _cq_result
        else:
            eig_val, Q = _cq_result
            Q_enc_out = Q

        if final_orientation == 'pca_joint':
            W_mlp_out = layer_adapter.get_mlp_output().weight.data
            eig_val, Q = reorder_q_by_joint_importance(eig_val, Q, W_mlp_out)
            Q_enc_out = Q  # pca_joint: encoder = decoder

        Q = Q.to(device=config.device, dtype=torch.float64)
        Q_enc_out = Q_enc_out.to(device=config.device, dtype=torch.float64)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)
            Q_enc_out = Q

        layer.mlp_shortcut_Q = nn.Parameter(
            torch.matmul(layer.mlp_shortcut_Q.to(device=config.device), Q_enc_out.to(dtype=dtype))
        )
        rotate_mlp_output(layer_adapter, Q_enc_out)

        if importance_collector is not None:
            W_out = layer_adapter.get_mlp_output().weight.data.float()
            col_norms_sq = (W_out ** 2).sum(dim=0).cpu()
            importance_collector.append({
                "eig_val": eig_val.detach().cpu(),
                "w_col_norms_sq": col_norms_sq,
            })

        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)])

        layer.to('cpu')
        cleanup_memory()

    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
    eig_val_collector: dict | None = None,
    importance_collector: list[dict[str, torch.Tensor]] | None = None,
    l21_lambda: float = 0.01,
) -> None:
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    # --- 统一的旋转矩阵计算 Helper ---
    def compute_Q(signals, layer_idx=None):
        if final_orientation == 'fisher':
            # Parallel 暂时简化为仅使用信号本身计算 Fisher 方向 (模拟) 或者回退到 PCA
            # 这里为了完整性，可以使用 PCA，因为 Parallel Fisher 比较复杂
            return pca_calc(signals, ignore_masks) 
        if final_orientation == 'entropy':
            return pca_calc(signals, ignore_masks)
        else:
            return pca_calc(signals, ignore_masks)
    # -------------------------------

    _, Q = compute_Q(inps)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
        
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        outputs = []
        layer = layer.to(config.device)
        for layer_args_batch, layer_kwargs_batch in zip(args, kwargs):
            layer_args_batch, layer_kwargs_batch = map_tensors([layer_args_batch, layer_kwargs_batch], device=config.device)
            out = layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            out = out.cpu()
            outputs.append(out)

        inps = outputs
        _, Q = compute_Q(inps)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))

        rotate_mlp_output(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))

        layer.attn_shortcut_Q = nn.Parameter(
            layer.attn_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)]
        )

        layer.to('cpu')
        cleanup_memory()

    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")



@torch.no_grad()


# ---------------------------------------------------------------------------
# Gradient-weighted PCA: Riemannian optimisation on O(d) with gradient objective
# ---------------------------------------------------------------------------

def collect_gradient_sensitivity(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader,
    n_batches: int = 4,
) -> list[torch.Tensor]:
    """Collect per-layer gradient sensitivity via backward pass.

    For each transformer layer l, computes:
        G_l[i] = E[ (dL/dh_l)_i^2 ]

    Implementation:
    1. Move full model to config.device (GPU) for fast inference.
    2. Patch decoder layer forwards to return tensors (not tuples) for
       backbone compatibility with newer transformers.
    3. Inject gradient anchor at embedding output so activations require grad
       without storing weight gradients.
    4. After collection, move model back to CPU (rotate_and_slice is layer-by-layer).

    Returns list of (d,) float32 tensors, one per transformer layer.
    """
    model   = model_adapter.model
    layers  = model_adapter.get_layers()
    n_layer = len(layers)
    d       = model_adapter.hidden_size

    grad_sq = [torch.zeros(d, dtype=torch.float32) for _ in range(n_layer)]
    layer_acts: dict[int, torch.Tensor] = {}
    anchor_holder: list[torch.Tensor] = []

    # --- Step 0: Move full model to GPU for fast inference ---
    model_orig_device = next(model.parameters()).device
    if model_orig_device.type == 'cpu' and config.device.type == 'cuda':
        logging.info("collect_gradient_sensitivity: moving full model to GPU")
        model.to(config.device)
    device = next(model.parameters()).device

    # --- Step 1: Freeze all parameters (avoid large weight gradient buffers) ---
    orig_req = {}
    for name, p in model.named_parameters():
        orig_req[name] = p.requires_grad
        p.requires_grad_(False)

    # --- Step 2: Patch decoder layer forwards to return tensors ---
    # Newer transformers LlamaModel expects: hidden_states = decoder_layer(...)
    # But CompressedDecoderLayer returns a tuple -> causes TypeError.
    orig_forwards = []
    for la in layers:
        orig_fwd = la.layer.forward
        orig_forwards.append(orig_fwd)
        def make_tensor_fwd(orig):
            def tensor_fwd(*args, **kwargs):
                out = orig(*args, **kwargs)
                return out[0] if isinstance(out, tuple) else out
            return tensor_fwd
        la.layer.forward = make_tensor_fwd(orig_fwd)

    # --- Step 3: Inject gradient anchor at embedding output ---
    def embed_fwd_hook(module, inp, out):
        anchor = out.detach().requires_grad_(True)
        anchor_holder.clear()
        anchor_holder.append(anchor)
        return anchor

    # --- Step 4: Hook decoder layer outputs to retain grad ---
    def make_layer_hook(idx: int):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.requires_grad:
                h.retain_grad()
            layer_acts[idx] = h
        return hook

    embed_module = model.get_input_embeddings()
    h_embed = embed_module.register_forward_hook(embed_fwd_hook)

    handles_layer = []
    for i, la in enumerate(layers):
        h = la.layer.register_forward_hook(make_layer_hook(i))
        handles_layer.append(h)

    model.eval()
    n_seen = 0

    for batch in dataloader:
        input_ids = batch.get("input_ids", batch.get("labels"))
        if input_ids is None:
            continue
        input_ids = input_ids.to(device)

        anchor_holder.clear()
        layer_acts.clear()

        with torch.enable_grad():
            out  = model(input_ids, labels=input_ids)
            loss = out.loss
            loss.backward()

        for i in range(n_layer):
            if i in layer_acts:
                h = layer_acts[i]
                if h.grad is not None:
                    g = h.grad.detach().float()
                    grad_sq[i] += g.reshape(-1, d).pow(2).mean(0).cpu()

        layer_acts.clear()
        anchor_holder.clear()
        model.zero_grad()
        n_seen += 1
        logging.info(f"collect_gradient_sensitivity: batch {n_seen}/{n_batches}")
        cleanup_memory()
        if n_seen >= n_batches:
            break

    # --- Cleanup: remove hooks, restore forwards, move model back to CPU ---
    h_embed.remove()
    for h in handles_layer:
        h.remove()

    for la, orig_fwd in zip(layers, orig_forwards):
        la.layer.forward = orig_fwd

    for name, p in model.named_parameters():
        p.requires_grad_(orig_req.get(name, True))

    if model_orig_device.type == 'cpu' and config.device.type == 'cuda':
        logging.info("collect_gradient_sensitivity: moving model back to CPU")
        model.to('cpu')
        cleanup_memory()

    denom = max(n_seen, 1)
    return [g / denom for g in grad_sq]


def collect_gradient_sensitivity_layerwise(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader,
    n_batches: int = 8,
) -> list[torch.Tensor]:
    """Memory-efficient gradient sensitivity via layer-by-layer forward + backward.

    Peak GPU memory: O(one_layer_weights + one_batch_activations)
    vs. full-model approach: O(all_layers + all_batch_activations).
    Suitable for 7B/13B models on 32 GB GPU.

    Algorithm:
        Phase 1  Phase 0 & 1: 正向传播与缓存 (Fwd & Cache).Forward pass layer-by-layer; store each layer's input h_l on CPU. 
        Phase 2  Compute initial gradient ∂L/∂h_L via lm_head (only final norm + head on GPU).
        Phase 3  Backward pass layer-by-layer in reverse order:
                 - Load layer_l to GPU, re-run forward with autograd enabled.
                 - Record G_l = E[(∂L/∂h_l)^2] from gradient at layer output.
                 - Propagate gradient to layer input, move layer back to CPU.
    初始梯度 就是最后一层的一个交叉熵求导，然后一层一层反向传播回去
    Returns list of (d,) float32 tensors, one per transformer layer.
    """
    import torch.nn.functional as F

    model   = model_adapter.model
    layers  = model_adapter.get_layers()
    n_layer = len(layers)
    d       = model_adapter.hidden_size
    pos     = layers[0].hidden_states_args_position  # hidden states position in args (usually 0)

    grad_sq = [torch.zeros(d, dtype=torch.float32) for _ in range(n_layer)]

    model.eval()
    orig_req = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Phase 0: collect embedding outputs and batch metadata ─────────────────
    all_inps    : list[torch.Tensor] = []   # embedding outputs (CPU)
    all_args    : list[tuple]        = []   # full args (CPU tensors)
    all_kwargs  : list[dict]         = []   # kwargs (CPU tensors)
    all_labels  : list[torch.Tensor] = []   # input_ids for loss

    n_collected = 0
    for batch in dataloader:
        if n_collected >= n_batches:
            break
        label_ids = batch.get("input_ids", batch.get("labels"))
        all_labels.append(label_ids.cpu())

        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        # Store everything on CPU
        all_inps.append(inp_batch.cpu())
        all_args.append(map_tensors(args_batch, device="cpu"))
        all_kwargs.append(map_tensors(kwargs_batch, device="cpu"))
        n_collected += 1

    n_batches_actual = n_collected

    # ── Phase 1: Forward pass layer-by-layer, store inputs on CPU ─────────────
    # layer_inputs[l][b] = hidden states ENTERING layer l, batch b (CPU)
    layer_inputs: list[list[torch.Tensor]] = [
        [None] * n_batches_actual for _ in range(n_layer)
    ]

    current_inps = [h.clone() for h in all_inps]

    for l, la in enumerate(layers):
        for b in range(n_batches_actual):
            layer_inputs[l][b] = current_inps[b]  # CPU

        la.layer.to(config.device)
        next_inps = []
        for b in range(n_batches_actual):
            h_in    = current_inps[b].to(config.device)
            args_b  = la.get_updated_args(h_in, map_tensors(all_args[b], device=config.device))
            kw_b    = map_tensors(all_kwargs[b], device=config.device)
            with torch.no_grad():
                out = la.layer(*args_b, **kw_b)
            h_out = (out[la.hidden_states_output_position]
                     if isinstance(out, tuple) else out)
            next_inps.append(h_out.cpu())
        la.layer.to("cpu")
        current_inps = next_inps
        cleanup_memory()
        logging.info(f"collect_gradient_sensitivity_layerwise: fwd {l+1}/{n_layer}")

    final_outputs = current_inps  # h_L for each batch (CPU)

    # ── Phase 2: Compute ∂L/∂h_L via lm_head ────────────────────────────────
    pre_head_norm = model_adapter.get_pre_head_layernorm()
    lm_head       = model_adapter.get_lm_head()
    pre_head_norm.to(config.device)
    lm_head.to(config.device)

    initial_grads: list[torch.Tensor] = []
    for b in range(n_batches_actual):
        h_L     = final_outputs[b].to(config.device)
        label_b = all_labels[b].to(config.device)
        h_L_req = h_L.detach().requires_grad_(True)
        with torch.enable_grad():
            normed  = pre_head_norm(h_L_req)
            logits  = lm_head(normed)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = label_b[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
            loss.backward()
        initial_grads.append(h_L_req.grad.cpu())

    pre_head_norm.to("cpu")
    lm_head.to("cpu")
    cleanup_memory()

    # ── Phase 3: Backward pass layer-by-layer in reverse ─────────────────────
    current_grads = initial_grads  # ∂L/∂h_l at current layer's OUTPUT (CPU)

    for l in range(n_layer - 1, -1, -1):
        la = layers[l]
        la.layer.to(config.device)

        next_grads: list[torch.Tensor] = []
        for b in range(n_batches_actual):
            grad_out = current_grads[b].to(config.device)

            # G_l = E[(∂L/∂h_l)^2]  (gradient at this layer's OUTPUT)
            g = grad_out.float().reshape(-1, d)
            grad_sq[l] += g.pow(2).mean(0).cpu()

            # Re-run layer l forward with autograd to get ∂L/∂h_{l-1}
            h_in     = layer_inputs[l][b].to(config.device)
            h_in_req = h_in.detach().requires_grad_(True)
            args_b   = la.get_updated_args(h_in_req, map_tensors(all_args[b], device=config.device))
            kw_b     = map_tensors(all_kwargs[b], device=config.device)
            with torch.enable_grad():
                out  = la.layer(*args_b, **kw_b)
            h_out = out[la.hidden_states_output_position] if isinstance(out, tuple) else out
            h_out.backward(grad_out)
            next_grads.append(h_in_req.grad.cpu())

        la.layer.to("cpu")
        current_grads = next_grads
        cleanup_memory()
        logging.info(f"collect_gradient_sensitivity_layerwise: bwd {l+1}/{n_layer}")

    for name, p in model.named_parameters():
        p.requires_grad_(orig_req.get(name, True))

    denom = max(n_batches_actual, 1)
    return [g / denom for g in grad_sq]



@torch.no_grad()
def gradient_pca_calc(
    X: list[torch.Tensor],
    ignore_masks: list[torch.Tensor] | None,
    grad_sensitivity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient-weighted PCA: maximise Tr(Q^T H_grad Q) on O(d).

    Objective:
        H_grad[i,j] = sqrt(G[i]) * H[i,j] * sqrt(G[j])
    where G[i] = E[(dL/dh)_i^2] is the per-coordinate gradient sensitivity.

    This is a smooth optimisation on the orthogonal group O(d); the exact
    solution is the eigendecomposition of H_grad (standard PCA on the
    gradient-reweighted covariance).  The gradient weighting ensures that
    directions critical for the downstream loss are preserved preferentially.

    Returns (rayleigh_quotients wrt original H, Q_grad d x d orthogonal).
    """
    cleanup_memory()
    H   = _build_H(X, ignore_masks)                                # d x d, float64
    d   = H.shape[0]
    G   = grad_sensitivity.to(device=H.device, dtype=H.dtype)      # d, float64
    G   = G.clamp(min=1e-10)
    g   = G.sqrt()                                                  # d

    # H_grad = diag(g) @ H @ diag(g)
    H_grad = g.unsqueeze(1) * H * g.unsqueeze(0)                   # d x d

    eig_v, eig_vec = torch.linalg.eigh(H_grad)
    order    = torch.argsort(eig_v, descending=True)
    Q_grad   = eig_vec[:, order]                                    # d x d orthogonal

    # Rayleigh quotients w.r.t. the *original* H (for eigenvalue reporting)
    rayleigh = torch.diag(Q_grad.T @ H @ Q_grad)
    return rayleigh, Q_grad

@torch.no_grad()
def oblique_pca_calc(
    X: list[torch.Tensor],
    ignore_masks: list[torch.Tensor] | None,
    grad_sensitivity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oblique projection: returns (eig_val, U_decoder, V_encoder) where V.T @ U = I_d.

    Constructs a biorthogonal pair (U, V) such that the oblique projector P = U[:,m] @ V[:,m].T
    approximates the input distribution while weighting important gradient directions.

    U: d x d decoder  -- right-multiply for input-side weights (W @ U)
    V: d x d encoder  -- left-multiply for output-side weights (V.T @ W)

    Biorthogonality V.T @ U = I ensures V.T is left-inverse of U.

    Algorithm:
        1. Standard PCA: Q = argmax_Q Tr(Q.T H Q), Q orthonormal (variance-preserving)
        2. Gradient-weighted encoder: V_raw = diag(G_norm) @ Q; V = V_raw / col_norms
        3. Biorthogonalization: U = Q @ inv(V.T @ Q) ensures V.T @ U = I
    """
    cleanup_memory()
    eig_val, Q = pca_calc(X, ignore_masks)   # (d,), (d x d) orthonormal, float64
    Q = Q.to(dtype=torch.float64)

    G = grad_sensitivity.to(device=Q.device, dtype=torch.float64).clamp(min=1e-10)
    G_norm = G / G.max()                     # normalize to [0, 1]

    # Gradient-weighted encoder: bias column directions toward high-gradient dims
    V_raw  = G_norm.unsqueeze(1) * Q        # d x d
    col_norms = V_raw.norm(dim=0, keepdim=True).clamp(min=1e-8)
    V = V_raw / col_norms                   # d x d, unit-norm columns

    # Biorthogonalize so that V.T @ U ~ I (with Tikhonov regularization for stability)
    # V.T @ Q = Q.T @ diag(G_norm) @ Q / col_norms (diagonal-dominant, well-conditioned for smooth G)
    VtQ = V.T @ Q                           # d x d

    # Regularized inverse: (VtQ + eps*I)^{-1} to bound condition number
    # This gives V.T @ U ~ I (exact when eps=0)
    eps = 1e-3
    try:
        U_raw = Q @ torch.linalg.inv(VtQ + eps * torch.eye(VtQ.shape[0], dtype=VtQ.dtype, device=VtQ.device))
    except torch.linalg.LinAlgError:
        logging.warning("oblique_pca_calc: inversion failed, falling back to standard PCA")
        return eig_val, Q, Q

    # Clip U column norms: if any column of U has norm > max_norm, scale it back.
    # This prevents weight explosion while approximately preserving biorthogonality.
    max_norm = 3.0
    u_col_norms = U_raw.norm(dim=0, keepdim=True)   # (1, d)
    scale = torch.clamp(u_col_norms / max_norm, min=1.0)  # scale >= 1 (only shrink large cols)
    U = U_raw / scale                                # d x d, bounded norms

    # Log condition numbers for diagnostics
    cond = float(u_col_norms.max() / u_col_norms.clamp(min=1e-10).min())
    if cond > 10.0:
        logging.warning(f"oblique_pca_calc: U col-norm ratio = {cond:.1f} (>10), eps clipping active")

    return eig_val, U, V


@torch.no_grad()
def collect_layer_pca_eigenvalues(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    *,
    apply_mask: bool = True,
) -> list[torch.Tensor]:
    """Collect PCA eigenvalue spectra for each layer WITHOUT any rotation or slicing.

    Runs calibration data through the unmodified original model and computes the
    covariance eigenvalues at each layer output (mlp output → next layer input).
    These uncontaminated eigenvalues give a reliable picture of each layer's
    intrinsic information content for energy-based dimension allocation.

    Returns:
        List of (d,) tensors, one per layer, eigenvalues sorted descending.
    """
    model_adapter.model.eval() 
    dtype = next(iter(model_adapter.model.parameters())).dtype # dtype 是为了后续可能的矩阵计算保持一致，虽然 PCA 计算内部会转换为 float64 以提高数值稳定性。

    inps, args, kwargs, ignore_masks = [], [], [], [] 
    # inps是每层的输入，args和kwargs是对应的参数，ignore_masks是可选的注意力掩码列表，用于在计算协方差时忽略填充位置。
    # inp hidden_states 是一个 batch 的输入，shape 通常是 (batch_size, seq_len, hidden_size)
    # args , 通常是attention_mask, position_ids
    # kwargs, 通常是use_cache, output_attentions等
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers() 
    # 返回的是一个包含所有 Transformer 层（Layers/Blocks） 的容器（通常是 torch.nn.ModuleList）

    layer_eig_vals: list[torch.Tensor] = []

    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Collecting PCA eigvals")):
        # Forward pass through this layer to get outputs
        _, inps = get_signals(layer_adapter, args, kwargs)  # get_signals 执行了 layer_adapter.forward()。
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])

        # Compute covariance of layer output activations
        eig_val, _ = pca_calc(inps, ignore_masks)   # 计算 PCA 特征值
        layer_eig_vals.append(eig_val.detach().cpu())
        cleanup_memory()
 
    return layer_eig_vals

def collect_layer_fisher_eigenvalues(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    *,
    apply_mask: bool = True,
    max_layers: int | None = None,
) -> dict[int, torch.Tensor]:
    """Collect Fisher eigenvalue spectra for each layer without slicing."""
    model_adapter.model.eval()

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    layers_to_process = len(layers) if max_layers is None else min(len(layers), int(max_layers))
    fisher_eigs: dict[int, torch.Tensor] = {}

    for idx in tqdm(range(layers_to_process), unit="layer", desc="Collecting Fisher eigvals"):
        layer_adapter = layers[idx]

        for batch_idx, inp in enumerate(inps):
            args[batch_idx] = layer_adapter.get_updated_args(inp, args[batch_idx])

        eig_val, _ = fisher_calc_real(
            model_adapter,
            idx,
            inps,
            args,
            kwargs,
            dataloader,
            target_norm_getter=lambda la: la.get_second_layernorm(),
            ignore_masks=ignore_masks,
        )
        fisher_eigs[idx] = eig_val.detach().cpu()

        _, inps = get_signals(layer_adapter, args, kwargs)
        cleanup_memory()

    return fisher_eigs


@torch.no_grad()
def rotate(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    collect_info: bool = False,
) -> list[dict[str, torch.Tensor]] | None:
    # 保持原样，只做 PCA 旋转。collect_info=True 时额外收集每层特征值与权重列范数。
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype 

    layers = model_adapter.get_layers()
    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    layer_info = [] if collect_info else None

    _, Q_1 = pca_calc(inps)
    Q_1 = Q_1.to(device=config.device)
    rotate_embeddings(model_adapter, Q_1)

    logging.info("Rotate layers")
    for layer_adapter in tqdm(layers, unit="layer", desc="Rotating"):
        layer = layer_adapter.layer
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
        mlp_ln_inputs, outs = get_signals(layer_adapter, args, kwargs)
        
        _, Q_3 = pca_calc(mlp_ln_inputs)
        Q_3 = Q_3.to(device=config.device)
        eig_val_5, Q_5 = pca_calc(outs)
        Q_5 = Q_5.to(device=config.device)

        rotate_attention_inputs(layer_adapter, Q_1)
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype))
        rotate_attention_output(layer_adapter, Q_3)
        rotate_mlp_input(layer_adapter, Q_3)
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype))
        rotate_mlp_output(layer_adapter, Q_5)

        if collect_info and layer_info is not None:
            W_out = layer_adapter.get_mlp_output().weight.data.float()
            col_norms_sq = (W_out ** 2).sum(dim=0).cpu()
            layer_info.append({
                "eig_val": eig_val_5.cpu(),
                "w_col_norms_sq": col_norms_sq,
            })

        cleanup_memory()

        inps = outs 
        Q_1 = Q_5 

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")
    return layer_info if collect_info else None

def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    # 保持原样
    model_adapter.model.eval()
    layers = model_adapter.get_layers()
    if not slicing_scheduler:
        if model_adapter.slicing_conf.const_dimension is not None:
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))
        if not model_adapter.parallel_blocks:
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[: slicing_scheduler.get_mlp_input_dimension(i), :])
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

        if model_adapter.parallel_blocks:
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)
            )
        else:
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)]
            )
            layer.mlp_shortcut_Q = nn.Parameter(
                layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(i)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)
            )

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

# ============================================================
# 改进 Fisher 方法 1: Fisher Diagonal Fast (加速版)
# ============================================================

@torch.no_grad()
def fisher_diag_fast_calc(
    model_adapter: ModelAdapter,
    layer_idx: int,
    inps: list[torch.Tensor],
    args: list[tuple],
    kwargs: list[dict],
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    target_norm_getter=None,
    ignore_masks: list[torch.Tensor] | None = None,
    n_samples: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Fisher Diagonal Fast Calculation
    
    核心改进：
    1. 只用 n_samples=32 个样本估计对角线（vs 全部 128 样本）
    2. 基于 Fisher 对角线重排序 PCA 特征向量
    
    优势：比完整 Fisher 快 2-3 倍，性能接近
    '''
    import torch
    from torch import nn
    cleanup_memory()
    
    # 第一步：快速计算 Fisher 对角线
    fisher_diag = None
    all_layers = model_adapter.get_layers()
    subsequent_layers = all_layers[layer_idx:]
    pre_head_ln = model_adapter.get_pre_head_layernorm()
    lm_head = model_adapter.get_lm_head()
    
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to(config.device)
    if pre_head_ln is not None:
        pre_head_ln.to(config.device)
    lm_head.to(config.device)
    
    data_iter = iter(dataloader)
    
    for i, X in enumerate(inps[:n_samples]):
        model_adapter.model.zero_grad(set_to_none=True)
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        
        labels = batch.get('labels', batch.get('input_ids'))
        if labels is None: continue
        labels = labels.to(config.device)
        
        X_clone = X.detach().clone().to(config.device).requires_grad_(True)
        batch_args = map_tensors(args[i], device=config.device)
        batch_kwargs = map_tensors(kwargs[i], device=config.device)
        
        def run_tail(hidden_state):
            h = hidden_state
            for layer_adapter in subsequent_layers:
                layer_args_updated = layer_adapter.get_updated_args(h, batch_args)
                out = layer_adapter.layer(*layer_args_updated, **batch_kwargs)
                h = out[layer_adapter.hidden_states_output_position] if isinstance(out, tuple) else out
            if pre_head_ln is not None:
                h = pre_head_ln(h)
            return lm_head(h)
        
        target_grad, target_input, hook_handles = [], [], []
        
        with torch.enable_grad():
            if target_norm_getter is not None and len(subsequent_layers) > 0:
                target_module = target_norm_getter(subsequent_layers[0])
                hook_handles.append(target_module.register_full_backward_hook(
                    lambda m, gi, go: target_grad.append(gi[0].detach()) if gi and gi[0] is not None else None
                ))
                hook_handles.append(target_module.register_forward_hook(
                    lambda m, i, o: target_input.append(i[0].detach()) if i and i[0] is not None else None
                ))
            
            logits = run_tail(X_clone)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]
            
            loss = nn.CrossEntropyLoss()(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            loss.backward()
        
        for h in hook_handles: h.remove()
        
        if target_grad and target_input:
            g = target_grad[0].double().reshape(-1, target_grad[0].shape[-1])
            x = target_input[0].double().reshape(-1, target_input[0].shape[-1])
            diag_batch = torch.mean((g * x)**2, dim=0)
            fisher_diag = diag_batch if fisher_diag is None else fisher_diag + diag_batch
    
    if fisher_diag is None:
        for layer_adapter in subsequent_layers:
            layer_adapter.layer.to('cpu')
        if pre_head_ln is not None:
            pre_head_ln.to('cpu')
        lm_head.to('cpu')
        return pca_calc(inps, ignore_masks)
    
    fisher_diag = fisher_diag / len(inps[:n_samples])
    
    # 第二步：计算标准 PCA 并用 Fisher 对角线重排序
    H = None
    total_samples = 0
    for idx, X_batch in enumerate(inps):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.double()
        n_samples_batch = X_batch.shape[0] * X_batch.shape[1]
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H + H_batch
        total_samples += n_samples_batch
    
    H = H / total_samples
    
    damp = 0.01 * torch.mean(torch.diag(H))
    diag_indices = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag_indices, diag_indices] = H[diag_indices, diag_indices] + damp
    
    eig_val, eig_vec = torch.linalg.eigh(H)
    
    fisher_diag = fisher_diag.to(device=config.device).double()
    d_in = eig_vec.shape[0]
    
    fisher_scores = []
    for i in range(d_in):
        vec_i = eig_vec[:, i]
        score = torch.sum(vec_i ** 2 * fisher_diag).item()
        fisher_scores.append(score)
    
    fisher_scores = torch.tensor(fisher_scores)
    fisher_index = torch.argsort(fisher_scores, descending=True)
    eig_val = eig_val[fisher_index]
    eig_vec = eig_vec[:, fisher_index]
    
    del H, fisher_diag
    cleanup_memory()
    
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to('cpu')
    if pre_head_ln is not None:
        pre_head_ln.to('cpu')
    lm_head.to('cpu')
    
    return eig_val, eig_vec


# ============================================================
# 改进 Fisher 方法 2: Fisher Saliency (更好的信息压缩)
# ============================================================

def fisher_saliency_calc(
    model_adapter: ModelAdapter,
    layer_idx: int,
    inps: list[torch.Tensor],
    args: list[tuple],
    kwargs: list[dict],
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    target_norm_getter=None,
    ignore_masks: list[torch.Tensor] | None = None,
    saliency_alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Saliency-Weighted Fisher Calculation
    
    核心改进：
    1. Fisher 权重：梯度范数（告诉哪些维度对损失重要）
    2. Saliency 权重：|激活 * 梯度|（告诉哪些维度活跃）
    3. 组合权重：(1-α) * Fisher + α * Saliency
    
    理论：既重要又活跃的方向应该优先保留
    '''
    import torch
    from torch import nn
    cleanup_memory()
    H = None
    all_layers = model_adapter.get_layers()
    subsequent_layers = all_layers[layer_idx:]
    pre_head_ln = model_adapter.get_pre_head_layernorm()
    lm_head = model_adapter.get_lm_head()
    
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to(config.device)
    if pre_head_ln is not None:
        pre_head_ln.to(config.device)
    lm_head.to(config.device)
    
    data_iter = iter(dataloader)
    
    for i, X in enumerate(inps):
        model_adapter.model.zero_grad(set_to_none=True)
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        
        labels = batch.get('labels', batch.get('input_ids'))
        if labels is None: continue
        labels = labels.to(config.device)
        
        X_clone = X.detach().clone().to(config.device).requires_grad_(True)
        batch_args = map_tensors(args[i], device=config.device)
        batch_kwargs = map_tensors(kwargs[i], device=config.device)
        
        def run_tail(hidden_state):
            h = hidden_state
            for layer_adapter in subsequent_layers:
                layer_args_updated = layer_adapter.get_updated_args(h, batch_args)
                out = layer_adapter.layer(*layer_args_updated, **batch_kwargs)
                h = out[layer_adapter.hidden_states_output_position] if isinstance(out, tuple) else out
            if pre_head_ln is not None:
                h = pre_head_ln(h)
            return lm_head(h)
        
        target_grad, target_input, hook_handles = [], [], []
        
        with torch.enable_grad():
            if target_norm_getter is not None and len(subsequent_layers) > 0:
                target_module = target_norm_getter(subsequent_layers[0])
                hook_handles.append(target_module.register_full_backward_hook(
                    lambda m, gi, go: target_grad.append(gi[0].detach()) if gi and gi[0] is not None else None
                ))
                hook_handles.append(target_module.register_forward_hook(
                    lambda m, i, o: target_input.append(i[0].detach()) if i and i[0] is not None else None
                ))
            
            logits = run_tail(X_clone)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]
            
            loss = nn.CrossEntropyLoss()(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            loss.backward()
        
        for h in hook_handles: h.remove()
        
        if target_grad and target_input:
            g = target_grad[0].double().reshape(-1, target_grad[0].shape[-1])
            x = target_input[0].double().reshape(-1, target_input[0].shape[-1])
            
            w_fisher = torch.norm(g, dim=1, keepdim=True)
            w_fisher = w_fisher / (w_fisher.mean() + 1e-8) if w_fisher.mean() > 1e-8 else w_fisher
            
            w_saliency = torch.abs(x * g).mean(dim=1, keepdim=True)
            w_saliency = w_saliency / (w_saliency.mean() + 1e-8) if w_saliency.mean() > 1e-8 else w_saliency
            
            w_combined = (1 - saliency_alpha) * w_fisher + saliency_alpha * w_saliency
            
            x_weighted = x * torch.sqrt(w_combined)
            H_batch = x_weighted.T @ x_weighted
        else:
            x = target_input[0] if target_input else X
            x = x.reshape(-1, x.shape[-1]).double().to(config.device)
            H_batch = x.T @ x
        
        H = H_batch if H is None else H + H_batch
    
    if H is None: raise ValueError('H is None! No batches processed?')
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    
    H_cpu = H.to('cpu')
    del H
    cleanup_memory()
    
    eig_val, eig_vec = torch.linalg.eigh(H_cpu)
    Q_fish = eig_vec.flip(1)
    eig_val = eig_val.flip(0)
    del H_cpu
    cleanup_memory()
    
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to('cpu')
    if pre_head_ln is not None:
        pre_head_ln.to('cpu')
    lm_head.to('cpu')
    
    return eig_val, Q_fish


# ============================================================
# 改进 Fisher 方法（预收集版本 - 避免维度不匹配）
# ============================================================



def fisher_fast_pca_calc(X, ignore_masks=None, fisher_diagonal=None):
    import torch
    
    config_device = X[0].device
    
    H = None
    total_samples = 0
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.double()
        n_samples = X_batch.shape[0] * X_batch.shape[1]
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H + H_batch
        total_samples += n_samples
    
    H = H / total_samples
    
    damp = 0.01 * torch.mean(torch.diag(H))
    diag_indices = torch.arange(H.shape[-1]).to(device=config_device)
    H[diag_indices, diag_indices] = H[diag_indices, diag_indices] + damp
    
    X_eig = torch.linalg.eigh(H)
    eig_val = X_eig[0]
    eigen_vec = X_eig[1]
    
    if fisher_diagonal is not None:
        fisher_diagonal = fisher_diagonal.to(device=config_device).double()
        
        d_in = eigen_vec.shape[0]
        fisher_scores = []
        
        for i in range(d_in):
            vec_i = eigen_vec[:, i]
            score = torch.sum(vec_i ** 2 * fisher_diagonal).item()
            fisher_scores.append(score)
        
        fisher_scores = torch.tensor(fisher_scores)
        fisher_index = torch.argsort(fisher_scores, descending=True)
        eig_val = eig_val[fisher_index]
        eigen_vec = eigen_vec[:, fisher_index]
    else:
        index = torch.argsort(eig_val, descending=True)
        eig_val = eig_val[index]
        eigen_vec = eigen_vec[:, index]
    
    del H
    
    return eig_val, eigen_vec


def fisher_saliency_pca_calc(X, ignore_masks=None, fisher_diagonal=None, saliency_diagonal=None, alpha=0.3):
    import torch
    
    config_device = X[0].device
    
    H = None
    total_samples = 0
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.double()
        n_samples = X_batch.shape[0] * X_batch.shape[1]
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H + H_batch
        total_samples += n_samples
    
    H = H / total_samples
    
    if fisher_diagonal is not None and saliency_diagonal is not None:
        fisher_diagonal = fisher_diagonal.to(device=config_device).double()
        saliency_diagonal = saliency_diagonal.to(device=config_device).double()
        
        fisher_norm = fisher_diagonal / (fisher_diagonal.mean() + 1e-8)
        saliency_norm = saliency_diagonal / (saliency_diagonal.mean() + 1e-8)
        
        combined_weight = (1 - alpha) * fisher_norm + alpha * saliency_norm
        
        H_weighted = H * torch.diag(combined_weight)
        
        damp = 0.01 * torch.mean(torch.diag(H_weighted))
        diag_indices = torch.arange(H_weighted.shape[-1]).to(device=config_device)
        H_weighted[diag_indices, diag_indices] = H_weighted[diag_indices, diag_indices] + damp
        
        X_eig = torch.linalg.eigh(H_weighted)
        eig_val = X_eig[0]
        eigen_vec = X_eig[1]
        
        del H_weighted
    else:
        damp = 0.01 * torch.mean(torch.diag(H))
        diag_indices = torch.arange(H.shape[-1]).to(device=config_device)
        H[diag_indices, diag_indices] = H[diag_indices, diag_indices] + damp
        
        X_eig = torch.linalg.eigh(H)
        eig_val = X_eig[0]
        eigen_vec = X_eig[1]
    
    del H
    
    index = torch.argsort(eig_val, descending=True)
    eig_val = eig_val[index]
    eigen_vec = eigen_vec[:, index]
    
    return eig_val, eigen_vec
