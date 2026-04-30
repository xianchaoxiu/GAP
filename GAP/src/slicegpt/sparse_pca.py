"""Alternative PCA rotation methods for SliceGPT.

Methods:
  admm_sparse_pca  -- ADMM on Stiefel + L2,1 (proper optimization)
                      min_{W in St(d,m)} -Tr(W^T H W) + lambda * ||W||_{2,1}
                      W-step: Riemannian GD (QR retraction)
                      Z-step: row soft-thresholding (prox of L_{2,1})
  sparse_pca       -- L2,1 importance-weighted PCA (legacy, deprecated)
  shrinkage_pca    -- Ledoit-Wolf shrinkage PCA
"""

import torch


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _standard_pca(H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (eig_val, Q) sorted descending."""
    eig_v, eig_vec = torch.linalg.eigh(H)
    order = torch.argsort(eig_v, descending=True)
    return eig_v[order], eig_vec[:, order]


def _qr_retract(W: torch.Tensor) -> torch.Tensor:
    """QR retraction onto Stiefel manifold. Fixes sign so diag(R) > 0."""
    Q, R = torch.linalg.qr(W)
    signs = R.diagonal().sign()
    signs[signs == 0] = 1.0
    return Q * signs.unsqueeze(0)   # d x m


def _riem_grad(H: torch.Tensor, W: torch.Tensor,
               A: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Riemannian gradient on Stiefel of:
        f(W) = -Tr(W^T H W) + rho/2 * ||W - A||_F^2

    Euclidean gradient:  grad_euc = -2HW + rho(W - A)
    Riemannian gradient: grad_euc - W * sym(W^T grad_euc)
    """
    g = -2.0 * (H @ W) + rho * (W - A)
    WtG = W.T @ g                            # m x m
    return g - W @ ((WtG + WtG.T) * 0.5)    # d x m


def _w_step(H: torch.Tensor, W: torch.Tensor, A: torch.Tensor,
            rho: float, n_inner: int, lr: float) -> torch.Tensor:
    """Riemannian gradient descent + QR retraction for the W-step."""
    for _ in range(n_inner):
        rg = _riem_grad(H, W, A, rho)
        W = _qr_retract(W - lr * rg)
    return W


def _z_step(R: torch.Tensor, thresh: float) -> torch.Tensor:
    """
    Row-wise soft thresholding: prox_{thresh * ||.||_{2,1}}(R)
        Z_i = max(0, 1 - thresh / ||R_i||) * R_i
    Closed-form proximal operator of L_{2,1} norm.
    """
    row_norms = R.norm(dim=1, keepdim=True).clamp(min=1e-12)  # d x 1
    scale = (1.0 - thresh / row_norms).clamp(min=0.0)
    return scale * R


def _extend_to_orthogonal(W_sparse: torch.Tensor) -> torch.Tensor:
    """
    Given W_sparse (d x m, first m columns of desired Q),
    fill the remaining d-m columns with random complement via QR.
    Returns d x d orthogonal Q.
    """
    d, m = W_sparse.shape
    complement = torch.randn(d, d - m, dtype=W_sparse.dtype, device=W_sparse.device)
    full = torch.cat([W_sparse, complement], dim=1)   # d x d
    return _qr_retract(full)


# ---------------------------------------------------------------------------
# Main: ADMM on Stiefel + L2,1
# ---------------------------------------------------------------------------

def admm_sparse_pca(
    H: torch.Tensor,
    m: int,
    lambda_: float = 0.01,
    rho: float = 1.0,
    max_outer: int = 100,
    n_inner: int = 5,
    lr: float | None = None,
    eps_converge: float = 1e-5,
    eps_row: float = 1e-3,
    adaptive_rho: bool = True,
    mu: float = 10.0,
    tau: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve:
        min_{W in St(d,m)} -Tr(W^T H W) + lambda * ||W||_{2,1}

    via ADMM splitting  W = Z:

        W-step : Riemannian GD on Stiefel (QR retraction)
        Z-step : row-wise soft thresholding  (closed-form prox of L_{2,1})
        U-step : dual ascent

    After convergence:
        S      = {i : ||W_i||_2 > eps_row}   (support set)
        Stage2 : standard PCA restricted to H[S,:][:,S]
        Stage3 : zero-pad V_S to d x m, QR-extend to d x d orthogonal Q

    Returns
    -------
    rayleigh : (d,) Rayleigh quotients  diag(Q^T H Q)  (proxy for eigenvalues)
    Q        : (d,d) orthogonal matrix  (first m cols are row-sparse w.r.t. S)
    """
    H = H.double()
    d = H.shape[0]

    # --- Initialise W with top-m eigenvectors of H ---
    eig_v, eig_vec = torch.linalg.eigh(H)
    order = torch.argsort(eig_v, descending=True)
    W = eig_vec[:, order[:m]].contiguous()   # d x m

    Z = W.clone()
    U = torch.zeros_like(W)

    # Auto learning rate: step size ~ 1 / (2*lambda_max(H) + rho)
    if lr is None:
        lmax = eig_v[-1].item()          # largest eigenvalue (H is PSD)
        lr = 1.0 / (2.0 * lmax + rho)

    # --- ADMM outer loop ---
    for k in range(max_outer):
        Z_old = Z.clone()

        # W-step: Riemannian GD
        A = Z - U
        W = _w_step(H, W, A, rho, n_inner, lr)

        # Z-step: row soft-thresholding
        R = W + U
        Z = _z_step(R, thresh=lambda_ / rho)

        # U-step: dual ascent
        U = U + W - Z

        # Residuals
        primal_res = (W - Z).norm().item()
        dual_res   = (rho * (Z - Z_old)).norm().item()

        # Adaptive rho (standard ADMM heuristic)
        if adaptive_rho:
            if primal_res > mu * dual_res:
                rho *= tau
                U   /= tau
                lr   = 1.0 / (2.0 * eig_v[-1].item() + rho)
            elif dual_res > mu * primal_res:
                rho /= tau
                U   *= tau
                lr   = 1.0 / (2.0 * eig_v[-1].item() + rho)

        if primal_res < eps_converge and dual_res < eps_converge:
            break

    # --- Stage 2: support set + restricted PCA ---
    row_norms = W.norm(dim=1)              # d
    support   = (row_norms > eps_row).nonzero(as_tuple=True)[0]

    if len(support) >= m:
        H_S = H[support][:, support]
        ev_S, evec_S = torch.linalg.eigh(H_S)
        ord_S = torch.argsort(ev_S, descending=True)
        V_S = evec_S[:, ord_S[:m]]        # |S| x m

        W_sparse = torch.zeros(d, m, dtype=H.dtype, device=H.device)
        W_sparse[support] = V_S
    else:
        # Fallback: standard PCA (support collapsed below m)
        W_sparse = eig_vec[:, order[:m]]

    # --- Stage 3: QR-extend to d x d ---
    Q = _extend_to_orthogonal(W_sparse)   # d x d

    # Rayleigh quotients as proxy for eigenvalues
    rayleigh = torch.diag(Q.T @ H @ Q)

    return rayleigh, Q


# ---------------------------------------------------------------------------
# Legacy methods (kept for reference, mostly failed)
# ---------------------------------------------------------------------------

def sparse_pca(H, m, lambda_=0.5, eps=1e-8):
    """L2,1-inspired importance-WEIGHTED PCA. Result: ~44 PPL. Deprecated."""
    eig_val_std, Q_std = _standard_pca(H)
    W_top = Q_std[:, :m]
    row_imp = (W_top ** 2).sum(dim=1)
    row_imp = row_imp / (row_imp.max() + eps)
    w = (row_imp + eps) ** (lambda_ / 2.0)
    H_w = w.unsqueeze(1) * H * w.unsqueeze(0).T
    eig_val_w, Q_w = _standard_pca(H_w)
    rayleigh = torch.diag(Q_w.T @ H @ Q_w)
    return rayleigh, Q_w


def shrinkage_pca(H, m, alpha=0.1, eps=1e-8):
    """Ledoit-Wolf shrinkage PCA. Result: ~18.2 PPL. Deprecated."""
    d = H.shape[0]
    mu = H.trace() / d
    H_s = (1 - alpha) * H + alpha * mu * torch.eye(d, dtype=H.dtype, device=H.device)
    eig_val_s, Q_s = _standard_pca(H_s)
    rayleigh = torch.diag(Q_s.T @ H @ Q_s)
    return rayleigh, Q_s
