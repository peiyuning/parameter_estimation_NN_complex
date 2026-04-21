#!/usr/bin/env python
"""在重新生成的轨迹上评估训练好的 non-IID pair-feature 估计器。"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_VALUE = 0.0
TRAIN_DTYPE = torch.float32


def _project_root() -> Path:
    root = Path.cwd().resolve()
    if root.name in {"notebooks", "train_notebooks", "executed_notebook", "scripts"}:
        root = root.parent
    return root


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tools import (  # noqa: E402
    build_noniid_three_level_kernel_model,
    evaluate_jump_kernel_matrix,
    kernel_stationary_distribution,
    trapz_integral,
)

DATA_ROOT = Path(os.environ.get("PARAMEST_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
OUTPUT_ROOT = Path(os.environ.get("PARAMEST_RESULT_ROOT", PROJECT_ROOT / "data" / "result_noniid")).resolve()
FIGURE_DIR = OUTPUT_ROOT / "figures"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODELS = ("v", "shelving", "lambda", "ladder")
MODEL_NAMES = tuple(
    x.strip().lower()
    for x in os.environ.get("PARAMEST_RESULT_MODELS", ",".join(DEFAULT_MODELS)).split(",")
    if x.strip()
)
SEED = int(os.environ.get("PARAMEST_RESULT_SEED", "42"))
TRAIN_FRACTION = float(os.environ.get("PARAMEST_TRAIN_FRACTION", "0.8"))
BATCH_SIZE = int(os.environ.get("PARAMEST_RESULT_BATCH_SIZE", "8192"))
MAX_EVAL_TRAJ = int(os.environ.get("PARAMEST_RESULT_MAX_TRAJ", "0"))
SCATTER_POINTS = int(os.environ.get("PARAMEST_RESULT_SCATTER_POINTS", "5000"))
BOUNDS_ENABLED = os.environ.get("PARAMEST_NONIID_BOUNDS", "1") == "1"
BOUND_MAX_THETA = int(os.environ.get("PARAMEST_BOUND_MAX_THETA", "6"))
BOUND_N_TRAJ = int(os.environ.get("PARAMEST_BOUND_N_TRAJ", "384"))
BOUND_LENGTHS = tuple(
    int(x)
    for x in os.environ.get("PARAMEST_BOUND_LENGTHS", "50").split(",")
    if x.strip()
)
BOUND_FD_DELTA = float(os.environ.get("PARAMEST_BOUND_FD_DELTA", "0.02"))
BOUND_FD_OMEGA = float(os.environ.get("PARAMEST_BOUND_FD_OMEGA", "0.05"))
BOUND_GRID_SIZE = int(os.environ.get("PARAMEST_BOUND_GRID_SIZE", "2048"))
BOUND_TAU_MAX_INIT = float(os.environ.get("PARAMEST_BOUND_TAU_MAX_INIT", "40.0"))
BOUND_TAU_MAX_LIMIT = float(os.environ.get("PARAMEST_BOUND_TAU_MAX_LIMIT", "2048.0"))
BOUND_DT_MAX = float(os.environ.get("PARAMEST_BOUND_DT_MAX", "0.5"))
BOUND_CAPTURE_TOL = float(os.environ.get("PARAMEST_BOUND_CAPTURE_TOL", "1e-3"))
BOUND_TRANSITION_TOL = float(os.environ.get("PARAMEST_BOUND_TRANSITION_TOL", "1e-3"))
BOUND_FISHER_STABILITY_TOL = float(os.environ.get("PARAMEST_BOUND_FISHER_STABILITY_TOL", "2e-2"))
BOUND_BCRB_STABILITY_TOL = float(os.environ.get("PARAMEST_BOUND_BCRB_STABILITY_TOL", "2e-2"))
DENSITY_EPS = float(os.environ.get("PARAMEST_DENSITY_EPS", "1e-300"))
ROUND_OFF_TOL = float(os.environ.get("PARAMEST_ROUND_OFF_TOL", "1e-10"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_THREADS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))

np.random.seed(SEED)
torch.manual_seed(SEED)


THREE_LEVEL_DEFAULTS = {
    "shelving": {
        "gamma_det": 1.0,
        "gamma_hidden": 0.20,
        "gamma_aux": 0.05,
        "delta_aux": 0.0,
        "omega_aux": 0.0,
    },
    "lambda": {
        "gamma_det": 1.0,
        "gamma_hidden": 0.20,
        "gamma_aux": 0.05,
        "delta_aux": 0.0,
        "omega_aux": 0.35,
    },
    "v": {
        "gamma_det": 1.0,
        "gamma_hidden": 0.20,
        "gamma_aux": 0.05,
        "delta_aux": 0.8,
        "omega_aux": 0.55,
    },
    "ladder": {
        "gamma_det": 1.0,
        "gamma_hidden": 0.20,
        "gamma_aux": 0.05,
        "delta_aux": 0.8,
        "omega_aux": 0.55,
    },
}


def _init_probe_values(init, K: int, dtype: torch.dtype) -> torch.Tensor:
    init_t = torch.as_tensor(init, dtype=dtype)
    if init_t.ndim == 1 and init_t.numel() == K:
        return init_t.clone()
    if init_t.ndim == 1 and init_t.numel() == 2:
        return torch.linspace(init_t[0], init_t[1], K, dtype=dtype)
    raise ValueError(f"Expected probe initializer with shape ({K},) or (2,), got {tuple(init_t.shape)}")


def _positive_raw_from_init(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = torch.clamp(x - eps, min=1e-6)
    return torch.log(torch.expm1(x))


class OneAndPairLaplaceFeatures(nn.Module):
    """单点与相邻 pair 的经验复 Laplace 特征。"""

    def __init__(
        self,
        K_one: int = 4,
        M_pair: int = 4,
        alpha_init=(0.2, 0.8),
        beta_init=(-2.0, 2.0),
        pair_alpha_init=(0.2, 0.8),
        pair_beta_init=(-2.0, 2.0),
        eps_alpha: float = 1e-4,
    ):
        super().__init__()
        self.K_one = int(K_one)
        self.M_pair = int(M_pair)
        self.eps_alpha = float(eps_alpha)

        alpha0 = _init_probe_values(alpha_init, self.K_one, TRAIN_DTYPE)
        beta0 = _init_probe_values(beta_init, self.K_one, TRAIN_DTYPE)
        self.raw_alpha = nn.Parameter(_positive_raw_from_init(alpha0, self.eps_alpha))
        self.beta = nn.Parameter(beta0)

        pair_alpha0 = _init_probe_values(pair_alpha_init, self.M_pair, TRAIN_DTYPE)
        pair_beta0 = _init_probe_values(pair_beta_init, self.M_pair, TRAIN_DTYPE)
        self.raw_pair_alpha_left = nn.Parameter(_positive_raw_from_init(pair_alpha0, self.eps_alpha))
        self.raw_pair_alpha_right = nn.Parameter(_positive_raw_from_init(torch.flip(pair_alpha0, dims=(0,)), self.eps_alpha))
        self.pair_beta_left = nn.Parameter(pair_beta0.clone())
        self.pair_beta_right = nn.Parameter(torch.flip(pair_beta0, dims=(0,)).clone())

        self.output_dim = 2 * self.K_one + 2 * self.M_pair + 2

    def one_parameters(self):
        alpha = F.softplus(self.raw_alpha) + self.eps_alpha
        return alpha, self.beta

    def pair_parameters(self):
        alpha_left = F.softplus(self.raw_pair_alpha_left) + self.eps_alpha
        alpha_right = F.softplus(self.raw_pair_alpha_right) + self.eps_alpha
        return alpha_left, self.pair_beta_left, alpha_right, self.pair_beta_right

    def forward(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        taus = taus.to(dtype=self.raw_alpha.dtype)
        mask = mask.bool()
        mask_f = mask.to(dtype=taus.dtype)

        N = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        alpha, beta = self.one_parameters()
        tau = taus.unsqueeze(-1)
        one_mask = mask_f.unsqueeze(-1)
        one_amp = torch.exp(-tau * alpha)
        one_phase = tau * beta
        one_re = (one_amp * torch.cos(one_phase) * one_mask).sum(dim=1) / N
        one_im = (one_amp * torch.sin(one_phase) * one_mask).sum(dim=1) / N

        if taus.shape[1] > 1:
            left_tau = taus[:, :-1].unsqueeze(-1)
            right_tau = taus[:, 1:].unsqueeze(-1)
            pair_mask = (mask[:, :-1] & mask[:, 1:]).to(dtype=taus.dtype).unsqueeze(-1)
            N_pair = pair_mask.sum(dim=1).clamp_min(1.0)
            alpha_l, beta_l, alpha_r, beta_r = self.pair_parameters()
            pair_amp = torch.exp(-left_tau * alpha_l - right_tau * alpha_r)
            pair_phase = left_tau * beta_l + right_tau * beta_r
            pair_re = (pair_amp * torch.cos(pair_phase) * pair_mask).sum(dim=1) / N_pair
            pair_im = (pair_amp * torch.sin(pair_phase) * pair_mask).sum(dim=1) / N_pair
            pair_len = torch.rsqrt(N_pair)
        else:
            B = taus.shape[0]
            pair_re = torch.zeros((B, self.M_pair), dtype=taus.dtype, device=taus.device)
            pair_im = torch.zeros((B, self.M_pair), dtype=taus.dtype, device=taus.device)
            pair_len = torch.ones((B, 1), dtype=taus.dtype, device=taus.device)

        one_len = torch.rsqrt(N)
        return torch.cat([one_re, one_im, pair_re, pair_im, one_len, pair_len], dim=1)


class NonIIDPairLaplaceEstimator(nn.Module):
    """单点 + 相邻 pair Laplace 特征，然后接 affine readout。"""

    def __init__(
        self,
        target_dim: int,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        K_one: int = 4,
        M_pair: int = 4,
        alpha_init=(0.2, 0.8),
        beta_init=(-2.0, 2.0),
        pair_alpha_init=(0.2, 0.8),
        pair_beta_init=(-2.0, 2.0),
    ):
        super().__init__()
        self.target_dim = int(target_dim)
        self.feat = OneAndPairLaplaceFeatures(
            K_one=K_one,
            M_pair=M_pair,
            alpha_init=alpha_init,
            beta_init=beta_init,
            pair_alpha_init=pair_alpha_init,
            pair_beta_init=pair_beta_init,
        )
        self.in_dim = self.feat.output_dim
        self.head = nn.Linear(self.in_dim, self.target_dim)

        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=TRAIN_DTYPE))
        self.register_buffer("target_std", torch.clamp(torch.as_tensor(target_std, dtype=TRAIN_DTYPE), min=1e-6))

    def encode_targets(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(device=self.target_mean.device, dtype=self.target_mean.dtype)
        return (y - self.target_mean) / self.target_std

    def decode_targets(self, y_std: torch.Tensor) -> torch.Tensor:
        y_std = y_std.to(device=self.target_mean.device, dtype=self.target_mean.dtype)
        return y_std * self.target_std + self.target_mean

    def forward_standardized(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.feat(taus, mask))

    def forward(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decode_targets(self.forward_standardized(taus, mask))

    @torch.no_grad()
    def probe_parameters(self) -> Dict[str, torch.Tensor]:
        one_alpha, one_beta = self.feat.one_parameters()
        pair_alpha_l, pair_beta_l, pair_alpha_r, pair_beta_r = self.feat.pair_parameters()
        return {
            "one_alpha": one_alpha.detach(),
            "one_beta": one_beta.detach(),
            "pair_alpha_left": pair_alpha_l.detach(),
            "pair_beta_left": pair_beta_l.detach(),
            "pair_alpha_right": pair_alpha_r.detach(),
            "pair_beta_right": pair_beta_r.detach(),
        }


def paramblocks(y_raw: np.ndarray) -> np.ndarray:
    if len(y_raw) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    changes = np.any(y_raw[1:] != y_raw[:-1], axis=1)
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    stops = np.r_[starts[1:], len(y_raw)]
    return np.stack([starts, stops], axis=1).astype(np.int64, copy=False)


def validation_indices(y_raw: np.ndarray, train_frac: float, seed: int) -> np.ndarray:
    blocks = paramblocks(y_raw)
    if len(blocks) < 2:
        raise ValueError("Need at least two parameter blocks for validation split.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    n_train = int(round(train_frac * len(blocks)))
    n_train = min(max(n_train, 1), len(blocks) - 1)
    val_blocks = blocks[order[n_train:]]
    return np.concatenate([np.arange(start, stop, dtype=np.int64) for start, stop in val_blocks])


def validation_blocks(y_raw: np.ndarray, train_frac: float, seed: int) -> np.ndarray:
    blocks = paramblocks(y_raw)
    if len(blocks) < 2:
        raise ValueError("Need at least two parameter blocks for validation split.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    n_train = int(round(train_frac * len(blocks)))
    n_train = min(max(n_train, 1), len(blocks) - 1)
    return blocks[order[n_train:]]


def maybe_subsample(indices: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    if max_count <= 0 or len(indices) <= max_count:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_count, replace=False))


def make_model_from_checkpoint(ckpt: Dict) -> NonIIDPairLaplaceEstimator:
    model = NonIIDPairLaplaceEstimator(
        target_dim=int(ckpt["target_dim"]),
        target_mean=ckpt["target_mean"],
        target_std=ckpt["target_std"],
        K_one=int(ckpt["K_one_probes"]),
        M_pair=int(ckpt["M_pair_probes"]),
        alpha_init=tuple(ckpt.get("alpha_init_range", (0.2, 0.8))),
        beta_init=tuple(ckpt.get("beta_init_range", (-2.0, 2.0))),
        pair_alpha_init=tuple(ckpt.get("pair_alpha_init_range", ckpt.get("alpha_init_range", (0.2, 0.8)))),
        pair_beta_init=tuple(ckpt.get("pair_beta_init_range", ckpt.get("beta_init_range", (-2.0, 2.0)))),
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(DEVICE).eval()


def as_batch_arrays(trajectories, params, row_idx: np.ndarray, target_columns: Iterable[int], length: int):
    tau = np.asarray(trajectories[row_idx, :length], dtype=np.float32)
    y = np.asarray(params[row_idx][:, list(target_columns)], dtype=np.float32)
    mask = np.isfinite(tau) & (tau > 0.0)
    tau = np.where(mask, tau, PAD_VALUE).astype(np.float32, copy=False)
    return tau, mask, y


@torch.no_grad()
def evaluate_lengths(model, trajectories, params, indices: np.ndarray, target_columns: List[int], lengths: List[int]):
    rows = []
    max_length = max(lengths)
    sqerr = {int(L): torch.zeros(model.target_dim, dtype=TRAIN_DTYPE, device=DEVICE) for L in lengths}
    count = {int(L): 0 for L in lengths}
    non_blocking = DEVICE.type == "cuda"

    tick = time.perf_counter()
    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start : start + BATCH_SIZE]
        tau_np, mask_np, y_np = as_batch_arrays(trajectories, params, batch_idx, target_columns, max_length)
        y = torch.from_numpy(y_np).to(DEVICE, dtype=TRAIN_DTYPE, non_blocking=non_blocking)
        tau_full = torch.from_numpy(tau_np).to(DEVICE, dtype=TRAIN_DTYPE, non_blocking=non_blocking)
        mask_full = torch.from_numpy(mask_np).to(DEVICE, dtype=torch.bool, non_blocking=non_blocking)

        for L in lengths:
            pred = model(tau_full[:, :L], mask_full[:, :L])
            sqerr[int(L)] += torch.sum((pred - y) ** 2, dim=0)
            count[int(L)] += int(y.shape[0])

    elapsed = time.perf_counter() - tick
    for L in lengths:
        rmse = torch.sqrt(sqerr[int(L)] / max(count[int(L)], 1)).detach().cpu().numpy()
        rows.append(
            {
                "L": int(L),
                "n_eval": int(count[int(L)]),
                "rmse_delta": float(rmse[0]),
                "rmse_omega": float(rmse[1]) if len(rmse) > 1 else float("nan"),
                "elapsed_s": float(elapsed),
            }
        )
    return rows


@torch.no_grad()
def prediction_sample(model, trajectories, params, indices: np.ndarray, target_columns: List[int], length: int, n_points: int, seed: int):
    chosen = maybe_subsample(indices, n_points, seed)
    pred_chunks = []
    target_chunks = []
    for start in range(0, len(chosen), BATCH_SIZE):
        batch_idx = chosen[start : start + BATCH_SIZE]
        tau_np, mask_np, y_np = as_batch_arrays(trajectories, params, batch_idx, target_columns, length)
        tau = torch.from_numpy(tau_np).to(DEVICE, dtype=TRAIN_DTYPE)
        mask = torch.from_numpy(mask_np).to(DEVICE, dtype=torch.bool)
        pred_chunks.append(model(tau, mask).detach().cpu().numpy())
        target_chunks.append(y_np)
    return np.concatenate(target_chunks, axis=0), np.concatenate(pred_chunks, axis=0)


def plot_rmse_by_length(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    specs = [("rmse_delta", "delta RMSE"), ("rmse_omega", "omega RMSE")]
    for ax, (col, title) in zip(axes, specs):
        for model_name, part in df.groupby("model"):
            part = part.sort_values("L")
            ax.plot(part["L"], part[col], marker="o", linewidth=1.8, label=model_name)
        ax.set_xlabel("trajectory length")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Non-IID pair-feature estimator validation RMSE")
    fig.savefig(FIGURE_DIR / "noniid_pair_rmse_by_length.png", dpi=220)
    plt.close(fig)


def plot_prediction_scatter(model_name: str, target: np.ndarray, pred: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    labels = ["delta", "omega"]
    for k, ax in enumerate(axes):
        ax.scatter(target[:, k], pred[:, k], s=4, alpha=0.35)
        lo = float(min(target[:, k].min(), pred[:, k].min()))
        hi = float(max(target[:, k].max(), pred[:, k].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        ax.set_xlabel(f"true {labels[k]}")
        ax.set_ylabel(f"predicted {labels[k]}")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{model_name}: predictions on validation trajectories")
    fig.savefig(FIGURE_DIR / f"{model_name}_prediction_scatter.png", dpi=220)
    plt.close(fig)


def plot_training_curves(model_name: str, history: Dict):
    train = history.get("train_std_mse", [])
    val = history.get("val_std_mse", [])
    if not train or not val:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(np.arange(1, len(train) + 1), train, label="train")
    ax.plot(np.arange(1, len(val) + 1), val, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("standardized MSE")
    ax.set_title(f"{model_name}: training curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURE_DIR / f"{model_name}_training_curve.png", dpi=220)
    plt.close(fig)


def build_theory_kernel_model(model_name: str, theta: np.ndarray):
    defaults = THREE_LEVEL_DEFAULTS[str(model_name).lower()]
    return build_noniid_three_level_kernel_model(
        delta=float(theta[0]),
        omega=float(theta[1]),
        scheme=str(model_name).lower(),
        gamma_det=defaults["gamma_det"],
        gamma_hidden=defaults["gamma_hidden"],
        gamma_aux=defaults["gamma_aux"],
        delta_aux=defaults["delta_aux"],
        omega_aux=defaults["omega_aux"],
    )


def make_bound_tau_grid(tau_max: float) -> np.ndarray:
    """为解析 Fisher 积分选择同一条 tau 网格。"""
    tau_max = float(tau_max)
    n_from_dt = int(np.ceil(tau_max / max(float(BOUND_DT_MAX), 1e-12))) + 1
    n_grid = max(int(BOUND_GRID_SIZE), n_from_dt, 2)
    return np.linspace(0.0, tau_max, n_grid, dtype=np.float64)


def relative_matrix_change(old: np.ndarray, new: np.ndarray) -> float:
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    denom = max(float(np.linalg.norm(new, ord="fro")), 1e-12)
    return float(np.linalg.norm(new - old, ord="fro") / denom)


def _integrate_pair_density(pair: np.ndarray, tau: np.ndarray) -> float:
    inner = trapz_integral(pair, tau, axis=1)
    return float(trapz_integral(inner, tau, axis=0))


def _real_nonnegative_from_roundoff(values: np.ndarray, name: str) -> np.ndarray:
    values = np.real_if_close(values, tol=1000)
    values = np.asarray(values)
    if np.iscomplexobj(values):
        imag_scale = float(np.max(np.abs(values.imag)))
        real_scale = max(1.0, float(np.max(np.abs(values.real))))
        if imag_scale > ROUND_OFF_TOL * real_scale:
            raise RuntimeError(f"{name} has non-negligible imaginary part: {imag_scale}")
        values = values.real

    values = np.asarray(values, dtype=np.float64)
    min_val = float(np.nanmin(values))
    scale = max(1.0, float(np.nanmax(np.abs(values))))
    if min_val < -ROUND_OFF_TOL * scale:
        raise RuntimeError(f"{name} has negative values beyond roundoff: min={min_val}")
    return np.where(values < 0.0, 0.0, values)


def exact_kernel_transition_matrix(kernel_model) -> np.ndarray:
    """解析计算 P_ij = int_0^infty K_ij(t) dt，不用 tau 网格截断。"""
    evals = np.asarray(kernel_model["evals"], dtype=np.complex128)
    coeff = np.asarray(kernel_model["coeff"], dtype=np.complex128)
    if np.any(np.abs(evals) <= 1e-14):
        raise RuntimeError("kernel generator has an eigenvalue too close to zero for analytic integration.")
    transition = np.sum(coeff * (-1.0 / evals)[None, None, :], axis=2)
    return _real_nonnegative_from_roundoff(transition, "analytic transition matrix")


def stationary_markov_waiting_densities(model_name: str, theta: np.ndarray, tau: np.ndarray) -> Dict[str, np.ndarray]:
    """
    从解析 kernel matrix 构造一阶马尔可夫 waiting-time law。

    K_{ij}(t) 的列 j 是上一观测 reset 态，行 i 是下一观测 reset 态。
    平稳 latent channel 分布为 pi_j。于是

        mu(t) = sum_{ij} K_{ij}(t) pi_j,
        p(t,t') = sum_i [sum_j K_{ij}(t) pi_j] [sum_k K_{ki}(t')],
        q(t'|t) = p(t,t') / mu(t).

    这些对象正对应用户给出的
    J_init + (N-1) J_pair 里的 mu、pi(t)q(t'|t) 和 q。
    """
    kernel_model = build_theory_kernel_model(model_name, theta)
    kernel = _real_nonnegative_from_roundoff(
        evaluate_jump_kernel_matrix(tau, kernel_model),
        "jump kernel matrix",
    )  # (nt, next_i, prev_j)

    grid_transition = trapz_integral(kernel, tau, axis=0)  # (next_i, prev_j), only for diagnostics.
    transition = exact_kernel_transition_matrix(kernel_model)
    latent_pi = kernel_stationary_distribution(transition)

    current_and_next_state = np.einsum("tij,j->ti", kernel, latent_pi)  # A_i(t)
    next_wait_given_state = np.sum(kernel, axis=1)  # B_i(t') = sum_k K_{ki}(t')
    pair = current_and_next_state @ next_wait_given_state.T  # (t, t')
    pair = _real_nonnegative_from_roundoff(pair, "stationary pair density")

    pair_area = _integrate_pair_density(pair, tau)
    if (not np.isfinite(pair_area)) or pair_area <= 0.0:
        raise RuntimeError(f"invalid pair density area: {pair_area}")

    mu = np.einsum("j,tj->t", latent_pi, np.sum(kernel, axis=1))
    mu = _real_nonnegative_from_roundoff(mu, "stationary one-point density")
    mu_area = float(trapz_integral(mu, tau, axis=0))
    q = pair / np.clip(mu[:, None], DENSITY_EPS, None)
    row_area = trapz_integral(q, tau, axis=1)
    transition_col_sums = np.sum(transition, axis=0)
    grid_transition_col_sums = np.sum(grid_transition, axis=0)

    return {
        "tau": np.asarray(tau, dtype=np.float64),
        "mu": np.maximum(mu, DENSITY_EPS),
        "pair": np.maximum(pair, DENSITY_EPS),
        "q": np.maximum(q, DENSITY_EPS),
        "latent_pi": latent_pi,
        "grid_transition": grid_transition,
        "transition": transition,
        "capture_area": np.asarray([pair_area, mu_area], dtype=np.float64),
        "q_row_area_min": np.asarray([float(np.min(row_area))], dtype=np.float64),
        "q_row_area_max": np.asarray([float(np.max(row_area))], dtype=np.float64),
        "transition_col_sum_min": np.asarray([float(np.min(transition_col_sums))], dtype=np.float64),
        "transition_col_sum_max": np.asarray([float(np.max(transition_col_sums))], dtype=np.float64),
        "grid_transition_col_sum_min": np.asarray([float(np.min(grid_transition_col_sums))], dtype=np.float64),
        "grid_transition_col_sum_max": np.asarray([float(np.max(grid_transition_col_sums))], dtype=np.float64),
    }


def finite_difference_density_family(model_name: str, theta: np.ndarray, tau: np.ndarray, fd_step: np.ndarray):
    base = stationary_markov_waiting_densities(model_name, theta, tau)
    plus = []
    minus = []
    for dim in range(len(fd_step)):
        h = np.zeros_like(theta, dtype=np.float64)
        h[dim] = float(fd_step[dim])
        plus.append(stationary_markov_waiting_densities(model_name, theta + h, tau))
        minus.append(stationary_markov_waiting_densities(model_name, theta - h, tau))
    return base, plus, minus


def density_family_diagnostics(base: Dict, plus: List[Dict], minus: List[Dict]) -> Dict[str, float]:
    family = [base] + list(plus) + list(minus)
    capture_pair = np.asarray([item["capture_area"][0] for item in family], dtype=np.float64)
    capture_mu = np.asarray([item["capture_area"][1] for item in family], dtype=np.float64)
    q_min = np.asarray([item["q_row_area_min"][0] for item in family], dtype=np.float64)
    q_max = np.asarray([item["q_row_area_max"][0] for item in family], dtype=np.float64)
    transition_err = np.asarray(
        [
            np.max(np.abs(np.asarray(item["grid_transition"], dtype=np.float64) - np.asarray(item["transition"], dtype=np.float64)))
            for item in family
        ],
        dtype=np.float64,
    )
    return {
        "capture_pair_min": float(np.min(capture_pair)),
        "capture_pair_max": float(np.max(capture_pair)),
        "capture_mu_min": float(np.min(capture_mu)),
        "capture_mu_max": float(np.max(capture_mu)),
        "q_row_area_min_family": float(np.min(q_min)),
        "q_row_area_max_family": float(np.max(q_max)),
        "transition_error_max_family": float(np.max(transition_err)),
        "capture_error_max_family": float(
            max(
                np.max(np.abs(capture_pair - 1.0)),
                np.max(np.abs(capture_mu - 1.0)),
                np.max(np.abs(q_min - 1.0)),
                np.max(np.abs(q_max - 1.0)),
            )
        ),
    }


def analytic_markov_fisher_from_density_family(base: Dict, plus: List[Dict], minus: List[Dict], length: int, tau: np.ndarray, fd_step: np.ndarray):
    dlog_mu = []
    dlog_q = []
    for dim in range(len(fd_step)):
        denom = 2.0 * float(fd_step[dim])
        dlog_mu.append(
            (np.log(np.clip(plus[dim]["mu"], DENSITY_EPS, None)) - np.log(np.clip(minus[dim]["mu"], DENSITY_EPS, None))) / denom
        )
        dlog_q.append(
            (np.log(np.clip(plus[dim]["q"], DENSITY_EPS, None)) - np.log(np.clip(minus[dim]["q"], DENSITY_EPS, None))) / denom
        )

    dlog_mu = np.stack(dlog_mu, axis=1)  # (nt, d)
    dlog_q = np.stack(dlog_q, axis=2)  # (nt, nt, d)

    dim = dlog_mu.shape[1]
    fisher_init = np.zeros((dim, dim), dtype=np.float64)
    fisher_pair = np.zeros((dim, dim), dtype=np.float64)
    mu = base["mu"]
    pair = base["pair"]

    for a in range(dim):
        for b in range(dim):
            fisher_init[a, b] = trapz_integral(mu * dlog_mu[:, a] * dlog_mu[:, b], tau, axis=0)
            integrand = pair * dlog_q[:, :, a] * dlog_q[:, :, b]
            fisher_pair[a, b] = trapz_integral(trapz_integral(integrand, tau, axis=1), tau, axis=0)

    fisher_init = 0.5 * (fisher_init + fisher_init.T)
    fisher_pair = 0.5 * (fisher_pair + fisher_pair.T)
    fisher_total = fisher_init + max(int(length) - 1, 0) * fisher_pair
    fisher_total = 0.5 * (fisher_total + fisher_total.T)
    return fisher_total, fisher_init, fisher_pair


def analytic_markov_fisher(model_name: str, theta: np.ndarray, length: int, tau: np.ndarray, fd_step: np.ndarray):
    base, plus, minus = finite_difference_density_family(model_name, theta, tau, fd_step)
    fisher_total, fisher_init, fisher_pair = analytic_markov_fisher_from_density_family(
        base=base,
        plus=plus,
        minus=minus,
        length=length,
        tau=tau,
        fd_step=fd_step,
    )
    return fisher_total, fisher_init, fisher_pair, base


@torch.no_grad()
def analytic_feature_mean_from_densities(model, densities: Dict, length: int, tau: np.ndarray) -> np.ndarray:
    mu = densities["mu"]
    pair = densities["pair"]

    probes = model.probe_parameters()
    one_alpha = probes["one_alpha"].detach().cpu().numpy().astype(np.float64)
    one_beta = probes["one_beta"].detach().cpu().numpy().astype(np.float64)
    pair_alpha_left = probes["pair_alpha_left"].detach().cpu().numpy().astype(np.float64)
    pair_beta_left = probes["pair_beta_left"].detach().cpu().numpy().astype(np.float64)
    pair_alpha_right = probes["pair_alpha_right"].detach().cpu().numpy().astype(np.float64)
    pair_beta_right = probes["pair_beta_right"].detach().cpu().numpy().astype(np.float64)

    tau_col = tau[:, None]
    one_amp = np.exp(-tau_col * one_alpha[None, :])
    one_phase = tau_col * one_beta[None, :]
    one_re = trapz_integral(mu[:, None] * one_amp * np.cos(one_phase), tau, axis=0)
    one_im = trapz_integral(mu[:, None] * one_amp * np.sin(one_phase), tau, axis=0)

    left_tau = tau[:, None, None]
    right_tau = tau[None, :, None]
    pair_amp = np.exp(-left_tau * pair_alpha_left[None, None, :] - right_tau * pair_alpha_right[None, None, :])
    pair_phase = left_tau * pair_beta_left[None, None, :] + right_tau * pair_beta_right[None, None, :]
    pair_weight = pair[:, :, None]
    pair_re = trapz_integral(trapz_integral(pair_weight * pair_amp * np.cos(pair_phase), tau, axis=1), tau, axis=0)
    pair_im = trapz_integral(trapz_integral(pair_weight * pair_amp * np.sin(pair_phase), tau, axis=1), tau, axis=0)

    one_len = np.asarray([1.0 / np.sqrt(max(int(length), 1))], dtype=np.float64)
    pair_len = np.asarray([1.0 / np.sqrt(max(int(length) - 1, 1))], dtype=np.float64)
    return np.concatenate([one_re, one_im, pair_re, pair_im, one_len, pair_len]).astype(np.float64, copy=False)


@torch.no_grad()
def analytic_feature_mean(model, model_name: str, theta: np.ndarray, length: int, tau: np.ndarray) -> np.ndarray:
    densities = stationary_markov_waiting_densities(model_name, theta, tau)
    return analytic_feature_mean_from_densities(model, densities, length, tau)


@torch.no_grad()
def analytic_estimator_mean_from_feature(model, feature_mean: np.ndarray) -> np.ndarray:
    weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
    bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
    target_mean = model.target_mean.detach().cpu().numpy().astype(np.float64)
    target_std = model.target_std.detach().cpu().numpy().astype(np.float64)
    standardized = weight @ feature_mean + bias
    return standardized * target_std + target_mean


@torch.no_grad()
def analytic_estimator_mean_from_densities(model, densities: Dict, length: int, tau: np.ndarray) -> np.ndarray:
    feature_mean = analytic_feature_mean_from_densities(model, densities, length, tau)
    return analytic_estimator_mean_from_feature(model, feature_mean)


@torch.no_grad()
def analytic_estimator_mean(model, model_name: str, theta: np.ndarray, length: int, tau: np.ndarray) -> np.ndarray:
    feature_mean = analytic_feature_mean(model, model_name, theta, length, tau)
    return analytic_estimator_mean_from_feature(model, feature_mean)


def analytic_estimator_jacobian(model, model_name: str, theta: np.ndarray, length: int, tau: np.ndarray, fd_step: np.ndarray) -> np.ndarray:
    columns = []
    for dim in range(len(fd_step)):
        h = np.zeros_like(theta, dtype=np.float64)
        h[dim] = float(fd_step[dim])
        eta_plus = analytic_estimator_mean(model, model_name, theta + h, length, tau)
        eta_minus = analytic_estimator_mean(model, model_name, theta - h, length, tau)
        columns.append((eta_plus - eta_minus) / (2.0 * float(fd_step[dim])))
    return np.column_stack(columns)


def analytic_estimator_jacobian_from_density_family(model, plus: List[Dict], minus: List[Dict], length: int, tau: np.ndarray, fd_step: np.ndarray) -> np.ndarray:
    columns = []
    for dim in range(len(fd_step)):
        eta_plus = analytic_estimator_mean_from_densities(model, plus[dim], length, tau)
        eta_minus = analytic_estimator_mean_from_densities(model, minus[dim], length, tau)
        columns.append((eta_plus - eta_minus) / (2.0 * float(fd_step[dim])))
    return np.column_stack(columns)


def compute_analytic_bound_object(model, model_name: str, theta: np.ndarray, length: int, tau: np.ndarray, fd_step: np.ndarray) -> Dict:
    density_base, density_plus, density_minus = finite_difference_density_family(
        model_name=model_name,
        theta=theta,
        tau=tau,
        fd_step=fd_step,
    )
    fisher, fisher_init, fisher_pair = analytic_markov_fisher_from_density_family(
        base=density_base,
        plus=density_plus,
        minus=density_minus,
        length=length,
        tau=tau,
        fd_step=fd_step,
    )
    fisher_inv = np.linalg.pinv(fisher, rcond=1e-8)
    eta_analytic = analytic_estimator_mean_from_densities(
        model=model,
        densities=density_base,
        length=length,
        tau=tau,
    )
    bias = eta_analytic - theta
    jac_eta = analytic_estimator_jacobian_from_density_family(
        model=model,
        plus=density_plus,
        minus=density_minus,
        length=length,
        tau=tau,
        fd_step=fd_step,
    )
    cov_bcrb = jac_eta @ fisher_inv @ jac_eta.T
    cov_bcrb = 0.5 * (cov_bcrb + cov_bcrb.T)
    mse_bcrb = cov_bcrb + np.outer(bias, bias)
    return {
        "tau": tau,
        "tau_max": float(tau[-1]),
        "grid_size": int(len(tau)),
        "density_base": density_base,
        "density_plus": density_plus,
        "density_minus": density_minus,
        "density_diagnostics": density_family_diagnostics(density_base, density_plus, density_minus),
        "fisher": fisher,
        "fisher_init": fisher_init,
        "fisher_pair": fisher_pair,
        "eta_analytic": eta_analytic,
        "bias": bias,
        "jac_eta": jac_eta,
        "mse_bcrb": mse_bcrb,
    }


def compute_adaptive_analytic_bound_object(model, model_name: str, theta: np.ndarray, length: int, fd_step: np.ndarray) -> Dict:
    tau_max = float(BOUND_TAU_MAX_INIT)
    limit = float(BOUND_TAU_MAX_LIMIT)
    previous = None
    last = None

    while tau_max <= limit * (1.0 + 1e-12):
        tau = make_bound_tau_grid(tau_max)
        current = compute_analytic_bound_object(
            model=model,
            model_name=model_name,
            theta=theta,
            length=length,
            tau=tau,
            fd_step=fd_step,
        )
        diag = current["density_diagnostics"]
        current["fisher_pair_rel_change"] = float("nan")
        current["fisher_total_rel_change"] = float("nan")
        current["bcrb_rel_change"] = float("nan")
        current["quadrature_converged"] = False

        if previous is not None:
            current["fisher_pair_rel_change"] = relative_matrix_change(previous["fisher_pair"], current["fisher_pair"])
            current["fisher_total_rel_change"] = relative_matrix_change(previous["fisher"], current["fisher"])
            current["bcrb_rel_change"] = relative_matrix_change(previous["mse_bcrb"], current["mse_bcrb"])
            current["quadrature_converged"] = bool(
                diag["transition_error_max_family"] <= float(BOUND_TRANSITION_TOL)
                and diag["capture_error_max_family"] <= float(BOUND_CAPTURE_TOL)
                and current["fisher_pair_rel_change"] <= float(BOUND_FISHER_STABILITY_TOL)
                and current["fisher_total_rel_change"] <= float(BOUND_FISHER_STABILITY_TOL)
                and current["bcrb_rel_change"] <= float(BOUND_BCRB_STABILITY_TOL)
            )
            if current["quadrature_converged"]:
                return current

        previous = current
        last = current
        tau_max *= 2.0

    if last is None:
        raise RuntimeError("adaptive quadrature did not evaluate any grid.")
    last["quadrature_converged"] = False
    return last


@torch.no_grad()
def predict_numpy(model, tau_np: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    tau_np = np.asarray(tau_np, dtype=np.float32)
    mask_np = np.isfinite(tau_np) & (tau_np > 0.0)
    tau_np = np.where(mask_np, tau_np, PAD_VALUE).astype(np.float32, copy=False)
    chunks = []
    for start in range(0, len(tau_np), batch_size):
        tau = torch.from_numpy(tau_np[start : start + batch_size]).to(DEVICE, dtype=TRAIN_DTYPE)
        mask = torch.from_numpy(mask_np[start : start + batch_size]).to(DEVICE, dtype=torch.bool)
        chunks.append(model(tau, mask).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def select_bound_blocks(params: np.ndarray, val_blocks: np.ndarray, metadata: Dict, fd_step: np.ndarray, max_theta: int, seed: int):
    delta_range = metadata.get("delta_range", [float(np.min(params[:, 0])), float(np.max(params[:, 0]))])
    omega_range = metadata.get("omega_range", [float(np.min(params[:, 1])), float(np.max(params[:, 1]))])
    lo = np.asarray([delta_range[0], omega_range[0]], dtype=np.float64) + fd_step
    hi = np.asarray([delta_range[1], omega_range[1]], dtype=np.float64) - fd_step

    eligible = []
    for start, stop in val_blocks:
        theta = np.asarray(params[start, :2], dtype=np.float64)
        if np.all(theta >= lo) and np.all(theta <= hi):
            eligible.append((int(start), int(stop)))
    if not eligible:
        return []

    rng = np.random.default_rng(seed)
    eligible = np.asarray(eligible, dtype=np.int64)
    if len(eligible) <= max_theta:
        return [tuple(x) for x in eligible]
    chosen = np.sort(rng.choice(len(eligible), size=int(max_theta), replace=False))
    return [tuple(x) for x in eligible[chosen]]


def compute_noniid_bound_rows(model, model_name: str, trajectories, params, val_blocks: np.ndarray, metadata: Dict, lengths: List[int]):
    if not BOUNDS_ENABLED:
        return []

    fd_step = np.asarray([BOUND_FD_DELTA, BOUND_FD_OMEGA], dtype=np.float64)
    selected_blocks = select_bound_blocks(
        params=np.asarray(params),
        val_blocks=val_blocks,
        metadata=metadata,
        fd_step=fd_step,
        max_theta=BOUND_MAX_THETA,
        seed=SEED + 211,
    )
    bound_lengths = [int(L) for L in (BOUND_LENGTHS or [max(lengths)]) if int(L) in set(map(int, lengths))]
    if not bound_lengths:
        bound_lengths = [max(map(int, lengths))]

    rows = []
    for block_id, (start, stop) in enumerate(selected_blocks):
        theta = np.asarray(params[start, :2], dtype=np.float64)
        block_idx = np.arange(start, stop, dtype=np.int64)
        block_idx = maybe_subsample(block_idx, BOUND_N_TRAJ, SEED + 300 + block_id)

        for length in bound_lengths:
            base_taus = np.asarray(trajectories[block_idx, :length], dtype=np.float32)
            base_pred = predict_numpy(model, base_taus)
            err = base_pred - theta[None, :]
            mse = (err.T @ err) / max(len(err), 1)
            eta_empirical = np.mean(base_pred, axis=0)

            bound_object = compute_adaptive_analytic_bound_object(
                model=model,
                model_name=model_name,
                theta=theta,
                length=length,
                fd_step=fd_step,
            )
            density_base = bound_object["density_base"]
            density_diag = bound_object["density_diagnostics"]
            fisher = bound_object["fisher"]
            fisher_init = bound_object["fisher_init"]
            fisher_pair = bound_object["fisher_pair"]
            eta_analytic = bound_object["eta_analytic"]
            bias = bound_object["bias"]
            mse_bcrb = bound_object["mse_bcrb"]

            trace_mse = float(np.trace(mse))
            trace_bcrb = float(np.trace(mse_bcrb))
            rows.append(
                {
                    "model": model_name,
                    "block_id": int(block_id),
                    "L": int(length),
                    "n_eval": int(len(block_idx)),
                    "grid_size": int(bound_object["grid_size"]),
                    "tau_max": float(bound_object["tau_max"]),
                    "quadrature_converged": bool(bound_object["quadrature_converged"]),
                    "delta": float(theta[0]),
                    "omega": float(theta[1]),
                    "mse_delta": float(mse[0, 0]),
                    "mse_omega": float(mse[1, 1]),
                    "trace_mse": trace_mse,
                    "eta_empirical_delta": float(eta_empirical[0]),
                    "eta_empirical_omega": float(eta_empirical[1]),
                    "eta_analytic_delta": float(eta_analytic[0]),
                    "eta_analytic_omega": float(eta_analytic[1]),
                    "bias_analytic_delta": float(bias[0]),
                    "bias_analytic_omega": float(bias[1]),
                    "bcrb_delta": float(mse_bcrb[0, 0]),
                    "bcrb_omega": float(mse_bcrb[1, 1]),
                    "trace_bcrb": trace_bcrb,
                    "ratio_trace_bcrb": trace_mse / max(trace_bcrb, 1e-12),
                    "ratio_delta_bcrb": float(mse[0, 0]) / max(float(mse_bcrb[0, 0]), 1e-12),
                    "ratio_omega_bcrb": float(mse[1, 1]) / max(float(mse_bcrb[1, 1]), 1e-12),
                    "fisher_00": float(fisher[0, 0]),
                    "fisher_01": float(fisher[0, 1]),
                    "fisher_11": float(fisher[1, 1]),
                    "fisher_init_00": float(fisher_init[0, 0]),
                    "fisher_init_01": float(fisher_init[0, 1]),
                    "fisher_init_11": float(fisher_init[1, 1]),
                    "fisher_pair_00": float(fisher_pair[0, 0]),
                    "fisher_pair_01": float(fisher_pair[0, 1]),
                    "fisher_pair_11": float(fisher_pair[1, 1]),
                    "capture_area_pair": float(density_base["capture_area"][0]),
                    "capture_area_mu": float(density_base["capture_area"][1]),
                    "capture_pair_min_family": float(density_diag["capture_pair_min"]),
                    "capture_pair_max_family": float(density_diag["capture_pair_max"]),
                    "capture_mu_min_family": float(density_diag["capture_mu_min"]),
                    "capture_mu_max_family": float(density_diag["capture_mu_max"]),
                    "q_row_area_min": float(density_base["q_row_area_min"][0]),
                    "q_row_area_max": float(density_base["q_row_area_max"][0]),
                    "q_row_area_min_family": float(density_diag["q_row_area_min_family"]),
                    "q_row_area_max_family": float(density_diag["q_row_area_max_family"]),
                    "transition_col_sum_min": float(density_base["transition_col_sum_min"][0]),
                    "transition_col_sum_max": float(density_base["transition_col_sum_max"][0]),
                    "grid_transition_col_sum_min": float(density_base["grid_transition_col_sum_min"][0]),
                    "grid_transition_col_sum_max": float(density_base["grid_transition_col_sum_max"][0]),
                    "transition_error_max_family": float(density_diag["transition_error_max_family"]),
                    "capture_error_max_family": float(density_diag["capture_error_max_family"]),
                    "fisher_pair_rel_change": float(bound_object["fisher_pair_rel_change"]),
                    "fisher_total_rel_change": float(bound_object["fisher_total_rel_change"]),
                    "bcrb_rel_change": float(bound_object["bcrb_rel_change"]),
                }
            )
            print(
                f"[{model_name}] bounds block={block_id + 1}/{len(selected_blocks)} "
                f"L={length} analytic learned/bCRB trace ratio={rows[-1]['ratio_trace_bcrb']:.3g} "
                f"converged={rows[-1]['quadrature_converged']} tau_max={rows[-1]['tau_max']:.1f}"
            )
    return rows


def main():
    print("project_root =", PROJECT_ROOT)
    print("data_root =", DATA_ROOT)
    print("output_root =", OUTPUT_ROOT)
    print("models =", MODEL_NAMES)
    print("device =", DEVICE)
    print("batch_size =", BATCH_SIZE)
    print("max_eval_traj =", MAX_EVAL_TRAJ if MAX_EVAL_TRAJ > 0 else "all validation")

    summary_rows = []
    probe_rows = []
    all_length_rows = []
    all_bound_rows = []

    for model_name in MODEL_NAMES:
        print(f"\n[{model_name}] loading checkpoint and data")
        ckpt_path = DATA_ROOT / "models" / model_name / f"noniid_pair_laplace_{model_name}.pt"
        data_dir = DATA_ROOT / "tragectories" / model_name
        traj_path = data_dir / "trajectories.npy"
        param_path = data_dir / "params.npy"
        meta_path = data_dir / "metadata.json"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        if not traj_path.exists() or not param_path.exists():
            raise FileNotFoundError(f"missing nonrenewal data in: {data_dir}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = make_model_from_checkpoint(ckpt)
        trajectories = np.load(traj_path, mmap_mode="r")
        params = np.load(param_path, mmap_mode="r")
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        target_columns = list(ckpt.get("target_columns") or [0, 1])
        y_all = np.asarray(params[:, target_columns], dtype=np.float32)
        val_blocks = validation_blocks(y_all, TRAIN_FRACTION, SEED)
        val_idx = validation_indices(y_all, TRAIN_FRACTION, SEED)
        val_idx = maybe_subsample(val_idx, MAX_EVAL_TRAJ, SEED + 17)
        lengths = [int(x) for x in ckpt.get("train_length_choices", [10, 15, 20, 30, 40, 50])]

        print(f"[{model_name}] val rows = {len(val_idx):,}, lengths = {lengths}")
        length_rows = evaluate_lengths(model, trajectories, params, val_idx, target_columns, lengths)
        for row in length_rows:
            row["model"] = model_name
        all_length_rows.extend(length_rows)

        max_length = max(lengths)
        target_sample, pred_sample = prediction_sample(
            model,
            trajectories,
            params,
            val_idx,
            target_columns,
            max_length,
            SCATTER_POINTS,
            SEED + 101,
        )
        plot_prediction_scatter(model_name, target_sample, pred_sample)
        plot_training_curves(model_name, ckpt.get("history", {}))

        best = ckpt.get("history", {}).get("best_val_std_mse", float("nan"))
        max_row = next(row for row in length_rows if row["L"] == max_length)
        summary_rows.append(
            {
                "model": model_name,
                "checkpoint": str(ckpt_path),
                "sampling_mode": metadata.get("sampling_mode", ""),
                "n_eval": int(max_row["n_eval"]),
                "L_max": int(max_length),
                "best_val_std_mse": float(best),
                "rmse_delta_Lmax": float(max_row["rmse_delta"]),
                "rmse_omega_Lmax": float(max_row["rmse_omega"]),
                "K_one": int(ckpt.get("K_one_probes", 0)),
                "M_pair": int(ckpt.get("M_pair_probes", 0)),
                "feature_dim": int(ckpt.get("feature_dim", 0)),
            }
        )

        probes = model.probe_parameters()
        for name, values in probes.items():
            for i, value in enumerate(values.detach().cpu().numpy().reshape(-1)):
                probe_rows.append({"model": model_name, "probe": name, "index": int(i), "value": float(value)})

        bound_rows = compute_noniid_bound_rows(
            model=model,
            model_name=model_name,
            trajectories=trajectories,
            params=params,
            val_blocks=val_blocks,
            metadata=metadata,
            lengths=lengths,
        )
        all_bound_rows.extend(bound_rows)

        print(f"[{model_name}] L={max_length} RMSE =", max_row["rmse_delta"], max_row["rmse_omega"])

    length_df = pd.DataFrame(all_length_rows).sort_values(["model", "L"])
    summary_df = pd.DataFrame(summary_rows).sort_values("model")
    probe_df = pd.DataFrame(probe_rows).sort_values(["model", "probe", "index"])
    bound_df = pd.DataFrame(all_bound_rows)
    if not bound_df.empty:
        bound_df = bound_df.sort_values(["model", "L", "block_id"])

    length_path = OUTPUT_ROOT / "noniid_pair_validation_by_length.csv"
    summary_path = OUTPUT_ROOT / "noniid_pair_validation_summary.csv"
    probe_path = OUTPUT_ROOT / "noniid_pair_learned_probes.csv"
    bound_path = OUTPUT_ROOT / "noniid_pair_sequence_bcrb.csv"
    length_df.to_csv(length_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    probe_df.to_csv(probe_path, index=False)
    if not bound_df.empty:
        bound_df.to_csv(bound_path, index=False)
    plot_rmse_by_length(length_df)

    print("\nSaved:")
    print(" ", length_path)
    print(" ", summary_path)
    print(" ", probe_path)
    if not bound_df.empty:
        print(" ", bound_path)
    print(" ", FIGURE_DIR)
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    return {
        "summary_df": summary_df,
        "length_df": length_df,
        "probe_df": probe_df,
        "bound_df": bound_df,
        "output_root": OUTPUT_ROOT,
        "figure_dir": FIGURE_DIR,
    }


RESULTS = main()
