#!/usr/bin/env python3
"""Build representative benchmark tables from scan_cache and runtime timing.

This script does two things:
1) Rebuild the existing representative benchmark tables (accuracy-centric).
2) Build extended runtime tables with:
   - Proposed estimator inference latency
   - Traditional MLP baseline inference latency
   - Parameter counts and speedup

Outputs are written under notebooks/scan_cache/benchmark_tables by default.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import jacrev, vmap

    HAS_TORCH_FUNC = True
except Exception:
    jacrev = None
    vmap = None
    HAS_TORCH_FUNC = False


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tools import build_model_theory_torch


TRAIN_DTYPE = torch.float64
THEORY_COMPLEX_DTYPE = torch.complex128
EPS_W = 1e-12
PAD_VALUE = 0.0

MODEL_DISPLAY = {
    "tls": "TLS",
}

BASE_COLUMNS = ["Model", "Parameters", "N", "Learned_MSE", "RbCRB", "RBar"]

# Global function pointer used by modelhead/dtangent, mirroring notebook logic.
laplace_transform_torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build representative benchmark tables with runtime metrics.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tls"],
        help="Model names to include. Only TLS has a renewal Torch theory at the moment.",
    )
    parser.add_argument("--delta", type=float, default=1.55, help="Representative delta point.")
    parser.add_argument("--omega", type=float, default=2.75, help="Representative omega point.")
    parser.add_argument("--n", type=int, default=50, help="Representative trajectory length N (same as L).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for latency timing.")
    parser.add_argument("--warmup-iters", type=int, default=2, help="Warmup iterations for latency timing.")
    parser.add_argument("--timed-iters", type=int, default=5, help="Timed iterations for latency timing.")
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=7,
        help="Number of repeated latency measurements for stability statistics.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Benchmark device.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks") / "scan_cache" / "benchmark_tables",
        help="Directory for benchmark table outputs.",
    )
    parser.add_argument(
        "--scan-cache-root",
        type=Path,
        default=Path("notebooks") / "scan_cache",
        help="Root directory containing per-model scan_df files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Data root containing models and trajectories subfolders.",
    )
    parser.add_argument(
        "--mlp-hidden",
        type=str,
        default="256,256",
        help="Comma-separated hidden dims for traditional MLP baseline.",
    )
    parser.add_argument(
        "--train-mlp-epochs",
        type=int,
        default=0,
        help="If >0, train MLP baseline and report training time + val MSE.",
    )
    parser.add_argument("--mlp-lr", type=float, default=2e-3, help="MLP training learning rate.")
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4, help="MLP training weight decay.")
    parser.add_argument("--mlp-batch-size", type=int, default=2048, help="MLP training batch size.")
    parser.add_argument(
        "--mlp-sample-cap",
        type=int,
        default=120_000,
        help="Max trajectories loaded per model when training MLP baseline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--ablation-sample-cap",
        type=int,
        default=40_000,
        help="Max trajectories per model for probe ablation MSE evaluation.",
    )
    parser.add_argument(
        "--ablation-batch-size",
        type=int,
        default=2048,
        help="Batch size for probe ablation MSE evaluation.",
    )
    parser.add_argument(
        "--ablation-max-k",
        type=int,
        default=0,
        help="If >0, evaluate probe ablation up to this M; 0 means up to checkpoint K.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def safe_torch_load(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _init_probe_values(init, K: int, dtype: torch.dtype) -> torch.Tensor:
    init_t = torch.as_tensor(init, dtype=dtype)
    if init_t.ndim == 1 and init_t.numel() == K:
        return init_t.clone()
    if init_t.ndim == 1 and init_t.numel() == 2:
        return torch.linspace(init_t[0], init_t[1], K, dtype=dtype)
    raise ValueError(f"Expected probe initializer with shape ({K},) or (2,), got {tuple(init_t.shape)}")


class ComplexLaplaceFeatures(nn.Module):
    """Complex Laplace probe layer from notebooks/2-train.ipynb."""

    def __init__(
        self,
        K: int = 3,
        alpha_init=(0.2, 0.8),
        beta_init=(-2.0, 2.0),
        eps_alpha: float = 1e-4,
    ):
        super().__init__()
        self.K = K
        self.eps_alpha = eps_alpha

        alpha0 = _init_probe_values(alpha_init, K, TRAIN_DTYPE)
        beta0 = _init_probe_values(beta_init, K, TRAIN_DTYPE)
        raw_alpha0 = torch.log(torch.expm1(torch.clamp(alpha0 - eps_alpha, min=1e-6)))

        self.raw_alpha = nn.Parameter(raw_alpha0)
        self.beta = nn.Parameter(beta0)
        self.output_dim = 2 * K + 1

    def forward(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        taus = taus.to(dtype=self.raw_alpha.dtype)
        mask = mask.to(dtype=self.raw_alpha.dtype)

        N = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        alpha = F.softplus(self.raw_alpha) + self.eps_alpha
        beta = self.beta

        tau = taus.unsqueeze(-1)
        m = mask.unsqueeze(-1)

        real_part = torch.exp(-tau * alpha) * torch.cos(tau * beta)
        imag_part = torch.exp(-tau * alpha) * torch.sin(tau * beta)

        re = (real_part * m).sum(dim=1) / N
        im = (imag_part * m).sum(dim=1) / N
        lf = torch.rsqrt(N)
        return torch.cat([re, im, lf], dim=1)


def modelhead(theta: torch.Tensor, s: torch.Tensor, eps: float = EPS_W):
    if laplace_transform_torch is None:
        raise RuntimeError("laplace_transform_torch is not set.")
    theta = theta.to(dtype=TRAIN_DTYPE)
    s = s.to(dtype=THEORY_COMPLEX_DTYPE)
    squeeze = theta.ndim == 1
    if squeeze:
        return laplace_transform_torch(theta, s, eps=eps)

    try:
        out = laplace_transform_torch(theta, s, eps=eps)
        if out.shape[0] == theta.shape[0]:
            return out
    except Exception:
        pass

    if HAS_TORCH_FUNC:
        return vmap(lambda th: laplace_transform_torch(th, s, eps=eps))(theta)
    outs = [laplace_transform_torch(theta_i, s, eps=eps) for theta_i in theta]
    return torch.stack(outs, dim=0)


def _delta_tangent_autograd(alpha: torch.Tensor, beta: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
    s = torch.complex(alpha, -beta)

    def _mu_parts(theta_local):
        mu = modelhead(theta_local, s)
        return torch.stack([torch.real(mu), torch.imag(mu)], dim=1)

    if HAS_TORCH_FUNC:
        jac = vmap(jacrev(_mu_parts))(theta_ref)
        d_re_d_delta = jac[:, :, 0, 0]
        d_im_d_delta = jac[:, :, 1, 0]
        out = torch.stack([d_re_d_delta, d_im_d_delta], dim=2).to(dtype=alpha.dtype)
        return out.detach()

    th_req = theta_ref.detach().clone().requires_grad_(True)

    def _mu_batch(th_batch):
        mu = modelhead(th_batch, s)
        return torch.stack([torch.real(mu), torch.imag(mu)], dim=2)

    jac = torch.autograd.functional.jacobian(
        _mu_batch,
        th_req,
        create_graph=False,
        strict=False,
        vectorize=True,
    )
    batch_idx = torch.arange(theta_ref.shape[0], device=theta_ref.device)
    jac_diag = jac[batch_idx, :, :, batch_idx, :]
    d_re_d_delta = jac_diag[:, :, 0, 0]
    d_im_d_delta = jac_diag[:, :, 1, 0]
    return torch.stack([d_re_d_delta, d_im_d_delta], dim=2).to(dtype=alpha.dtype)


def dtangent(alpha: torch.Tensor, beta: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
    squeeze = theta_ref.ndim == 1
    if squeeze:
        theta_ref = theta_ref.unsqueeze(0)
    theta_ref = theta_ref.to(device=alpha.device, dtype=alpha.dtype)
    g = _delta_tangent_autograd(alpha, beta, theta_ref)
    if not torch.isfinite(g).all():
        raise RuntimeError("Non-finite tangent values encountered in autograd path.")
    return g[0] if squeeze else g


class LaplaceLinearEstimator(nn.Module):
    """Block-structured estimator from notebooks/2-train.ipynb."""

    def __init__(
        self,
        out_dim: int = 2,
        K: int = 3,
        alpha_init=(0.2, 0.8),
        beta_init=(-2.0, 2.0),
        target_mean: Optional[torch.Tensor] = None,
        target_std: Optional[torch.Tensor] = None,
        omega_init_scale: float = 0.1,
        delta_steps: int = 4,
    ):
        super().__init__()
        if target_mean is None or target_std is None:
            raise ValueError("target_mean and target_std must be provided explicitly.")
        if out_dim != 2:
            raise ValueError("Expected out_dim=2 for [delta, omega].")

        self.feat = ComplexLaplaceFeatures(K=K, alpha_init=alpha_init, beta_init=beta_init)
        self.in_dim = self.feat.output_dim
        self.K = int(K)
        self.delta_steps = int(delta_steps)

        self.delta_coeff = nn.Parameter(torch.zeros(self.K, dtype=TRAIN_DTYPE))
        self.delta_length_weight = nn.Parameter(torch.zeros((), dtype=TRAIN_DTYPE))
        self.delta_bias = nn.Parameter(torch.zeros((), dtype=TRAIN_DTYPE))

        self.omega_dir_raw = nn.Parameter(torch.randn(self.K, 2, dtype=TRAIN_DTYPE) * omega_init_scale)
        self.omega_coeff = nn.Parameter(torch.zeros(self.K, dtype=TRAIN_DTYPE))
        self.omega_length_weight = nn.Parameter(torch.zeros((), dtype=TRAIN_DTYPE))
        self.omega_bias = nn.Parameter(torch.zeros((), dtype=TRAIN_DTYPE))

        self.register_buffer("head_theta_ref", torch.as_tensor(target_mean, dtype=TRAIN_DTYPE))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=TRAIN_DTYPE))
        self.register_buffer("target_std", torch.clamp(torch.as_tensor(target_std, dtype=TRAIN_DTYPE), min=1e-6))

    def omegadirs(self) -> torch.Tensor:
        return self.omega_dir_raw / torch.clamp(torch.linalg.norm(self.omega_dir_raw, dim=1, keepdim=True), min=1e-6)

    def omegaweights(self) -> torch.Tensor:
        dirs = self.omegadirs()
        return self.omega_coeff.unsqueeze(1) * dirs

    def deltadirs(self, theta_ref: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.feat.raw_alpha) + self.feat.eps_alpha
        beta = self.feat.beta
        q = self.omegadirs()
        g = dtangent(alpha, beta, theta_ref)
        if g.ndim == 2:
            proj = g - torch.sum(g * q, dim=1, keepdim=True) * q
            proj_norm = torch.linalg.norm(proj, dim=1, keepdim=True)
            fallback = torch.stack([-q[:, 1], q[:, 0]], dim=1)
            return torch.where(proj_norm > 1e-6, proj / proj_norm.clamp_min(1e-6), fallback)

        q = q.unsqueeze(0)
        proj = g - torch.sum(g * q, dim=2, keepdim=True) * q
        proj_norm = torch.linalg.norm(proj, dim=2, keepdim=True)
        fallback = torch.stack([-q[..., 1], q[..., 0]], dim=2)
        return torch.where(proj_norm > 1e-6, proj / proj_norm.clamp_min(1e-6), fallback)

    def splitx(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return x[:, : self.K], x[:, self.K : 2 * self.K], x[:, -1]

    def omegastd(self, x: torch.Tensor) -> torch.Tensor:
        xre, xim, xlen = self.splitx(x)
        omega_probe = self.omegaweights()
        return xre @ omega_probe[:, 0] + xim @ omega_probe[:, 1] + xlen * self.omega_length_weight + self.omega_bias

    def deltastd(self, x: torch.Tensor, omega_std: torch.Tensor) -> torch.Tensor:
        xre, xim, xlen = self.splitx(x)
        omega = self.decode_targets(torch.stack([torch.zeros_like(omega_std), omega_std], dim=1))[:, 1]
        delta = self.target_mean[0].expand_as(omega)

        for _ in range(self.delta_steps):
            theta_ref = torch.stack([delta, omega], dim=1)
            dirs = self.deltadirs(theta_ref)
            delta_std = torch.sum(xre * (self.delta_coeff.unsqueeze(0) * dirs[:, :, 0]), dim=1)
            delta_std = delta_std + torch.sum(xim * (self.delta_coeff.unsqueeze(0) * dirs[:, :, 1]), dim=1)
            delta_std = delta_std + xlen * self.delta_length_weight + self.delta_bias
            delta = self.decode_targets(torch.stack([delta_std, omega_std], dim=1))[:, 0]

        return delta_std

    def decode_targets(self, y_std: torch.Tensor) -> torch.Tensor:
        y_std = y_std.to(device=self.target_mean.device, dtype=self.target_mean.dtype)
        return y_std * self.target_std + self.target_mean

    def forward_standardized(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.feat(taus, mask)
        omega_std = self.omegastd(x)
        delta_std = self.deltastd(x, omega_std)
        return torch.stack([delta_std, omega_std], dim=1)

    def forward(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decode_targets(self.forward_standardized(taus, mask))

    @torch.no_grad()
    def probe_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.feat.raw_alpha) + self.feat.eps_alpha
        beta = self.feat.beta
        return alpha.detach(), beta.detach()


def adaptstate(state_dict):
    upgraded = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in state_dict.items()}
    legacy_omega = "omega_coeff" not in upgraded and "omega_gate_logits" in upgraded and "omega_scale_raw" in upgraded
    if legacy_omega:
        gates = torch.softmax(upgraded["omega_gate_logits"].to(dtype=TRAIN_DTYPE), dim=0)
        scale = F.softplus(upgraded["omega_scale_raw"].to(dtype=TRAIN_DTYPE))
        upgraded["omega_coeff"] = scale * gates
    upgraded.pop("omega_gate_logits", None)
    upgraded.pop("omega_scale_raw", None)
    return upgraded, legacy_omega


class TraditionalMLPBaseline(nn.Module):
    """Simple traditional NN baseline that consumes [taus, mask] flattened."""

    def __init__(
        self,
        length: int,
        hidden_dims: Sequence[int],
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        input_mean: Optional[torch.Tensor] = None,
        input_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.length = int(length)
        in_dim = 2 * self.length
        dims = [in_dim, *[int(h) for h in hidden_dims], 2]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], dtype=TRAIN_DTYPE))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1], dtype=TRAIN_DTYPE))
        self.net = nn.Sequential(*layers)

        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=TRAIN_DTYPE))
        self.register_buffer("target_std", torch.clamp(torch.as_tensor(target_std, dtype=TRAIN_DTYPE), min=1e-6))
        if input_mean is None:
            input_mean = torch.zeros(in_dim, dtype=TRAIN_DTYPE)
        if input_std is None:
            input_std = torch.ones(in_dim, dtype=TRAIN_DTYPE)
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=TRAIN_DTYPE))
        self.register_buffer("input_std", torch.clamp(torch.as_tensor(input_std, dtype=TRAIN_DTYPE), min=1e-6))

    def encode_inputs(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_tau = taus[:, : self.length].to(dtype=TRAIN_DTYPE)
        x_mask = mask[:, : self.length].to(dtype=TRAIN_DTYPE)
        x = torch.cat([x_tau, x_mask], dim=1)
        return (x - self.input_mean) / self.input_std

    def encode_targets(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(dtype=TRAIN_DTYPE)
        return (y - self.target_mean) / self.target_std

    def decode_targets(self, y_std: torch.Tensor) -> torch.Tensor:
        y_std = y_std.to(dtype=TRAIN_DTYPE)
        return y_std * self.target_std + self.target_mean

    def forward_standardized(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode_inputs(taus, mask))

    def forward(self, taus: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decode_targets(self.forward_standardized(taus, mask))


@dataclass
class LatencyResult:
    batch_ms: float
    sample_us: float


@dataclass
class LatencyStats:
    batch_ms_mean: float
    batch_ms_std: float
    sample_us_mean: float
    sample_us_std: float


def benchmark_latency(
    model: nn.Module,
    taus: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
) -> LatencyResult:
    model.eval()
    taus = taus.to(device=device, dtype=TRAIN_DTYPE)
    mask = mask.to(device=device)

    with torch.no_grad():
        for _ in range(max(0, warmup_iters)):
            _ = model(taus, mask)
    maybe_sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max(1, timed_iters)):
            _ = model(taus, mask)
    maybe_sync(device)
    elapsed = time.perf_counter() - t0
    batch_ms = (elapsed / max(1, timed_iters)) * 1000.0
    sample_us = batch_ms * 1000.0 / max(1, taus.shape[0])
    return LatencyResult(batch_ms=batch_ms, sample_us=sample_us)


def benchmark_latency_repeated(
    model: nn.Module,
    taus: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
    repeats: int,
) -> LatencyStats:
    batch_vals: List[float] = []
    sample_vals: List[float] = []
    for _ in range(max(1, int(repeats))):
        lr = benchmark_latency(
            model=model,
            taus=taus,
            mask=mask,
            device=device,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        batch_vals.append(float(lr.batch_ms))
        sample_vals.append(float(lr.sample_us))

    batch_arr = np.asarray(batch_vals, dtype=float)
    sample_arr = np.asarray(sample_vals, dtype=float)
    return LatencyStats(
        batch_ms_mean=float(np.mean(batch_arr)),
        batch_ms_std=float(np.std(batch_arr, ddof=0)),
        sample_us_mean=float(np.mean(sample_arr)),
        sample_us_std=float(np.std(sample_arr, ddof=0)),
    )


def paramblocks(y_raw: np.ndarray) -> np.ndarray:
    y_raw = np.asarray(y_raw, dtype=np.float64)
    params = y_raw[:, :2]
    if len(params) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    changes = np.any(params[1:] != params[:-1], axis=1)
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    stops = np.r_[starts[1:], len(params)]
    return np.stack([starts, stops], axis=1).astype(np.int64, copy=False)


def splitblocks(taus_raw: np.ndarray, y_raw: np.ndarray, train_frac: float, seed: int):
    blocks = paramblocks(y_raw)
    if len(blocks) < 2:
        raise ValueError("Need at least two parameter blocks for train/val split.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    n_train = int(round(train_frac * len(blocks)))
    n_train = min(max(n_train, 1), len(blocks) - 1)

    train_blocks = blocks[order[:n_train]]
    val_blocks = blocks[order[n_train:]]

    def gather(arr, chosen_blocks):
        parts = [np.asarray(arr[start:stop]) for start, stop in chosen_blocks]
        return np.concatenate(parts, axis=0)

    x_train = gather(taus_raw, train_blocks)
    y_train = gather(y_raw, train_blocks)
    x_val = gather(taus_raw, val_blocks)
    y_val = gather(y_raw, val_blocks)
    return x_train, x_val, y_train, y_val


def parse_hidden_dims(spec: str) -> List[int]:
    dims = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not dims:
        raise ValueError("mlp-hidden must contain at least one integer.")
    return dims


def add_trace_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["learned_trace_mse"] = out["exact_mse_delta"] + out["exact_mse_omega"]
    out["bcrb_trace_mse"] = out["bcrb_mse_delta"] + out["bcrb_mse_omega"]
    out["bar_trace_mse"] = out["barankin_mse_delta"] + out["barankin_mse_omega"]
    out["ratio_bcrb_trace"] = out["learned_trace_mse"] / out["bcrb_trace_mse"]
    out["ratio_bar_trace"] = out["learned_trace_mse"] / out["bar_trace_mse"]
    return out


def load_representative_row(scan_csv: Path, n_obs: int, delta: float, omega: float) -> pd.Series:
    if not scan_csv.exists():
        raise FileNotFoundError(f"scan_df not found: {scan_csv}")
    df = pd.read_csv(scan_csv)
    df_n = df.loc[df["L"] == int(n_obs)].copy()
    if df_n.empty:
        raise ValueError(f"No rows with L={n_obs} in {scan_csv}")

    tol = 1e-12
    exact = df_n.loc[np.isclose(df_n["delta"], delta, atol=tol) & np.isclose(df_n["omega"], omega, atol=tol)]
    if not exact.empty:
        return exact.iloc[0]

    # Fallback to nearest point if exact grid point is not present.
    dist = (df_n["delta"] - delta).abs() + (df_n["omega"] - omega).abs()
    idx = int(dist.idxmin())
    return df_n.loc[idx]


def load_ablation_subset(
    model_name: str,
    data_root: Path,
    n_obs: int,
    sample_cap: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path_tau = data_root / "tragectories" / model_name / "trajectories.npy"
    path_param = data_root / "tragectories" / model_name / "params.npy"
    if not path_tau.exists() or not path_param.exists():
        raise FileNotFoundError(f"missing trajectory/param data for model={model_name}")
    tau_all = np.load(path_tau, mmap_mode="r")
    param_all = np.load(path_param, mmap_mode="r")
    n_total = min(int(tau_all.shape[0]), int(param_all.shape[0]))
    n_take = min(int(sample_cap), n_total)
    rng = np.random.default_rng(seed)
    if n_take < n_total:
        idx = np.sort(rng.choice(n_total, size=n_take, replace=False))
    else:
        idx = np.arange(n_total)
    taus_np = np.asarray(tau_all[idx, : int(n_obs)], dtype=np.float64)
    y_np = np.asarray(param_all[idx, :2], dtype=np.float64)
    mask_np = np.asarray(taus_np > 0.0, dtype=np.bool_)
    return torch.from_numpy(taus_np), torch.from_numpy(mask_np), torch.from_numpy(y_np)


def evaluate_model_trace_mse(
    model: nn.Module,
    taus: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    n = int(y.shape[0])
    total_sq = 0.0
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            sl = slice(start, min(start + int(batch_size), n))
            taus_b = taus[sl].to(device=device, dtype=TRAIN_DTYPE)
            mask_b = mask[sl].to(device=device)
            y_b = y[sl].to(device=device, dtype=TRAIN_DTYPE)
            pred = model(taus_b, mask_b)
            sq = torch.sum((pred - y_b) ** 2, dim=1)
            total_sq += float(torch.sum(sq).item())
    return total_sq / max(1, n)


def build_reduced_probe_model(model: LaplaceLinearEstimator, keep_m: int, device: torch.device) -> LaplaceLinearEstimator:
    keep_m = int(keep_m)
    new_model = copy.deepcopy(model).to(device)
    st = new_model.state_dict()
    if "delta_coeff" in st:
        st["delta_coeff"][keep_m:] = 0.0
    if "omega_coeff" in st:
        st["omega_coeff"][keep_m:] = 0.0
    # For strictness, also neutralize unused directions to avoid accidental coupling.
    if "omega_dir_raw" in st and st["omega_dir_raw"].ndim == 2:
        st["omega_dir_raw"][keep_m:, :] = 0.0
    new_model.load_state_dict(st)
    new_model.eval()
    return new_model


def _score_jacobian(theta_ref: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    def feat_mu(theta_local: torch.Tensor) -> torch.Tensor:
        mu = modelhead(theta_local, s)
        return torch.cat([torch.real(mu), torch.imag(mu)], dim=0)

    jac = torch.autograd.functional.jacobian(feat_mu, theta_ref, create_graph=False, strict=False)
    return jac.to(dtype=TRAIN_DTYPE)


def projection_retention_diagnostics_by_prefix(
    model: LaplaceLinearEstimator,
    theta_ref: torch.Tensor,
) -> Dict[int, Dict[str, float]]:
    alpha, beta = model.probe_parameters()
    s = torch.complex(alpha.to(dtype=TRAIN_DTYPE), -beta.to(dtype=TRAIN_DTYPE)).to(dtype=THEORY_COMPLEX_DTYPE)
    K = int(alpha.numel())
    theta_ref = theta_ref.to(dtype=TRAIN_DTYPE, device=alpha.device)

    J_full = _score_jacobian(theta_ref=theta_ref, s=s)  # (2K, 2), columns are G_i(.;theta)
    full_col_energy = torch.sum(J_full * J_full, dim=0).to(dtype=TRAIN_DTYPE)  # [delta, omega]
    full_col_energy = torch.clamp(full_col_energy, min=1e-18)
    full_total_energy = torch.clamp(torch.sum(full_col_energy), min=1e-18)

    out: Dict[int, Dict[str, float]] = {}
    for m in range(1, K + 1):
        idx = torch.cat([torch.arange(m), torch.arange(K, K + m)]).to(device=J_full.device)
        J_m = J_full[idx]
        proj_col_energy = torch.sum(J_m * J_m, dim=0).to(dtype=TRAIN_DTYPE)
        proj_total_energy = torch.sum(proj_col_energy)

        ret_delta = float((proj_col_energy[0] / full_col_energy[0]).item())
        ret_omega = float((proj_col_energy[1] / full_col_energy[1]).item())
        ret_total = float((proj_total_energy / full_total_energy).item())

        # Clamp to [0, 1] for numerical safety.
        ret_delta = float(min(1.0, max(0.0, ret_delta)))
        ret_omega = float(min(1.0, max(0.0, ret_omega)))
        ret_total = float(min(1.0, max(0.0, ret_total)))

        out[m] = {
            "retained_fisher_fraction": ret_total,
            "retained_fisher_fraction_delta": ret_delta,
            "retained_fisher_fraction_omega": ret_omega,
            "projection_loss_rel": float(max(0.0, 1.0 - ret_total)),
            "projection_loss_rel_delta": float(max(0.0, 1.0 - ret_delta)),
            "projection_loss_rel_omega": float(max(0.0, 1.0 - ret_omega)),
        }
    return out


def local_projection_diagnostics_grid(
    model: LaplaceLinearEstimator,
    theta_grid: np.ndarray,
    max_m: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    max_m = max(1, int(max_m))
    for theta in np.asarray(theta_grid, dtype=np.float64):
        theta_ref = torch.as_tensor(theta, dtype=TRAIN_DTYPE, device=model.head_theta_ref.device)
        ret_map = projection_retention_diagnostics_by_prefix(model=model, theta_ref=theta_ref)
        for m in range(1, max_m + 1):
            row = {
                "delta": float(theta[0]),
                "omega": float(theta[1]),
                "M": int(m),
                **ret_map[m],
            }
            rows.append(row)
    return rows


def summarize_local_projection_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "retained_fisher_fraction",
        "retained_fisher_fraction_delta",
        "retained_fisher_fraction_omega",
        "projection_loss_rel",
        "projection_loss_rel_delta",
        "projection_loss_rel_omega",
    ]
    rows: List[Dict[str, float]] = []
    for (model_name, K, M), g in df.groupby(["Model", "K", "M"], sort=True):
        row: Dict[str, float] = {
            "Model": model_name,
            "K": int(K),
            "M": int(M),
            "num_grid_points": int(g.shape[0]),
        }
        for col in metric_cols:
            vals = np.asarray(g[col], dtype=float)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_p05"] = float(np.percentile(vals, 5))
            row[f"{col}_p95"] = float(np.percentile(vals, 95))
        rows.append(row)
    return pd.DataFrame(rows)


def build_accuracy_row(display_name: str, delta: float, omega: float, n_obs: int, scan_row: pd.Series) -> Dict[str, float]:
    learned_mse = float(scan_row["exact_mse_delta"] + scan_row["exact_mse_omega"])
    rbcrb = float(
        (scan_row["exact_mse_delta"] + scan_row["exact_mse_omega"])
        / (scan_row["bcrb_mse_delta"] + scan_row["bcrb_mse_omega"])
    )
    rbar = float(
        (scan_row["exact_mse_delta"] + scan_row["exact_mse_omega"])
        / (scan_row["barankin_mse_delta"] + scan_row["barankin_mse_omega"])
    )
    return {
        "Model": display_name,
        "Parameters": f"(δ={delta:.2f}, ω={omega:.2f})",
        "N": int(n_obs),
        "Learned_MSE": learned_mse,
        "RbCRB": rbcrb,
        "RBar": rbar,
    }


def load_checkpoint_model(
    model_name: str,
    data_root: Path,
    device: torch.device,
) -> Tuple[LaplaceLinearEstimator, Dict]:
    global laplace_transform_torch
    _, laplace_transform_torch = build_model_theory_torch(model_name)

    ckpt_path = data_root / "models" / model_name / f"pole_residue_block_head_{model_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")

    model_state_raw = ckpt["model_state"]
    model_state, _ = adaptstate(model_state_raw)
    target_mean = model_state["target_mean"].to(dtype=TRAIN_DTYPE)
    target_std = model_state["target_std"].to(dtype=TRAIN_DTYPE)
    K = int(ckpt.get("K_probes", 4))

    model = LaplaceLinearEstimator(
        out_dim=2,
        K=K,
        alpha_init=ckpt.get("alpha_init_range", (0.1, 1.5)),
        beta_init=ckpt.get("beta_init_range", (-5.0, 5.0)),
        target_mean=target_mean,
        target_std=target_std,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, ckpt


def trajectory_batch_for_timing(
    model_name: str,
    data_root: Path,
    batch_size: int,
    n_obs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    path_tau = data_root / "tragectories" / model_name / "trajectories.npy"
    if not path_tau.exists():
        raise FileNotFoundError(f"trajectory file not found: {path_tau}")
    tau_all = np.load(path_tau, mmap_mode="r")
    n_take = min(int(batch_size), int(tau_all.shape[0]))
    taus_np = np.asarray(tau_all[:n_take, : int(n_obs)], dtype=np.float64)
    mask_np = np.asarray(taus_np > 0.0, dtype=np.bool_)
    taus = torch.from_numpy(taus_np)
    mask = torch.from_numpy(mask_np)
    return taus, mask


def checkpoint_train_compute_seconds(ckpt: Dict) -> float:
    history = ckpt.get("history", None)
    if not isinstance(history, dict):
        return float("nan")
    epoch_timing = history.get("epoch_timing", None)
    if not isinstance(epoch_timing, list):
        return float("nan")
    total = 0.0
    for item in epoch_timing:
        if isinstance(item, dict):
            total += float(item.get("compute_time", 0.0))
    return total


def train_mlp_baseline(
    model_name: str,
    data_root: Path,
    n_obs: int,
    hidden_dims: Sequence[int],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    sample_cap: int,
    seed: int,
) -> Tuple[TraditionalMLPBaseline, float, float]:
    path_tau = data_root / "tragectories" / model_name / "trajectories.npy"
    path_param = data_root / "tragectories" / model_name / "params.npy"
    tau_all = np.load(path_tau, mmap_mode="r")
    param_all = np.load(path_param, mmap_mode="r")
    n_total = min(int(tau_all.shape[0]), int(param_all.shape[0]))
    n_select = min(int(sample_cap), n_total)

    rng = np.random.default_rng(seed)
    if n_select < n_total:
        idx = np.sort(rng.choice(n_total, size=n_select, replace=False))
    else:
        idx = np.arange(n_total)

    taus_raw = np.asarray(tau_all[idx, : int(n_obs)], dtype=np.float64)
    y_raw = np.asarray(param_all[idx, :2], dtype=np.float64)
    mask_raw = (taus_raw > 0.0).astype(np.float64)
    x_raw = np.concatenate([taus_raw, mask_raw], axis=1)

    x_train, x_val, y_train, y_val = splitblocks(x_raw, y_raw, train_frac=0.8, seed=seed)
    x_mean = torch.as_tensor(x_train.mean(axis=0), dtype=TRAIN_DTYPE)
    x_std = torch.as_tensor(np.clip(x_train.std(axis=0), 1e-6, None), dtype=TRAIN_DTYPE)
    y_mean = torch.as_tensor(y_train.mean(axis=0), dtype=TRAIN_DTYPE)
    y_std = torch.as_tensor(np.clip(y_train.std(axis=0), 1e-6, None), dtype=TRAIN_DTYPE)

    model = TraditionalMLPBaseline(
        length=int(n_obs),
        hidden_dims=hidden_dims,
        target_mean=y_mean,
        target_std=y_std,
        input_mean=x_mean,
        input_std=x_std,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train_t = torch.as_tensor(x_train, dtype=TRAIN_DTYPE, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=TRAIN_DTYPE, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=TRAIN_DTYPE, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=TRAIN_DTYPE, device=device)

    # Reconstruct taus/mask tensors expected by forward.
    def _split_input(x):
        taus = x[:, : int(n_obs)]
        mask = x[:, int(n_obs) :].to(dtype=torch.bool)
        return taus, mask

    num_train = int(x_train_t.shape[0])
    idx_full = torch.arange(num_train, device=device)
    t0 = time.perf_counter()
    for _ in range(max(0, int(epochs))):
        perm = idx_full[torch.randperm(num_train, device=device)]
        for start in range(0, num_train, int(batch_size)):
            batch_idx = perm[start : start + int(batch_size)]
            xb = x_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            taus_b, mask_b = _split_input(xb)
            pred_std = model.forward_standardized(taus_b, mask_b)
            loss = F.mse_loss(pred_std, model.encode_targets(yb))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    maybe_sync(device)
    train_sec = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        taus_v, mask_v = _split_input(x_val_t)
        pred = model(taus_v, mask_v)
        val_mse = torch.mean(torch.sum((pred - y_val_t) ** 2, dim=1)).item()
    return model, float(train_sec), float(val_mse)


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(col: str, val) -> str:
        if pd.isna(val):
            return "NA"
        if col in {"Learned_MSE", "TraditionalNN_Val_MSE", "learned_trace_mse"}:
            return f"{float(val):.6e}"
        if col in {"RbCRB", "RBar", "ratio_bcrb_trace", "ratio_bar_trace"}:
            return f"{float(val):.4f}"
        if col.startswith("retained_fisher_fraction") or col.startswith("projection_loss_rel"):
            return f"{float(val):.4f}"
        if col.endswith("_mean") or col.endswith("_std") or col.endswith("_median") or col.endswith("_p05") or col.endswith("_p95"):
            return f"{float(val):.4f}"
        if col in {"empirical_trace_mse", "empirical_trace_mse_over_full"}:
            return f"{float(val):.6f}"
        if col.endswith("_ms"):
            return f"{float(val):.2f}"
        if "_us_" in col:
            return f"{float(val):.2f}"
        if col.endswith("_x"):
            return f"{float(val):.2f}"
        if col.endswith("_s"):
            return f"{float(val):.1f}"
        if col.endswith("_Params"):
            return f"{int(val)}"
        if col in {"N", "L", "num_grid_points", "M", "K"}:
            return f"{int(val)}"
        return str(val)

    header_map = {
        "RbCRB": r"$R_{\mathrm{bCRB}}$",
        "RBar": r"$R_{\mathrm{Bar}}$",
        "ratio_bcrb_trace": r"$R_{\mathrm{bCRB}}$",
        "ratio_bar_trace": r"$R_{\mathrm{Bar}}$",
    }

    def esc(s: str) -> str:
        return s.replace("_", r"\_")

    cols = list(df.columns)
    col_spec = " ".join(["l" if c in {"Model", "Parameters"} else "r" for c in cols])
    lines = [rf"\begin{{tabular}}{{{col_spec}}}", r"\hline"]
    header_cells = [header_map.get(c, esc(c)) for c in cols]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        cells = [fmt(c, row[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = resolve_device(args.device)
    hidden_dims = parse_hidden_dims(args.mlp_hidden)
    scan_root = (REPO_ROOT / args.scan_cache_root).resolve()
    data_root = (REPO_ROOT / args.data_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_base: List[Dict] = []
    rows_runtime: List[Dict] = []
    rows_latency_stability: List[Dict] = []
    rows_feature_ablation: List[Dict] = []
    rows_global_allL: List[Dict] = []
    rows_global_byL: List[Dict] = []
    rows_projection_local: List[Dict] = []

    for model_name in args.models:
        if model_name not in MODEL_DISPLAY:
            raise ValueError(f"Unknown model: {model_name}")

        scan_csv = scan_root / model_name / f"{model_name}_scan_df.csv"
        scan_df = pd.read_csv(scan_csv)
        scan_df = add_trace_metrics(scan_df)
        row = load_representative_row(scan_csv, n_obs=args.n, delta=args.delta, omega=args.omega)
        base = build_accuracy_row(
            display_name=MODEL_DISPLAY[model_name],
            delta=float(args.delta),
            omega=float(args.omega),
            n_obs=int(args.n),
            scan_row=row,
        )
        rows_base.append(base)

        # Global ratio stats over the full scan grid.
        ratios_all = scan_df[["ratio_bcrb_trace", "ratio_bar_trace"]]
        rows_global_allL.append(
            {
                "Model": MODEL_DISPLAY[model_name],
                "num_grid_points": int(scan_df.shape[0]),
                "RbCRB_mean": float(ratios_all["ratio_bcrb_trace"].mean()),
                "RbCRB_median": float(ratios_all["ratio_bcrb_trace"].median()),
                "RbCRB_p95": float(np.percentile(ratios_all["ratio_bcrb_trace"], 95)),
                "RBar_mean": float(ratios_all["ratio_bar_trace"].mean()),
                "RBar_median": float(ratios_all["ratio_bar_trace"].median()),
                "RBar_p95": float(np.percentile(ratios_all["ratio_bar_trace"], 95)),
            }
        )
        for L, df_L in scan_df.groupby("L"):
            rows_global_byL.append(
                {
                    "Model": MODEL_DISPLAY[model_name],
                    "L": int(L),
                    "num_grid_points": int(df_L.shape[0]),
                    "RbCRB_mean": float(df_L["ratio_bcrb_trace"].mean()),
                    "RbCRB_median": float(df_L["ratio_bcrb_trace"].median()),
                    "RbCRB_p95": float(np.percentile(df_L["ratio_bcrb_trace"], 95)),
                    "RBar_mean": float(df_L["ratio_bar_trace"].mean()),
                    "RBar_median": float(df_L["ratio_bar_trace"].median()),
                    "RBar_p95": float(np.percentile(df_L["ratio_bar_trace"], 95)),
                }
            )

        proposed_model, ckpt = load_checkpoint_model(model_name=model_name, data_root=data_root, device=device)
        taus_batch, mask_batch = trajectory_batch_for_timing(
            model_name=model_name,
            data_root=data_root,
            batch_size=args.batch_size,
            n_obs=args.n,
        )
        prop_latency_stats = benchmark_latency_repeated(
            model=proposed_model,
            taus=taus_batch,
            mask=mask_batch,
            device=device,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            repeats=args.latency_repeats,
        )
        proposed_params = int(sum(p.numel() for p in proposed_model.parameters()))
        proposed_train_compute_s = checkpoint_train_compute_seconds(ckpt)

        if int(args.train_mlp_epochs) > 0:
            mlp_model, mlp_train_s, mlp_val_mse = train_mlp_baseline(
                model_name=model_name,
                data_root=data_root,
                n_obs=int(args.n),
                hidden_dims=hidden_dims,
                device=device,
                epochs=int(args.train_mlp_epochs),
                lr=float(args.mlp_lr),
                weight_decay=float(args.mlp_weight_decay),
                batch_size=int(args.mlp_batch_size),
                sample_cap=int(args.mlp_sample_cap),
                seed=int(args.seed),
            )
        else:
            state = ckpt["model_state"]
            target_mean = state["target_mean"].to(dtype=TRAIN_DTYPE)
            target_std = state["target_std"].to(dtype=TRAIN_DTYPE)
            mlp_model = TraditionalMLPBaseline(
                length=int(args.n),
                hidden_dims=hidden_dims,
                target_mean=target_mean,
                target_std=target_std,
                input_mean=None,
                input_std=None,
            ).to(device)
            mlp_train_s = float("nan")
            mlp_val_mse = float("nan")

        mlp_latency_stats = benchmark_latency_repeated(
            model=mlp_model,
            taus=taus_batch,
            mask=mask_batch,
            device=device,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            repeats=args.latency_repeats,
        )
        mlp_params = int(sum(p.numel() for p in mlp_model.parameters()))

        runtime_row = dict(base)
        runtime_row.update(
            {
                "Proposed_Params": proposed_params,
                "TraditionalNN_Params": mlp_params,
                "Proposed_TrainCompute_s": proposed_train_compute_s,
                "TraditionalNN_Train_s": mlp_train_s,
                "TraditionalNN_Val_MSE": mlp_val_mse,
                "Proposed_Infer_ms": prop_latency_stats.batch_ms_mean,
                "TraditionalNN_Infer_ms": mlp_latency_stats.batch_ms_mean,
                "Proposed_Infer_us_per_sample": prop_latency_stats.sample_us_mean,
                "TraditionalNN_Infer_us_per_sample": mlp_latency_stats.sample_us_mean,
                "TraditionalNN_over_Proposed_x": mlp_latency_stats.batch_ms_mean / max(prop_latency_stats.batch_ms_mean, 1e-12),
                "Proposed_Infer_ms_std": prop_latency_stats.batch_ms_std,
                "TraditionalNN_Infer_ms_std": mlp_latency_stats.batch_ms_std,
                "Proposed_Infer_us_per_sample_std": prop_latency_stats.sample_us_std,
                "TraditionalNN_Infer_us_per_sample_std": mlp_latency_stats.sample_us_std,
            }
        )
        rows_runtime.append(runtime_row)
        rows_latency_stability.append(
            {
                "Model": MODEL_DISPLAY[model_name],
                "N": int(args.n),
                "batch_size": int(args.batch_size),
                "repeats": int(args.latency_repeats),
                "Proposed_Infer_ms_mean": prop_latency_stats.batch_ms_mean,
                "Proposed_Infer_ms_std": prop_latency_stats.batch_ms_std,
                "TraditionalNN_Infer_ms_mean": mlp_latency_stats.batch_ms_mean,
                "TraditionalNN_Infer_ms_std": mlp_latency_stats.batch_ms_std,
                "Proposed_Infer_us_per_sample_mean": prop_latency_stats.sample_us_mean,
                "Proposed_Infer_us_per_sample_std": prop_latency_stats.sample_us_std,
                "TraditionalNN_Infer_us_per_sample_mean": mlp_latency_stats.sample_us_mean,
                "TraditionalNN_Infer_us_per_sample_std": mlp_latency_stats.sample_us_std,
            }
        )

        # Feature-efficiency ablation (prefix probes M=1..Kmax).
        taus_ab, mask_ab, y_ab = load_ablation_subset(
            model_name=model_name,
            data_root=data_root,
            n_obs=int(args.n),
            sample_cap=int(args.ablation_sample_cap),
            seed=int(args.seed),
        )
        mse_full = evaluate_model_trace_mse(
            model=proposed_model,
            taus=taus_ab,
            mask=mask_ab,
            y=y_ab,
            device=device,
            batch_size=int(args.ablation_batch_size),
        )
        K_total = int(proposed_model.K)
        K_eval = K_total if int(args.ablation_max_k) <= 0 else min(K_total, int(args.ablation_max_k))
        theta_ref = torch.as_tensor([float(args.delta), float(args.omega)], dtype=TRAIN_DTYPE, device=device)
        ret_map = projection_retention_diagnostics_by_prefix(proposed_model, theta_ref=theta_ref)

        # Local diagnostics over the full parameter grid used by the scan cache.
        theta_grid = (
            scan_df[["delta", "omega"]]
            .drop_duplicates()
            .sort_values(["delta", "omega"])
            .to_numpy(dtype=np.float64, copy=True)
        )
        local_rows = local_projection_diagnostics_grid(
            model=proposed_model,
            theta_grid=theta_grid,
            max_m=K_eval,
        )
        for r in local_rows:
            rows_projection_local.append(
                {
                    "Model": MODEL_DISPLAY[model_name],
                    "K": int(K_total),
                    **r,
                }
            )

        for M in range(1, K_eval + 1):
            reduced = build_reduced_probe_model(proposed_model, keep_m=M, device=device)
            mse_m = evaluate_model_trace_mse(
                model=reduced,
                taus=taus_ab,
                mask=mask_ab,
                y=y_ab,
                device=device,
                batch_size=int(args.ablation_batch_size),
            )
            ret_diag = dict(ret_map[M])
            rows_feature_ablation.append(
                {
                    "Model": MODEL_DISPLAY[model_name],
                    "N": int(args.n),
                    "K": int(K_total),
                    "M": int(M),
                    **ret_diag,
                    "empirical_trace_mse": float(mse_m),
                    "empirical_trace_mse_over_full": float(mse_m / max(mse_full, 1e-18)),
                }
            )

        print(
            f"[{model_name}] proposed={prop_latency_stats.batch_ms_mean:.2f}±{prop_latency_stats.batch_ms_std:.2f} ms/batch, "
            f"mlp={mlp_latency_stats.batch_ms_mean:.2f}±{mlp_latency_stats.batch_ms_std:.2f} ms/batch, "
            f"speedup={runtime_row['TraditionalNN_over_Proposed_x']:.2f}x"
        )

    base_df = pd.DataFrame(rows_base, columns=BASE_COLUMNS)
    runtime_cols = BASE_COLUMNS + [
        "Proposed_Params",
        "TraditionalNN_Params",
        "Proposed_TrainCompute_s",
        "TraditionalNN_Train_s",
        "TraditionalNN_Val_MSE",
        "Proposed_Infer_ms",
        "Proposed_Infer_ms_std",
        "TraditionalNN_Infer_ms",
        "TraditionalNN_Infer_ms_std",
        "Proposed_Infer_us_per_sample",
        "Proposed_Infer_us_per_sample_std",
        "TraditionalNN_Infer_us_per_sample",
        "TraditionalNN_Infer_us_per_sample_std",
        "TraditionalNN_over_Proposed_x",
    ]
    runtime_df = pd.DataFrame(rows_runtime, columns=runtime_cols)
    latency_stability_df = pd.DataFrame(rows_latency_stability)
    feature_ablation_df = pd.DataFrame(rows_feature_ablation)
    global_allL_df = pd.DataFrame(rows_global_allL)
    global_byL_df = pd.DataFrame(rows_global_byL).sort_values(["Model", "L"]).reset_index(drop=True)
    projection_local_df = pd.DataFrame(rows_projection_local).sort_values(["Model", "M", "delta", "omega"]).reset_index(drop=True)
    projection_local_stats_df = summarize_local_projection_diagnostics(projection_local_df)

    # Existing outputs (legacy/base).
    five_base_csv = output_dir / "representative_benchmark_five_models.csv"
    five_base_tex = output_dir / "representative_benchmark_five_models.tex"

    # New runtime outputs.
    five_runtime_csv = output_dir / "representative_benchmark_five_models_runtime.csv"
    five_runtime_tex = output_dir / "representative_benchmark_five_models_runtime.tex"
    latency_stability_csv = output_dir / "representative_benchmark_latency_stability.csv"
    latency_stability_tex = output_dir / "representative_benchmark_latency_stability.tex"
    feature_ablation_csv = output_dir / "representative_benchmark_feature_ablation.csv"
    feature_ablation_tex = output_dir / "representative_benchmark_feature_ablation.tex"
    projection_local_csv = output_dir / "representative_benchmark_projection_local_diagnostics.csv"
    projection_local_stats_csv = output_dir / "representative_benchmark_projection_local_stats.csv"
    projection_local_stats_tex = output_dir / "representative_benchmark_projection_local_stats.tex"
    global_allL_csv = output_dir / "representative_benchmark_global_ratio_stats_allL.csv"
    global_allL_tex = output_dir / "representative_benchmark_global_ratio_stats_allL.tex"
    global_byL_csv = output_dir / "representative_benchmark_global_ratio_stats_byL.csv"
    global_byL_tex = output_dir / "representative_benchmark_global_ratio_stats_byL.tex"

    base_df.to_csv(five_base_csv, index=False)
    write_latex_table(base_df, five_base_tex)

    runtime_df.to_csv(five_runtime_csv, index=False)
    write_latex_table(runtime_df, five_runtime_tex)

    latency_stability_df.to_csv(latency_stability_csv, index=False)
    write_latex_table(latency_stability_df, latency_stability_tex)

    feature_ablation_df.to_csv(feature_ablation_csv, index=False)
    write_latex_table(feature_ablation_df, feature_ablation_tex)

    projection_local_df.to_csv(projection_local_csv, index=False)
    projection_local_stats_df.to_csv(projection_local_stats_csv, index=False)
    write_latex_table(projection_local_stats_df, projection_local_stats_tex)

    global_allL_df.to_csv(global_allL_csv, index=False)
    write_latex_table(global_allL_df, global_allL_tex)

    global_byL_df.to_csv(global_byL_csv, index=False)
    write_latex_table(global_byL_df, global_byL_tex)

    print(f"Saved: {plus_base_csv}")
    print(f"Saved: {plus_base_tex}")
    print(f"Saved: {five_base_csv}")
    print(f"Saved: {five_base_tex}")
    print(f"Saved: {plus_runtime_csv}")
    print(f"Saved: {plus_runtime_tex}")
    print(f"Saved: {five_runtime_csv}")
    print(f"Saved: {five_runtime_tex}")
    print(f"Saved: {latency_stability_csv}")
    print(f"Saved: {latency_stability_tex}")
    print(f"Saved: {feature_ablation_csv}")
    print(f"Saved: {feature_ablation_tex}")
    print(f"Saved: {projection_local_csv}")
    print(f"Saved: {projection_local_stats_csv}")
    print(f"Saved: {projection_local_stats_tex}")
    print(f"Saved: {global_allL_csv}")
    print(f"Saved: {global_allL_tex}")
    print(f"Saved: {global_byL_csv}")
    print(f"Saved: {global_byL_tex}")


if __name__ == "__main__":
    main()
