import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.integrate import cumulative_trapezoid, trapezoid as scipy_trapezoid

try:
    import torch
except Exception:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tools import (
    build_noniid_three_level_kernel_model,
    evaluate_detected_kernel_matrix,
    kernel_conditional_transition_matrix,
    kernel_stationary_distribution,
    tls_waiting_time_real,
)


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

MODEL_ORDER = ["tls", "shelving", "lambda", "v", "ladder"]


def trapz_integral(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    if hasattr(np, "trapz"):
        return np.trapz(y, x, axis=axis)
    return scipy_trapezoid(y, x, axis=axis)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate trajectory datasets for TLS and three-level models.")
    parser.add_argument("--output-root", type=Path, default=Path("data") / "tragectories")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--n-trajectories", type=int, default=1_000_000)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--n-param-points", type=int, default=2048)
    parser.add_argument("--delta-min", type=float, default=0.0)
    parser.add_argument("--delta-max", type=float, default=3.0)
    parser.add_argument("--omega-min", type=float, default=0.25)
    parser.add_argument("--omega-max", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=4096)
    parser.add_argument("--tau-max-init", type=float, default=40.0)
    parser.add_argument("--tau-max-limit", type=float, default=1024.0)
    parser.add_argument("--tail-tol", type=float, default=1e-4)
    parser.add_argument("--min-captured-area", type=float, default=0.995)
    parser.add_argument("--max-redraw-attempts", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device_arg == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda is not available.")
    return device_arg


def make_waiting_time_fn(model_name):
    if model_name == "tls":
        return lambda tau, delta, omega: tls_waiting_time_real(tau, delta=delta, omega=omega, eps=1e-300)
    raise ValueError(f"renewal inverse-CDF sampling is only available for TLS, got model={model_name!r}")


def build_three_level_kernel_model(model_name, delta, omega):
    defaults = THREE_LEVEL_DEFAULTS[model_name]
    return build_noniid_three_level_kernel_model(
        delta=delta,
        omega=omega,
        scheme=model_name,
        gamma_det=defaults["gamma_det"],
        gamma_hidden=defaults["gamma_hidden"],
        gamma_aux=defaults["gamma_aux"],
        delta_aux=defaults["delta_aux"],
        omega_aux=defaults["omega_aux"],
    )


def build_three_level_noniid_tables(kernel_model, grid_size, tau_max_init, tau_max_limit, tail_tol):
    tau_max = float(tau_max_init)
    tau = None
    kernel = None
    pdf_by_state = None
    areas = None

    while True:
        tau = np.linspace(0.0, tau_max, int(grid_size), dtype=np.float64)
        kernel = np.real_if_close(evaluate_detected_kernel_matrix(tau, kernel_model))
        kernel = np.maximum(np.asarray(kernel, dtype=np.float64), 0.0)
        if kernel.ndim != 3:
            raise RuntimeError(f"expected detected kernel matrix with shape (T, n, n), got shape={kernel.shape}")
        pdf_by_state = np.sum(kernel, axis=1)
        areas = trapz_integral(pdf_by_state, tau, axis=0)
        if float(np.min(areas)) >= 1.0 - float(tail_tol) or tau_max >= float(tau_max_limit):
            break
        tau_max *= 2.0

    if tau is None or kernel is None or pdf_by_state is None or areas is None:
        raise RuntimeError("failed to build noniid three-level sampling tables")

    cdf_by_state = cumulative_trapezoid(pdf_by_state, tau, axis=0, initial=0.0)
    final = cdf_by_state[-1]
    if np.any(~np.isfinite(final)) or np.any(final <= 0.0):
        raise RuntimeError(f"invalid noniid CDF normalizers: {final}")
    cdf_by_state = cdf_by_state / final[None, :]

    inverse_cdfs = []
    n_states = int(cdf_by_state.shape[1])
    for state in range(n_states):
        cdf_state = cdf_by_state[:, state]
        keep = np.r_[True, np.diff(cdf_state) > 1e-14]
        tau_state = tau[keep]
        cdf_state = cdf_state[keep]
        cdf_state[0] = 0.0
        cdf_state[-1] = 1.0
        inverse_cdfs.append((tau_state, cdf_state))

    transition = kernel_conditional_transition_matrix(trapz_integral(kernel, tau, axis=0))
    stationary = kernel_stationary_distribution(transition)
    area_weighted = float(np.dot(stationary, np.asarray(areas, dtype=np.float64)))

    return {
        "tau": tau,
        "kernel": kernel,
        "inverse_cdfs": inverse_cdfs,
        "transition": transition,
        "stationary": stationary,
        "areas": np.asarray(areas, dtype=np.float64),
        "area_weighted": area_weighted,
        "tau_max": float(tau_max),
    }


def sample_three_level_noniid_trajectories(rng, tables, n_traj, length):
    n_traj = int(n_traj)
    length = int(length)
    if n_traj <= 0:
        return np.empty((0, length), dtype=np.float32)

    tau_grid = tables["tau"]
    kernel = tables["kernel"]
    inverse_cdfs = tables["inverse_cdfs"]
    transition = tables["transition"]
    stationary = tables["stationary"]
    n_states = len(inverse_cdfs)

    states = rng.choice(n_states, size=n_traj, p=stationary)
    out = np.empty((n_traj, length), dtype=np.float32)

    for step in range(length):
        waits = np.empty(n_traj, dtype=np.float64)
        u = rng.random(n_traj, dtype=np.float64)

        for state in range(n_states):
            mask = states == state
            if not np.any(mask):
                continue
            tau_state, cdf_state = inverse_cdfs[state]
            waits[mask] = np.interp(u[mask], cdf_state, tau_state)

        out[:, step] = waits.astype(np.float32, copy=False)

        next_states = np.empty_like(states)
        for state in range(n_states):
            mask = states == state
            if not np.any(mask):
                continue
            waits_state = waits[mask]
            probs = np.empty((waits_state.shape[0], n_states), dtype=np.float64)
            for nxt in range(n_states):
                probs[:, nxt] = np.interp(waits_state, tau_grid, kernel[:, nxt, state])
            probs = np.maximum(probs, 0.0)
            row_sum = np.sum(probs, axis=1, keepdims=True)
            zero_rows = row_sum[:, 0] <= 0.0
            if np.any(zero_rows):
                probs[zero_rows, :] = transition[:, state][None, :]
                row_sum = np.sum(probs, axis=1, keepdims=True)
            probs = probs / np.clip(row_sum, 1e-12, None)
            cumulative = np.cumsum(probs, axis=1)
            draws = rng.random(cumulative.shape[0], dtype=np.float64)[:, None]
            next_states[mask] = np.sum(draws > cumulative, axis=1)

        states = next_states

    return out


def build_inverse_cdf(waiting_time_fn, delta, omega, grid_size, tau_max_init, tau_max_limit, tail_tol):
    tau_max = float(tau_max_init)
    tau = None
    pdf = None
    area = 0.0

    while True:
        tau = np.linspace(0.0, tau_max, int(grid_size), dtype=np.float64)
        pdf = np.asarray(waiting_time_fn(tau, delta=delta, omega=omega), dtype=np.float64)
        pdf = np.maximum(pdf, 0.0)
        area = float(trapz_integral(pdf, tau))

        if area >= 1.0 - tail_tol or tau_max >= tau_max_limit:
            break
        tau_max *= 2.0

    if tau is None or pdf is None:
        raise RuntimeError("failed to build inverse CDF")

    cdf = cumulative_trapezoid(pdf, tau, initial=0.0)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0.0:
        raise RuntimeError(f"invalid CDF for delta={delta}, omega={omega}, area={area}")
    cdf = cdf / cdf[-1]

    keep = np.r_[True, np.diff(cdf) > 1e-14]
    tau = tau[keep]
    cdf = cdf[keep]
    cdf[0] = 0.0
    cdf[-1] = 1.0
    return tau, cdf, area, tau_max


def sample_trajectories_from_inverse_cdf_cpu(rng, tau, cdf, n_traj, length):
    u = rng.random((int(n_traj), int(length)), dtype=np.float64)
    samples = np.interp(u.reshape(-1), cdf, tau).reshape(int(n_traj), int(length))
    return samples.astype(np.float32, copy=False)


def sample_trajectories_from_inverse_cdf_gpu(torch_gen, tau, cdf, n_traj, length, device):
    tau_t = torch.as_tensor(tau, device=device, dtype=torch.float64)
    cdf_t = torch.as_tensor(cdf, device=device, dtype=torch.float64)
    u = torch.rand((int(n_traj), int(length)), generator=torch_gen, device=device, dtype=torch.float64)
    flat = u.reshape(-1)
    idx = torch.searchsorted(cdf_t, flat, right=False)
    idx = idx.clamp(min=1, max=cdf_t.numel() - 1)
    c0 = cdf_t[idx - 1]
    c1 = cdf_t[idx]
    t0 = tau_t[idx - 1]
    t1 = tau_t[idx]
    weight = (flat - c0) / torch.clamp(c1 - c0, min=1e-12)
    out = (t0 + weight * (t1 - t0)).reshape(int(n_traj), int(length))
    return out.to(dtype=torch.float32).cpu().numpy()


def sample_trajectories_from_inverse_cdf(rng, tau, cdf, n_traj, length, device, torch_gen=None):
    if device == "cuda":
        if torch_gen is None:
            raise ValueError("torch_gen must be provided for CUDA sampling.")
        return sample_trajectories_from_inverse_cdf_gpu(torch_gen, tau, cdf, n_traj, length, device=device)
    return sample_trajectories_from_inverse_cdf_cpu(rng, tau, cdf, n_traj, length)


def allocate_counts(n_total, n_param_points):
    base = int(n_total) // int(n_param_points)
    rem = int(n_total) % int(n_param_points)
    counts = np.full(int(n_param_points), base, dtype=np.int64)
    counts[:rem] += 1
    return counts


def ensure_clean_target_dir(target_dir: Path, overwrite: bool):
    target_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for name in ["trajectories.npy", "params.npy", "param_points.npy", "metadata.json"]:
        path = target_dir / name
        if path.exists():
            path.unlink()


def sample_valid_param_point(rng, waiting_time_fn, args):
    for attempt in range(1, int(args.max_redraw_attempts) + 1):
        delta = float(rng.uniform(args.delta_min, args.delta_max))
        omega = float(rng.uniform(args.omega_min, args.omega_max))
        tau, cdf, area, tau_max = build_inverse_cdf(
            waiting_time_fn=waiting_time_fn,
            delta=delta,
            omega=omega,
            grid_size=args.grid_size,
            tau_max_init=args.tau_max_init,
            tau_max_limit=args.tau_max_limit,
            tail_tol=args.tail_tol,
        )
        if area >= float(args.min_captured_area):
            return delta, omega, tau, cdf, area, tau_max, attempt
    raise RuntimeError(
        "failed to draw a valid parameter point within the redraw budget; "
        f"last constraint was min_captured_area={args.min_captured_area}"
    )


def sample_valid_param_point_three_level_noniid(rng, model_name, args):
    for attempt in range(1, int(args.max_redraw_attempts) + 1):
        delta = float(rng.uniform(args.delta_min, args.delta_max))
        omega = float(rng.uniform(args.omega_min, args.omega_max))
        kernel_model = build_three_level_kernel_model(model_name=model_name, delta=delta, omega=omega)
        tables = build_three_level_noniid_tables(
            kernel_model=kernel_model,
            grid_size=args.grid_size,
            tau_max_init=args.tau_max_init,
            tau_max_limit=args.tau_max_limit,
            tail_tol=args.tail_tol,
        )
        area = float(tables["area_weighted"])
        if area >= float(args.min_captured_area):
            return delta, omega, tables, area, float(tables["tau_max"]), attempt
    raise RuntimeError(
        f"failed to draw a valid noniid parameter point for model={model_name} within redraw budget; "
        f"last constraint was min_captured_area={args.min_captured_area}"
    )


def generate_model_dataset(model_name, args):
    target_dir = args.output_root / model_name
    ensure_clean_target_dir(target_dir, overwrite=args.overwrite)

    trajectories_path = target_dir / "trajectories.npy"
    params_path = target_dir / "params.npy"
    param_points_path = target_dir / "param_points.npy"
    metadata_path = target_dir / "metadata.json"

    if (not args.overwrite) and all(p.exists() for p in [trajectories_path, params_path, param_points_path, metadata_path]):
        print(f"[{model_name}] target files already exist, skipping")
        return

    model_seed = int(args.seed + 1000 * MODEL_ORDER.index(model_name))
    rng = np.random.default_rng(model_seed)
    device = resolve_device(args.device)
    is_three_level = model_name in THREE_LEVEL_DEFAULTS
    sampling_mode = "noniid" if is_three_level else "renewal"
    draw_device = "cpu" if sampling_mode == "noniid" else device

    torch_gen = None
    if sampling_mode == "renewal" and device == "cuda":
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(model_seed)
    if sampling_mode == "noniid" and device == "cuda":
        print(f"[{model_name}] noniid three-level sampling is CPU-only; using CPU trajectory draws.")

    waiting_time_fn = make_waiting_time_fn(model_name) if model_name == "tls" else None

    n_total = int(args.n_trajectories)
    n_param_points = min(int(args.n_param_points), n_total)
    counts = allocate_counts(n_total, n_param_points)
    rng.shuffle(counts)

    param_points = np.empty((n_param_points, 2), dtype=np.float32)

    traj_mm = open_memmap(trajectories_path, mode="w+", dtype=np.float32, shape=(n_total, int(args.length)))
    param_mm = open_memmap(params_path, mode="w+", dtype=np.float32, shape=(n_total, 2))

    rows_done = 0
    areas = []
    tau_max_values = []
    redraw_attempts = []
    t0 = time.time()

    for idx, n_rep in enumerate(counts, start=1):
        if sampling_mode == "renewal":
            delta, omega, tau, cdf, area, tau_max, attempts = sample_valid_param_point(
                rng=rng,
                waiting_time_fn=waiting_time_fn,
                args=args,
            )
            block = sample_trajectories_from_inverse_cdf(
                rng=rng,
                tau=tau,
                cdf=cdf,
                n_traj=int(n_rep),
                length=int(args.length),
                device=device,
                torch_gen=torch_gen,
            )
        else:
            delta, omega, tables, area, tau_max, attempts = sample_valid_param_point_three_level_noniid(
                rng=rng,
                model_name=model_name,
                args=args,
            )
            block = sample_three_level_noniid_trajectories(
                rng=rng,
                tables=tables,
                n_traj=int(n_rep),
                length=int(args.length),
            )

        param_points[idx - 1, 0] = np.float32(delta)
        param_points[idx - 1, 1] = np.float32(omega)

        row_slice = slice(rows_done, rows_done + int(n_rep))
        traj_mm[row_slice] = block
        param_mm[row_slice, 0] = delta
        param_mm[row_slice, 1] = omega
        rows_done += int(n_rep)
        areas.append(float(area))
        tau_max_values.append(float(tau_max))
        redraw_attempts.append(int(attempts))

        if idx % 128 == 0 or idx == n_param_points:
            elapsed = time.time() - t0
            rate = rows_done / max(elapsed, 1e-9)
            print(
                f"[{model_name}] param_points={idx}/{n_param_points} "
                f"rows={rows_done}/{n_total} rate={rate:,.0f} traj/s "
                f"area_min={min(areas):.6f}"
            )

    traj_mm.flush()
    param_mm.flush()
    np.save(param_points_path, param_points)

    metadata = {
        "model": model_name,
        "device": draw_device,
        "requested_device": device,
        "sampling_mode": sampling_mode,
        "n_trajectories": n_total,
        "trajectory_length": int(args.length),
        "n_param_points": n_param_points,
        "delta_range": [float(args.delta_min), float(args.delta_max)],
        "omega_range": [float(args.omega_min), float(args.omega_max)],
        "seed": model_seed,
        "grid_size": int(args.grid_size),
        "tau_max_init": float(args.tau_max_init),
        "tau_max_limit": float(args.tau_max_limit),
        "tail_tol": float(args.tail_tol),
        "min_captured_area": float(args.min_captured_area),
        "max_redraw_attempts": int(args.max_redraw_attempts),
        "mean_area_before_renorm": float(np.mean(areas)),
        "min_area_before_renorm": float(np.min(areas)),
        "max_area_before_renorm": float(np.max(areas)),
        "mean_tau_max": float(np.mean(tau_max_values)),
        "max_tau_max": float(np.max(tau_max_values)),
        "mean_redraw_attempts": float(np.mean(redraw_attempts)),
        "max_redraw_attempts_used": int(np.max(redraw_attempts)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[{model_name}] done -> {target_dir}")


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"output_root = {args.output_root}")
    print(f"models = {args.models}")
    print(f"n_trajectories = {args.n_trajectories:,}, length = {args.length}, n_param_points = {args.n_param_points}")

    for model_name in args.models:
        generate_model_dataset(model_name, args)


if __name__ == "__main__":
    main()
