#!/usr/bin/env python3
"""Plot projection-retention diagnostics from benchmark CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["TLS", "V", "Ladder", "Λ", "Shelving"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot local projection diagnostics.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("notebooks") / "scan_cache" / "benchmark_tables",
        help="Directory containing representative_benchmark_projection_local_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks") / "scan_cache" / "benchmark_tables" / "figures",
        help="Directory for plot outputs.",
    )
    parser.add_argument("--heatmap-m", type=int, default=1, help="Probe prefix M used for local heatmaps.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    return parser.parse_args()


def _sanitize_name(name: str) -> str:
    out = name.lower().replace(" ", "_").replace("-", "_")
    out = out.replace("λ", "lambda").replace("Λ", "lambda")
    return out


def _ordered_models(models: List[str]) -> List[str]:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def plot_local_heatmaps(df_local: pd.DataFrame, output_dir: Path, heatmap_m: int, dpi: int) -> List[Path]:
    paths: List[Path] = []
    metrics = [
        ("retained_fisher_fraction", r"$\mathrm{Ret}(M;\theta)$"),
        ("retained_fisher_fraction_delta", r"$\mathrm{Ret}_{\delta}(M;\theta)$"),
        ("retained_fisher_fraction_omega", r"$\mathrm{Ret}_{\omega}(M;\theta)$"),
    ]
    models = _ordered_models(df_local["Model"].drop_duplicates().tolist())
    for model in models:
        dfm = df_local.loc[(df_local["Model"] == model) & (df_local["M"] == int(heatmap_m))].copy()
        if dfm.empty:
            continue
        fig, axes = plt.subplots(1, len(metrics), figsize=(14.0, 3.8), constrained_layout=True)
        for ax, (col, title) in zip(axes, metrics):
            pivot = (
                dfm.pivot(index="delta", columns="omega", values=col)
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
            if pivot.empty:
                continue
            arr = pivot.to_numpy(dtype=float)
            im = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                extent=[
                    float(pivot.columns.min()),
                    float(pivot.columns.max()),
                    float(pivot.index.min()),
                    float(pivot.index.max()),
                ],
            )
            ax.set_title(title)
            ax.set_xlabel(r"$\omega$")
            ax.set_ylabel(r"$\delta$")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.set_ylabel("fraction")
        fig.suptitle(f"{model}: local retention landscapes at M={int(heatmap_m)}")
        out = output_dir / f"projection_local_heatmaps_m{int(heatmap_m)}_{_sanitize_name(model)}.png"
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def plot_stats_by_m(df_stats: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    metric_specs = [
        ("retained_fisher_fraction", "Total"),
        ("retained_fisher_fraction_delta", r"$\delta$"),
        ("retained_fisher_fraction_omega", r"$\omega$"),
    ]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(15.5, 4.6), constrained_layout=False)
    models = _ordered_models(df_stats["Model"].drop_duplicates().tolist())
    for ax, (metric, title) in zip(axes, metric_specs):
        for model in models:
            g = df_stats.loc[df_stats["Model"] == model].sort_values("M")
            if g.empty:
                continue
            x = g["M"].to_numpy(dtype=float)
            y_mean = g[f"{metric}_mean"].to_numpy(dtype=float)
            y_lo = g[f"{metric}_p05"].to_numpy(dtype=float)
            y_hi = g[f"{metric}_p95"].to_numpy(dtype=float)
            ax.plot(x, y_mean, marker="o", linewidth=1.8, label=model)
            ax.fill_between(x, y_lo, y_hi, alpha=0.16)
        ax.set_title(f"Retained fraction ({title})")
        ax.set_xlabel("M (kept probe blocks)")
        ax.set_ylabel("retained fraction")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=min(6, max(1, len(labels))),
        frameon=False,
        title="Model",
    )
    # Reserve headroom for the figure-level legend to avoid overlap with subplot titles.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    out = output_dir / "projection_retention_stats_by_m.png"
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_csv = input_dir / "representative_benchmark_projection_local_diagnostics.csv"
    stats_csv = input_dir / "representative_benchmark_projection_local_stats.csv"
    if not local_csv.exists():
        raise FileNotFoundError(f"Missing diagnostics CSV: {local_csv}")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Missing diagnostics stats CSV: {stats_csv}")

    df_local = pd.read_csv(local_csv)
    df_stats = pd.read_csv(stats_csv)
    required_local = {"Model", "delta", "omega", "M", "retained_fisher_fraction", "retained_fisher_fraction_delta", "retained_fisher_fraction_omega"}
    required_stats = {"Model", "M", "retained_fisher_fraction_mean", "retained_fisher_fraction_p05", "retained_fisher_fraction_p95"}
    if not required_local.issubset(df_local.columns):
        missing = sorted(required_local.difference(df_local.columns))
        raise RuntimeError(f"Local diagnostics CSV is missing columns: {missing}")
    if not required_stats.issubset(df_stats.columns):
        missing = sorted(required_stats.difference(df_stats.columns))
        raise RuntimeError(f"Stats CSV is missing columns: {missing}")

    heatmaps = plot_local_heatmaps(
        df_local=df_local,
        output_dir=output_dir,
        heatmap_m=max(1, int(args.heatmap_m)),
        dpi=int(args.dpi),
    )
    stats_fig = plot_stats_by_m(df_stats=df_stats, output_dir=output_dir, dpi=int(args.dpi))

    print("Saved plots:")
    for p in heatmaps:
        print(f"  {p}")
    print(f"  {stats_fig}")


if __name__ == "__main__":
    main()
