from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {
                key: float(value) if value not in ("", "nan", "NaN") else float("nan")
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def first_epoch_at(rows: list[dict], target: float) -> int | None:
    return next((int(row["epoch"]) for row in rows if row.get("AP", -1) >= target), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare COCO baseline and BQR-DN V2 runs.")
    parser.add_argument("baseline_history", type=Path)
    parser.add_argument("v2_history", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison"))
    args = parser.parse_args()
    baseline = read_history(args.baseline_history)
    v2 = read_history(args.v2_history)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    colors = {"Baseline": "#2563eb", "BQR-DN V2": "#16a34a"}
    runs = {"Baseline": baseline, "BQR-DN V2": v2}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for label, rows in runs.items():
        epochs = [row["epoch"] for row in rows]
        for metric in ("AP", "AP50", "AP75"):
            axes[0, 0].plot(epochs, [row[metric] for row in rows], label=f"{label} {metric}")
        for metric in ("loss_ce", "loss_bbox", "loss_giou"):
            axes[0, 1].plot(epochs, [row[metric] for row in rows], label=f"{label} {metric}")
        axes[1, 0].plot(epochs, [row["epoch_train_time_seconds"] / 60 for row in rows], marker="o", color=colors[label], label=label)
        axes[1, 1].plot(epochs, [row["peak_gpu_memory_mb"] / 1024 for row in rows], marker="o", color=colors[label], label=label)
    titles = ("COCO AP convergence", "Final-head training losses", "Train minutes / epoch", "Peak GPU memory (GiB)")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.savefig(args.output_dir / "baseline_vs_v2.png", dpi=180)
    plt.close(fig)

    baseline_final = baseline[-1]["AP"]
    summary = {
        "baseline_final": baseline[-1],
        "bqr_dn_v2_final": v2[-1],
        "delta_AP": v2[-1]["AP"] - baseline_final,
        "baseline_90pct_epoch": first_epoch_at(baseline, baseline_final * 0.90),
        "v2_90pct_epoch": first_epoch_at(v2, baseline_final * 0.90),
        "baseline_95pct_epoch": first_epoch_at(baseline, baseline_final * 0.95),
        "v2_95pct_epoch": first_epoch_at(v2, baseline_final * 0.95),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
