from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


METRICS = ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR100")


def _read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        parsed: dict = {}
        for key, value in row.items():
            if value in ("", None):
                continue
            if value in ("True", "False"):
                parsed[key] = value == "True"
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        result.append(parsed)
    return result


def _validate_checkpoints(baseline: Path | None, bqr: Path | None) -> dict:
    if baseline is None or bqr is None:
        return {}
    left = torch.load(baseline, map_location="cpu")
    right = torch.load(bqr, map_location="cpu")
    checks = {
        "comparison_fingerprint": left.get("comparison_fingerprint")
        == right.get("comparison_fingerprint"),
        "detector_fingerprint": left.get("detector_fingerprint")
        == right.get("detector_fingerprint"),
        "initialization_fingerprint": left.get("initialization_fingerprint")
        == right.get("initialization_fingerprint"),
        "different_methods": left.get("method") == "baseline"
        and right.get("method") == "bqr",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid experiment pair: {', '.join(failed)}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and BQR histories")
    parser.add_argument("baseline_history", type=Path)
    parser.add_argument("bqr_history", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--bqr-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checks = _validate_checkpoints(args.baseline_checkpoint, args.bqr_checkpoint)
    baseline = {int(row["epoch"]): row for row in _read_history(args.baseline_history)}
    bqr = {int(row["epoch"]): row for row in _read_history(args.bqr_history)}
    epochs = sorted(set(baseline) & set(bqr))
    comparisons = []
    for epoch in epochs:
        if "AP" not in baseline[epoch] or "AP" not in bqr[epoch]:
            continue
        row = {"epoch": epoch}
        for metric in METRICS:
            if metric in baseline[epoch] and metric in bqr[epoch]:
                row[f"baseline_{metric}"] = baseline[epoch][metric]
                row[f"bqr_{metric}"] = bqr[epoch][metric]
                row[f"delta_{metric}"] = bqr[epoch][metric] - baseline[epoch][metric]
        comparisons.append(row)
    if not comparisons:
        raise ValueError("No common evaluated epochs were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {"pair_checks": checks, "epochs": comparisons, "final": comparisons[-1]}
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric in ("AP", "AP75"):
        axes[0].plot(
            [row["epoch"] for row in comparisons],
            [row[f"baseline_{metric}"] for row in comparisons],
            marker="o",
            label=f"baseline {metric}",
        )
        axes[0].plot(
            [row["epoch"] for row in comparisons],
            [row[f"bqr_{metric}"] for row in comparisons],
            marker="o",
            linestyle="--",
            label=f"BQR {metric}",
        )
    axes[0].set(xlabel="epoch", ylabel="COCO metric", title="Validation curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for metric in ("AP", "AP75"):
        axes[1].plot(
            [row["epoch"] for row in comparisons],
            [row[f"delta_{metric}"] for row in comparisons],
            marker="o",
            label=f"BQR - baseline {metric}",
        )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set(xlabel="epoch", ylabel="delta", title="BQR improvement")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
