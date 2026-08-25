from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify baseline/V2 comparison invariants.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("v2", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    v2 = torch.load(args.v2, map_location="cpu", weights_only=False)
    checks = {
        "methods": {baseline.get("method"), v2.get("method")} == {"baseline", "bqr_dn_v2"},
        "comparison_fingerprint": baseline.get("comparison_fingerprint") == v2.get("comparison_fingerprint"),
        "detector_fingerprint": baseline.get("detector_fingerprint") == v2.get("detector_fingerprint"),
        "initialization_fingerprint": (
            baseline.get("initialization_fingerprint")
            and baseline.get("initialization_fingerprint")
            == v2.get("initialization_fingerprint")
        ),
        "upstream_commit": baseline.get("upstream_commit") == v2.get("upstream_commit"),
        "completed_epochs": (
            int(baseline.get("epoch", -1)) + 1 == int(baseline["config"]["epochs"])
            and int(v2.get("epoch", -1)) + 1 == int(v2["config"]["epochs"])
        ),
    }
    report = {"ok": all(checks.values()), "checks": checks}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if not report["ok"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
