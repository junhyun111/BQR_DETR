from __future__ import annotations

import argparse
from pathlib import Path

from bqr_dn.config import ExperimentConfig
from bqr_dn.distributed import cleanup_distributed, initialize_distributed
from bqr_dn.model import ensure_common_initialization


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shared detector initialization")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--torch-cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    context = initialize_distributed()
    try:
        config = ExperimentConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            torch_cache=args.torch_cache,
            method="baseline",
            seed=args.seed,
        )
        payload = ensure_common_initialization(config, context)
        if context.is_main:
            print(
                f"prepared={config.initialization_path} "
                f"fingerprint={payload['initialization_fingerprint']}",
                flush=True,
            )
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
