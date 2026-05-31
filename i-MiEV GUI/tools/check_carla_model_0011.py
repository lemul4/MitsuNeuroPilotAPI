"""Check that CARLA/LEAD checkpoint directory can load model_0011.pth only."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint directory passed to leaderboard_wrapper --checkpoint")
    parser.add_argument("--model-file", default="model_0011.pth")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"FAIL: checkpoint directory does not exist: {ckpt}")
    if not ckpt.is_dir():
        raise SystemExit(f"FAIL: checkpoint is not a directory: {ckpt}")
    cfg = ckpt / "config.json"
    if not cfg.exists():
        raise SystemExit(f"FAIL: config.json missing in checkpoint directory: {cfg}")
    target = ckpt / args.model_file
    if not target.exists():
        raise SystemExit(f"FAIL: {args.model_file} missing in checkpoint directory: {target}")

    all_models = sorted(p.name for p in ckpt.glob("model*.pth"))
    print(f"checkpoint: {ckpt}")
    print(f"config:     {cfg}")
    print(f"models:     {all_models}")
    print(f"selected:   {args.model_file}")
    if len(all_models) > 1:
        print("NOTE: without MITSU_CARLA_MODEL_FILE the current OpenLoopInference loads all model*.pth as an ensemble.")
        print(f"Set:  $env:MITSU_CARLA_MODEL_FILE = \"{args.model_file}\"")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
