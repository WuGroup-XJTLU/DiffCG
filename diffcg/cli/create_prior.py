from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diffcg.prior.targets import load_polymer_targets
from diffcg.prior.training import PriorConfig, pretrain_potentials


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain tabulated CG potentials from target distributions.")
    p.add_argument("--base_dir", required=True, help="Base dataset directory (contains T*/ and polymer/)")
    p.add_argument("--temperature", type=int, required=True)
    p.add_argument("--output", required=True, help="Output directory for params and plots")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    targets = load_polymer_targets(args.base_dir, args.temperature)
    cfg = PriorConfig(temperature=float(args.temperature), num_updates=args.steps, seed=args.seed)

    params_dict, grids = pretrain_potentials(targets, cfg)
    np.save(out / "pretrained_params.npy", params_dict)

    meta = {
        "temperature": args.temperature,
        "steps": args.steps,
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


