"""Price one new spot vector using a verified frozen research checkpoint."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model
from .estimators import fit_controls, path_samples, summarize
from .neural import gradient_callback
from .provenance import seed_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Model metadata .json, not an arbitrary pickle")
    parser.add_argument("--spots", type=float, nargs="+", required=True)
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=901)
    args = parser.parse_args()
    if args.paths < 3 or args.steps < 1:
        parser.error("Need >=3 paths and >=1 step")
    torch.set_num_threads(1)
    model, original_market, metadata = load_model(args.model)
    market = replace(original_market, spot=tuple(args.spots))
    grad = gradient_callback(model)
    py, pc = path_samples(market, args.paths, args.steps, seed_for("cli_pilot", args.seed), grad)
    beta = fit_controls(py, pc)
    y, c = path_samples(market, args.paths, args.steps, seed_for("cli_eval", args.seed), grad)
    answer = dict(model=metadata["id"], market=market.to_dict(), pilot_paths=args.paths,
                  evaluation_paths=args.paths, time_steps=args.steps, seed=args.seed,
                  beta=beta.tolist(), plain=summarize(y), combined=summarize(y - c @ beta),
                  notice="Research estimator; interval is asymptotic, not an investment recommendation.")
    print(json.dumps(answer, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
