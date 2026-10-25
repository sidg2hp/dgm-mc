"""End-to-end benchmark: new models, independent pilots, immutable raw records."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .estimators import fit_controls, path_samples, rqmc, summarize, terminal_samples
from .market import equicorrelated, geometric_log_gradient, geometric_price
from .neural import gradient_callback, train
from .provenance import manifest, seed_for, sha256, write_json


def record(method, values, seconds, paths, **metadata):
    return dict(method=method, **summarize(values), seconds=seconds, paths=paths,
                sampling_unit="independent_path", **metadata)


def reference(market, config, key):
    if market.d == 1:
        return dict(price=float(geometric_price(market)[0]), se=0.0, method="analytical", paths=0)
    return dict(method="independent_rqmc_reference", **rqmc(
        market, config["reference_points"], config["reference_scrambles"], seed_for("reference", key)))


def classical(market, config, key, ref):
    rows = []
    for repeat in range(config["repeats"]):
        common = dict(d=market.d, scenario=key, repeat=repeat, reference_price=ref["price"], reference_se=ref["se"])
        seed = seed_for("classical_eval", key, repeat)
        begin = perf_counter()
        y, _ = terminal_samples(market, config["paths"], seed)
        rows.append(record("MC", y, perf_counter() - begin, config["paths"], **common))
        begin = perf_counter()
        y, _ = terminal_samples(market, config["paths"], seed, antithetic=True)
        av = record("Antithetic", y, perf_counter() - begin, config["paths"], **common)
        av["sampling_unit"] = "independent_antithetic_pair"
        rows.append(av)
        begin = perf_counter()
        py, pc = terminal_samples(market, config["pilot_paths"], seed_for("geo_pilot", key, repeat))
        beta = fit_controls(py, pc[:, None])
        pilot_seconds = perf_counter() - begin
        begin = perf_counter()
        y, c = terminal_samples(market, config["paths"], seed)
        values = y - beta[0] * c
        rows.append(record("Geometric CV", values, perf_counter() - begin, config["paths"],
                           pilot_paths=config["pilot_paths"], pilot_seconds=pilot_seconds,
                           beta=beta.tolist(), **common))
        sobol = rqmc(market, config["paths"] // config["scrambles"], config["scrambles"],
                     seed_for("rqmc_eval", key, repeat))
        rows.append(dict(method="RQMC", **sobol, **common))
    return rows


def neural_evaluation(market, model, config, key, ref, architecture, training_seed):
    rows = []
    gradient = gradient_callback(model)
    for repeat in range(config["repeats"]):
        pilot_seed = seed_for("neural_pilot", key, training_seed, repeat)
        eval_seed = seed_for("neural_eval", key, training_seed, repeat)
        begin = perf_counter()
        py, pc = path_samples(market, config["pilot_paths"], config["time_steps"], pilot_seed, gradient)
        beta_neural = fit_controls(py, pc[:, 1:])
        beta_combined = fit_controls(py, pc)
        pilot_seconds = perf_counter() - begin
        begin = perf_counter()
        y, controls = path_samples(market, config["paths"], config["time_steps"], eval_seed, gradient)
        elapsed = perf_counter() - begin
        common = dict(d=market.d, scenario=key, repeat=repeat, training_seed=training_seed,
                      architecture=architecture, time_steps=config["time_steps"],
                      reference_price=ref["price"], reference_se=ref["se"],
                      pilot_seed=pilot_seed, eval_seed=eval_seed,
                      pilot_seconds=pilot_seconds, pilot_paths=config["pilot_paths"])
        for name, coefficients in (("Neural fixed", [0.0, 1.0]),
                                   ("Neural pilot", [0.0, float(beta_neural[0])]),
                                   ("Combined CV", beta_combined.tolist())):
            fixed = name == "Neural fixed"
            row = record(name, y - controls @ coefficients, elapsed, config["paths"],
                         beta=coefficients, control_means=controls.mean(axis=0).tolist(),
                         paired_payoff_variance=float(y.var(ddof=1)), **common)
            row["variance_ratio"] = float(y.var(ddof=1) / max(row["variance"], 1e-300))
            if fixed:
                row["pilot_paths"] = 0
                row["pilot_seconds"] = 0.0
            rows.append(row)
    return rows


def calibration(config):
    rows = []
    # This section uses an analytic integrand, never labels it a learned result.
    for d in config["calibration_dimensions"]:
        market = equicorrelated(d)
        truth = float(geometric_price(market)[0])
        cases = [(steps, 1.0) for steps in config["calibration_grids"]]
        cases += [(config["time_steps"], scale) for scale in (-1.0, 0.0, 0.5, 2.0)]
        for steps, scale in cases:
            for repeat in range(config["calibration_repeats"]):
                seed = seed_for("calibration", d, steps, repeat)
                grad = lambda t, x, scale=scale: scale * geometric_log_gradient(market, t, x)
                y, c = path_samples(market, config["paths"], steps, seed, grad, kind="geometric")
                estimate = summarize(y - c[:, 1])
                rows.append(dict(d=d, steps=steps, scale=scale, repeat=repeat, seed=seed,
                                 exact_price=truth, **estimate,
                                 covered=bool(estimate["ci_low"] <= truth <= estimate["ci_high"]),
                                 variance_ratio=float(y.var(ddof=1) / max(estimate["variance"], 1e-300))))
    return rows


def validate_config(config):
    required_positive = ["paths", "pilot_paths", "time_steps", "repeats", "scrambles", "reference_points",
                         "reference_scrambles", "calibration_repeats", "threads"]
    for key in required_positive:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["paths"] % config["scrambles"]:
        raise ValueError("paths must divide evenly into independent scrambles")
    points = config["paths"] // config["scrambles"]
    if points < 2 or points & (points - 1) or config["scrambles"] < 2:
        raise ValueError("Each scramble needs a power-of-two point count and at least two scrambles")
    if config["pilot_paths"] < 3 or config["paths"] % 2:
        raise ValueError("Invalid pilot or antithetic budget")


def run(config, output, command):
    validate_config(config)
    output = Path(output)
    # Never merge old and new results or silently resume experiments.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    write_json(output / "manifest.json", manifest(config, command))
    records, direct, references, models = [], [], {}, []
    start = perf_counter()
    for d in sorted(set(config["dimensions"] + config["classical_dimensions"])):
        scenarios = []
        # Heterogeneous spots, not the equal-asset diagonal except in d=1.
        for money in config["moneyness"]:
            spots = money * np.exp(np.linspace(-0.12, 0.12, d)) if d > 1 else np.array([money])
            market = equicorrelated(d, spot=spots)
            key = f"d{d}_m{money:g}"
            ref = reference(market, config, key)
            references[key] = dict(market=market.to_dict(), **ref)
            scenarios.append((market, key, ref))
            records.extend(classical(market, config, key, ref))
        print(f"classical d={d} done", flush=True)
        if d not in config["dimensions"]:
            continue
        for architecture in config["architectures"]:
            for training_seed in config["training_seeds"]:
                training_market = equicorrelated(d)
                model, metadata = train(training_market, config["training"], training_seed, architecture)
                model_id = f"{architecture}_d{d}_seed{training_seed}"
                checkpoint = output / "models" / (model_id + ".pt")
                checkpoint.parent.mkdir(exist_ok=True)
                torch.save(dict(state_dict=model.state_dict(), metadata=metadata), checkpoint)
                metadata.update(id=model_id, checkpoint_sha256=sha256(checkpoint))
                models.append(metadata)
                write_json(output / "models" / (model_id + ".json"), metadata)
                for market, key, ref in scenarios:
                    with torch.no_grad():
                        price = model(torch.zeros((1, 1), dtype=torch.float64),
                                      torch.tensor(market.log_spot[None, :], dtype=torch.float64)).item()
                    direct.append(dict(model=model_id, architecture=architecture, training_seed=training_seed,
                                       d=d, scenario=key, price=price, reference_price=ref["price"], reference_se=ref["se"]))
                    records.extend(neural_evaluation(market, model, config, key, ref, architecture, training_seed))
                print(f"{model_id}: training {metadata['seconds']:.1f}s; evaluation done", flush=True)
        # Persist completed blocks but only mark complete after all experiments.
        write_json(output / "records.partial.json", records)
    write_json(output / "references.json", references)
    write_json(output / "direct_prices.json", direct)
    write_json(output / "records.json", records)
    print("analytic calibration starting", flush=True)
    write_json(output / "calibration.json", calibration(config))
    write_json(output / "completion.json", dict(status="complete", seconds=perf_counter() - start,
                                               models=len(models), records=len(records),
                                               artifact_sha256={p.relative_to(output).as_posix(): sha256(p)
                                                                for p in sorted(output.rglob("*"))
                                                                if p.is_file() and not p.name.endswith("partial.json")}))
    print(f"complete: {len(records)} estimate records, {len(models)} models", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output, "python -m dgm_mc.benchmark " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
