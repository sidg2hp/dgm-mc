"""Independent analytical price/Greek and learned-control coverage extension."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.special import ndtr

from .checkpoint import load_model
from .estimators import fit_controls, path_samples, summarize
from .market import geometric_price
from .neural import derivative, gradient_callback
from .provenance import manifest, seed_for, sha256, write_json
from .report import dump_csv, wilson


def run(model_dir, output, repeats=80, paths=1024, steps=8):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = dict(repeats=repeats, paths=paths, steps=steps, spots=101, times=[0.0, 0.5, 0.9])
    write_json(output / "manifest.json", manifest(config, "python -m dgm_mc.diagnostics"))
    greek_rows, coverage_rows, summaries = [], [], []
    begin = perf_counter()
    for meta_path in sorted(model_dir.glob("*_d1_seed*.json")):
        model, market, metadata = load_model(meta_path)
        gradient = gradient_callback(model)
        for time in config["times"]:
            spots = np.linspace(.6, 1.4, config["spots"])
            x = torch.tensor(np.log(spots / market.strike)[:, None], dtype=torch.float64, requires_grad=True)
            t = torch.full((len(x), 1), time, dtype=torch.float64, requires_grad=True)
            u = model(t, x)
            ux = derivative(u, x)
            uxx = derivative(ux, x)
            prices = u.detach().numpy().ravel()
            delta = ux.detach().numpy().ravel() / spots
            gamma = (uxx - ux).detach().numpy().ravel() / spots ** 2
            tau, sigma = market.maturity - time, market.sigma[0]
            d1 = (np.log(spots / market.strike) + (market.rate + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
            exact_p = geometric_price(market, np.log(spots / market.strike)[:, None], tau)
            exact_d = ndtr(d1)
            exact_g = np.exp(-d1 ** 2 / 2) / (np.sqrt(2 * np.pi) * spots * sigma * np.sqrt(tau))
            for i, spot in enumerate(spots):
                greek_rows.append(dict(model=metadata["id"], time=time, spot=float(spot), price=float(prices[i]),
                                       delta=float(delta[i]), gamma=float(gamma[i]), reference_price=float(exact_p[i]),
                                       reference_delta=float(exact_d[i]), reference_gamma=float(exact_g[i])))
            summaries.append(dict(model=metadata["id"], time=time,
                                  price_relative_l2=float(np.linalg.norm(prices - exact_p) / np.linalg.norm(exact_p)),
                                  delta_relative_l2=float(np.linalg.norm(delta - exact_d) / np.linalg.norm(exact_d)),
                                  gamma_relative_l2=float(np.linalg.norm(gamma - exact_g) / np.linalg.norm(exact_g)),
                                  negative_gamma_count=int((gamma < 0).sum()), negative_price_count=int((prices < 0).sum())))
        truth = float(geometric_price(market)[0])
        for repeat in range(repeats):
            py, pc = path_samples(market, paths, steps, seed_for("diagnostic_pilot", metadata["id"], repeat), gradient)
            beta = fit_controls(py, pc[:, 1:])[0]
            y, c = path_samples(market, paths, steps, seed_for("diagnostic_eval", metadata["id"], repeat), gradient)
            for method, coefficient in (("fixed", 1.0), ("pilot", float(beta))):
                result = summarize(y - coefficient * c[:, 1])
                coverage_rows.append(dict(model=metadata["id"], repeat=repeat, method=method,
                                          coefficient=coefficient, exact_price=truth,
                                          covered=result["ci_low"] <= truth <= result["ci_high"], **result))
        print(f"diagnostics {metadata['id']} done", flush=True)
    if not greek_rows:
        raise ValueError("No 1D model metadata found")
    cov_summary = []
    for model in sorted({r["model"] for r in coverage_rows}):
        for method in ("fixed", "pilot"):
            rows = [r for r in coverage_rows if r["model"] == model and r["method"] == method]
            errors = np.array([r["price"] - r["exact_price"] for r in rows])
            count = sum(r["covered"] for r in rows)
            low, high = wilson(count, len(rows))
            cov_summary.append(dict(model=model, method=method, repetitions=len(rows), coverage=count / len(rows),
                                    coverage_low=low, coverage_high=high, bias=float(errors.mean()),
                                    bias_mcse=float(errors.std(ddof=1) / np.sqrt(len(rows))),
                                    rmse=float(np.sqrt(np.mean(errors ** 2)))))
    for name, rows in (("greeks_raw", greek_rows), ("greeks_summary", summaries),
                       ("learned_coverage_raw", coverage_rows), ("learned_coverage_summary", cov_summary)):
        write_json(output / (name + ".json"), rows)
        dump_csv(output / (name + ".csv"), rows)
    write_json(output / "completion.json", dict(status="complete", seconds=perf_counter() - begin,
                                               source_models={p.name: sha256(p) for p in sorted(model_dir.glob("*_d1_seed*"))},
                                               artifact_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--paths", type=int, default=1024)
    args = parser.parse_args()
    if args.repeats < 2 or args.paths < 3:
        parser.error("Need >=2 repetitions and >=3 paths")
    run(args.models, args.output, args.repeats, args.paths)


if __name__ == "__main__":
    main()
