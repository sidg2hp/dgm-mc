"""Compile measured evidence; refuse incomplete or hash-inconsistent runs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .provenance import ROOT, sha256, write_json


def load_verified(folder):
    completion = json.loads((folder / "completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("Only completed runs may be reported")
    for name, expected in completion["artifact_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return {name: json.loads((folder / (name + ".json")).read_text())
            for name in ("records", "direct_prices", "references", "calibration", "manifest", "completion")}


def dump_csv(path, rows):
    if not rows:
        return
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_records(records, models):
    groups = defaultdict(list)
    for row in records:
        groups[(row["scenario"], row.get("architecture", "classical"), row["method"])].append(row)
    output = []
    for (scenario, arch, method), rows in sorted(groups.items(), key=lambda item: (item[1][0]["d"], item[0])):
        errors = np.array([r["price"] - r["reference_price"] for r in rows])
        seeds = sorted({r["training_seed"] for r in rows if "training_seed" in r})
        train_seconds = [models[f"{arch}_d{rows[0]['d']}_seed{s}"]["seconds"] for s in seeds]
        per_seed_rmse = [float(np.sqrt(np.mean([(r["price"] - r["reference_price"]) ** 2
                                               for r in rows if r.get("training_seed") == s]))) for s in seeds]
        warm = float(np.mean([r["seconds"] + r.get("pilot_seconds", 0) for r in rows]))
        mean_se2 = float(np.mean([r["se"] ** 2 for r in rows]))
        output.append(dict(scenario=scenario, d=rows[0]["d"], architecture=arch, method=method,
                           estimates=len(rows), training_seeds=len(seeds),
                           bias_vs_reference=float(errors.mean()), rmse_vs_reference=float(np.sqrt(np.mean(errors ** 2))),
                           rms_reported_se=float(np.sqrt(mean_se2)), reference_se=rows[0]["reference_se"],
                           empirical_price_sd=float(np.std([r["price"] for r in rows], ddof=1)),
                           warm_seconds=warm, training_seconds=float(np.mean(train_seconds)) if seeds else 0,
                           cold_seconds=warm + (float(np.mean(train_seconds)) if seeds else 0),
                           variance_time=mean_se2 * warm,
                           training_seed_rmse_sd=float(np.std(per_seed_rmse, ddof=1)) if len(seeds) > 1 else None))
    return output


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count ** 2)) / (1 + z * z / count)
    return float(center - half), float(center + half)


def summarize_calibration(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["d"], row["steps"], row["scale"])].append(row)
    output = []
    for (d, steps, scale), rows in sorted(groups.items()):
        errors = np.array([r["price"] - r["exact_price"] for r in rows])
        successes = sum(r["covered"] for r in rows)
        low, high = wilson(successes, len(rows))
        output.append(dict(d=d, steps=steps, scale=scale, repetitions=len(rows),
                           bias=float(errors.mean()), bias_mcse=float(errors.std(ddof=1) / np.sqrt(len(rows))),
                           rmse=float(np.sqrt(np.mean(errors ** 2))), coverage=successes / len(rows),
                           coverage_low=low, coverage_high=high,
                           mean_variance_ratio=float(np.mean([r["variance_ratio"] for r in rows]))))
    return output


def direct_summary(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["d"], r["architecture"], r["training_seed"])].append(r)
    return [dict(d=d, architecture=arch, training_seed=seed, states=len(rows),
                 relative_l2=float(np.linalg.norm([r["price"] - r["reference_price"] for r in rows]) /
                                   np.linalg.norm([r["reference_price"] for r in rows])),
                 negative_prices=sum(r["price"] < 0 for r in rows))
            for (d, arch, seed), rows in sorted(groups.items())]


def render_tables(destination, summary, calibration, direct, config):
    destination.mkdir(parents=True, exist_ok=True)
    lines = [r"% Generated from immutable raw records; do not edit.",
             r"\begin{tabular}{rllrrr}", r"\toprule",
             r"$d$ & Method & Arch. & RMSE & RMS SE & Warm ms \\", r"\midrule"]
    methods = {"MC", "Antithetic", "Geometric CV", "RQMC", "Neural pilot", "Combined CV"}
    for r in summary:
        if r["d"] not in config["dimensions"] or not r["scenario"].endswith("_m1") or r["method"] not in methods:
            continue
        lines.append(f"{r['d']} & {r['method']} & {r['architecture'].upper()} & {r['rmse_vs_reference']:.2e} & {r['rms_reported_se']:.2e} & {r['warm_seconds'] * 1000:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (destination / "performance_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule", r"$d$ & Steps & VR factor & Coverage & Wilson 95\% interval \\", r"\midrule"]
    for r in calibration:
        if r["scale"] == 1:
            lines.append(f"{r['d']} & {r['steps']} & {r['mean_variance_ratio']:.1f} & {r['coverage']:.3f} & [{r['coverage_low']:.3f}, {r['coverage_high']:.3f}] " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (destination / "calibration_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = [r"\begin{tabular}{rlrr}", r"\toprule", r"$d$ & Architecture & Mean rel. $L^2$ (\%) & Seed SD (pp) \\", r"\midrule"]
    for d in config["dimensions"]:
        for arch in config["architectures"]:
            vals = [r["relative_l2"] * 100 for r in direct if r["d"] == d and r["architecture"] == arch]
            lines.append(f"{d} & {arch.upper()} & {np.mean(vals):.2f} & {np.std(vals, ddof=1) if len(vals) > 1 else 0:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (destination / "direct_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_figures(destination, summary, calibration, config):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    dims = config["dimensions"]
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 3.5), squeeze=False)
    for ax, d in zip(axes[0], dims):
        for r in summary:
            if r["d"] != d or not r["scenario"].endswith("_m1") or r["architecture"] == "mlp" or r["method"] == "Neural fixed":
                continue
            # Roundoff-limited d=1 geometric controls are shown at the floor.
            ax.scatter(r["warm_seconds"], max(r["rmse_vs_reference"], 1e-14), label=r["method"], s=35)
        ax.set(xscale="log", yscale="log", title=f"d = {d}", xlabel="Evaluation + pilot (seconds)", ylabel="RMSE vs independent reference")
        ax.grid(alpha=.2)
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "cost_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    dims = config["calibration_dimensions"]
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 3.4), squeeze=False)
    for ax, d in zip(axes[0], dims):
        rows = [r for r in calibration if r["d"] == d and r["scale"] == 1]
        ax.errorbar([r["steps"] for r in rows], [r["coverage"] for r in rows],
                    yerr=[[r["coverage"] - r["coverage_low"] for r in rows],
                          [r["coverage_high"] - r["coverage"] for r in rows]], fmt="o-", capsize=3)
        ax.axhline(.95, color="gray", linestyle="--")
        ax.set(title=f"Analytic control, d = {d}", xlabel="Control time steps", ylabel="Empirical 95% coverage", ylim=(.65, 1.01), xscale="log")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(destination / "coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def run(folder, paper_dir=None):
    data = load_verified(folder)
    config = data["manifest"]["config"]
    models = {p.stem: json.loads(p.read_text()) for p in (folder / "models").glob("*.json")}
    summary = summarize_records(data["records"], models)
    calibration = summarize_calibration(data["calibration"])
    direct = direct_summary(data["direct_prices"])
    destination = folder / "reports"
    destination.mkdir(exist_ok=True)
    for name, rows in (("estimates", data["records"]), ("summary", summary),
                       ("calibration", calibration), ("direct", direct)):
        dump_csv(destination / (name + ".csv"), rows)
    write_json(destination / "summary.json", dict(performance=summary, calibration=calibration, direct=direct))
    text = ["# Measured results", "", "Generated from a completed, hash-verified run. No legacy performance is included.", "",
            f"- Fresh models: {data['completion']['models']}; estimate records: {len(data['records'])}.",
            f"- Analytic calibration records: {len(data['calibration'])}.",
            f"- Total recorded wall time: {data['completion']['seconds']:.1f} seconds.",
            f"- Training: {config['training']['steps']} steps; seeds {config['training_seeds']}.",
            f"- Evaluation paths: {config['paths']}; pilot paths: {config['pilot_paths']}; time steps: {config['time_steps']}.",
            "", "## Central moneyness scenario", "",
            "RMSE is against the recorded independent reference, not necessarily exact truth.",
            "Warm cost includes pilot; cold cost adds full training. No equal-compute claim.", "",
            "| d | Method | Architecture | RMSE | RMS SE | Warm ms | Cold s |",
            "|---|---|---|---:|---:|---:|---:|"]
    for r in summary:
        if r["scenario"].endswith("_m1"):
            text.append(f"| {r['d']} | {r['method']} | {r['architecture']} | {r['rmse_vs_reference']:.3g} | {r['rms_reported_se']:.3g} | {r['warm_seconds'] * 1000:.2f} | {r['cold_seconds']:.2f} |")
    text += ["", "## Interpretation limits", "",
             "- Three states per dimension are not a full-dimensional surface validation.",
             "- Classical d=25/50/100 results are not learned-model scaling evidence.",
             "- The geometric control is exact for a one-asset basket; roundoff-level errors are expected.",
             "- Forty calibration replications give wide coverage intervals; no coverage guarantee is inferred.",
             "- Repeated evaluation paths are not additional training seeds. Per-seed dispersion is in summary.csv.",
             "- Negative findings are retained. Statistical validity is distinct from efficiency and novelty."]
    (destination / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    render_tables(destination, summary, calibration, direct, config)
    render_figures(destination, summary, calibration, config)
    if paper_dir is not None:
        render_tables(paper_dir, summary, calibration, direct, config)
        render_figures(paper_dir, summary, calibration, config)
    lineage = dict(input_completion_sha256=sha256(folder / "completion.json"),
                   report_source_sha256=sha256(Path(__file__)),
                   input_manifest_sha256=sha256(folder / "manifest.json"),
                   outputs={p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name != "lineage.json"})
    write_json(destination / "lineage.json", lineage)
    if paper_dir is not None:
        write_json(paper_dir / "lineage.json", lineage)
    print(f"Compiled {len(summary)} scenario/method summaries into {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path)
    args = parser.parse_args()
    run(args.input, args.paper_dir)


if __name__ == "__main__":
    main()
