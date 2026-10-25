"""Generate diagnostic tables and figures from the independent extension."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .provenance import sha256, write_json


def run(folder, destination):
    completion = json.loads((folder / "completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("Incomplete diagnostics")
    for filename, digest in completion["artifact_sha256"].items():
        if sha256(folder / filename) != digest:
            raise ValueError(f"Diagnostic hash mismatch: {filename}")
    summaries = json.loads((folder / "greeks_summary.json").read_text())
    coverage = json.loads((folder / "learned_coverage_summary.json").read_text())
    raw = json.loads((folder / "greeks_raw.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Architecture & $t$ & Price rel. $L^2$ (\%) & Delta (\%) & Gamma (\%) \\", r"\midrule"]
    for architecture in ("dgm", "mlp"):
        for time in (0.0, 0.5, 0.9):
            rows = [r for r in summaries if r["model"].startswith(architecture) and r["time"] == time]
            numbers = [np.mean([r[k] for r in rows]) * 100 for k in ("price_relative_l2", "delta_relative_l2", "gamma_relative_l2")]
            lines.append(f"{architecture.upper()} & {time:.1f} & {numbers[0]:.2f} & {numbers[1]:.2f} & {numbers[2]:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (destination / "greeks_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model / seed & Coverage & Wilson 95\% interval & Bias & Bias MCSE \\", r"\midrule"]
    for r in coverage:
        if r["method"] != "pilot":
            continue
        name = r["model"].replace("_d1_seed", " / ").upper()
        lines.append(f"{name} & {r['coverage']:.4f} & [{r['coverage_low']:.3f}, {r['coverage_high']:.3f}] & {r['bias']:.2e} & {r['bias_mcse']:.2e} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (destination / "learned_coverage_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
    for column, time in enumerate((0.0, 0.5, 0.9)):
        for row_index, quantity in enumerate(("delta", "gamma")):
            ax = axes[row_index, column]
            exact = sorted([r for r in raw if r["time"] == time and r["model"] == "dgm_d1_seed11"], key=lambda r: r["spot"])
            ax.plot([r["spot"] for r in exact], [r["reference_" + quantity] for r in exact], "k--", label="Exact")
            for architecture, color in (("dgm", "tab:blue"), ("mlp", "tab:orange")):
                arrays = []
                spots = []
                for seed in (11, 29, 47):
                    selected = sorted([r for r in raw if r["time"] == time and r["model"] == f"{architecture}_d1_seed{seed}"], key=lambda r: r["spot"])
                    spots = [r["spot"] for r in selected]
                    arrays.append([r[quantity] for r in selected])
                values = np.asarray(arrays)
                ax.plot(spots, values.mean(axis=0), label=architecture.upper(), color=color)
                ax.fill_between(spots, values.min(axis=0), values.max(axis=0), alpha=.15, color=color)
            ax.set(title=f"t = {time:g}", ylabel=quantity.capitalize())
            ax.grid(alpha=.2)
            if row_index == 1:
                ax.set_xlabel("Spot / strike")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "greeks.pdf", bbox_inches="tight")
    plt.close(fig)
    write_json(destination / "diagnostics_lineage.json", dict(input_completion_sha256=sha256(folder / "completion.json"),
                                                             report_source_sha256=sha256(Path(__file__)),
                                                             outputs={n: sha256(destination / n) for n in
                                                                      ("greeks.pdf", "greeks_table.tex", "learned_coverage_table.tex")}))
    print(f"Diagnostic figures and tables generated in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
