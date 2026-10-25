"""Verify committed results, report lineage, checkpoint metadata, and paper."""
import json
from pathlib import Path

from .checkpoint import load_model
from .provenance import ROOT, sha256
from .report import load_verified


def verify_hashes(folder, mapping):
    for name, digest in mapping.items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Integrity failure: {folder.name}/{name}")


def main():
    run = ROOT / "results/cpu_verified"
    data = load_verified(run)
    config = data["manifest"]["config"]
    scenarios = len(config["moneyness"])
    classical = len(set(config["dimensions"] + config["classical_dimensions"])) * scenarios * config["repeats"] * 4
    neural = len(config["dimensions"]) * len(config["architectures"]) * len(config["training_seeds"]) * scenarios * config["repeats"] * 3
    if len(data["records"]) != classical + neural:
        raise ValueError("Estimate count does not match declared protocol")
    for meta in sorted((run / "models").glob("*.json")):
        load_model(meta)
    reports = run / "reports"
    lineage = json.loads((reports / "lineage.json").read_text())
    verify_hashes(reports, lineage["outputs"])
    if sha256(run / "completion.json") != lineage["input_completion_sha256"]:
        raise ValueError("Report lineage does not match run")
    diagnostics = ROOT / "results/diagnostics"
    completion = json.loads((diagnostics / "completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("Incomplete diagnostics")
    verify_hashes(diagnostics, completion["artifact_sha256"])
    verify_hashes(run / "models", completion["source_models"])
    paper = ROOT / "paper"
    receipt = json.loads((paper / "build_receipt.json").read_text())
    verify_hashes(paper, {"main.tex": receipt["source_sha256"], "references.bib": receipt["bibliography_sha256"],
                          "main.pdf": receipt["pdf_sha256"]})
    verify_hashes(paper / "generated", receipt["generated_sha256"])
    print(f"Verified {len(data['records'])} main estimates, all checkpoints, diagnostics, reports and paper hashes.")


if __name__ == "__main__":
    main()
