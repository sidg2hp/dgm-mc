"""Fingerprint inherited public research artifacts and check numerical claims.

This optional command requires the original sibling trees. All other commands
are standalone. No source bytes are copied, and no checkpoints are loaded.
"""
import argparse
import json
from pathlib import Path

from .market import equicorrelated, geometric_price
from .provenance import sha256, write_json


def inventory(parent):
    items = []
    for name in ("dgm_option_pricing", "neurips_submission"):
        root = parent / name
        for p in sorted(root.rglob("*")):
            relative = p.relative_to(root)
            if any(part in {".git", "venv", ".venv", "__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".yml", ".md", ".tex", ".bib", ".toml"}:
                items.append(dict(path=f"{name}/{relative.as_posix()}", bytes=p.stat().st_size, sha256=sha256(p)))
    return items


def check_claims(parent):
    legacy = parent / "neurips_submission"
    summary = json.loads((legacy / "results/hybrid_mc/hybrid_mc_summary.json").read_text())
    row = summary["1"]
    exact = float(geometric_price(equicorrelated(1))[0])
    wins, total, atm_wins = 0, 0, 0
    import numpy as np
    comparisons = []
    for d in (1, 2, 3, 5):
        path = legacy / f"results/paper/comparison/d_{d}/comparison_results.json"
        data = json.loads(path.read_text())
        # Keys are checked explicitly; no heuristic metric substitutions.
        dg = np.asarray(data["dgm_prices"])
        zh = np.asarray(data["zhou_prices"])
        mc = np.asarray(data["mc_prices"])
        win = np.abs(dg - mc) < np.abs(zh - mc)
        wins += int(win.sum())
        atm_wins += int(win[2])
        total += len(win)
        comparisons.append(dict(d=d, dgm_mre=float(np.mean(abs(dg - mc) / abs(mc))),
                                dgm_relative_l2=float(np.linalg.norm(dg - mc) / np.linalg.norm(mc)),
                                zhou_mre=float(np.mean(abs(zh - mc) / abs(mc))), dgm_wins=int(win.sum())))
    return dict(label="legacy_artifact_checks_not_new_experiments", exact_1d=exact,
                legacy_hybrid_price=row["cv_price"], legacy_hybrid_se=row["cv_se"],
                relative_bias=(row["cv_price"] - exact) / exact,
                error_in_reported_se=(row["cv_price"] - exact) / row["cv_se"],
                comparisons=comparisons, dgm_wins=wins, comparisons_count=total, atm_wins=atm_wins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("audit"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    current = inventory(args.legacy_parent)
    if args.verify:
        stored = json.loads((args.output / "source_manifest.json").read_text(encoding="utf-8"))
        if stored != current:
            raise SystemExit("Legacy manifest changed; investigate before claiming preservation")
        print(f"Verified {len(current)} original source/result files unchanged")
        return
    if (args.output / "source_manifest.json").exists():
        raise FileExistsError("Audit already exists; use --verify or a new output")
    checks = check_claims(args.legacy_parent)
    write_json(args.output / "source_manifest.json", current)
    write_json(args.output / "legacy_checks.json", checks)
    print(f"Inventoried {len(current)} artifacts; legacy 1D bias {checks['relative_bias']:.3%}")


if __name__ == "__main__":
    main()
