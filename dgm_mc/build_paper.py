"""Compile the manuscript without installing packages or changing global PATH."""
import os
from pathlib import Path
import shutil
import subprocess

from .provenance import ROOT, sha256, write_json


def main():
    paper = ROOT / "paper"
    env = os.environ.copy()
    # Some Windows setups have a compiler executable (rather than its directory)
    # in PATH. MiKTeX font discovery fails on it; sanitize only this child process.
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                 if p and Path(p.strip('"')).is_dir())
    latex = shutil.which("pdflatex", path=env["PATH"])
    bibtex = shutil.which("bibtex", path=env["PATH"])
    if latex is None or bibtex is None:
        raise SystemExit("Install a TeX distribution with pdflatex and bibtex to build the PDF.")
    version = subprocess.run([latex, "--version"], capture_output=True, text=True, env=env, check=True).stdout
    miktex = "MiKTeX" in version
    flags = ["-interaction=nonstopmode", "-halt-on-error"]
    flags += ["-disable-installer", "-disable-write18"] if miktex else ["-no-shell-escape"]
    commands = [[latex, *flags, "main.tex"], [bibtex, *(["-disable-installer"] if miktex else []), "main"],
                [latex, *flags, "main.tex"], [latex, *flags, "main.tex"]]
    for command in commands:
        result = subprocess.run(command, cwd=paper, env=env, capture_output=True, text=True, timeout=120)
        if result.returncode:
            print(result.stdout[-10000:])
            print(result.stderr[-3000:])
            raise SystemExit(result.returncode)
    log = (paper / "main.log").read_text(errors="replace")
    problems = [line for line in log.splitlines() if "undefined" in line.lower() or "Overfull" in line]
    if problems:
        raise SystemExit("Build needs cleanup: " + "; ".join(problems))
    write_json(paper / "build_receipt.json", dict(status="passed", engine=version.splitlines()[0],
                                                commands=[[Path(c[0]).name, *c[1:]] for c in commands],
                                                source_sha256=sha256(paper / "main.tex"),
                                                bibliography_sha256=sha256(paper / "references.bib"),
                                                pdf_sha256=sha256(paper / "main.pdf"),
                                                generated_sha256={p.name: sha256(p) for p in sorted((paper / "generated").iterdir()) if p.is_file()}))
    print("Built paper/main.pdf; no undefined references or overfull boxes.")


if __name__ == "__main__":
    main()
