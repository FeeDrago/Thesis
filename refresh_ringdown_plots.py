"""Refresh every ringdown figure from saved poles, preserving both scenarios.

Run from anywhere: python refresh_ringdown_plots.py
Use --system kundur / ieee39 to refresh just one system.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
KUNDUR = ROOT / "PreliminaryInvestigation"
IEEE39 = ROOT / "IEEE39/analysis/Load03_Pplus2_50s_0.2_to_end_reset"


def figures(roots):
    return [path for root in roots for directory in ("plots", "stats", "clustering")
            for path in (root / directory).rglob("*")
            if path.suffix.lower() in (".pdf", ".png")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("both", "kundur", "ieee39"), default="both")
    args = parser.parse_args()
    roots = ([KUNDUR] if args.system == "kundur" else [IEEE39] if args.system == "ieee39" else [KUNDUR, IEEE39])
    for root in roots:
        if not (root / "results.csv").is_file():
            parser.error(f"Saved results missing: {root / 'results.csv'}")

    # Verify numerical outputs as well as coverage, since this is a visual refresh.
    protected = [root / "results.csv" for root in roots]
    protected += [path for root in roots for path in root.rglob("mad.csv")]
    before_data = {path: digest(path) for path in protected}
    before_figures = {path: path.stat().st_mtime_ns for path in figures(roots)}
    audit_dir = ROOT / ".codex_tmp/ringdown_style"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"exports_{time.time_ns()}.jsonl"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["RINGDOWN_STYLE_AUDIT"] = str(audit_path)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[variable] = "1"
    commands = []
    if KUNDUR in roots:
        commands += [["PreliminaryInvestigation/mp_plotter.py"],
                     ["PreliminaryInvestigation/stats.py"],
                     ["PreliminaryInvestigation/clustering_analysis.py"]]
    if IEEE39 in roots:
        commands += [["IEEE39/analyze_ieee39.py", "--scenario", "load03", "--skip-matrix-pencil",
                      "--analysis-dir", "analysis/Load03_Pplus2_50s_0.2_to_end_reset", "--clustering", "--plots"]]
    for command in commands:
        print("Refreshing: " + " ".join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)

    stale = [str(path.relative_to(ROOT)) for path, stamp in before_figures.items()
             if path.stat().st_mtime_ns <= stamp]
    changed_data = [str(path.relative_to(ROOT)) for path, sha in before_data.items() if digest(path) != sha]
    exports = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    bad_legends = [record["path"] for record in exports
                   if not record["legends_inside_canvas"] or not record["legends_outside_axes"]
                   or any(ax["internal_legend"] for ax in record["axes"])]
    modal_axes = [ax for record in exports for ax in record["axes"] if ax["kind"] == "modal"]
    modal_views = {json.dumps([ax[key] for key in ("xlim", "ylim", "xticks", "yticks", "xscale", "yscale")])
                   for ax in modal_axes}
    summary = dict(figures_exported=len(exports), files_refreshed=len(figures(roots)),
                   stale_figures=stale, changed_results_or_mad=changed_data,
                   legend_problems=bad_legends, modal_axis_presets=len(modal_views), audit=str(audit_path))
    print(json.dumps(summary, indent=2), flush=True)
    if stale or changed_data or bad_legends or len(modal_views) != 1:
        raise SystemExit("Refresh verification failed; see the summary above.")


if __name__ == "__main__":
    main()
