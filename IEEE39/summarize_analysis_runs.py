import argparse
import json
from pathlib import Path

import pandas as pd

from analysis_evaluator import build_evaluation_payload, load_json, resolve_ieee39_path


ANALYSIS_ROOT = Path(__file__).resolve().parent / "analysis"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize IEEE39 analysis folders using metadata from analysis_config.json and scenario.json.")
    parser.add_argument("--analysis-dir", nargs="+", default=None, help="Specific analysis folders. Relative paths are resolved from IEEE39.")
    parser.add_argument("--load", default=None, help="Convenience load filter. Matches either analysis scenario name like load03 or source load name like 'Load 03'.")
    parser.add_argument("--load-name", default=None, help="Filter by source load name, e.g. 'Load 03'.")
    parser.add_argument("--dp-percent", type=float, default=None, help="Filter by source dp_percent.")
    parser.add_argument("--dq-percent", type=float, default=None, help="Filter by source dq_percent.")
    parser.add_argument("--event-time", type=float, default=None, help="Filter by source event_time_s.")
    parser.add_argument("--duration", type=float, default=None, help="Filter by source sim_stop_time_s.")
    parser.add_argument("--scenario-name", default=None, help="Filter by analysis scenario name, e.g. load03.")
    parser.add_argument("--include", default="*", help="Glob pattern under IEEE39/analysis when --analysis-dir is omitted. Default: *")
    parser.add_argument("--output-dir", default="analysis/summary", help="Output directory relative to IEEE39, or an absolute path. Default: analysis/summary")
    parser.add_argument("--modal-weight", type=float, default=3.0, help="Relative weight of modal ranking in the aggregate overall score. Default: 3.")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="Relative weight of reconstruction ranking in the aggregate overall score. Default: 1.")
    return parser.parse_args()


def _select_analysis_dirs(args):
    if args.analysis_dir:
        selected = []
        for raw in args.analysis_dir:
            path = resolve_ieee39_path(raw)
            if not path.exists() or not path.is_dir():
                raise SystemExit(f"Analysis folder does not exist: {path}")
            selected.append(path)
        return selected

    if not ANALYSIS_ROOT.exists():
        raise SystemExit(f"Analysis root not found: {ANALYSIS_ROOT}")
    return [path for path in sorted(ANALYSIS_ROOT.glob(args.include)) if path.is_dir()]


def _match_numeric(actual, expected, tol=1e-9):
    if expected is None:
        return True
    if actual is None:
        return False
    return abs(float(actual) - float(expected)) <= tol


def _include_payload(payload, args):
    summary = payload["summary"]
    if args.load is not None:
        load_filter = str(args.load).strip().lower().replace(" ", "")
        scenario_name = str(summary.get("scenario_name", "")).strip().lower().replace(" ", "")
        load_name = str(summary.get("load_name", "")).strip().lower().replace(" ", "")
        if load_filter not in {scenario_name, load_name}:
            return False
    if args.load_name is not None and str(summary.get("load_name")) != str(args.load_name):
        return False
    if args.scenario_name is not None and str(summary.get("scenario_name")) != str(args.scenario_name):
        return False
    if not _match_numeric(summary.get("dp_percent"), args.dp_percent):
        return False
    if not _match_numeric(summary.get("dq_percent"), args.dq_percent):
        return False
    if not _match_numeric(summary.get("event_time_s"), args.event_time):
        return False
    if not _match_numeric(summary.get("sim_stop_time_s"), args.duration):
        return False
    return True


def _ensure_payload(folder):
    config_path = folder / "analysis_config.json"
    config = load_json(config_path)
    payload = config.get("evaluation")
    if payload:
        return payload
    return build_evaluation_payload(folder)


def _rank_summary(df, modal_weight=3.0, reconstruction_weight=1.0):
    if df.empty:
        return df
    modal_sorted = df.sort_values(
        ["modal_mid_modes", "modal_strong_modes", "modal_loose_modes", "best_mean_R2", "mean_R2", "negative_R2_count"],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    modal_sorted["modal_rank"] = modal_sorted.index + 1

    recon_sorted = df.sort_values(
        ["best_mean_R2", "best_min_R2", "mean_R2", "negative_R2_count"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    recon_sorted["reconstruction_rank"] = recon_sorted.index + 1

    merged = df.merge(modal_sorted[["analysis_folder", "modal_rank"]], on="analysis_folder")
    merged = merged.merge(recon_sorted[["analysis_folder", "reconstruction_rank"]], on="analysis_folder")
    merged["unweighted_overall_score"] = merged["modal_rank"] + merged["reconstruction_rank"]
    merged["modal_weight"] = float(modal_weight)
    merged["reconstruction_weight"] = float(reconstruction_weight)
    merged["weighted_overall_score"] = (
        merged["modal_rank"] * float(modal_weight)
        + merged["reconstruction_rank"] * float(reconstruction_weight)
    )
    return merged.sort_values(
        ["weighted_overall_score", "unweighted_overall_score", "modal_rank", "reconstruction_rank", "analysis_folder"],
        kind="stable",
    )


def main():
    args = parse_args()
    folders = _select_analysis_dirs(args)
    output_dir = resolve_ieee39_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for folder in folders:
        config_path = folder / "analysis_config.json"
        results_path = folder / "results.csv"
        report_path = folder / "stats" / "comprehensive_report.csv"
        if not config_path.exists() or not results_path.exists() or not report_path.exists():
            continue
        payload = _ensure_payload(folder)
        if _include_payload(payload, args):
            rows.append(payload["summary"])

    summary_df = _rank_summary(pd.DataFrame(rows), modal_weight=args.modal_weight, reconstruction_weight=args.reconstruction_weight)
    summary_csv = output_dir / "run_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = output_dir / "summary.json"
    payload = {
        "filters": {
            "load": args.load,
            "load_name": args.load_name,
            "dp_percent": args.dp_percent,
            "dq_percent": args.dq_percent,
            "event_time": args.event_time,
            "duration": args.duration,
            "scenario_name": args.scenario_name,
            "include": args.include,
            "modal_weight": args.modal_weight,
            "reconstruction_weight": args.reconstruction_weight,
        },
        "evaluated_runs": int(len(summary_df)),
        "top_modal_run": None if summary_df.empty else summary_df.nsmallest(1, "modal_rank").iloc[0]["analysis_folder"],
        "top_reconstruction_run": None if summary_df.empty else summary_df.nsmallest(1, "reconstruction_rank").iloc[0]["analysis_folder"],
        "top_unweighted_run": None if summary_df.empty else summary_df.nsmallest(1, "unweighted_overall_score").iloc[0]["analysis_folder"],
        "top_weighted_run": None if summary_df.empty else summary_df.nsmallest(1, "weighted_overall_score").iloc[0]["analysis_folder"],
        "run_summary_csv": summary_csv.name,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved summary to: {summary_csv}")
    print(f"Saved summary JSON to: {summary_json}")


if __name__ == "__main__":
    main()
