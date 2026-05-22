import argparse
import json
from pathlib import Path
from textwrap import dedent

import pandas as pd

from analysis_evaluator import build_evaluation_payload, load_json, resolve_ieee39_path


ANALYSIS_ROOT = Path(__file__).resolve().parent / "analysis"
SUMMARY_ROOT = ANALYSIS_ROOT / "summaries"
DEFAULT_DP_PERCENT = 2.0
DEFAULT_DQ_PERCENT = 0.0
DEFAULT_EVENT_TIME_S = 0.0
DEFAULT_DURATION_S = 50.0
COMPACT_SUMMARY_COLUMNS = [
    "analysis_folder",
    "load_name",
    "dp_percent",
    "dq_percent",
    "event_time_s",
    "sim_stop_time_s",
    "mean_R2",
    "best_mean_R2",
    "best_min_R2",
    "negative_R2_count",
    "modal_loose_modes",
    "modal_mid_modes",
    "modal_strong_modes",
    "modal_rank",
    "reconstruction_rank",
    "unweighted_overall_score",
    "weighted_overall_score",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize IEEE39 analysis folders using only metadata stored in each analysis folder's\n"
            "analysis_config.json and linked source scenario.json."
        ),
        epilog=dedent(
            """
            Modes:
              1. Metadata scan mode: use --load to search under IEEE39/analysis with resolved metadata defaults.
              2. Explicit folder mode: use --analysis-dir and also give a custom --output-dir name.

            Metadata scan defaults:
              - dp_percent = 2
              - dq_percent = 0
              - event_time = 0
              - duration = 50
              These defaults match the standard IEEE39 generate/analyze defaults.

            Examples:
              python IEEE39/summarize_analysis_runs.py --load load03
              python IEEE39/summarize_analysis_runs.py --load load03 --modal-weight 5 --reconstruction-weight 1
              python IEEE39/summarize_analysis_runs.py --load "Load 03" --dp-percent 2 --dq-percent 0 --event-time 0 --duration 50
              python IEEE39/summarize_analysis_runs.py --analysis-dir analysis/Load03_Pplus2_50s_0_to_end_reset analysis/Load03_Pplus2_50s_0.4_to_end_reset --output-dir analysis/summaries/summary_load03_manual_compare

            Notes:
              - Choose exactly one selection mode: either --load or --analysis-dir.
              - If --output-dir is omitted in metadata scan mode, the folder name is derived automatically under IEEE39/analysis/summaries.
              - In explicit folder mode, --output-dir is required and should be a custom name so the output is not mistaken for a general summary.
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--analysis-dir", nargs="+", default=None, help="Explicit analysis folders to compare. Relative paths are resolved from IEEE39. Cannot be combined with --load.")
    parser.add_argument("--load", default=None, help="Metadata scan selector. Matches either analysis scenario name like load03 or source load name like 'Load 03'. Cannot be combined with --analysis-dir.")
    parser.add_argument("--load-name", default=None, help="Optional exact source load-name filter from scenario.json, e.g. 'Load 03'.")
    parser.add_argument("--dp-percent", type=float, default=None, help=f"Source dp_percent filter. Default in metadata scan mode: {DEFAULT_DP_PERCENT:g}.")
    parser.add_argument("--dq-percent", type=float, default=None, help=f"Source dq_percent filter. Default in metadata scan mode: {DEFAULT_DQ_PERCENT:g}.")
    parser.add_argument("--event-time", type=float, default=None, help=f"Source event_time_s filter. Default in metadata scan mode: {DEFAULT_EVENT_TIME_S:g}.")
    parser.add_argument("--duration", type=float, default=None, help=f"Source sim_stop_time_s filter. Default in metadata scan mode: {DEFAULT_DURATION_S:g}.")
    parser.add_argument("--output-dir", default=None, help="Explicit output directory relative to IEEE39, or an absolute path. Required in explicit-folder mode. Optional in metadata scan mode, where the default is under analysis/summaries/.")
    parser.add_argument("--modal-weight", type=float, default=3.0, help="Relative weight of modal ranking in the aggregate overall score. Default: 3.")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="Relative weight of reconstruction ranking in the aggregate overall score. Default: 1.")
    return parser.parse_args()


def _sanitize_suffix_part(value):
    text = str(value).strip().lower().replace(" ", "")
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text)


def _format_value(value):
    return f"{float(value):g}" if isinstance(value, (int, float)) or str(value).replace('.', '', 1).replace('-', '', 1).isdigit() else str(value)


def validate_and_resolve_args(args):
    explicit_mode = bool(args.analysis_dir)
    load_mode = args.load is not None

    if explicit_mode and load_mode:
        raise SystemExit("Use either --load or --analysis-dir, not both.")

    if not explicit_mode and not load_mode:
        raise SystemExit(
            "Choose one selection mode: either '--load load03' for metadata scan mode or '--analysis-dir <folder> [<folder> ...]' for explicit-folder mode."
        )

    if explicit_mode:
        if not args.output_dir:
            raise SystemExit(
                "When using --analysis-dir, you must also give a custom --output-dir so the summary is not mistaken for a general metadata-based summary."
            )
        resolved = {
            "selection_mode": "explicit_analysis_dirs",
            "load": args.load,
            "load_name": args.load_name,
            "dp_percent": args.dp_percent,
            "dq_percent": args.dq_percent,
            "event_time": args.event_time,
            "duration": args.duration,
        }
        return resolved

    resolved = {
        "selection_mode": "metadata_scan",
        "load": args.load,
        "load_name": args.load_name,
        "dp_percent": DEFAULT_DP_PERCENT if args.dp_percent is None else args.dp_percent,
        "dq_percent": DEFAULT_DQ_PERCENT if args.dq_percent is None else args.dq_percent,
        "event_time": DEFAULT_EVENT_TIME_S if args.event_time is None else args.event_time,
        "duration": DEFAULT_DURATION_S if args.duration is None else args.duration,
    }
    return resolved


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
    return [path for path in sorted(ANALYSIS_ROOT.iterdir()) if path.is_dir()]


def _resolve_output_dir(args, resolved):
    if args.output_dir:
        return resolve_ieee39_path(args.output_dir)

    base_name = (
        f"summary_{_sanitize_suffix_part(resolved['load'])}"
        f"_dp{_format_value(resolved['dp_percent'])}"
        f"_dq{_format_value(resolved['dq_percent'])}"
        f"_evt{_format_value(resolved['event_time'])}"
        f"_dur{_format_value(resolved['duration'])}"
    )
    if resolved.get("load_name"):
        base_name += f"_src-{_sanitize_suffix_part(resolved['load_name'])}"
    return SUMMARY_ROOT / base_name


def _compact_summary(df):
    if df.empty:
        return pd.DataFrame(columns=COMPACT_SUMMARY_COLUMNS)
    compact_columns = [column for column in COMPACT_SUMMARY_COLUMNS if column in df.columns]
    return df.loc[:, compact_columns].copy()


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
    resolved = validate_and_resolve_args(args)
    args.load = resolved["load"]
    args.load_name = resolved["load_name"]
    args.dp_percent = resolved["dp_percent"]
    args.dq_percent = resolved["dq_percent"]
    args.event_time = resolved["event_time"]
    args.duration = resolved["duration"]
    folders = _select_analysis_dirs(args)
    output_dir = _resolve_output_dir(args, resolved)
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
    compact_df = _compact_summary(summary_df)
    summary_csv = output_dir / "run_summary.csv"
    summary_df_full_csv = output_dir / "run_summary_full.csv"
    compact_df.to_csv(summary_csv, index=False)
    summary_df.to_csv(summary_df_full_csv, index=False)

    summary_json = output_dir / "summary.json"
    payload = {
        "selection_mode": resolved["selection_mode"],
        "filters": {
            "load": args.load,
            "load_name": args.load_name,
            "dp_percent": args.dp_percent,
            "dq_percent": args.dq_percent,
            "event_time": args.event_time,
            "duration": args.duration,
            "modal_weight": args.modal_weight,
            "reconstruction_weight": args.reconstruction_weight,
        },
        "analysis_dirs": [str(folder) for folder in folders] if args.analysis_dir else None,
        "evaluated_runs": int(len(summary_df)),
        "top_modal_run": None if summary_df.empty else summary_df.nsmallest(1, "modal_rank").iloc[0]["analysis_folder"],
        "top_reconstruction_run": None if summary_df.empty else summary_df.nsmallest(1, "reconstruction_rank").iloc[0]["analysis_folder"],
        "top_unweighted_run": None if summary_df.empty else summary_df.nsmallest(1, "unweighted_overall_score").iloc[0]["analysis_folder"],
        "top_weighted_run": None if summary_df.empty else summary_df.nsmallest(1, "weighted_overall_score").iloc[0]["analysis_folder"],
        "run_summary_csv": summary_csv.name,
        "run_summary_full_csv": summary_df_full_csv.name,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved summary to: {summary_csv}")
    print(f"Saved summary JSON to: {summary_json}")


if __name__ == "__main__":
    main()
