import argparse
import csv
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two synthetic validation runs with emphasis on ground-truth performance.")
    parser.add_argument("--run-a", required=True, help="First run directory")
    parser.add_argument("--run-b", required=True, help="Second run directory")
    parser.add_argument("--comparison-label", required=True, help="Output label for the comparison")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "validation_outputs" / "comparisons"), help="Root folder for comparison outputs")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing comparison directory")
    return parser.parse_args()


def _load_csv(run_dir, name):
    path = Path(run_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_label(run_dir, config):
    if config and config.get("run_label"):
        return str(config["run_label"])
    return Path(run_dir).name


def _numeric_or_blank(value):
    return "" if pd.isna(value) else float(value)


def _int_or_blank(value):
    return "" if pd.isna(value) else int(value)


def _bool_or_blank(value):
    return "" if pd.isna(value) else bool(value)


def _winner_lower(a, b, label_a, label_b):
    if pd.isna(a) or pd.isna(b):
        return ""
    if float(a) < float(b):
        return label_a
    if float(b) < float(a):
        return label_b
    return "tie"


def _winner_higher(a, b, label_a, label_b):
    if pd.isna(a) or pd.isna(b):
        return ""
    if float(a) > float(b):
        return label_a
    if float(b) > float(a):
        return label_b
    return "tie"


def _overall_method_winner(row, label_a, label_b):
    matched_a = row.get("Matched_All_Truth_Modes_a")
    matched_b = row.get("Matched_All_Truth_Modes_b")
    if pd.notna(matched_a) and pd.notna(matched_b) and bool(matched_a) != bool(matched_b):
        return label_a if bool(matched_a) else label_b

    mean_2d_a = row.get("Mean_2D_Error_a")
    mean_2d_b = row.get("Mean_2D_Error_b")
    winner = _winner_lower(mean_2d_a, mean_2d_b, label_a, label_b)
    if winner:
        return winner

    return _winner_higher(row.get("R2_a"), row.get("R2_b"), label_a, label_b)


def _count_winners(rows, key):
    counts = {}
    for row in rows:
        winner = row.get(key)
        if winner in (None, "", "tie"):
            continue
        counts[winner] = counts.get(winner, 0) + 1
    return counts


def _winner_counts(rows, key, label_a, label_b):
    counts = {label_a: 0, label_b: 0, "tie": 0}
    for row in rows:
        winner = row.get(key)
        if winner in counts:
            counts[winner] += 1
    return counts


def main():
    args = parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    out_dir = Path(args.output_root) / args.comparison_label
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Comparison directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_a = _load_csv(run_a, "case_metrics.csv")
    metrics_b = _load_csv(run_b, "case_metrics.csv")
    matches_a = _load_csv(run_a, "matched_modes.csv")
    matches_b = _load_csv(run_b, "matched_modes.csv")
    config_a = _load_json_if_exists(run_a / "run_config.json")
    config_b = _load_json_if_exists(run_b / "run_config.json")
    summary_a = _load_json_if_exists(run_a / "summary.json")
    summary_b = _load_json_if_exists(run_b / "summary.json")

    label_a = _run_label(run_a, config_a)
    label_b = _run_label(run_b, config_b)

    metric_keys = ["Case", "Method"]
    merged_metrics = metrics_a.merge(metrics_b, on=metric_keys, suffixes=("_a", "_b"), how="outer", indicator=True)

    metric_rows = []
    for _, row in merged_metrics.iterrows():
        metric_rows.append(
            {
                "Case": row["Case"],
                "Method": row["Method"],
                "Presence": row["_merge"],
                "Selected_Order_A": _int_or_blank(row.get("Selected_Order_a")),
                "Selected_Order_B": _int_or_blank(row.get("Selected_Order_b")),
                "Matched_All_Truth_Modes_A": _bool_or_blank(row.get("Matched_All_Truth_Modes_a")),
                "Matched_All_Truth_Modes_B": _bool_or_blank(row.get("Matched_All_Truth_Modes_b")),
                "R2_A": _numeric_or_blank(row.get("R2_a")),
                "R2_B": _numeric_or_blank(row.get("R2_b")),
                "Better_R2": _winner_higher(row.get("R2_a"), row.get("R2_b"), label_a, label_b),
                "RMSE_A": _numeric_or_blank(row.get("RMSE_a")),
                "RMSE_B": _numeric_or_blank(row.get("RMSE_b")),
                "Better_RMSE": _winner_lower(row.get("RMSE_a"), row.get("RMSE_b"), label_a, label_b),
                "Mean_Freq_Error_Hz_A": _numeric_or_blank(row.get("Mean_Freq_Error_Hz_a")),
                "Mean_Freq_Error_Hz_B": _numeric_or_blank(row.get("Mean_Freq_Error_Hz_b")),
                "Better_Mean_Freq_Error": _winner_lower(row.get("Mean_Freq_Error_Hz_a"), row.get("Mean_Freq_Error_Hz_b"), label_a, label_b),
                "Mean_Damping_Error_A": _numeric_or_blank(row.get("Mean_Damping_Error_a")),
                "Mean_Damping_Error_B": _numeric_or_blank(row.get("Mean_Damping_Error_b")),
                "Better_Mean_Damping_Error": _winner_lower(row.get("Mean_Damping_Error_a"), row.get("Mean_Damping_Error_b"), label_a, label_b),
                "Mean_2D_Error_A": _numeric_or_blank(row.get("Mean_2D_Error_a")),
                "Mean_2D_Error_B": _numeric_or_blank(row.get("Mean_2D_Error_b")),
                "Better_Mean_2D_Error": _winner_lower(row.get("Mean_2D_Error_a"), row.get("Mean_2D_Error_b"), label_a, label_b),
                "Overall_Better_Run": _overall_method_winner(row, label_a, label_b),
            }
        )

    match_keys = ["Case", "Method", "truth_mode"]
    merged_matches = matches_a.merge(matches_b, on=match_keys, suffixes=("_a", "_b"), how="outer", indicator=True)
    match_rows = []
    for _, row in merged_matches.iterrows():
        match_rows.append(
            {
                "Case": row["Case"],
                "Method": row["Method"],
                "truth_mode": row["truth_mode"],
                "Presence": row["_merge"],
                "truth_frequency_hz": _numeric_or_blank(row.get("truth_frequency_hz_a") if pd.notna(row.get("truth_frequency_hz_a")) else row.get("truth_frequency_hz_b")),
                "truth_damping": _numeric_or_blank(row.get("truth_damping_a") if pd.notna(row.get("truth_damping_a")) else row.get("truth_damping_b")),
                "frequency_error_hz_A": _numeric_or_blank(row.get("frequency_error_hz_a")),
                "frequency_error_hz_B": _numeric_or_blank(row.get("frequency_error_hz_b")),
                "Better_Frequency_Error": _winner_lower(row.get("frequency_error_hz_a"), row.get("frequency_error_hz_b"), label_a, label_b),
                "damping_error_A": _numeric_or_blank(row.get("damping_error_a")),
                "damping_error_B": _numeric_or_blank(row.get("damping_error_b")),
                "Better_Damping_Error": _winner_lower(row.get("damping_error_a"), row.get("damping_error_b"), label_a, label_b),
                "distance_2d_A": _numeric_or_blank(row.get("distance_2d_a")),
                "distance_2d_B": _numeric_or_blank(row.get("distance_2d_b")),
                "Better_2D_Distance": _winner_lower(row.get("distance_2d_a"), row.get("distance_2d_b"), label_a, label_b),
            }
        )

    metric_fieldnames = list(metric_rows[0].keys()) if metric_rows else []
    match_fieldnames = list(match_rows[0].keys()) if match_rows else []
    _write_csv(out_dir / "case_metrics_comparison.csv", metric_rows, metric_fieldnames)
    _write_csv(out_dir / "matched_modes_comparison.csv", match_rows, match_fieldnames)

    summary = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "run_a_label": label_a,
        "run_b_label": label_b,
        "run_a_config": config_a,
        "run_b_config": config_b,
        "run_a_summary": summary_a,
        "run_b_summary": summary_b,
        "total_case_method_comparisons": len(metric_rows),
        "total_truth_mode_comparisons": len(match_rows),
        "overall_method_wins": _winner_counts(metric_rows, "Overall_Better_Run", label_a, label_b),
        "better_r2_wins": _winner_counts(metric_rows, "Better_R2", label_a, label_b),
        "better_rmse_wins": _winner_counts(metric_rows, "Better_RMSE", label_a, label_b),
        "better_mean_2d_error_wins": _winner_counts(metric_rows, "Better_Mean_2D_Error", label_a, label_b),
        "better_frequency_error_wins": _winner_counts(match_rows, "Better_Frequency_Error", label_a, label_b),
        "better_damping_error_wins": _winner_counts(match_rows, "Better_Damping_Error", label_a, label_b),
        "better_2d_distance_wins": _winner_counts(match_rows, "Better_2D_Distance", label_a, label_b),
        "case_metrics_comparison_csv": "case_metrics_comparison.csv",
        "matched_modes_comparison_csv": "matched_modes_comparison.csv",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Comparison saved to: {out_dir}")
    print(f"  - {out_dir / 'case_metrics_comparison.csv'}")
    print(f"  - {out_dir / 'matched_modes_comparison.csv'}")
    print(f"  - {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
