import argparse
import csv
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two synthetic validation runs.")
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


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    metric_keys = ["Case", "Method"]
    merged_metrics = metrics_a.merge(metrics_b, on=metric_keys, suffixes=("_a", "_b"), how="outer", indicator=True)

    metric_rows = []
    for _, row in merged_metrics.iterrows():
        metric_rows.append(
            {
                "Case": row["Case"],
                "Method": row["Method"],
                "Presence": row["_merge"],
                "Selected_Order_Diff": "" if pd.isna(row.get("Selected_Order_a")) or pd.isna(row.get("Selected_Order_b")) else int(row["Selected_Order_b"] - row["Selected_Order_a"]),
                "R2_Abs_Diff": "" if pd.isna(row.get("R2_a")) or pd.isna(row.get("R2_b")) else abs(float(row["R2_b"]) - float(row["R2_a"])),
                "RMSE_Abs_Diff": "" if pd.isna(row.get("RMSE_a")) or pd.isna(row.get("RMSE_b")) else abs(float(row["RMSE_b"]) - float(row["RMSE_a"])),
                "Mean_Freq_Error_Abs_Diff": "" if pd.isna(row.get("Mean_Freq_Error_Hz_a")) or pd.isna(row.get("Mean_Freq_Error_Hz_b")) else abs(float(row["Mean_Freq_Error_Hz_b"]) - float(row["Mean_Freq_Error_Hz_a"])),
                "Mean_Damping_Error_Abs_Diff": "" if pd.isna(row.get("Mean_Damping_Error_a")) or pd.isna(row.get("Mean_Damping_Error_b")) else abs(float(row["Mean_Damping_Error_b"]) - float(row["Mean_Damping_Error_a"])),
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
                "Estimated_Frequency_Abs_Diff_Hz": "" if pd.isna(row.get("estimated_frequency_hz_a")) or pd.isna(row.get("estimated_frequency_hz_b")) else abs(float(row["estimated_frequency_hz_b"]) - float(row["estimated_frequency_hz_a"])),
                "Estimated_Damping_Abs_Diff": "" if pd.isna(row.get("estimated_damping_a")) or pd.isna(row.get("estimated_damping_b")) else abs(float(row["estimated_damping_b"]) - float(row["estimated_damping_a"])),
                "Frequency_Error_Abs_Diff_Hz": "" if pd.isna(row.get("frequency_error_hz_a")) or pd.isna(row.get("frequency_error_hz_b")) else abs(float(row["frequency_error_hz_b"]) - float(row["frequency_error_hz_a"])),
                "Damping_Error_Abs_Diff": "" if pd.isna(row.get("damping_error_a")) or pd.isna(row.get("damping_error_b")) else abs(float(row["damping_error_b"]) - float(row["damping_error_a"])),
            }
        )

    _write_csv(
        out_dir / "case_metrics_diff.csv",
        metric_rows,
        [
            "Case",
            "Method",
            "Presence",
            "Selected_Order_Diff",
            "R2_Abs_Diff",
            "RMSE_Abs_Diff",
            "Mean_Freq_Error_Abs_Diff",
            "Mean_Damping_Error_Abs_Diff",
        ],
    )
    _write_csv(
        out_dir / "matched_modes_diff.csv",
        match_rows,
        [
            "Case",
            "Method",
            "truth_mode",
            "Presence",
            "Estimated_Frequency_Abs_Diff_Hz",
            "Estimated_Damping_Abs_Diff",
            "Frequency_Error_Abs_Diff_Hz",
            "Damping_Error_Abs_Diff",
        ],
    )

    summary = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "metric_rows": len(metric_rows),
        "match_rows": len(match_rows),
        "max_r2_abs_diff": max((row["R2_Abs_Diff"] for row in metric_rows if row["R2_Abs_Diff"] != ""), default=0.0),
        "max_rmse_abs_diff": max((row["RMSE_Abs_Diff"] for row in metric_rows if row["RMSE_Abs_Diff"] != ""), default=0.0),
        "max_estimated_frequency_abs_diff_hz": max((row["Estimated_Frequency_Abs_Diff_Hz"] for row in match_rows if row["Estimated_Frequency_Abs_Diff_Hz"] != ""), default=0.0),
        "max_estimated_damping_abs_diff": max((row["Estimated_Damping_Abs_Diff"] for row in match_rows if row["Estimated_Damping_Abs_Diff"] != ""), default=0.0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Comparison saved to: {out_dir}")
    print(f"  - {out_dir / 'case_metrics_diff.csv'}")
    print(f"  - {out_dir / 'matched_modes_diff.csv'}")
    print(f"  - {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
