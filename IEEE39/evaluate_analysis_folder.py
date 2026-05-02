import argparse
from pathlib import Path

from analysis_evaluator import update_analysis_config_with_evaluation, resolve_ieee39_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one or more IEEE39 analysis folders and write the result into analysis_config.json.")
    parser.add_argument("--analysis-dir", nargs="+", required=True, help="Analysis folder path(s). Relative paths are resolved from IEEE39.")
    return parser.parse_args()


def main():
    args = parse_args()
    for raw in args.analysis_dir:
        folder = resolve_ieee39_path(raw)
        if not folder.exists() or not folder.is_dir():
            raise SystemExit(f"Analysis folder does not exist: {folder}")
        payload = update_analysis_config_with_evaluation(folder)
        summary = payload["summary"]
        print(f"Updated evaluation in: {folder / 'analysis_config.json'}")
        print(
            f"  modal_mid_modes={summary.get('modal_mid_modes')} | "
            f"best_mean_R2={summary.get('best_mean_R2')} | "
            f"negative_R2_count={summary.get('negative_R2_count')}"
        )


if __name__ == "__main__":
    main()
