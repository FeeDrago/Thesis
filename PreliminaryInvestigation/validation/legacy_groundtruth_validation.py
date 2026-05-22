import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from groundtruth_validation import (
    DEFAULT_CASES,
    MODE_FREQ_EPS_HZ,
    _build_cases,
    _match_truth_to_estimates,
    _select_oscillatory_modes,
    run_validation_suite,
)
from legacy_matrix_pencil_reference import apply_matrix_pencil_fixed_order, determine_MP_order


SCRIPT_DIR = Path(__file__).resolve().parent


def _find_case(case_name):
    return next(case for case in DEFAULT_CASES if case["name"] == case_name)


def _evaluate_method(case_name, t, y_ref, truth_modes, spec):
    case = _find_case(case_name)
    try:
        if spec["kind"] == "fixed":
            requested_order = int(spec["order"])
            selected_order = requested_order
            freq, sigma, y_est, _, _, amplitudes = apply_matrix_pencil_fixed_order(y=y_ref, t=t, order=requested_order)
            auto_tau = ""
        else:
            auto_tau = float(spec["tau"])
            selected_order = determine_MP_order(
                t,
                y_ref,
                auto_tau,
                rate=10,
                max_order=int(case.get("auto_max_order", 6)),
            )
            requested_order = selected_order
            freq, sigma, y_est, _, _, amplitudes = apply_matrix_pencil_fixed_order(y=y_ref, t=t, order=selected_order)
    except Exception as exc:
        return {
            "Case": case_name,
            "Method": spec["method"],
            "Requested_Order": int(spec.get("order", 0)) if spec["kind"] == "fixed" else "",
            "Selected_Order": "",
            "Auto_Tau": "" if spec["kind"] == "fixed" else float(spec["tau"]),
            "Estimated_Mode_Count": 0,
            "Truth_Mode_Count": len(truth_modes),
            "R2": "",
            "RMSE": "",
            "Mean_Freq_Error_Hz": "",
            "Max_Freq_Error_Hz": "",
            "Mean_Damping_Error": "",
            "Max_Damping_Error": "",
            "Mean_2D_Error": "",
            "Max_2D_Error": "",
            "Matched_All_Truth_Modes": False,
            "Status": f"failed: {type(exc).__name__}: {exc}",
        }, [], []

    phases = np.angle(amplitudes)
    oscillatory_modes = _select_oscillatory_modes(freq, sigma, np.abs(amplitudes), phases)
    matched_rows = _match_truth_to_estimates(truth_modes, oscillatory_modes)

    if matched_rows is None:
        return {
            "Case": case_name,
            "Method": spec["method"],
            "Requested_Order": requested_order,
            "Selected_Order": selected_order,
            "Auto_Tau": auto_tau,
            "Estimated_Mode_Count": len(oscillatory_modes),
            "Truth_Mode_Count": len(truth_modes),
            "R2": float(r2_score(y_ref, y_est)),
            "RMSE": float(np.sqrt(mean_squared_error(y_ref, y_est))),
            "Mean_Freq_Error_Hz": "",
            "Max_Freq_Error_Hz": "",
            "Mean_Damping_Error": "",
            "Max_Damping_Error": "",
            "Mean_2D_Error": "",
            "Max_2D_Error": "",
            "Matched_All_Truth_Modes": False,
            "Status": "ok_but_not_enough_modes",
        }, [], oscillatory_modes

    freq_errors = [row["frequency_error_hz"] for row in matched_rows]
    damping_errors = [row["damping_error"] for row in matched_rows]
    distances = [row["distance_2d"] for row in matched_rows]
    return {
        "Case": case_name,
        "Method": spec["method"],
        "Requested_Order": requested_order,
        "Selected_Order": selected_order,
        "Auto_Tau": auto_tau,
        "Estimated_Mode_Count": len(oscillatory_modes),
        "Truth_Mode_Count": len(truth_modes),
        "R2": float(r2_score(y_ref, y_est)),
        "RMSE": float(np.sqrt(mean_squared_error(y_ref, y_est))),
        "Mean_Freq_Error_Hz": float(np.mean(freq_errors)),
        "Max_Freq_Error_Hz": float(np.max(freq_errors)),
        "Mean_Damping_Error": float(np.mean(damping_errors)),
        "Max_Damping_Error": float(np.max(damping_errors)),
        "Mean_2D_Error": float(np.mean(distances)),
        "Max_2D_Error": float(np.max(distances)),
        "Matched_All_Truth_Modes": True,
        "Status": "ok",
    }, matched_rows, oscillatory_modes


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic ground-truth validation with the legacy Matrix Pencil implementation.")
    parser.add_argument("--run-label", required=True, help="Unique label for this run")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "validation_outputs_legacy"), help="Root folder for outputs")
    parser.add_argument("--cases", nargs="*", default=["all"], help="Case names to run, or 'all'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run directory")
    return parser.parse_args()


def main():
    args = parse_args()
    cases = _build_cases(args.cases)

    output_root = Path(args.output_root)
    run_dir = output_root / args.run_label
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    config = {
        "run_label": args.run_label,
        "seed": args.seed,
        "cases": [case["name"] for case in cases],
        "mode_freq_eps_hz": MODE_FREQ_EPS_HZ,
        "implementation": "legacy_professor_style",
    }
    run_validation_suite(run_dir, cases, rng, _evaluate_method, config)

    print(f"Legacy validation run saved to: {run_dir}")


if __name__ == "__main__":
    main()
