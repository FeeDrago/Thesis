import argparse
import csv
import json
import math
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy.signal import detrend
from sklearn.metrics import mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
PRELIM_DIR = SCRIPT_DIR.parent
if str(PRELIM_DIR) not in sys.path:
    sys.path.insert(0, str(PRELIM_DIR))

from matrix_pencil import apply_matrix_pencil_fixed_order, determine_MP_order, filter_signal  # noqa: E402


MODE_FREQ_EPS_HZ = 1e-6

DEFAULT_CASES = [
    {
        "name": "two_mode_clean",
        "dt": 0.01,
        "t_max": 30.0,
        "drop_before_s": 0.1,
        "filter_fc_hz": 10.0,
        "filter_order": 15,
        "noise_std": 0.0,
        "truth_modes": [
            {"amplitude": 2.0, "sigma_abs": 0.1102, "frequency_hz": 0.25, "phase_rad": 1.5 * np.pi},
            {"amplitude": 2.0, "sigma_abs": 0.1596, "frequency_hz": 0.39, "phase_rad": 0.5 * np.pi},
        ],
        "fixed_orders": [2, 4, 6],
        "taus": [1.0, 0.1, 0.01],
        "auto_max_order": 6,
    },
    {
        "name": "two_mode_noisy",
        "dt": 0.01,
        "t_max": 30.0,
        "drop_before_s": 0.1,
        "filter_fc_hz": 10.0,
        "filter_order": 15,
        "noise_std": 0.1,
        "truth_modes": [
            {"amplitude": 2.0, "sigma_abs": 0.1102, "frequency_hz": 0.25, "phase_rad": 1.5 * np.pi},
            {"amplitude": 2.0, "sigma_abs": 0.1596, "frequency_hz": 0.39, "phase_rad": 0.5 * np.pi},
        ],
        "fixed_orders": [2, 4, 6],
        "taus": [1.0, 0.1, 0.01],
        "auto_max_order": 6,
    },
    {
        "name": "three_mode_noisy_close",
        "dt": 0.01,
        "t_max": 35.0,
        "drop_before_s": 0.1,
        "filter_fc_hz": 10.0,
        "filter_order": 15,
        "noise_std": 0.06,
        "truth_modes": [
            {"amplitude": 2.6, "sigma_abs": 0.12, "frequency_hz": 0.54, "phase_rad": 1.2},
            {"amplitude": 1.4, "sigma_abs": 0.62, "frequency_hz": 1.08, "phase_rad": -0.6},
            {"amplitude": 1.1, "sigma_abs": 0.64, "frequency_hz": 1.12, "phase_rad": 0.9},
        ],
        "fixed_orders": [3, 4, 6],
        "taus": [1.0, 0.1, 0.01],
        "auto_max_order": 6,
    },
]

CASE_METRIC_FIELDNAMES = [
    "Case",
    "Method",
    "Requested_Order",
    "Selected_Order",
    "Auto_Tau",
    "Estimated_Mode_Count",
    "Truth_Mode_Count",
    "R2",
    "RMSE",
    "Mean_Freq_Error_Hz",
    "Max_Freq_Error_Hz",
    "Mean_Damping_Error",
    "Max_Damping_Error",
    "Mean_2D_Error",
    "Max_2D_Error",
    "Matched_All_Truth_Modes",
    "Status",
]

MATCHED_MODE_FIELDNAMES = [
    "Case",
    "Method",
    "truth_mode",
    "truth_frequency_hz",
    "truth_damping",
    "estimated_mode_rank",
    "estimated_frequency_hz",
    "estimated_damping",
    "frequency_error_hz",
    "damping_error",
    "distance_2d",
]

RAW_MODE_FIELDNAMES = [
    "Case",
    "Method",
    "mode_rank",
    "frequency_hz",
    "damping",
    "amplitude",
    "phase_rad",
]


def _truth_mode_to_estimation_sign(mode):
    return {
        "amplitude": float(mode["amplitude"]),
        "damping": -float(mode["sigma_abs"]),
        "frequency_hz": float(mode["frequency_hz"]),
        "phase_rad": float(mode["phase_rad"]),
    }


def _generate_signal(case, rng):
    dt = float(case["dt"])
    t_max = float(case["t_max"])
    t = np.linspace(0.0, t_max, int(t_max / dt) + 1)

    truth_modes = [_truth_mode_to_estimation_sign(mode) for mode in case["truth_modes"]]

    y_clean = np.zeros_like(t)
    for mode in truth_modes:
        y_clean += (
            mode["amplitude"]
            * np.exp(mode["damping"] * t)
            * np.cos(2 * np.pi * mode["frequency_hz"] * t + mode["phase_rad"])
        )

    noise_std = float(case.get("noise_std", 0.0))
    y_noisy = y_clean.copy()
    if noise_std > 0.0:
        y_noisy = y_noisy + noise_std * rng.standard_normal(len(t))

    return t, y_clean, y_noisy, truth_modes


def _preprocess_signal(t, y, case):
    mask = t > float(case.get("drop_before_s", 0.0))
    t_proc = t[mask].copy()
    y_proc = y[mask].copy()

    t_proc = t_proc - t_proc[0]
    y_proc = detrend(y_proc)
    y_proc = filter_signal(
        y_proc,
        t_proc,
        fc=float(case.get("filter_fc_hz", 10.0)),
        N=int(case.get("filter_order", 15)),
    )
    return t_proc, y_proc


def _method_specs(case):
    specs = []
    for order in case["fixed_orders"]:
        specs.append({"method": f"Order {order}", "kind": "fixed", "order": int(order)})
    for tau in case["taus"]:
        specs.append({"method": f"Tau {tau}", "kind": "auto", "tau": float(tau)})
    return specs


def _reconstruct_signal(t, frequencies, dampings, amplitudes, phases):
    y_est = np.zeros_like(t, dtype=float)
    for f, s, a, p in zip(frequencies, dampings, amplitudes, phases):
        y_est += 2.0 * a * np.exp(s * t) * np.cos(2 * np.pi * f * t + p)
    return y_est


def _select_oscillatory_modes(freq, sigma, amplitudes, phases):
    selected = []
    for idx, (f, s, a, p) in enumerate(zip(freq, sigma, amplitudes, phases), start=1):
        if float(f) <= MODE_FREQ_EPS_HZ:
            continue
        selected.append(
            {
                "mode_rank": idx,
                "frequency_hz": float(f),
                "damping": float(s),
                "amplitude": float(np.abs(a)),
                "phase_rad": float(p),
            }
        )
    return selected


def _match_truth_to_estimates(truth_modes, estimated_modes):
    truth_count = len(truth_modes)
    if len(estimated_modes) < truth_count:
        return None

    best = None
    best_cost = None
    estimate_indices = range(len(estimated_modes))

    for subset in combinations(estimate_indices, truth_count):
        for perm in permutations(subset):
            rows = []
            total_cost = 0.0
            for truth_idx, est_idx in enumerate(perm):
                truth = truth_modes[truth_idx]
                est = estimated_modes[est_idx]
                freq_err = abs(est["frequency_hz"] - truth["frequency_hz"])
                damping_err = abs(est["damping"] - truth["damping"])
                distance = math.hypot(freq_err, damping_err)
                total_cost += distance
                rows.append(
                    {
                        "truth_mode": truth_idx + 1,
                        "truth_frequency_hz": truth["frequency_hz"],
                        "truth_damping": truth["damping"],
                        "estimated_mode_rank": est["mode_rank"],
                        "estimated_frequency_hz": est["frequency_hz"],
                        "estimated_damping": est["damping"],
                        "frequency_error_hz": freq_err,
                        "damping_error": damping_err,
                        "distance_2d": distance,
                    }
                )
            if best_cost is None or total_cost < best_cost:
                best_cost = total_cost
                best = rows
    return best


def _evaluate_method(case_name, t, y_ref, truth_modes, spec):
    case = next(case for case in DEFAULT_CASES if case["name"] == case_name)

    try:
        if spec["kind"] == "fixed":
            requested_order = int(spec["order"])
            selected_order = requested_order
            freq, sigma, y_est, _, _, amplitudes = apply_matrix_pencil_fixed_order(t=t, y=y_ref, order=requested_order)
            auto_tau = ""
        else:
            auto_tau = float(spec["tau"])
            selected_order = determine_MP_order(
                t,
                y_ref,
                auto_tau,
                rate=10,
                max_order=int(case.get("auto_max_order", 12)),
            )
            requested_order = selected_order
            freq, sigma, y_est, _, _, amplitudes = apply_matrix_pencil_fixed_order(t=t, y=y_ref, order=selected_order)
    except Exception as exc:
        metric_row = {
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
        }
        return metric_row, [], []

    phases = np.angle(amplitudes)
    oscillatory_modes = _select_oscillatory_modes(freq, sigma, amplitudes, phases)
    matched_rows = _match_truth_to_estimates(truth_modes, oscillatory_modes)

    if matched_rows is None:
        metric_row = {
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
        }
        return metric_row, [], oscillatory_modes

    freq_errors = [row["frequency_error_hz"] for row in matched_rows]
    damping_errors = [row["damping_error"] for row in matched_rows]
    distances = [row["distance_2d"] for row in matched_rows]
    metric_row = {
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
    }
    return metric_row, matched_rows, oscillatory_modes


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_cases(selected_case_names):
    all_cases = {case["name"]: case for case in DEFAULT_CASES}
    if not selected_case_names or selected_case_names == ["all"]:
        return list(DEFAULT_CASES)
    missing = [name for name in selected_case_names if name not in all_cases]
    if missing:
        raise ValueError(f"Unknown case(s): {', '.join(missing)}")
    return [all_cases[name] for name in selected_case_names]


def _build_run_summary(metric_rows):
    pass_rows = [row for row in metric_rows if row["Matched_All_Truth_Modes"]]
    summary = {
        "total_methods": len(metric_rows),
        "methods_with_full_truth_match": len(pass_rows),
        "best_methods_by_case": {},
    }
    for case in {row["Case"] for row in metric_rows}:
        case_rows = [row for row in pass_rows if row["Case"] == case]
        if not case_rows:
            continue
        best_row = min(case_rows, key=lambda row: (row["Mean_2D_Error"], -row["R2"]))
        summary["best_methods_by_case"][case] = {
            "method": best_row["Method"],
            "selected_order": best_row["Selected_Order"],
            "mean_freq_error_hz": best_row["Mean_Freq_Error_Hz"],
            "max_freq_error_hz": best_row["Max_Freq_Error_Hz"],
            "mean_damping_error": best_row["Mean_Damping_Error"],
            "max_damping_error": best_row["Max_Damping_Error"],
            "r2": best_row["R2"],
            "rmse": best_row["RMSE"],
        }
    return summary


def run_validation_suite(run_dir, cases, rng, evaluate_method, run_config):
    metric_rows = []
    matched_rows = []
    raw_mode_rows = []

    for case in cases:
        t, _, y_noisy, truth_modes = _generate_signal(case, rng)
        t_proc, y_proc = _preprocess_signal(t, y_noisy, case)

        for spec in _method_specs(case):
            metric_row, method_matches, oscillatory_modes = evaluate_method(case["name"], t_proc, y_proc, truth_modes, spec)
            metric_rows.append(metric_row)

            for row in method_matches:
                matched_rows.append({"Case": case["name"], "Method": spec["method"], **row})

            for row in oscillatory_modes:
                raw_mode_rows.append({"Case": case["name"], "Method": spec["method"], **row})

    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    _write_csv(run_dir / "case_metrics.csv", metric_rows, CASE_METRIC_FIELDNAMES)
    _write_csv(run_dir / "matched_modes.csv", matched_rows, MATCHED_MODE_FIELDNAMES)
    _write_csv(run_dir / "raw_modes.csv", raw_mode_rows, RAW_MODE_FIELDNAMES)
    summary = _build_run_summary(metric_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic ground-truth Matrix Pencil validation.")
    parser.add_argument("--run-label", required=True, help="Unique label for this run, e.g. windows_run_01 or wsl_run_01")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "validation_outputs"), help="Root folder for validation outputs")
    parser.add_argument("--cases", nargs="*", default=["all"], help="Case names to run, or 'all'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noisy synthetic cases")
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
    }
    run_validation_suite(run_dir, cases, rng, _evaluate_method, config)

    print(f"Validation run saved to: {run_dir}")
    print("Files:")
    print(f"  - {run_dir / 'case_metrics.csv'}")
    print(f"  - {run_dir / 'matched_modes.csv'}")
    print(f"  - {run_dir / 'raw_modes.csv'}")
    print(f"  - {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
