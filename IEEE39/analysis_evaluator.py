import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

REFERENCE_MODES = {
    "Mode 1": {"Frequency": 0.6062, "Damping": -0.0800, "relevant_generators": ["g1", "g8", "g9", "g10"]},
    "Mode 2": {"Frequency": 0.9497, "Damping": -0.1065, "relevant_generators": ["g1", "g4", "g5", "g6", "g7", "g8", "g9"]},
    "Mode 3": {"Frequency": 1.0312, "Damping": -0.2558, "relevant_generators": ["g2", "g3", "g4", "g5"]},
    "Mode 4": {"Frequency": 1.1211, "Damping": -0.3373, "relevant_generators": ["g2", "g3", "g6", "g7"]},
    "Mode 5": {"Frequency": 1.3155, "Damping": -0.4033, "relevant_generators": ["g2", "g3"]},
    "Mode 6": {"Frequency": 1.2851, "Damping": -0.3458, "relevant_generators": ["g1", "g8", "g9"]},
    "Mode 7": {"Frequency": 1.4953, "Damping": -0.7033, "relevant_generators": ["g4", "g5"]},
    "Mode 8": {"Frequency": 1.5202, "Damping": -0.6010, "relevant_generators": ["g4", "g5", "g6", "g7"]},
    "Mode 9": {"Frequency": 1.5468, "Damping": -0.6376, "relevant_generators": ["g1", "g8"]},
}

MATCH_LEVELS = {
    "loose": {"freq_tol_hz": 0.08, "damping_tol": 0.15},
    "mid": {"freq_tol_hz": 0.05, "damping_tol": 0.10},
    "strong": {"freq_tol_hz": 0.03, "damping_tol": 0.05},
}


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_ieee39_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == SCRIPT_DIR.name:
        return SCRIPT_DIR.parent / path
    return SCRIPT_DIR / path


def _scenario_metadata(config):
    scenario_meta = {
        "scenario_name": config.get("name"),
        "analysis_method": config.get("analysis_method"),
        "data_dir": config.get("data_dir"),
        "time_window_s": config.get("time_window_s"),
        "time_cross": config.get("time_cross"),
        "resolved_time_cross": config.get("resolved_time_cross"),
    }

    scenario_json_ref = config.get("data_scenario_json")
    if scenario_json_ref:
        scenario_json_path = resolve_ieee39_path(scenario_json_ref)
        if scenario_json_path.exists():
            data_cfg = load_json(scenario_json_path)
            scenario_meta.update({
                "source_scenario_name": data_cfg.get("scenario_name"),
                "load_name": data_cfg.get("load_name"),
                "dp_percent": data_cfg.get("dp_percent"),
                "dq_percent": data_cfg.get("dq_percent"),
                "event_time_s": data_cfg.get("event_time_s"),
                "sim_stop_time_s": data_cfg.get("sim_stop_time_s"),
            })

    return scenario_meta


def best_reconstruction_rows(report_df):
    if report_df.empty:
        return pd.DataFrame(columns=["Gen", "Signal", "Method", "R2", "RMSE", "Poles"])
    best_idx = report_df.groupby(["Gen", "Signal"])["R2"].idxmax()
    return report_df.loc[best_idx].copy().sort_values(["R2", "Gen", "Signal"])


def mode_match_rows(results_df, reference_modes=None):
    filtered = results_df[results_df["Frequency"] > 0.1].copy()
    rows = []
    recovered_counts = {level: 0 for level in MATCH_LEVELS}
    recovered_names = {level: [] for level in MATCH_LEVELS}
    reference_modes = REFERENCE_MODES if reference_modes is None else reference_modes

    for mode_name, ref in reference_modes.items():
        freq_ref = float(ref["Frequency"])
        damping_ref = float(ref["Damping"])
        relevant_generators = list(ref.get("relevant_generators", []))

        tmp = filtered.copy()
        if relevant_generators:
            tmp = tmp[tmp["Gen"].isin(relevant_generators)].copy()

        if tmp.empty:
            row = {
                "mode": mode_name,
                "reference_frequency_hz": freq_ref,
                "reference_damping": damping_ref,
                "relevant_generators": relevant_generators,
                "best_gen": None,
                "best_signal": None,
                "best_method": None,
                "best_frequency_hz": None,
                "best_damping": None,
                "abs_frequency_error_hz": None,
                "abs_damping_error": None,
                "distance_2d": None,
            }
            for level in MATCH_LEVELS:
                row[f"{level}_match_count"] = 0
            rows.append(row)
            continue

        tmp["abs_frequency_error_hz"] = (tmp["Frequency"] - freq_ref).abs()
        tmp["abs_damping_error"] = (tmp["Damping"] - damping_ref).abs()
        tmp["distance_2d"] = (tmp["abs_frequency_error_hz"] ** 2 + tmp["abs_damping_error"] ** 2) ** 0.5
        best = tmp.nsmallest(1, "distance_2d").iloc[0]

        row = {
            "mode": mode_name,
            "reference_frequency_hz": freq_ref,
            "reference_damping": damping_ref,
            "relevant_generators": relevant_generators,
            "best_gen": best["Gen"],
            "best_signal": best["Signal"],
            "best_method": best["Method"],
            "best_frequency_hz": float(best["Frequency"]),
            "best_damping": float(best["Damping"]),
            "abs_frequency_error_hz": float(best["abs_frequency_error_hz"]),
            "abs_damping_error": float(best["abs_damping_error"]),
            "distance_2d": float(best["distance_2d"]),
        }

        for level, thresholds in MATCH_LEVELS.items():
            match_mask = (
                (tmp["abs_frequency_error_hz"] <= thresholds["freq_tol_hz"])
                & (tmp["abs_damping_error"] <= thresholds["damping_tol"])
            )
            match_count = int(match_mask.sum())
            row[f"{level}_match_count"] = match_count
            if match_count > 0:
                recovered_counts[level] += 1
                recovered_names[level].append(mode_name)

        rows.append(row)

    return pd.DataFrame(rows), recovered_counts, recovered_names


def _reference_modes_from_config(config):
    reference_modes = config.get("reference_modes")
    if isinstance(reference_modes, dict) and reference_modes:
        return reference_modes
    return REFERENCE_MODES


def build_evaluation_payload(analysis_folder):
    folder = Path(analysis_folder)
    results_path = folder / "results.csv"
    report_path = folder / "stats" / "comprehensive_report.csv"
    config_path = folder / "analysis_config.json"
    if not results_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Missing required files in {folder}: results.csv and analysis_config.json"
        )

    results_df = pd.read_csv(results_path)
    config = load_json(config_path)
    report_df = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame(columns=["Gen", "Signal", "Method", "R2", "RMSE", "Poles"])
    reference_modes = _reference_modes_from_config(config)
    mode_df, recovered_counts, recovered_names = mode_match_rows(results_df, reference_modes=reference_modes)
    best_df = best_reconstruction_rows(report_df) if not report_df.empty else pd.DataFrame(columns=["Gen", "Signal", "Method", "R2", "RMSE", "Poles"])

    summary = {
        "analysis_folder": folder.name,
        **_scenario_metadata(config),
        "has_reconstruction_report": bool(report_path.exists()),
        "report_row_count": int(len(report_df)),
        "results_row_count": int(len(results_df)),
        "mean_R2": float(report_df["R2"].mean()) if not report_df.empty else None,
        "median_R2": float(report_df["R2"].median()) if not report_df.empty else None,
        "min_R2": float(report_df["R2"].min()) if not report_df.empty else None,
        "negative_R2_count": int((report_df["R2"] < 0).sum()) if not report_df.empty else 0,
        "lt_0_8_R2_count": int((report_df["R2"] < 0.8).sum()) if not report_df.empty else 0,
        "best_mean_R2": float(best_df["R2"].mean()) if not best_df.empty else None,
        "best_min_R2": float(best_df["R2"].min()) if not best_df.empty else None,
        "best_voltage_R2": float(best_df[best_df["Signal"] == "Voltage"]["R2"].mean()) if not best_df.empty else None,
        "best_current_R2": float(best_df[best_df["Signal"] == "Current"]["R2"].mean()) if not best_df.empty else None,
        "best_active_power_R2": float(best_df[best_df["Signal"] == "Active Power"]["R2"].mean()) if not best_df.empty else None,
        "best_reactive_power_R2": float(best_df[best_df["Signal"] == "Reactive Power"]["R2"].mean()) if not best_df.empty else None,
        "modal_loose_modes": recovered_counts["loose"],
        "modal_mid_modes": recovered_counts["mid"],
        "modal_strong_modes": recovered_counts["strong"],
        "modal_loose_recovered": recovered_names["loose"],
        "modal_mid_recovered": recovered_names["mid"],
        "modal_strong_recovered": recovered_names["strong"],
    }

    worst_best = best_df[["Gen", "Signal", "Method", "R2", "RMSE", "Poles"]].head(10).to_dict(orient="records")
    best_reconstruction_by_signal = best_df.sort_values(["Gen", "Signal"])[["Gen", "Signal", "Method", "R2", "RMSE", "Poles"]].to_dict(orient="records")
    mode_matches = mode_df.to_dict(orient="records")

    payload = {
        "summary": summary,
        "reference_modes": reference_modes,
        "match_levels": MATCH_LEVELS,
        "mode_matches": mode_matches,
        "best_reconstruction_by_signal": best_reconstruction_by_signal,
        "worst_best_reconstruction": worst_best,
    }
    return payload


def update_analysis_config_with_evaluation(analysis_folder):
    folder = Path(analysis_folder)
    config_path = folder / "analysis_config.json"
    config = load_json(config_path)
    payload = build_evaluation_payload(folder)
    config["evaluation"] = payload
    save_json(config_path, config)
    return payload
