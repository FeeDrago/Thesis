import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
PRELIM_DIR = REPO_DIR / "PreliminaryInvestigation"

if str(PRELIM_DIR) not in sys.path:
    sys.path.insert(0, str(PRELIM_DIR))


AMBIENT_DEFAULT_SIGNALS = {
    "s:ut in p.u.": "Voltage",
    "s:cur1 in p.u.": "Current",
}
AMBIENT_DEFAULT_ORDER_GROUPS = [
    {"name": "orders1", "orders": list(range(2, 32, 2))},
    {"name": "orders2", "orders": list(range(10, 50, 5))},
]
AMBIENT_DEFAULT_DOWNSAMPLE_HZ = 5.0
AMBIENT_DEFAULT_LPF_HZ = 2.0
AMBIENT_DEFAULT_DETREND = True
AMBIENT_DEFAULT_CLUSTERING_METHODS = ["kmeans", "kmedoids", "optics"]
AMBIENT_DEFAULT_CLUSTERING_SCOPE = {"global": False, "by_control_area": True}
AMBIENT_DEFAULT_OPTICS_SETTINGS = {
    "premerge_enabled": True,
    "premerge_scope": "Gen+Signal",
    "merge_radius_scaled": 0.20,
    "merge_min_distinct_orders": 2,
    "min_samples_min": 5,
    "min_samples_max": 20,
    "xi": 0.05,
}
AMBIENT_REFERENCE_MODES = {
    "Mode 1": {"Frequency": 0.6062, "Damping": -0.0800, "Damping_Factor": 0.0210, "Generator_Involvement": "1-9 vs. 10", "relevant_areas": [1, 2, 3]},
    "Mode 2": {"Frequency": 0.9497, "Damping": -0.1065, "Damping_Factor": 0.0178, "Generator_Involvement": "1,8 and 9 vs. 4,5,6 and 7", "relevant_areas": [1, 2]},
    "Mode 3": {"Frequency": 1.0312, "Damping": -0.2558, "Damping_Factor": 0.0395, "Generator_Involvement": "2 and 3 vs. 4 and 5", "relevant_areas": [2, 3]},
    "Mode 4": {"Frequency": 1.1211, "Damping": -0.3373, "Damping_Factor": 0.0478, "Generator_Involvement": "2 and 3 vs. 6 and 7", "relevant_areas": [2, 3]},
    "Mode 5": {"Frequency": 1.3155, "Damping": -0.4033, "Damping_Factor": 0.0487, "Generator_Involvement": "2 vs. 3", "relevant_areas": [2]},
    "Mode 6": {"Frequency": 1.2851, "Damping": -0.3458, "Damping_Factor": 0.0428, "Generator_Involvement": "1 vs. 8 and 9", "relevant_areas": [1]},
    "Mode 7": {"Frequency": 1.4953, "Damping": -0.7033, "Damping_Factor": 0.0747, "Generator_Involvement": "4 vs. 5", "relevant_areas": [3]},
    "Mode 8": {"Frequency": 1.5202, "Damping": -0.6010, "Damping_Factor": 0.0628, "Generator_Involvement": "5 and 7 vs. 4 and 6", "relevant_areas": [3]},
    "Mode 9": {"Frequency": 1.5468, "Damping": -0.6376, "Damping_Factor": 0.0655, "Generator_Involvement": "1 vs. 8", "relevant_areas": [1]},
}
CONTROL_AREAS = {
    "area_1": ["g1", "g8", "g9", "g10"],
    "area_2": ["g2", "g3"],
    "area_3": ["g4", "g5", "g6", "g7"],
}
MIN_REQUIRED_SAMPLES = 32
MIN_HANKEL_COLUMNS = 16
FREQ_EPS_HZ = 1e-6
RESULT_COLUMNS = [
    "Scenario",
    "Gen",
    "Signal",
    "Method",
    "AnalysisMethod",
    "Order",
    "ModeIndex",
    "Frequency",
    "Damping",
    "Amplitude",
    "Phase",
    "DampingRatio",
    "DiscreteEigenvalueReal",
    "DiscreteEigenvalueImag",
    "DiscreteEigenvalueMagnitude",
    "Stable",
    "SingularValue",
    "SingularValueEnergyRatio",
    "OrderSingularValueEnergyRatio",
    "StatePredictionRMSE",
    "OutputPredictionRMSE",
    "OutputFitPercent",
]
ORDER_SUMMARY_COLUMNS = [
    "Scenario",
    "Gen",
    "Signal",
    "Order",
    "ModesIdentified",
    "StableModes",
    "MeanOutputPredictionRMSE",
    "MeanStatePredictionRMSE",
    "OutputFitPercent",
    "OrderSingularValueEnergyRatio",
    "Status",
    "Message",
]


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _path_for_metadata(path):
    try:
        return Path(path).relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path)


def _timing_entry(seconds, skipped=False):
    total_seconds = max(0.0, float(seconds))
    minutes = int(total_seconds // 60)
    seconds_part = total_seconds - (minutes * 60)
    return {
        "seconds": round(total_seconds, 6),
        "min_sec": f"{minutes:02d}:{seconds_part:04.1f}",
        "skipped": bool(skipped),
    }


def _resolve_clustering_scope(scope_name):
    if scope_name == "none":
        return {"global": False, "by_control_area": False}
    if scope_name == "global":
        return {"global": True, "by_control_area": False}
    if scope_name == "both":
        return {"global": True, "by_control_area": True}
    return {"global": False, "by_control_area": True}


def _resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == BASE_DIR.name:
        return REPO_DIR / path
    return BASE_DIR / path


def _read_numeric_csv(csv_path):
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _time_mask(time_values, mask_config):
    mask_config = mask_config or {}
    mask = np.ones(len(time_values), dtype=bool)
    if mask_config.get("start") is not None:
        mask &= time_values > float(mask_config["start"])
    if mask_config.get("start_inclusive") is not None:
        mask &= time_values >= float(mask_config["start_inclusive"])
    if mask_config.get("end") is not None:
        mask &= time_values < float(mask_config["end"])
    if mask_config.get("end_inclusive") is not None:
        mask &= time_values <= float(mask_config["end_inclusive"])
    return mask


def _parse_area_names_to_indices(area_names):
    indices = []
    for area_name in area_names:
        text = str(area_name).strip()
        if not text:
            continue
        try:
            indices.append(int(text.split("_")[-1]))
        except (TypeError, ValueError):
            continue
    return indices


def _load_generated_reference_modes(data_dir):
    modal_csv = _resolve_path(data_dir) / "modal" / "electromechanical_modes_stable_oscillatory.csv"
    if not modal_csv.exists():
        return None

    df = pd.read_csv(modal_csv)
    if df.empty:
        return None

    required_columns = {"ModeIndex", "FrequencyHz", "Damping"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise SystemExit(
            f"Generated electromechanical modes file is missing required columns {sorted(missing)}: {modal_csv}"
        )

    reference_modes = {}
    for _, row in df.iterrows():
        try:
            mode_index = int(row["ModeIndex"])
            frequency = float(row["FrequencyHz"])
            damping = float(row["Damping"])
        except (TypeError, ValueError):
            continue

        mode_name = f"Mode {mode_index}"
        participating_generators = [
            entry.strip() for entry in str(row.get("ParticipatingGenerators", "")).split(";")
            if entry and entry.strip()
        ]
        participating_areas = [
            entry.strip() for entry in str(row.get("ParticipatingAreas", "")).split(";")
            if entry and entry.strip()
        ]
        reference_modes[mode_name] = {
            "Frequency": frequency,
            "Damping": damping,
            "ModeIndex": mode_index,
            "RealPart": None if pd.isna(row.get("RealPart")) else float(row.get("RealPart")),
            "ImagPart": None if pd.isna(row.get("ImagPart")) else float(row.get("ImagPart")),
            "PhiSpeedRatio": None if pd.isna(row.get("PhiSpeedRatio")) else float(row.get("PhiSpeedRatio")),
            "Generator_Involvement": str(row.get("ParticipatingGenerators", "")).strip(),
            "relevant_generators": participating_generators,
            "relevant_areas": _parse_area_names_to_indices(participating_areas),
        }

    return _path_for_metadata(modal_csv), reference_modes if reference_modes else None


def _load_reference_modes(data_dir):
    generated = _load_generated_reference_modes(data_dir)
    if generated is not None:
        source, reference_modes = generated
        if reference_modes:
            return source, reference_modes
    return "built_in", dict(AMBIENT_REFERENCE_MODES)


def _reference_modes_for_control_area(reference_modes, area_name):
    try:
        area_idx = int(str(area_name).split("_")[-1])
    except (TypeError, ValueError):
        return dict(reference_modes)

    filtered = {}
    for mode_name, mode_data in reference_modes.items():
        relevant_areas = mode_data.get("relevant_areas")
        if not relevant_areas or area_idx in relevant_areas:
            filtered[mode_name] = dict(mode_data)
    return filtered


def _save_combined_reference_mad_summary(area_root, reference_modes):
    assignment_files = sorted(area_root.glob("area_*/reference_mad/mode_estimates_with_reference_assignment.csv"))
    if not assignment_files:
        return

    combined_dir = area_root / "reference_mad"
    combined_dir.mkdir(parents=True, exist_ok=True)

    assigned_df = pd.concat([pd.read_csv(path) for path in assignment_files], ignore_index=True)
    assigned_df.to_csv(combined_dir / "mode_estimates_with_reference_assignment.csv", index=False)

    summary = (
        assigned_df.groupby("Reference_Mode", as_index=False)
        .agg(
            Reference_Frequency=("Reference_Frequency", "first"),
            Reference_Damping=("Reference_Damping", "first"),
            Count=("Distance_to_Reference", "size"),
            MAD=("Distance_to_Reference", "median"),
            Mean_Distance=("Distance_to_Reference", "mean"),
            Max_Distance=("Distance_to_Reference", "max"),
        )
    )

    mode_names = list(reference_modes.keys())
    complete_summary = pd.DataFrame({
        "Reference_Mode": mode_names,
        "Reference_Frequency": [float(reference_modes[name]["Frequency"]) for name in mode_names],
        "Reference_Damping": [float(reference_modes[name]["Damping"]) for name in mode_names],
    }).merge(
        summary,
        on=["Reference_Mode", "Reference_Frequency", "Reference_Damping"],
        how="left",
    )
    complete_summary["Count"] = complete_summary["Count"].fillna(0).astype(int)
    complete_summary.to_csv(combined_dir / "reference_mad_summary_overall.csv", index=False)


def _raw_sample_hz(time_values):
    diffs = np.diff(np.asarray(time_values, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        raise SystemExit("Could not infer a valid sample rate from the ambient time axis.")
    return float(1.0 / np.median(diffs))


def _maybe_downsample(t, y, target_hz):
    raw_hz = _raw_sample_hz(t)
    if target_hz is None or target_hz <= 0.0 or raw_hz <= target_hz:
        return t, y, raw_hz, raw_hz, 1

    factor = max(1, int(round(raw_hz / float(target_hz))))
    if factor == 1:
        return t, y, raw_hz, raw_hz, 1

    return t[::factor], y[::factor], raw_hz, float(raw_hz / factor), factor


def _maybe_lowpass(y, sample_hz, cutoff_hz):
    if cutoff_hz is None or cutoff_hz <= 0.0:
        return y
    nyquist_hz = 0.5 * float(sample_hz)
    if nyquist_hz <= 0.0:
        return y
    normalized = float(cutoff_hz) / nyquist_hz
    if normalized >= 1.0:
        return y
    b, a = butter(4, normalized, btype="low")
    return filtfilt(b, a, y)


def preprocess_ambient_signal(df, column_name, time_mask_config, detrend_enabled, downsample_hz, lowpass_hz):
    time_all = df.iloc[:, 0].to_numpy(dtype=float)
    signal_all = df[column_name].to_numpy(dtype=float)
    mask = _time_mask(time_all, time_mask_config)
    if not np.any(mask):
        return None, None, None

    t = time_all[mask].copy()
    y = signal_all[mask].copy()
    valid = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(valid) < MIN_REQUIRED_SAMPLES:
        return None, None, None

    t = t[valid]
    y = y[valid]
    if detrend_enabled:
        y = detrend(y)

    t, y, raw_hz, effective_hz, downsample_factor = _maybe_downsample(t, y, downsample_hz)
    y = _maybe_lowpass(y, effective_hz, lowpass_hz)
    if len(t) < MIN_REQUIRED_SAMPLES:
        return None, None, None

    meta = {
        "raw_sample_hz": raw_hz,
        "effective_sample_hz": effective_hz,
        "downsample_factor": downsample_factor,
        "selected_samples": int(len(t)),
        "time_start_s": float(t[0]),
        "time_end_s": float(t[-1]),
        "mean_after_preprocessing": float(np.mean(y)),
        "std_after_preprocessing": float(np.std(y)),
    }
    return t, y, meta


def _build_hankel(signal_values, block_rows):
    y = np.asarray(signal_values, dtype=float).reshape(-1)
    columns = y.size - (2 * block_rows) + 1
    if columns < MIN_HANKEL_COLUMNS:
        raise ValueError(
            f"Ambient signal is too short for N4SID order sweep with block_rows={block_rows}; need at least {2 * block_rows + MIN_HANKEL_COLUMNS - 1} samples."
        )
    return np.vstack([y[idx:idx + columns] for idx in range(2 * block_rows)])


def _fit_modal_amplitudes_and_phases(t, y, modal_rows):
    if not modal_rows:
        return []

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t_rel = t - t[0]
    basis_columns = []
    valid_rows = []
    for row in modal_rows:
        sigma = float(row["Damping"])
        omega = 2.0 * np.pi * float(row["Frequency"])
        envelope = np.exp(sigma * t_rel)
        basis_columns.append(envelope * np.cos(omega * t_rel))
        basis_columns.append(envelope * np.sin(omega * t_rel))
        valid_rows.append(row)

    design = np.column_stack(basis_columns)
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    fitted_rows = []
    for idx, row in enumerate(valid_rows):
        alpha = float(coeffs[2 * idx])
        beta = float(coeffs[(2 * idx) + 1])
        amplitude = 0.5 * float(np.sqrt((alpha ** 2) + (beta ** 2)))
        phase = float(np.arctan2(-beta, alpha))
        fitted_row = dict(row)
        fitted_row["Amplitude"] = amplitude
        fitted_row["Phase"] = phase
        fitted_rows.append(fitted_row)
    return fitted_rows


def identify_n4sid_modes(t, y, dt_s, order):
    order = int(order)
    if order < 2:
        raise ValueError("N4SID order must be at least 2.")

    block_rows = max(order + 4, 12)
    hankel = _build_hankel(y, block_rows)
    U, singular_values, Vt = np.linalg.svd(hankel, full_matrices=False)
    if order >= len(singular_values):
        raise ValueError(f"N4SID order {order} exceeds available numerical rank {len(singular_values) - 1}.")

    U1 = U[:, :order]
    S1 = singular_values[:order]
    V1 = Vt[:order, :]

    sqrt_s1 = np.diag(np.sqrt(S1))
    observability = U1 @ sqrt_s1
    state_sequence = sqrt_s1 @ V1

    output_dim = 1
    obs_upper = observability[:-output_dim, :]
    obs_lower = observability[output_dim:, :]
    a_matrix = np.linalg.lstsq(obs_upper, obs_lower, rcond=None)[0]
    c_matrix = observability[:output_dim, :]

    x_k = state_sequence[:, :-1].T
    x_next = state_sequence[:, 1:].T
    y_k = np.asarray(y[:x_k.shape[0]], dtype=float).reshape(-1, 1)
    x_next_hat = x_k @ a_matrix
    y_hat = x_k @ c_matrix.T

    state_rmse = float(np.sqrt(np.mean((x_next - x_next_hat) ** 2)))
    output_rmse = float(np.sqrt(np.mean((y_k - y_hat) ** 2)))
    output_var = float(np.var(y_k))
    fit_percent = None
    if output_var > 0.0:
        fit_percent = float(max(0.0, 100.0 * (1.0 - (output_rmse ** 2) / output_var)))

    eigvals = np.linalg.eigvals(a_matrix)
    poles = np.log(eigvals) / float(dt_s)
    total_sv_energy = float(np.sum(singular_values ** 2))
    order_sv_energy = float(np.sum(S1 ** 2))
    modes = []
    for mode_index, pole in enumerate(poles, start=1):
        if not np.isfinite(pole.real) or not np.isfinite(pole.imag):
            continue
        frequency_hz = abs(float(np.imag(pole))) / (2.0 * np.pi)
        damping = float(np.real(pole))
        if frequency_hz <= FREQ_EPS_HZ:
            continue
        discrete_eig = eigvals[mode_index - 1]
        dominant_sv = float(S1[min(mode_index - 1, len(S1) - 1)])
        singular_value_energy_ratio = None
        if total_sv_energy > 0.0:
            singular_value_energy_ratio = float((dominant_sv ** 2) / total_sv_energy)
        damping_ratio = None
        pole_mag = float(np.abs(pole))
        if pole_mag > 0.0:
            damping_ratio = float(-damping / pole_mag)

        modes.append({
            "Order": order,
            "ModeIndex": int(mode_index),
            "Frequency": frequency_hz,
            "Damping": damping,
            "Amplitude": None,
            "Phase": None,
            "DampingRatio": damping_ratio,
            "DiscreteEigenvalueReal": float(np.real(discrete_eig)),
            "DiscreteEigenvalueImag": float(np.imag(discrete_eig)),
            "DiscreteEigenvalueMagnitude": float(np.abs(discrete_eig)),
            "Stable": bool(np.abs(discrete_eig) < 1.0),
            "SingularValue": dominant_sv,
            "SingularValueEnergyRatio": singular_value_energy_ratio,
            "OrderSingularValueEnergyRatio": None if total_sv_energy <= 0.0 else float(order_sv_energy / total_sv_energy),
            "StatePredictionRMSE": state_rmse,
            "OutputPredictionRMSE": output_rmse,
            "OutputFitPercent": fit_percent,
        })

    modes = _fit_modal_amplitudes_and_phases(t=t, y=y, modal_rows=modes)

    summary = {
        "Order": order,
        "ModesIdentified": int(len(modes)),
        "StableModes": int(sum(1 for row in modes if row["Stable"])),
        "MeanOutputPredictionRMSE": output_rmse,
        "MeanStatePredictionRMSE": state_rmse,
        "OutputFitPercent": fit_percent,
        "OrderSingularValueEnergyRatio": None if total_sv_energy <= 0.0 else float(order_sv_energy / total_sv_energy),
    }
    return modes, summary


def _run_clustering_pipeline(results_path, output_path, reference_modes, methods, optics_settings=None):
    from clustering_analysis import (
        _load_screened_data,
        _save_reference_mad_outputs,
        run_kmeans_modal_analysis,
        run_kmedoids_modal_analysis,
        run_optics_modal_analysis,
    )

    requested_methods = list(methods or [])
    output_path.mkdir(parents=True, exist_ok=True)
    timings = {}

    screen_start = time.perf_counter()
    df_for_mad = _load_screened_data(str(results_path), str(output_path))
    timings["screen_and_load"] = _timing_entry(time.perf_counter() - screen_start)

    ref_start = time.perf_counter()
    if df_for_mad is not None:
        _save_reference_mad_outputs(df_for_mad, str(output_path), reference_modes=reference_modes)
    timings["reference_mad"] = _timing_entry(time.perf_counter() - ref_start, skipped=df_for_mad is None)

    runners = {
        "kmeans": run_kmeans_modal_analysis,
        "kmedoids": run_kmedoids_modal_analysis,
        "optics": run_optics_modal_analysis,
    }
    for method in requested_methods:
        started = time.perf_counter()
        if method == "optics":
            runners[method](
                str(results_path),
                str(output_path),
                reference_modes=reference_modes,
                optics_settings=optics_settings,
            )
        else:
            runners[method](str(results_path), str(output_path), reference_modes=reference_modes)
        timings[method] = _timing_entry(time.perf_counter() - started)

    timings["total"] = _timing_entry(sum(entry["seconds"] for entry in timings.values()))
    return timings


def run_ambient_clustering_for_results(output_dir, results_path, df_results, reference_modes, methods, optics_settings=None, clustering_scope=None):
    if df_results.empty:
        print(f"No ambient N4SID results for {output_dir}; skipping clustering.")
        return {}

    timings = {}
    scope = dict(clustering_scope or AMBIENT_DEFAULT_CLUSTERING_SCOPE)

    if scope.get("global", False):
        global_out = output_dir / "clustering" / "global"
        timings["global"] = _run_clustering_pipeline(
            results_path=results_path,
            output_path=global_out,
            reference_modes=reference_modes,
            methods=methods,
            optics_settings=optics_settings,
        )

    if scope.get("by_control_area", False):
        area_root = output_dir / "clustering" / "by_control_area"
        area_timings = {}
        for area_name, gens in CONTROL_AREAS.items():
            area_out = area_root / area_name
            area_out.mkdir(parents=True, exist_ok=True)
            area_df = df_results[df_results["Gen"].isin(gens)].copy()
            area_reference_modes = _reference_modes_for_control_area(reference_modes, area_name)
            if area_df.empty:
                area_timings[area_name] = {"total": _timing_entry(0.0, skipped=True)}
                continue

            area_results_path = area_out / "results.csv"
            area_df.to_csv(area_results_path, index=False)
            _save_json(area_out / "control_area.json", {"name": area_name, "generators": gens})
            area_timings[area_name] = _run_clustering_pipeline(
                results_path=area_results_path,
                output_path=area_out,
                reference_modes=area_reference_modes,
                methods=methods,
                optics_settings=optics_settings,
            )

        _save_combined_reference_mad_summary(area_root, reference_modes)
        timings["by_control_area"] = area_timings
    return timings


def resolve_ambient_settings(scenario, args):
    if args.n4sid_orders is not None:
        order_groups = [{"name": "custom_orders", "orders": [int(order) for order in args.n4sid_orders]}]
    else:
        order_groups = [
            {"name": str(group["name"]), "orders": [int(order) for order in group["orders"]]}
            for group in AMBIENT_DEFAULT_ORDER_GROUPS
        ]

    for group in order_groups:
        if not group["orders"]:
            raise SystemExit(f"Ambient N4SID order group '{group['name']}' is empty.")

    reference_source, reference_modes = _load_reference_modes(scenario["data_dir"])
    clustering_methods = list(args.clustering_methods) if args.clustering_methods is not None else list(AMBIENT_DEFAULT_CLUSTERING_METHODS)
    signals = dict(scenario.get("columns") or AMBIENT_DEFAULT_SIGNALS)
    if not signals:
        raise SystemExit("Ambient N4SID requires at least one signal.")

    optics_settings = dict(AMBIENT_DEFAULT_OPTICS_SETTINGS)
    if args.merge_radius is not None:
        optics_settings["merge_radius_scaled"] = float(args.merge_radius)
    clustering_scope = _resolve_clustering_scope(getattr(args, "clustering_scope", "areas"))

    return {
        "analysis_method": "n4sid",
        "order_groups": order_groups,
        "ambient_preprocessing": {
            "detrend": not bool(args.ambient_no_detrend),
            "downsample_hz": float(args.ambient_downsample_hz) if args.ambient_downsample_hz is not None else float(AMBIENT_DEFAULT_DOWNSAMPLE_HZ),
            "low_pass_hz": float(args.ambient_lpf_hz) if args.ambient_lpf_hz is not None else float(AMBIENT_DEFAULT_LPF_HZ),
        },
        "clustering_methods": clustering_methods,
        "clustering_scope": clustering_scope,
        "optics_settings": optics_settings,
        "reference_modes_source": reference_source,
        "reference_modes": reference_modes,
        "signals": signals,
    }


def run_ambient_n4sid_for_scenario(name, scenario, args):
    data_dir = _resolve_path(scenario["data_dir"])
    base_output_dir = _resolve_path(scenario["output_dir"])
    base_output_dir.mkdir(parents=True, exist_ok=True)
    generated_config = _load_json(data_dir / "scenario.json") if (data_dir / "scenario.json").exists() else None
    generators = list(scenario.get("generators") or [f"g{i}" for i in range(1, 11)])
    settings = resolve_ambient_settings(scenario, args)

    preprocess_cfg = settings["ambient_preprocessing"]
    signals = settings["signals"]
    time_mask = dict(scenario.get("time_mask") or {})
    analysis_start = time.perf_counter()
    sweep_summaries = []

    for order_group in settings["order_groups"]:
        output_dir = base_output_dir / order_group["name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        results_rows = []
        order_summary_rows = []
        signal_summary_rows = []
        per_signal_timings = {}
        sweep_start = time.perf_counter()

        for gen in generators:
            csv_path = data_dir / f"{gen}.csv"
            if not csv_path.exists():
                print(f"File missing: {csv_path}", flush=True)
                continue

            df = _read_numeric_csv(csv_path)
            for column_name, signal_label in signals.items():
                if column_name not in df.columns:
                    print(f"Column {column_name} missing in {gen}", flush=True)
                    continue

                signal_start = time.perf_counter()
                t, y, preprocess_meta = preprocess_ambient_signal(
                    df=df,
                    column_name=column_name,
                    time_mask_config=time_mask,
                    detrend_enabled=bool(preprocess_cfg["detrend"]),
                    downsample_hz=float(preprocess_cfg["downsample_hz"]),
                    lowpass_hz=float(preprocess_cfg["low_pass_hz"]),
                )
                if t is None or y is None or preprocess_meta is None:
                    print(f"Not enough samples for ambient N4SID on {gen} {signal_label}", flush=True)
                    continue

                dt_s = float(np.median(np.diff(t)))
                signal_summary_rows.append({
                    "Scenario": name,
                    "Gen": gen,
                    "Signal": signal_label,
                    "SelectedSamples": int(preprocess_meta["selected_samples"]),
                    "TimeStart_s": preprocess_meta["time_start_s"],
                    "TimeEnd_s": preprocess_meta["time_end_s"],
                    "RawSampleHz": preprocess_meta["raw_sample_hz"],
                    "EffectiveSampleHz": preprocess_meta["effective_sample_hz"],
                    "DownsampleFactor": int(preprocess_meta["downsample_factor"]),
                    "MeanAfterPreprocessing": preprocess_meta["mean_after_preprocessing"],
                    "StdAfterPreprocessing": preprocess_meta["std_after_preprocessing"],
                })

                order_timing_rows = {}
                for order in order_group["orders"]:
                    order_start = time.perf_counter()
                    try:
                        modes, order_summary = identify_n4sid_modes(t=t, y=y, dt_s=dt_s, order=order)
                    except ValueError as exc:
                        order_summary_rows.append({
                            "Scenario": name,
                            "Gen": gen,
                            "Signal": signal_label,
                            "Order": int(order),
                            "ModesIdentified": 0,
                            "StableModes": 0,
                            "MeanOutputPredictionRMSE": None,
                            "MeanStatePredictionRMSE": None,
                            "OutputFitPercent": None,
                            "OrderSingularValueEnergyRatio": None,
                            "Status": "error",
                            "Message": str(exc),
                        })
                        order_timing_rows[str(order)] = _timing_entry(time.perf_counter() - order_start, skipped=True)
                        continue

                    for row in modes:
                        results_rows.append({
                            "Scenario": name,
                            "Gen": gen,
                            "Signal": signal_label,
                            "Method": f"Order {order}",
                            "AnalysisMethod": "n4sid",
                            **row,
                        })

                    order_summary_rows.append({
                        "Scenario": name,
                        "Gen": gen,
                        "Signal": signal_label,
                        **order_summary,
                        "Status": "ok",
                        "Message": None,
                    })
                    order_timing_rows[str(order)] = _timing_entry(time.perf_counter() - order_start)

                per_signal_timings.setdefault(gen, {})[signal_label] = {
                    "total_signal": _timing_entry(time.perf_counter() - signal_start),
                    "orders": order_timing_rows,
                }

        df_results = pd.DataFrame(results_rows, columns=RESULT_COLUMNS)
        df_order_summary = pd.DataFrame(order_summary_rows, columns=ORDER_SUMMARY_COLUMNS)
        results_path = output_dir / "results.csv"
        order_summary_path = output_dir / "order_summary.csv"
        df_results.to_csv(results_path, index=False)
        df_order_summary.to_csv(order_summary_path, index=False)

        clustering_details = {}
        skip_clustering_value = getattr(args, "skip_clustering", None)
        clustering_enabled = not bool(skip_clustering_value) if skip_clustering_value is not None else True
        if clustering_enabled:
            clustering_details = run_ambient_clustering_for_results(
                output_dir=output_dir,
                results_path=results_path,
                df_results=df_results,
                reference_modes=settings["reference_modes"],
                methods=settings["clustering_methods"],
                optics_settings=settings["optics_settings"],
                clustering_scope=settings["clustering_scope"],
            )
        clustering_total_seconds = 0.0
        if clustering_enabled:
            global_seconds = clustering_details.get("global", {}).get("total", {}).get("seconds", 0.0)
            area_seconds = sum(
                area_entry.get("total", {}).get("seconds", 0.0)
                for area_entry in clustering_details.get("by_control_area", {}).values()
            )
            clustering_total_seconds = float(global_seconds + area_seconds)

        time_window = None
        if signal_summary_rows:
            time_window = {
                "start_s": float(min(row["TimeStart_s"] for row in signal_summary_rows)),
                "end_s": float(max(row["TimeEnd_s"] for row in signal_summary_rows)),
            }

        sweep_config = {
            "name": name,
            "analysis_method": "n4sid",
            "order_group_name": order_group["name"],
            "data_dir": _path_for_metadata(data_dir),
            "output_dir": _path_for_metadata(output_dir),
            "data_scenario_json": _path_for_metadata(data_dir / "scenario.json") if generated_config else None,
            "disturbance_type": None if generated_config is None else generated_config.get("disturbance_type"),
            "time_mask": time_mask,
            "time_window_s": time_window,
            "time_reset_to_zero": False,
            "generators_used": generators,
            "signals_used": list(signals.values()),
            "columns": signals,
            "n4sid_orders": list(order_group["orders"]),
            "ambient_preprocessing": {
                **preprocess_cfg,
                "signals": list(signals.values()),
            },
            "clustering_methods": settings["clustering_methods"],
            "clustering_scope": dict(settings["clustering_scope"]),
            "optics_settings": settings["optics_settings"],
            "reference_modes_source": settings["reference_modes_source"],
            "reference_modes": settings["reference_modes"],
            "signal_summaries": signal_summary_rows,
            "timings": {
                "n4sid": _timing_entry(time.perf_counter() - sweep_start),
                "per_generator_signal": per_signal_timings,
                "clustering": _timing_entry(
                    clustering_total_seconds,
                    skipped=not clustering_enabled,
                ),
                "clustering_details": clustering_details,
            },
        }
        _save_json(output_dir / "analysis_config.json", sweep_config)
        sweep_summaries.append({
            "name": order_group["name"],
            "orders": list(order_group["orders"]),
            "output_dir": _path_for_metadata(output_dir),
            "results_csv": _path_for_metadata(results_path),
            "order_summary_csv": _path_for_metadata(order_summary_path),
            "row_count": int(len(df_results)),
        })

    analysis_config = {
        "name": name,
        "analysis_method": "n4sid",
        "data_dir": _path_for_metadata(data_dir),
        "output_dir": _path_for_metadata(base_output_dir),
        "data_scenario_json": _path_for_metadata(data_dir / "scenario.json") if generated_config else None,
        "disturbance_type": None if generated_config is None else generated_config.get("disturbance_type"),
        "time_mask": time_mask,
        "time_reset_to_zero": False,
        "generators_used": generators,
        "signals_used": list(signals.values()),
        "columns": signals,
        "n4sid_order_groups": settings["order_groups"],
        "ambient_preprocessing": {
            **preprocess_cfg,
            "signals": list(signals.values()),
        },
        "clustering_methods": settings["clustering_methods"],
        "clustering_scope": dict(settings["clustering_scope"]),
        "optics_settings": settings["optics_settings"],
        "reference_modes_source": settings["reference_modes_source"],
        "reference_modes": settings["reference_modes"],
        "sweeps": sweep_summaries,
        "timings": {
            "n4sid": _timing_entry(time.perf_counter() - analysis_start),
        },
    }
    _save_json(base_output_dir / "analysis_config.json", analysis_config)
    return base_output_dir, None, pd.DataFrame(), analysis_config


def load_existing_ambient_results_for_scenario(name, scenario):
    base_output_dir = _resolve_path(scenario["output_dir"])
    config_path = base_output_dir / "analysis_config.json"
    if not config_path.exists():
        raise SystemExit(
            f"Cannot skip N4SID for '{name}' because ambient analysis_config.json does not exist: {config_path}"
        )

    analysis_config = _load_json(config_path)
    sweeps = analysis_config.get("sweeps") or []
    if not sweeps:
        raise SystemExit(
            f"Cannot skip N4SID for '{name}' because no sweeps are recorded in: {config_path}"
        )

    for sweep in sweeps:
        results_path = _resolve_path(sweep["results_csv"])
        if not results_path.exists():
            raise SystemExit(
                f"Cannot skip N4SID for '{name}' because sweep results are missing: {results_path}"
            )

    analysis_config.setdefault("timings", {})
    analysis_config["timings"]["n4sid"] = _timing_entry(0.0, skipped=True)
    return base_output_dir, None, pd.DataFrame(), analysis_config
