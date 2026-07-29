import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate, detrend


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
AMBIENT_DEFAULT_DETREND = True
AMBIENT_DEFAULT_CLUSTERING_METHODS = ["kmeans", "kmedoids", "optics", "dbscan", "hdbscan", "gmm", "agglomerative"]
AMBIENT_DEFAULT_CLUSTERING_SCOPE = {"global": False, "by_control_area": True}
AMBIENT_DEFAULT_OPTICS_SETTINGS = {
    "pm_values": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "xi_values": [round(value, 2) for value in np.arange(0.02, 0.401, 0.02)],
    "multiply_by_orders": True,
    "min_npts": 2,
    "min_assigned_ratio": 0.50,
}
AMBIENT_DEFAULT_DBSCAN_SETTINGS = {
    "pe_values": [round(value, 3) for value in np.arange(0.01, 0.051, 0.005)],
    "pm_values": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "multiply_by_orders": True,
    "min_npts": 2,
    "min_assigned_ratio": 0.50,
}
AMBIENT_DEFAULT_HDBSCAN_SETTINGS = {
    "min_cluster_size": 20,
    "min_samples": 10,
    "cluster_selection_method": "eom",
    "metric": "euclidean",
    "allow_single_cluster": False,
    "copy": True,
}
AMBIENT_DEFAULT_GMM_SETTINGS = {
    "covariance_type": "full",
    "init_params": "kmeans",
    "n_init": 1,
    "random_state": 42,
    "max_iter": 100,
    "reg_covar": 1e-6,
}
AMBIENT_DEFAULT_AGGLOMERATIVE_SETTINGS = {
    "linkage": "ward",
    "metric": "euclidean",
    "compute_distances": False,
}
AMBIENT_PAPER_MAD_SETTINGS = {
    "definition": "median(abs(lambda_hat - lambda_reference))",
    "pole_coordinates": "sigma_j_omega",
    "omega_from_frequency": "2*pi*frequency_hz",
    "units": "rad/s",
    "noise_policy": "exclude_density_noise",
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
BLOCK_ROWS_MIN = 20
BLOCK_ROWS_MARGIN = 6
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
CLUSTERING_SELECTION_SUMMARY_COLUMNS = [
    "Scenario", "OrderGroup", "Orders", "Area", "Generators", "GeneratorCount",
    "ReferenceModeCount", "ReferenceModes", "Method", "Status",
    "SelectedK", "SilhouetteSelectedK", "Pe", "Pm", "Epsilon", "Xi",
    "MinPts", "MinSamples", "MinClusterSize", "CovarianceType", "Linkage",
    "Clusters", "NoisePoints", "AssignedPoints",
    "AssignedRatio", "Silhouette", "SelectionReason", "ObjectiveName",
    "ObjectiveValue",
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
    for sequential_index, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            mode_index = int(row["ModeIndex"])
            frequency = float(row["FrequencyHz"])
            damping = float(row["Damping"])
        except (TypeError, ValueError):
            continue

        mode_name = f"Mode {sequential_index}"
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
            "ModeIndex": sequential_index,
            "PowerFactoryModeIndex": mode_index,
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


def _selected_summary_row(metrics_path):
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None
    selected_column = next((column for column in ("Selected", "selected") if column in metrics.columns), None)
    if selected_column is None:
        return None
    selected = metrics[metrics[selected_column].astype(str).str.lower().eq("true")]
    return None if selected.empty else selected.iloc[0]


def _summary_value(row, column):
    if row is None or column not in row or pd.isna(row[column]):
        return None
    return row[column]


def _ambient_summary_base_row(scenario, order_group, orders, area_name, generators, reference_modes, method):
    return {
        "Scenario": scenario,
        "OrderGroup": order_group,
        "Orders": ", ".join(str(order) for order in orders),
        "Area": area_name,
        "Generators": ", ".join(generators),
        "GeneratorCount": len(generators),
        "ReferenceModeCount": len(reference_modes),
        "ReferenceModes": "; ".join(reference_modes),
        "Method": method,
        "Status": "ok",
    }


def _append_density_summary_row(rows, base_row, method_dir, method_name):
    metrics_name = "dbscan_metrics_summary.csv" if method_name == "DBSCAN" else "optics_metrics_summary.csv"
    selected = _selected_summary_row(method_dir / metrics_name)
    row = dict(base_row)
    if selected is None:
        row["Status"] = "missing_selection"
    else:
        for column in (
            "Pe", "Pm", "Epsilon", "Xi", "MinPts", "MinSamples", "Clusters",
            "NoisePoints", "AssignedPoints", "AssignedRatio", "Silhouette", "SelectionReason",
        ):
            row[column] = _summary_value(selected, column)
    rows.append(row)


def _append_partitioning_summary_row(rows, base_row, method_dir, method_name):
    metrics_name = "kmeans_metrics_summary.csv" if method_name == "K-Means" else "kmedoids_metrics_summary.csv"
    selected = _selected_summary_row(method_dir / metrics_name)
    row = dict(base_row)
    if selected is None:
        metrics_path = method_dir / metrics_name
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            selected_mask = metrics.get("k_selected_by_max_chord", pd.Series(False, index=metrics.index))
            selected_rows = metrics[selected_mask.astype(str).str.lower().eq("true")]
            selected = None if selected_rows.empty else selected_rows.iloc[0]
    if selected is None:
        row["Status"] = "missing_selection"
    else:
        objective = "WCSS" if method_name == "K-Means" else "Cost"
        row["SelectedK"] = _summary_value(selected, "k")
        row["Clusters"] = _summary_value(selected, "k")
        row["ObjectiveName"] = objective
        row["ObjectiveValue"] = _summary_value(selected, objective)

    silhouette_path = method_dir.parent / "silhouette" / "silhouette_optimal_k_summary.csv"
    if silhouette_path.exists():
        silhouette = pd.read_csv(silhouette_path)
        match = silhouette[silhouette["Method"].astype(str).str.lower().eq(method_name.lower())]
        if not match.empty:
            row["SilhouetteSelectedK"] = _summary_value(match.iloc[0], "k_opt")
            row["Silhouette"] = _summary_value(match.iloc[0], "Silhouette")
    rows.append(row)


def _append_fixed_method_summary_row(rows, base_row, method_dir, metrics_name, method_name):
    selected = _selected_summary_row(method_dir / metrics_name)
    row = dict(base_row)
    if selected is None:
        row["Status"] = "missing_selection"
    else:
        for column in (
            "SelectedK", "MinSamples", "MinClusterSize", "CovarianceType", "Linkage",
            "Clusters", "NoisePoints", "AssignedPoints", "AssignedRatio", "Silhouette",
            "SelectionReason",
        ):
            row[column] = _summary_value(selected, column)
        if method_name == "GMM":
            row["ObjectiveName"] = "BIC"
            row["ObjectiveValue"] = _summary_value(selected, "BIC")
    rows.append(row)


def save_ambient_clustering_selection_summary(base_output_dir, analysis_config=None):
    """Write one comparable clustering-selection row per order group, area, and method."""
    base_output_dir = Path(base_output_dir)
    if analysis_config is None:
        analysis_config = _load_json(base_output_dir / "analysis_config.json")

    scenario = str(analysis_config.get("name", "ambient"))
    all_reference_modes = dict(analysis_config.get("reference_modes") or {})
    rows = []
    for sweep in analysis_config.get("sweeps", []):
        sweep_dir = _resolve_path(sweep["output_dir"])
        sweep_config_path = sweep_dir / "analysis_config.json"
        sweep_config = _load_json(sweep_config_path) if sweep_config_path.exists() else {}
        order_group = str(sweep.get("name", sweep_config.get("order_group_name", sweep_dir.name)))
        orders = list(sweep.get("orders", sweep_config.get("n4sid_orders", [])))
        methods = list(sweep_config.get("clustering_methods", analysis_config.get("clustering_methods", [])))
        reference_modes = dict(sweep_config.get("reference_modes") or all_reference_modes)
        area_root = sweep_dir / "clustering" / "by_control_area"

        for area_name, generators in CONTROL_AREAS.items():
            area_dir = area_root / area_name
            area_reference_modes = _reference_modes_for_control_area(reference_modes, area_name)
            if "kmeans" in methods:
                _append_partitioning_summary_row(
                    rows,
                    _ambient_summary_base_row(scenario, order_group, orders, area_name, generators, area_reference_modes, "K-Means"),
                    area_dir / "kmeans",
                    "K-Means",
                )
            if "kmedoids" in methods:
                _append_partitioning_summary_row(
                    rows,
                    _ambient_summary_base_row(scenario, order_group, orders, area_name, generators, area_reference_modes, "K-Medoids"),
                    area_dir / "kmedoids",
                    "K-Medoids",
                )
            if "dbscan" in methods:
                _append_density_summary_row(
                    rows,
                    _ambient_summary_base_row(scenario, order_group, orders, area_name, generators, area_reference_modes, "DBSCAN"),
                    area_dir / "dbscan",
                    "DBSCAN",
                )
            if "optics" in methods:
                _append_density_summary_row(
                    rows,
                    _ambient_summary_base_row(scenario, order_group, orders, area_name, generators, area_reference_modes, "OPTICS"),
                    area_dir / "optics",
                    "OPTICS",
                )
            if "hdbscan" in methods:
                _append_fixed_method_summary_row(
                    rows,
                    _ambient_summary_base_row(scenario, order_group, orders, area_name, generators, area_reference_modes, "HDBSCAN"),
                    area_dir / "hdbscan",
                    "hdbscan_metrics_summary.csv",
                    "HDBSCAN",
                )
            if "gmm" in methods:
                _append_fixed_method_summary_row(
                    rows,
                    _ambient_summary_base_row(scenario, order_group, orders, area_name, generators, area_reference_modes, "GMM"),
                    area_dir / "gmm",
                    "gmm_metrics_summary.csv",
                    "GMM",
                )
            if "agglomerative" in methods:
                _append_fixed_method_summary_row(
                    rows,
                    _ambient_summary_base_row(scenario, order_group, orders, area_name, generators, area_reference_modes, "Agglomerative"),
                    area_dir / "agglomerative",
                    "agglomerative_metrics_summary.csv",
                    "Agglomerative",
                )

    summary = pd.DataFrame(rows, columns=CLUSTERING_SELECTION_SUMMARY_COLUMNS)
    summary.to_csv(base_output_dir / "clustering_selection_summary.csv", index=False)
    return summary


def _save_aggregated_paper_mad(output_dir, reference_modes, method_collectors):
    """Write one Eq. (10)-style MAD table per method across all relevant areas."""
    output_dir = Path(output_dir)
    mode_names = list(reference_modes)
    for method, assignments in method_collectors.items():
        assignment_df = pd.DataFrame(assignments)
        rows = []
        for mode in mode_names:
            distances = (
                assignment_df.loc[assignment_df["Mode"] == mode, "Distance_rad_s"]
                if not assignment_df.empty else pd.Series(dtype=float)
            )
            rows.append({
                "Mode": mode,
                "Estimates": int(len(distances)),
                "MAD": None if distances.empty else float(distances.median()),
            })
        mad_dir = output_dir / "mad" / method
        mad_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["Mode", "Estimates", "MAD"]).to_csv(mad_dir / "mad.csv", index=False)


def _raw_sample_hz(time_values):
    diffs = np.diff(np.asarray(time_values, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        raise SystemExit("Could not infer a valid sample rate from the ambient time axis.")
    return float(1.0 / np.median(diffs))


def _decimation_stages(factor):
    """
    Split a decimation factor into stages <= 13, as recommended for
    scipy.signal.decimate. E.g. 20 -> [10, 2], 100 -> [10, 10].
    """
    stages = []
    remaining = int(factor)
    while remaining > 13:
        for candidate in range(13, 1, -1):
            if remaining % candidate == 0:
                stages.append(candidate)
                remaining //= candidate
                break
        else:
            # Prime factor > 13: decimate in one large stage (still filtered).
            stages.append(remaining)
            remaining = 1
    if remaining > 1:
        stages.append(remaining)
    return stages


def _maybe_downsample(t, y, target_hz):
    """
    Anti-aliased downsampling: a zero-phase FIR low-pass at the new Nyquist
    frequency is applied BEFORE each decimation stage (via scipy.signal.decimate),
    so out-of-band content cannot fold into the modal band as spurious modes.
    """
    raw_hz = _raw_sample_hz(t)
    if target_hz is None or float(target_hz) <= 0.0 or raw_hz <= float(target_hz) * (1.0 + 1e-9):
        return t, y, raw_hz, raw_hz, 1

    factor = max(1, int(round(raw_hz / float(target_hz))))
    if factor == 1:
        return t, y, raw_hz, raw_hz, 1

    y_dec = np.asarray(y, dtype=float)
    for stage in _decimation_stages(factor):
        if len(y_dec) <= 20 * stage + 1:
            # Too short for the FIR anti-alias filter; stop decimating further.
            break
        y_dec = decimate(y_dec, stage, ftype="fir", zero_phase=True)

    achieved_factor = int(round(len(y) / max(len(y_dec), 1)))
    achieved_factor = max(1, achieved_factor)
    t_dec = t[::achieved_factor][: len(y_dec)]
    y_dec = y_dec[: len(t_dec)]
    return t_dec, y_dec, raw_hz, float(raw_hz / achieved_factor), achieved_factor


def preprocess_ambient_signal(df, column_name, time_mask_config, detrend_enabled, downsample_hz):
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


def default_block_rows_for_orders(orders):
    max_order = max(int(order) for order in orders)
    return max(BLOCK_ROWS_MIN, max_order + BLOCK_ROWS_MARGIN)


def identify_n4sid_modes(t, y, dt_s, order, block_rows=None):
    order = int(order)
    if order < 2:
        raise ValueError("N4SID order must be at least 2.")

    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    dt_s = float(dt_s)

    if block_rows is None:
        block_rows = max(order + BLOCK_ROWS_MARGIN, BLOCK_ROWS_MIN)
    i = int(block_rows)
    if order >= i:
        i = order + 2

    n_cols = y.size - (2 * i) + 1
    if n_cols < MIN_HANKEL_COLUMNS:
        raise ValueError(
            f"Ambient signal is too short for N4SID order sweep with block_rows={i}; "
            f"need at least {2 * i + MIN_HANKEL_COLUMNS - 1} samples."
        )

    hankel = np.vstack([y[idx:idx + n_cols] for idx in range(2 * i)]) / np.sqrt(n_cols)

    Q, R = np.linalg.qr(hankel.T)
    L = R.T
    Q_rows = Q.T

    L21 = L[i:, :i]
    U, singular_values, _ = np.linalg.svd(L21, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("N4SID projection is rank deficient; signal may be constant.")
    positive_rank = int(np.sum(singular_values > (singular_values[0] * 1e-12)))
    if order > positive_rank:
        raise ValueError(f"N4SID order {order} exceeds available numerical rank {positive_rank}.")

    U1 = U[:, :order]
    S1 = singular_values[:order]

    gamma = U1 * np.sqrt(S1)[None, :]
    projection = (L21 @ Q_rows[:i, :]) * np.sqrt(n_cols)
    x_hat = np.linalg.pinv(gamma) @ projection

    a_matrix = np.linalg.lstsq(gamma[:-1, :], gamma[1:, :], rcond=None)[0]
    c_matrix = gamma[:1, :]

    x_k = x_hat[:, :-1].T
    x_next = x_hat[:, 1:].T
    x_next_hat = x_k @ a_matrix
    state_rmse = float(np.sqrt(np.mean((x_next - x_next_hat) ** 2)))

    y_segment = y[i:i + x_hat.shape[1]]
    y_hat = (c_matrix @ x_hat).ravel()[:y_segment.size]
    output_rmse = float(np.sqrt(np.mean((y_segment - y_hat) ** 2)))
    output_var = float(np.var(y_segment))
    fit_percent = None
    if output_var > 0.0:
        fit_percent = float(max(0.0, 100.0 * (1.0 - (output_rmse ** 2) / output_var)))

    eigvals, eigvecs = np.linalg.eig(a_matrix)
    total_sv_energy = float(np.sum(singular_values ** 2))
    order_sv_energy = float(np.sum(S1 ** 2))
    order_sv_ratio = None if total_sv_energy <= 0.0 else float(order_sv_energy / total_sv_energy)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = eigvals.astype(complex)
        safe_z = np.where(np.abs(z) > 0.0, z, np.nan)
        poles = np.log(safe_z) / dt_s

    observability = (c_matrix @ eigvecs).ravel()

    modes = []
    mode_counter = 0
    for pole_idx, pole in enumerate(poles):
        if not np.isfinite(pole.real) or not np.isfinite(pole.imag):
            continue
        # Keep only the positive-frequency member of each conjugate pair.
        if np.imag(pole) <= 0.0:
            continue
        frequency_hz = float(np.imag(pole)) / (2.0 * np.pi)
        if frequency_hz <= FREQ_EPS_HZ:
            continue

        damping = float(np.real(pole))
        discrete_eig = eigvals[pole_idx]
        damping_ratio = None
        pole_mag = float(np.abs(pole))
        if pole_mag > 0.0:
            damping_ratio = float(-damping / pole_mag)

        mode_counter += 1
        modes.append({
            "Order": order,
            "ModeIndex": int(mode_counter),
            "Frequency": frequency_hz,
            "Damping": damping,
            "Amplitude": float(np.abs(observability[pole_idx])),
            "Phase": float(np.angle(observability[pole_idx])),
            "DampingRatio": damping_ratio,
            "DiscreteEigenvalueReal": float(np.real(discrete_eig)),
            "DiscreteEigenvalueImag": float(np.imag(discrete_eig)),
            "DiscreteEigenvalueMagnitude": float(np.abs(discrete_eig)),
            "Stable": bool(np.abs(discrete_eig) < 1.0),
            "SingularValue": None,
            "SingularValueEnergyRatio": None,
            "OrderSingularValueEnergyRatio": order_sv_ratio,
            "StatePredictionRMSE": state_rmse,
            "OutputPredictionRMSE": output_rmse,
            "OutputFitPercent": fit_percent,
        })

    summary = {
        "Order": order,
        "ModesIdentified": int(len(modes)),
        "StableModes": int(sum(1 for row in modes if row["Stable"])),
        "MeanOutputPredictionRMSE": output_rmse,
        "MeanStatePredictionRMSE": state_rmse,
        "OutputFitPercent": fit_percent,
        "OrderSingularValueEnergyRatio": order_sv_ratio,
    }
    return modes, summary


def _run_clustering_pipeline(results_path, output_path, reference_modes, methods, optics_settings=None, dbscan_settings=None, hdbscan_settings=None, gmm_settings=None, agglomerative_settings=None, paper_mad_collectors=None):
    from clustering_analysis import (
        _load_screened_data,
        run_kmeans_modal_analysis,
        run_kmedoids_modal_analysis,
        run_optics_modal_analysis,
        run_dbscan_modal_analysis,
        run_hdbscan_modal_analysis,
        run_gmm_modal_analysis,
        run_agglomerative_modal_analysis,
        run_silhouette_analysis,
    )

    requested_methods = list(methods or [])
    output_path.mkdir(parents=True, exist_ok=True)
    timings = {}
    selections = {}

    screen_start = time.perf_counter()
    df_for_mad = _load_screened_data(str(results_path), str(output_path))
    timings["screen_and_load"] = _timing_entry(time.perf_counter() - screen_start)

    runners = {
        "kmeans": run_kmeans_modal_analysis,
        "kmedoids": run_kmedoids_modal_analysis,
        "optics": run_optics_modal_analysis,
        "dbscan": run_dbscan_modal_analysis,
        "hdbscan": run_hdbscan_modal_analysis,
        "gmm": run_gmm_modal_analysis,
        "agglomerative": run_agglomerative_modal_analysis,
    }
    for method in requested_methods:
        started = time.perf_counter()
        paper_mad_collector = None if paper_mad_collectors is None else paper_mad_collectors.setdefault(method, [])
        if method == "optics":
            selection = runners[method](
                str(results_path),
                str(output_path),
                reference_modes=reference_modes,
                optics_settings=optics_settings,
                paper_mad_collector=paper_mad_collector,
            )
        elif method == "dbscan":
            selection = runners[method](
                str(results_path),
                str(output_path),
                reference_modes=reference_modes,
                dbscan_settings=dbscan_settings,
                paper_mad_collector=paper_mad_collector,
            )
        elif method == "hdbscan":
            selection = runners[method](str(results_path), str(output_path), reference_modes=reference_modes, hdbscan_settings=hdbscan_settings, paper_mad_collector=paper_mad_collector)
        elif method == "gmm":
            selection = runners[method](str(results_path), str(output_path), reference_modes=reference_modes, gmm_settings=gmm_settings, paper_mad_collector=paper_mad_collector)
        elif method == "agglomerative":
            selection = runners[method](str(results_path), str(output_path), reference_modes=reference_modes, agglomerative_settings=agglomerative_settings, paper_mad_collector=paper_mad_collector)
        else:
            selection = runners[method](str(results_path), str(output_path), reference_modes=reference_modes, paper_mad_collector=paper_mad_collector)
        timings[method] = _timing_entry(time.perf_counter() - started)
        if selection is not None:
            selections[method] = selection

    silhouette_skipped = True
    silhouette_elapsed = 0.0
    if {"kmeans", "kmedoids"}.issubset(set(requested_methods)):
        silhouette_start = time.perf_counter()
        run_silhouette_analysis(str(results_path), str(output_path), reference_modes=reference_modes)
        silhouette_elapsed = time.perf_counter() - silhouette_start
        silhouette_skipped = False
    timings["silhouette"] = _timing_entry(silhouette_elapsed, skipped=silhouette_skipped)

    timings["total"] = _timing_entry(sum(entry["seconds"] for entry in timings.values()))
    timings["selections"] = selections
    return timings


def run_ambient_clustering_for_results(output_dir, results_path, df_results, reference_modes, methods, optics_settings=None, dbscan_settings=None, hdbscan_settings=None, gmm_settings=None, agglomerative_settings=None, clustering_scope=None):
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
            dbscan_settings=dbscan_settings,
            hdbscan_settings=hdbscan_settings,
            gmm_settings=gmm_settings,
            agglomerative_settings=agglomerative_settings,
        )

    if scope.get("by_control_area", False):
        area_root = output_dir / "clustering" / "by_control_area"
        area_timings = {}
        paper_mad_collectors = {method: [] for method in methods}
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
                dbscan_settings=dbscan_settings,
                hdbscan_settings=hdbscan_settings,
                gmm_settings=gmm_settings,
                agglomerative_settings=agglomerative_settings,
                paper_mad_collectors=paper_mad_collectors,
            )

        _save_aggregated_paper_mad(output_dir, reference_modes, paper_mad_collectors)
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
    dbscan_settings = dict(AMBIENT_DEFAULT_DBSCAN_SETTINGS)
    hdbscan_settings = dict(AMBIENT_DEFAULT_HDBSCAN_SETTINGS)
    gmm_settings = dict(AMBIENT_DEFAULT_GMM_SETTINGS)
    agglomerative_settings = dict(AMBIENT_DEFAULT_AGGLOMERATIVE_SETTINGS)
    clustering_scope = _resolve_clustering_scope(getattr(args, "clustering_scope", "areas"))

    return {
        "analysis_method": "n4sid",
        "order_groups": order_groups,
        "ambient_preprocessing": {
            "detrend": not bool(args.ambient_no_detrend),
            "downsample_hz": float(args.ambient_downsample_hz) if args.ambient_downsample_hz is not None else float(AMBIENT_DEFAULT_DOWNSAMPLE_HZ),
        },
        "clustering_methods": clustering_methods,
        "clustering_scope": clustering_scope,
        "optics_settings": optics_settings,
        "dbscan_settings": dbscan_settings,
        "hdbscan_settings": hdbscan_settings,
        "gmm_settings": gmm_settings,
        "agglomerative_settings": agglomerative_settings,
        "paper_mad": dict(AMBIENT_PAPER_MAD_SETTINGS),
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
        group_block_rows = default_block_rows_for_orders(order_group["orders"])

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
                        modes, order_summary = identify_n4sid_modes(
                            t=t, y=y, dt_s=dt_s, order=order, block_rows=group_block_rows
                        )
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
                dbscan_settings=settings["dbscan_settings"],
                hdbscan_settings=settings["hdbscan_settings"],
                gmm_settings=settings["gmm_settings"],
                agglomerative_settings=settings["agglomerative_settings"],
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
            "n4sid_block_rows": int(group_block_rows),
            "pole_mapping": "log",
            "conjugate_pairs_deduplicated": True,
            "ambient_preprocessing": {
                **preprocess_cfg,
                "signals": list(signals.values()),
            },
            "clustering_methods": settings["clustering_methods"],
            "clustering_scope": dict(settings["clustering_scope"]),
            "optics_settings": settings["optics_settings"],
            "dbscan_settings": settings["dbscan_settings"],
            "hdbscan_settings": settings["hdbscan_settings"],
            "gmm_settings": settings["gmm_settings"],
            "agglomerative_settings": settings["agglomerative_settings"],
            "paper_mad": settings["paper_mad"],
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
        "pole_mapping": "log",
        "conjugate_pairs_deduplicated": True,
        "ambient_preprocessing": {
            **preprocess_cfg,
            "signals": list(signals.values()),
        },
        "clustering_methods": settings["clustering_methods"],
        "clustering_scope": dict(settings["clustering_scope"]),
        "optics_settings": settings["optics_settings"],
        "dbscan_settings": settings["dbscan_settings"],
        "hdbscan_settings": settings["hdbscan_settings"],
        "gmm_settings": settings["gmm_settings"],
        "agglomerative_settings": settings["agglomerative_settings"],
        "paper_mad": settings["paper_mad"],
        "reference_modes_source": settings["reference_modes_source"],
        "reference_modes": settings["reference_modes"],
        "sweeps": sweep_summaries,
        "timings": {
            "n4sid": _timing_entry(time.perf_counter() - analysis_start),
        },
    }
    _save_json(base_output_dir / "analysis_config.json", analysis_config)
    save_ambient_clustering_selection_summary(base_output_dir, analysis_config)
    return base_output_dir, None, pd.DataFrame(), analysis_config


def load_existing_ambient_results_for_scenario(name, scenario):
    base_output_dir = _resolve_path(scenario["output_dir"])
    config_path = base_output_dir / "analysis_config.json"
    if not config_path.exists():
        raise SystemExit(
            f"Cannot skip N4SID for '{name}' because ambient analysis_config.json does not exist: {config_path}"
        )

    analysis_config = _load_json(config_path)
    reference_source, reference_modes = _load_reference_modes(scenario["data_dir"])
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
    analysis_config["reference_modes_source"] = reference_source
    analysis_config["reference_modes"] = reference_modes
    _save_json(config_path, analysis_config)

    for sweep in sweeps:
        sweep_config_path = _resolve_path(sweep["output_dir"]) / "analysis_config.json"
        if not sweep_config_path.exists():
            continue
        sweep_config = _load_json(sweep_config_path)
        sweep_config["reference_modes_source"] = reference_source
        sweep_config["reference_modes"] = reference_modes
        _save_json(sweep_config_path, sweep_config)

    return base_output_dir, None, pd.DataFrame(), analysis_config
