import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run IEEE39 Matrix Pencil and clustering analysis.")
    parser.add_argument("--scenario", nargs="+", default=["all"], help="Scenario keys, existing results folder names, or 'all'.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print available scenario keys and exit.")
    parser.add_argument("--list-analysis", action="store_true", help="Print existing IEEE39 analysis folders and exit.")
    parser.add_argument("--skip-clustering", action="store_true", help="Only run Matrix Pencil extraction.")
    parser.add_argument("--clustering-scope", choices=["both", "global", "areas", "none"], default="both", help="Choose clustering output scope. Default: both.")
    parser.add_argument("--skip-matrix-pencil", action="store_true", help="Reuse an existing results.csv instead of recomputing Matrix Pencil poles.")
    parser.add_argument("--results-file", default=None, help="Existing pole results CSV to use with --skip-matrix-pencil. Default: output_dir/results.csv.")
    parser.add_argument("--analysis-dir", default=None, help="Existing analysis directory to use with --skip-matrix-pencil.")
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate IEEE39 modal maps and reconstruction plots.")
    parser.add_argument("--data-dir", default=None, help="Data directory relative to IEEE39, or an absolute path. Use with one scenario.")
    parser.add_argument("--output-dir", default=None, help="Output directory relative to IEEE39, or an absolute path. Use with one scenario.")
    parser.add_argument("--time-start", type=float, default=None, help="Inclusive analysis start time in seconds. Default: 0.2.")
    parser.add_argument("--time-end", type=float, default=None, help="Inclusive analysis end time in seconds. Default: last CSV timestamp.")
    parser.add_argument("--no-reset-time", action="store_true", help="Do not shift the selected time window to start at zero.")
    return parser


def parse_args():
    return build_arg_parser().parse_args()


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parse_args()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
PRELIM_DIR = REPO_DIR / "PreliminaryInvestigation"

MATRIX_PENCIL_PATH = PRELIM_DIR / "matrix_pencil.py"


def path_for_metadata(path):
    try:
        return Path(path).relative_to(BASE_DIR).as_posix()
    except ValueError:
        return Path(path).name

if not PRELIM_DIR.exists():
    raise RuntimeError(f"PreliminaryInvestigation folder not found: {PRELIM_DIR}")
if not MATRIX_PENCIL_PATH.exists():
    raise RuntimeError(f"matrix_pencil.py not found: {MATRIX_PENCIL_PATH}")

prelim_path = str(PRELIM_DIR)
if prelim_path not in sys.path:
    sys.path.insert(0, prelim_path)

matrix_pencil_spec = importlib.util.spec_from_file_location("preliminary_matrix_pencil", MATRIX_PENCIL_PATH)
if matrix_pencil_spec is None or matrix_pencil_spec.loader is None:
    raise RuntimeError(f"Could not load matrix_pencil.py from: {MATRIX_PENCIL_PATH}")
matrix_pencil = importlib.util.module_from_spec(matrix_pencil_spec)
matrix_pencil_spec.loader.exec_module(matrix_pencil)

apply_matrix_pencil_fixed_order = matrix_pencil.apply_matrix_pencil_fixed_order
apply_matrix_pencil_fixed_order_prepared = matrix_pencil.apply_matrix_pencil_fixed_order_prepared
determine_MP_order = matrix_pencil.determine_MP_order
filter_signal = matrix_pencil.filter_signal
prepare_matrix_pencil = matrix_pencil.prepare_matrix_pencil

from plot_style import (
    apply_thesis_style,
    style_axis,
    SIGNAL_COLORS,
)

apply_thesis_style()


COLUMNS = {
    "s:ut in p.u.": "Voltage",
    "s:cur1 in p.u.": "Current",
    "s:P1 in MW": "Active Power",
    "s:Q1 in Mvar": "Reactive Power",
}

IEEE39_GENERATORS = [f"g{i}" for i in range(1, 11)]
AUTO_ORDER_DECIMATION = 10
DEFAULT_TIME_START_S = 0.2
RECON_X_LIMS = (0, 50)
RECON_TICK_LABEL_SIZE = 30
RECON_AXIS_LABEL_SIZE = 34
METHOD_ORDER = ["Order 2", "Order 4", "Order 6", "Tau 1", "Tau 0.1", "Tau 0.01"]
RECONSTRUCTION_ROWS = [("Order 2", "Tau 1"), ("Order 4", "Tau 0.1"), ("Order 6", "Tau 0.01")]
SIGNAL_LABELS = {
    "Voltage": r"$\Delta V$ [p.u.]",
    "Current": r"$\Delta \mathrm{I}$ [p.u.]",
    "Active Power": r"$\Delta P$ [MW]",
    "Reactive Power": r"$\Delta Q$ [Mvar]",
}

CONTROL_AREAS = {
    "area_1": ["g1", "g8", "g9", "g10"],
    "area_2": ["g2", "g3"],
    "area_3": ["g4", "g5", "g6", "g7"],
}

IEEE39_REFERENCE_MODES = {
    "Mode 1": {"Frequency": 0.6062, "Damping": -0.0800, "Damping_Factor": 0.0210, "Generator_Involvement": "1-9 vs. 10", "DRGA_Peak_Value": 17.8},
    "Mode 2": {"Frequency": 0.9497, "Damping": -0.1065, "Damping_Factor": 0.0178, "Generator_Involvement": "1,8 and 9 vs. 4,5,6 and 7", "DRGA_Peak_Value": 4.3},
    "Mode 3": {"Frequency": 1.0312, "Damping": -0.2558, "Damping_Factor": 0.0395, "Generator_Involvement": "2 and 3 vs. 4 and 5", "DRGA_Peak_Value": 2.3},
    "Mode 4": {"Frequency": 1.1211, "Damping": -0.3373, "Damping_Factor": 0.0478, "Generator_Involvement": "2 and 3 vs. 6 and 7", "DRGA_Peak_Value": 0.8},
    "Mode 5": {"Frequency": 1.3155, "Damping": -0.4033, "Damping_Factor": 0.0487, "Generator_Involvement": "2 vs. 3", "DRGA_Peak_Value": 2.6},
    "Mode 6": {"Frequency": 1.2851, "Damping": -0.3458, "Damping_Factor": 0.0428, "Generator_Involvement": "1 vs. 8 and 9", "DRGA_Peak_Value": 3.0},
    "Mode 7": {"Frequency": 1.4953, "Damping": -0.7033, "Damping_Factor": 0.0747, "Generator_Involvement": "4 vs. 5", "DRGA_Peak_Value": None},
    "Mode 8": {"Frequency": 1.5202, "Damping": -0.6010, "Damping_Factor": 0.0628, "Generator_Involvement": "5 and 7 vs. 4 and 6", "DRGA_Peak_Value": None},
    "Mode 9": {"Frequency": 1.5468, "Damping": -0.6376, "Damping_Factor": 0.0655, "Generator_Involvement": "1 vs. 8", "DRGA_Peak_Value": None},
}

DEFAULT_SCENARIOS = {
    "load29": {
        "data_dir": "results/Load29_Pplus2_50s",
        "output_dir": "analysis/Load29_Pplus2_50s",
        "time_mask": {"start_inclusive": DEFAULT_TIME_START_S, "reset_time": True},
        "generators": IEEE39_GENERATORS,
        "columns": COLUMNS,
        "fixed_orders": [2, 4, 6],
        "taus": [1, 0.1, 0.01],
        "auto_order_decimation": AUTO_ORDER_DECIMATION,
        "filter": {"fc": 10, "N": 15},
        "clustering": {"global": True, "by_control_area": True},
    },
    "load03": {
        "data_dir": "results/Load03_Pplus2_50s",
        "output_dir": "analysis/Load03_Pplus2_50s",
        "time_mask": {"start_inclusive": DEFAULT_TIME_START_S, "reset_time": True},
        "generators": IEEE39_GENERATORS,
        "columns": COLUMNS,
        "fixed_orders": [2, 4, 6],
        "taus": [1, 0.1, 0.01],
        "auto_order_decimation": AUTO_ORDER_DECIMATION,
        "filter": {"fc": 10, "N": 15},
        "clustering": {"global": True, "by_control_area": True},
    },
    "load24": {
        "data_dir": "results/Load24_Pplus2_50s",
        "output_dir": "analysis/Load24_Pplus2_50s",
        "time_mask": {"start_inclusive": DEFAULT_TIME_START_S, "reset_time": True},
        "generators": IEEE39_GENERATORS,
        "columns": COLUMNS,
        "fixed_orders": [2, 4, 6],
        "taus": [1, 0.1, 0.01],
        "auto_order_decimation": AUTO_ORDER_DECIMATION,
        "filter": {"fc": 10, "N": 15},
        "clustering": {"global": True, "by_control_area": True},
    },
}


def _resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _format_time_value(value):
    return f"{float(value):g}".replace("-", "m")


def _time_mask_suffix(time_mask):
    time_mask = time_mask or {}
    start = time_mask.get("start_inclusive", time_mask.get("start"))
    end = time_mask.get("end_inclusive", time_mask.get("end"))
    reset = time_mask.get("reset_time", True)

    start_part = _format_time_value(start) if start is not None else "start"
    end_part = _format_time_value(end) if end is not None else "end"
    reset_part = "reset" if reset else "noreset"

    return f"{start_part}_to_{end_part}_{reset_part}"


def _analysis_output_dir(scenario):
    output_dir = _resolve_path(scenario.get("output_dir", "analysis"))
    if scenario.get("output_dir_explicit"):
        return output_dir

    return output_dir.parent / f"{output_dir.name}_{_time_mask_suffix(scenario.get('time_mask'))}"


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

    if "start" in mask_config and mask_config["start"] is not None:
        mask &= time_values > float(mask_config["start"])
    if "start_inclusive" in mask_config and mask_config["start_inclusive"] is not None:
        mask &= time_values >= float(mask_config["start_inclusive"])
    if "end" in mask_config and mask_config["end"] is not None:
        mask &= time_values < float(mask_config["end"])
    if "end_inclusive" in mask_config and mask_config["end_inclusive"] is not None:
        mask &= time_values <= float(mask_config["end_inclusive"])

    return mask


def _time_window_description(time_values, mask_config):
    mask_config = mask_config or {}
    finite_time = time_values[np.isfinite(time_values)]
    if finite_time.size == 0:
        return None

    start = mask_config.get("start_inclusive", mask_config.get("start"))
    end = mask_config.get("end_inclusive", mask_config.get("end"))

    if start is None:
        start = float(np.min(finite_time))
    if end is None:
        end = float(np.max(finite_time))

    return {"start_s": float(start), "end_s": float(end)}


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_scenario_json(data_dir):
    scenario_json = data_dir / "scenario.json"
    if not scenario_json.exists():
        return None

    return _load_json(scenario_json)


def _scenario_generators_from_json(config):
    if not config or not config.get("csv_files"):
        return None

    generators = []
    for csv_info in config["csv_files"]:
        file_name = Path(str(csv_info["file"]).replace("\\", "/")).stem
        generators.append(file_name)

    return generators or None


def _scenario_runtime_config(scenario):
    data_dir = _resolve_path(scenario["data_dir"])
    output_dir = _analysis_output_dir(scenario)
    generated_config = _load_scenario_json(data_dir)
    generators = _scenario_generators_from_json(generated_config) or scenario.get("generators", IEEE39_GENERATORS)
    columns = scenario.get("columns", COLUMNS)

    return data_dir, output_dir, generated_config, generators, columns


def _preprocess_signal(df, column_name, scenario):
    time_all = df.iloc[:, 0].to_numpy(dtype=float)
    signal_all = df[column_name].to_numpy(dtype=float)
    mask = _time_mask(time_all, scenario.get("time_mask"))

    if not np.any(mask):
        return None, None

    t = time_all[mask].copy()
    y = signal_all[mask].copy()
    if scenario.get("time_mask", {}).get("reset_time", True):
        t = t - t[0]

    valid = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(valid) < 4:
        return None, None

    t = t[valid]
    y = y[valid]
    y = detrend(y)
    filter_config = scenario.get("filter", {"fc": 10, "N": 15})
    y = filter_signal(y, t, fc=float(filter_config.get("fc", 10)), N=int(filter_config.get("N", 15)))
    y = y - np.mean(y)

    return t, y


def _r2_score(y_true, y_pred):
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total == 0:
        return np.nan
    return 1.0 - residual / total


def _save_current_figure(path_base, filename):
    for subdir, extension in [("png", "png"), ("pdf", "pdf")]:
        out_dir = path_base / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"{filename}.{extension}", dpi=300 if extension == "png" else None, bbox_inches="tight")


def _reconstruct_signal(t, modes):
    y_est = np.zeros_like(t)
    for _, mode in modes.iterrows():
        y_est += 2 * mode["Amplitude"] * np.exp(mode["Damping"] * t) * np.cos(
            2 * np.pi * mode["Frequency"] * t + mode["Phase"]
        )
    return y_est


def generate_ieee39_comprehensive_report(df_results, scenario):
    data_dir, output_dir, _, generators, columns = _scenario_runtime_config(scenario)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    report_columns = ["Gen", "Signal", "Method", "R2", "RMSE", "Poles"]

    metrics = []
    if df_results.empty:
        report = pd.DataFrame(metrics, columns=report_columns)
        report.to_csv(stats_dir / "comprehensive_report.csv", index=False)
        return report

    inv_columns = {label: csv_col for csv_col, label in columns.items()}

    for gen in generators:
        csv_path = data_dir / f"{gen}.csv"
        if not csv_path.exists():
            continue

        df = _read_numeric_csv(csv_path)
        for signal in columns.values():
            source_col = inv_columns[signal]
            if source_col not in df.columns:
                continue

            t, y_ref = _preprocess_signal(df, source_col, scenario)
            if t is None or y_ref is None:
                continue

            for method in METHOD_ORDER:
                modes = df_results[
                    (df_results["Gen"] == gen)
                    & (df_results["Signal"] == signal)
                    & (df_results["Method"] == method)
                ]
                if modes.empty:
                    continue

                y_est = _reconstruct_signal(t, modes)
                rmse = float(np.sqrt(np.mean((y_ref - y_est) ** 2)))
                r2 = float(_r2_score(y_ref, y_est))
                metrics.append({
                    "Gen": gen,
                    "Signal": signal,
                    "Method": method,
                    "R2": r2,
                    "RMSE": rmse,
                    "Poles": int(len(modes)),
                })

    report = pd.DataFrame(metrics, columns=report_columns)
    report.to_csv(stats_dir / "comprehensive_report.csv", index=False)
    return report


def generate_ieee39_plots(df_results, scenario):
    if df_results.empty:
        print("No Matrix Pencil results available; skipping IEEE39 plots.")
        return

    data_dir, output_dir, _, generators, columns = _scenario_runtime_config(scenario)
    plots_dir = output_dir / "plots"
    modal_maps_dir = plots_dir / "modal_maps"
    recon_dir = plots_dir / "reconstruction_grids"

    for gen in generators:
        gen_data = df_results[df_results["Gen"] == gen]
        if gen_data.empty:
            continue

        plt.figure(figsize=(10, 6))
        for signal in columns.values():
            signal_data = gen_data[gen_data["Signal"] == signal]
            if signal_data.empty:
                continue
            plt.scatter(
                signal_data["Damping"],
                signal_data["Frequency"],
                s=60,
                alpha=0.6,
                label=signal,
                c=SIGNAL_COLORS.get(signal),
                edgecolors="k",
            )
        plt.axvline(0, color="red", linestyle="-", alpha=0.3)
        plt.title(f"Combined Modal Map: Generator {gen.upper()}")
        plt.xlabel("Damping (Sigma) [rad/s]")
        plt.ylabel("Frequency [Hz]")
        plt.legend()
        style_axis(plt.gca())
        _save_current_figure(modal_maps_dir, f"{gen}_combined_modal_map")
        plt.close()

    plt.figure(figsize=(11, 7))
    for signal in columns.values():
        signal_data = df_results[df_results["Signal"] == signal]
        if signal_data.empty:
            continue
        plt.scatter(
            signal_data["Damping"],
            signal_data["Frequency"],
            s=60,
            alpha=0.6,
            label=signal,
            c=SIGNAL_COLORS.get(signal),
            edgecolors="k",
        )
    plt.axvline(0, color="red", linestyle="-", alpha=0.3)
    plt.title("IEEE39 System-Wide Modal Map")
    plt.xlabel("Damping (Sigma) [rad/s]")
    plt.ylabel("Frequency [Hz]")
    style_axis(plt.gca())
    plt.legend()
    _save_current_figure(modal_maps_dir, "system_modal_map")
    plt.close()

    inv_columns = {label: csv_col for csv_col, label in columns.items()}
    for gen in generators:
        csv_path = data_dir / f"{gen}.csv"
        if not csv_path.exists():
            continue

        df = _read_numeric_csv(csv_path)
        for signal in columns.values():
            source_col = inv_columns[signal]
            if source_col not in df.columns:
                continue

            t, y_ref = _preprocess_signal(df, source_col, scenario)
            if t is None or y_ref is None:
                continue

            fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
            fig.suptitle(
                f"Reconstruction Accuracy: {gen.upper()} - {signal}\nLeft: Fixed Orders | Right: Adaptive Tau",
                fontweight="bold",
                y=0.98,
            )

            for row_idx, (order_method, tau_method) in enumerate(RECONSTRUCTION_ROWS):
                for col_idx, method in enumerate([order_method, tau_method]):
                    ax = axes[row_idx, col_idx]
                    ax.set_xlim(*RECON_X_LIMS)
                    ax.tick_params(axis="both", labelsize=RECON_TICK_LABEL_SIZE)

                    modes = df_results[
                        (df_results["Gen"] == gen)
                        & (df_results["Signal"] == signal)
                        & (df_results["Method"] == method)
                    ]

                    if modes.empty:
                        ax.text(0.5, 0.5, "No Data Found", ha="center")
                        continue

                    y_est = _reconstruct_signal(t, modes)
                    rmse = float(np.sqrt(np.mean((y_ref - y_est) ** 2)))
                    r2 = float(_r2_score(y_ref, y_est))

                    ax.plot(t, y_ref, color="black", alpha=0.3, linewidth=2, label="Original (Filtered)")
                    ax.plot(t, y_est, "--", color="red", linewidth=1.5, label=f"MP Estimate ($R^2$={r2:.4f})")
                    ax.set_title(f"Method: {method} (RMSE: {rmse:.2e})", fontweight="semibold")
                    ax.legend(loc="upper right")
                    ax.grid(True, linestyle=":", alpha=0.75, linewidth=1.3, color="gray")

                    if col_idx == 0:
                        ax.set_ylabel(SIGNAL_LABELS.get(signal, signal), fontsize=RECON_AXIS_LABEL_SIZE)
                    if row_idx == 2:
                        ax.set_xlabel("Time (s)", fontsize=RECON_AXIS_LABEL_SIZE)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            _save_current_figure(recon_dir, f"{gen}_{signal.replace(' ', '_')}_reconstruction")
            plt.close(fig)


def run_matrix_pencil_for_scenario(name, scenario):
    data_dir, output_dir, generated_config, generators, columns = _scenario_runtime_config(scenario)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_orders = scenario.get("fixed_orders", [2, 4, 6])
    taus = scenario.get("taus", [1, 0.1, 0.01])
    auto_order_decimation = int(
        scenario.get("auto_order_decimation", scenario.get("order_rate", AUTO_ORDER_DECIMATION))
    )
    filter_config = scenario.get("filter", {"fc": 10, "N": 15})

    results = []
    stats_lines = []
    time_window = None

    for gen in generators:
        csv_path = data_dir / f"{gen}.csv"
        if not csv_path.exists():
            print(f"File missing: {csv_path}")
            continue

        print(f"Generator: {gen}", flush=True)
        df = _read_numeric_csv(csv_path)
        time_all = df.iloc[:, 0].to_numpy(dtype=float)
        if time_window is None:
            time_window = _time_window_description(time_all, scenario.get("time_mask"))
        mask = _time_mask(time_all, scenario.get("time_mask"))

        if not np.any(mask):
            print(f"No samples left after time mask for {gen}")
            continue

        time_col = time_all[mask].copy()
        if scenario.get("time_mask", {}).get("reset_time", True):
            time_col = time_col - time_col[0]

        for col, signal in columns.items():
            if col not in df.columns:
                print(f"Column {col} missing in {gen}")
                continue

            print(f"Gen: {gen}, Signal: {signal}", flush=True)
            signal_col = df[col].to_numpy(dtype=float)[mask].copy()
            valid = np.isfinite(time_col) & np.isfinite(signal_col)
            if np.count_nonzero(valid) < 4:
                print(f"Not enough finite samples for {gen} {signal}")
                continue

            t = time_col[valid]
            y = signal_col[valid]
            y = detrend(y)
            mean_after_detrend = float(np.mean(y))
            y = filter_signal(y, t, fc=float(filter_config.get("fc", 10)), N=int(filter_config.get("N", 15)))
            mean_after_lpf = float(np.mean(y))
            y = y - np.mean(y)
            mean_after_demean = float(np.mean(y))
            prepared_mp = prepare_matrix_pencil(y, t)

            stats_lines.append({
                "Scenario": name,
                "Gen": gen,
                "Signal": signal,
                "Mean after detrend": mean_after_detrend,
                "Mean after LPF": mean_after_lpf,
                "Mean after demean": mean_after_demean,
            })

            for order in fixed_orders:
                freq, sigma, _, _, _, amplitudes = apply_matrix_pencil_fixed_order_prepared(prepared_mp, order=order)
                for f, s, amplitude in zip(freq, sigma, amplitudes):
                    if f > 0:
                        results.append({
                            "Scenario": name,
                            "Gen": gen,
                            "Signal": signal,
                            "Method": f"Order {order}",
                            "Frequency": float(f),
                            "Damping": float(s),
                            "Amplitude": float(np.abs(amplitude)),
                            "Phase": float(np.angle(amplitude)),
                        })

            for tau in taus:
                order = determine_MP_order(t, y, tau, rate=auto_order_decimation)
                freq, sigma, _, _, _, amplitudes = apply_matrix_pencil_fixed_order_prepared(prepared_mp, order=order)
                for f, s, amplitude in zip(freq, sigma, amplitudes):
                    if f > 0:
                        results.append({
                            "Scenario": name,
                            "Gen": gen,
                            "Signal": signal,
                            "Method": f"Tau {tau}",
                            "Frequency": float(f),
                            "Damping": float(s),
                            "Amplitude": float(np.abs(amplitude)),
                            "Phase": float(np.angle(amplitude)),
                        })

    df_results = pd.DataFrame(results)
    results_path = output_dir / "results.csv"
    df_results.to_csv(results_path, index=False)
    metadata_scenario = dict(scenario)
    for key in ("data_dir", "output_dir"):
        if metadata_scenario.get(key):
            metadata_scenario[key] = path_for_metadata(_resolve_path(metadata_scenario[key]))
    _save_json(output_dir / "analysis_config.json", {
        "name": name,
        **metadata_scenario,
        "data_scenario_json": path_for_metadata(data_dir / "scenario.json") if generated_config else None,
        "generators_used": generators,
        "auto_order_decimation": auto_order_decimation,
        "time_window_s": time_window,
        "time_reset_to_zero": scenario.get("time_mask", {}).get("reset_time", True),
        "signal_means": stats_lines,
    })

    return output_dir, results_path, df_results


def run_clustering_for_scenario(output_dir, results_path, df_results, scenario):
    from clustering_analysis import (
        _load_screened_data,
        _save_reference_mad_outputs,
        run_kmeans_modal_analysis,
        run_kmedoids_modal_analysis,
        run_silhouette_analysis,
    )

    clustering_config = scenario.get("clustering", {})
    if df_results.empty:
        print(f"No Matrix Pencil results for {output_dir}; skipping clustering.")
        return

    if clustering_config.get("global", True):
        global_out = output_dir / "clustering" / "global"
        df_for_mad = _load_screened_data(str(results_path), str(global_out))
        if df_for_mad is not None:
            _save_reference_mad_outputs(df_for_mad, str(global_out), reference_modes=IEEE39_REFERENCE_MODES)
        run_kmeans_modal_analysis(str(results_path), str(global_out))
        run_kmedoids_modal_analysis(str(results_path), str(global_out))
        run_silhouette_analysis(str(results_path), str(global_out))

    if clustering_config.get("by_control_area", True):
        area_root = output_dir / "clustering" / "by_control_area"
        for area_name, gens in CONTROL_AREAS.items():
            area_out = area_root / area_name
            area_out.mkdir(parents=True, exist_ok=True)
            area_df = df_results[df_results["Gen"].isin(gens)].copy()
            if area_df.empty:
                print(f"No data for {area_name}; skipping.")
                continue

            area_results_path = area_out / "results.csv"
            area_df.to_csv(area_results_path, index=False)
            _save_json(area_out / "control_area.json", {"name": area_name, "generators": gens})

            df_for_mad = _load_screened_data(str(area_results_path), str(area_out))
            if df_for_mad is not None:
                _save_reference_mad_outputs(df_for_mad, str(area_out), reference_modes=IEEE39_REFERENCE_MODES)
            run_kmeans_modal_analysis(str(area_results_path), str(area_out))
            run_kmedoids_modal_analysis(str(area_results_path), str(area_out))
            run_silhouette_analysis(str(area_results_path), str(area_out))


def apply_existing_analysis_config(scenario, results_path, args):
    config_path = results_path.parent / "analysis_config.json"
    if not config_path.exists():
        return None

    config = _load_json(config_path)

    if not args.data_dir and config.get("data_dir"):
        scenario["data_dir"] = config["data_dir"]

    if args.time_start is None and args.time_end is None and not args.no_reset_time and config.get("time_mask"):
        scenario["time_mask"] = config["time_mask"]

    for key in ["filter", "columns", "fixed_orders", "taus", "auto_order_decimation"]:
        if key in config:
            scenario[key] = config[key]

    return config


def load_existing_results_for_scenario(name, scenario, results_file, args):
    if args.analysis_dir:
        output_dir = _resolve_path(args.analysis_dir)
        scenario["output_dir"] = str(output_dir)
        scenario["output_dir_explicit"] = True
    else:
        _, output_dir, _, _, _ = _scenario_runtime_config(scenario)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = _resolve_path(results_file) if results_file else output_dir / "results.csv"

    if not results_path.exists():
        raise SystemExit(
            f"Cannot skip Matrix Pencil for '{name}' because results file does not exist: {results_path}"
        )

    if results_file and not args.output_dir:
        output_dir = results_path.parent
        scenario["output_dir"] = str(output_dir)
        scenario["output_dir_explicit"] = True

    config = apply_existing_analysis_config(scenario, results_path, args)

    if config is None:
        print(
            f"Warning: no analysis_config.json found next to {results_path}; "
            "using the current CLI/default time mask for reports and reconstructions.",
            flush=True,
        )

    df_results = pd.read_csv(results_path)
    return output_dir, results_path, df_results


def list_analysis_folders():
    analysis_root = BASE_DIR / "analysis"
    if not analysis_root.exists():
        print(f"No analysis folder found: {analysis_root}")
        return

    for folder in sorted(path for path in analysis_root.iterdir() if path.is_dir()):
        results_path = folder / "results.csv"
        config_path = folder / "analysis_config.json"
        details = []

        if results_path.exists():
            details.append("results.csv")
        if config_path.exists():
            config = _load_json(config_path)
            time_mask = config.get("time_mask", {})
            time_window = config.get("time_window_s", {})
            reset = time_mask.get("reset_time", config.get("time_reset_to_zero"))
            start = time_mask.get("start_inclusive", time_window.get("start_s"))
            end = time_mask.get("end_inclusive", time_window.get("end_s", "end"))
            details.append(f"time_start={start}")
            details.append(f"time_end={end}")
            details.append(f"reset={reset}")

        suffix = f" ({', '.join(details)})" if details else ""
        print(f"{folder.name}{suffix}")


def _scenario_from_results_folder(folder_name):
    data_dir = _resolve_path(f"results/{folder_name}")
    if not data_dir.exists():
        return None

    return {
        "data_dir": f"results/{folder_name}",
        "output_dir": f"analysis/{folder_name}",
        "time_mask": {"start_inclusive": DEFAULT_TIME_START_S, "reset_time": True},
        "generators": IEEE39_GENERATORS,
        "columns": COLUMNS,
        "fixed_orders": [2, 4, 6],
        "taus": [1, 0.1, 0.01],
        "auto_order_decimation": AUTO_ORDER_DECIMATION,
        "filter": {"fc": 10, "N": 15},
        "clustering": {"global": True, "by_control_area": True},
    }


def select_scenarios(names, allow_custom=False):
    if not names or names == ["all"]:
        return {name: dict(scenario) for name, scenario in DEFAULT_SCENARIOS.items()}

    selected = {}
    for name in names:
        if name in DEFAULT_SCENARIOS:
            selected[name] = dict(DEFAULT_SCENARIOS[name])
            continue

        folder_scenario = _scenario_from_results_folder(name)
        if folder_scenario is not None:
            selected[name] = folder_scenario
            continue

        if allow_custom:
            selected[name] = _scenario_from_results_folder(name) or {
                "data_dir": f"results/{name}",
                "output_dir": f"analysis/{name}",
                "time_mask": {"start_inclusive": DEFAULT_TIME_START_S, "reset_time": True},
                "generators": IEEE39_GENERATORS,
                "columns": COLUMNS,
                "fixed_orders": [2, 4, 6],
                "taus": [1, 0.1, 0.01],
                "auto_order_decimation": AUTO_ORDER_DECIMATION,
                "filter": {"fc": 10, "N": 15},
                "clustering": {"global": True, "by_control_area": True},
            }
            continue

        if name not in DEFAULT_SCENARIOS:
            available = ", ".join(DEFAULT_SCENARIOS.keys())
            raise SystemExit(
                f"Unknown scenario '{name}'. Available defaults: {available}. "
                f"You can also pass an existing IEEE39/results folder name."
            )
    return selected


def apply_cli_overrides(selected, args):
    if args.data_dir and len(selected) != 1:
        raise SystemExit("--data-dir can only be used with exactly one --scenario.")
    if args.output_dir and len(selected) != 1:
        raise SystemExit("--output-dir can only be used with exactly one --scenario.")
    if args.analysis_dir and len(selected) != 1:
        raise SystemExit("--analysis-dir can only be used with exactly one --scenario.")

    for scenario in selected.values():
        if args.data_dir:
            scenario["data_dir"] = args.data_dir
        if args.output_dir:
            scenario["output_dir"] = args.output_dir
            scenario["output_dir_explicit"] = True

        time_mask = dict(scenario.get("time_mask", {}))
        if args.time_start is not None:
            time_mask.pop("start", None)
            time_mask["start_inclusive"] = args.time_start
        if args.time_end is not None:
            time_mask.pop("end", None)
            time_mask["end_inclusive"] = args.time_end
        time_mask["reset_time"] = not args.no_reset_time
        scenario["time_mask"] = time_mask

        if args.clustering_scope == "none" or args.skip_clustering:
            scenario["clustering"] = {"global": False, "by_control_area": False}
        elif args.clustering_scope == "global":
            scenario["clustering"] = {"global": True, "by_control_area": False}
        elif args.clustering_scope == "areas":
            scenario["clustering"] = {"global": False, "by_control_area": True}
        elif args.clustering_scope == "both":
            scenario["clustering"] = {"global": True, "by_control_area": True}


def main():
    args = parse_args()
    if args.list_scenarios:
        for name, scenario in DEFAULT_SCENARIOS.items():
            print(f"{name}: {scenario['data_dir']}")
        return
    if args.list_analysis:
        list_analysis_folders()
        return

    start = time.time()
    selected = select_scenarios(args.scenario, allow_custom=bool(args.data_dir))
    apply_cli_overrides(selected, args)

    for name, scenario in selected.items():
        print("=" * 80, flush=True)
        print(f"Analyzing scenario: {name}", flush=True)
        scenario_start = time.time()

        if args.skip_matrix_pencil:
            output_dir, results_path, df_results = load_existing_results_for_scenario(name, scenario, args.results_file, args)
        else:
            output_dir, results_path, df_results = run_matrix_pencil_for_scenario(name, scenario)

        generate_ieee39_comprehensive_report(df_results, scenario)

        if not args.skip_plots:
            generate_ieee39_plots(df_results, scenario)

        if scenario.get("clustering", {}).get("global", False) or scenario.get("clustering", {}).get("by_control_area", False):
            run_clustering_for_scenario(output_dir, results_path, df_results, scenario)

        scenario_elapsed = time.time() - scenario_start
        print(
            f"Scenario {name} finished in "
            f"{scenario_elapsed // 60:.0f} minutes and {scenario_elapsed % 60:.1f} seconds",
            flush=True,
        )

    elapsed = time.time() - start
    print("-" * 30, f"Execution Time: {elapsed // 60:.0f} minutes and {elapsed % 60:.1f} seconds", "-" * 30)


if __name__ == "__main__":
    main()
