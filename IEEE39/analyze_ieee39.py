import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from textwrap import dedent


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run IEEE39 Matrix Pencil analysis on generated CSV results.\n\n"
            "The main selector is --scenario, which can point either to a preset alias, an existing results folder,\n"
            "or a custom run label when used together with --data-dir."
        ),
        epilog=dedent(
            """
            Scenario input forms:
              1. Preset alias: load29
              2. Multiple preset aliases: load03 load24
              3. All presets: all
              4. Existing results folder name: Load29_Pplus2_50s
              5. Custom label with explicit data path: --scenario load20_custom --data-dir results/Load20_Pplus2_50s

            Examples:
              python IEEE39/analyze_ieee39.py --scenario load29
              python IEEE39/analyze_ieee39.py --scenario load03 load24
              python IEEE39/analyze_ieee39.py --scenario all
              python IEEE39/analyze_ieee39.py --scenario Load29_Pplus2_50s
              python IEEE39/analyze_ieee39.py --scenario load20_custom --data-dir results/Load20_Pplus2_50s --output-dir analysis/Load20_Pplus2_50s
              python IEEE39/analyze_ieee39.py --scenario load29 --time-cross global --time-cross-reference g2:Current --plots
              python IEEE39/analyze_ieee39.py --scenario load29 --fixed-orders 2 4 6 8 --taus 1 0.1 0.01
              python IEEE39/analyze_ieee39.py --scenario Load29_Pplus2_50s --skip-matrix-pencil --analysis-dir analysis/Load29_Pplus2_50s_0_to_end_reset
              python IEEE39/analyze_ieee39.py --scenario ambient_seed1997 --data-dir results/Ambient_Mag0.1_T600s_dt10ms_seed1997 --output-dir analysis/ambient_seed1997 --analysis-method n4sid --n4sid-orders 10 20 30 40 50 --ambient-downsample-hz 5 --ambient-lpf-hz 2 --clustering --clustering-methods kmeans kmedoids optics

            Notes:
              - --scenario is required for actual analysis runs; the script no longer defaults silently to 'all'.
              - If --data-dir is used, --scenario becomes just a label for the run.
              - --analysis-method auto dispatches to ambient N4SID when scenario.json says disturbance_type=ambient; otherwise it uses Matrix Pencil.
              - Fixed Matrix Pencil orders can be overridden with --fixed-orders; default: 2 4 6 8.
              - Adaptive tau values can be overridden with --taus; default: 1 0.1 0.01.
              - Without --output-dir, the analysis folder name is extended automatically with the selected time-window mode.
              - --scenario load29 runs a fresh Matrix Pencil analysis on IEEE39/results/Load29_Pplus2_50s and writes to a derived folder under IEEE39/analysis.
              - --skip-matrix-pencil requires --analysis-dir and reuses that folder's existing results.csv; it still regenerates reports, optional plots, and optional clustering.
              - Matrix Pencil analysis enables clustering and plots by default, both with by-control-area scope for clustering.
              - Ambient N4SID analysis enables clustering and plots by default, both with by-control-area scope for clustering.
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=None,
        help=(
            "Scenario selector. Accepts preset aliases like 'load29', exact IEEE39/results folder names like\n"
            "'Load29_Pplus2_50s', custom labels used with --data-dir, or 'all'. Required unless using --list-scenarios or --list-analysis."
        ),
    )
    parser.add_argument("--list-scenarios", action="store_true", help="Print the available preset scenario aliases and exit.")
    parser.add_argument("--list-analysis", action="store_true", help="Print existing IEEE39 analysis folders and exit.")
    parser.set_defaults(skip_clustering=None, skip_plots=None)
    parser.add_argument("--skip-clustering", dest="skip_clustering", action="store_true", help="Skip clustering output.")
    parser.add_argument("--clustering", dest="skip_clustering", action="store_false", help="Enable clustering output. Matrix Pencil default: on and by control area. Ambient default: on and by control area.")
    parser.add_argument("--clustering-scope", choices=["both", "global", "areas", "none"], default="areas", help="Choose clustering output scope. Default: areas.")
    parser.add_argument("--skip-matrix-pencil", action="store_true", help="Reuse an existing results.csv instead of recomputing Matrix Pencil poles.")
    parser.add_argument("--skip-n4sid", action="store_true", help="Reuse existing ambient N4SID sweep results instead of recomputing them.")
    parser.add_argument("--analysis-dir", default=None, help="Existing analysis directory to reuse with --skip-matrix-pencil. Relative paths are resolved from IEEE39.")
    parser.add_argument("--skip-plots", dest="skip_plots", action="store_true", help="Skip IEEE39 plot outputs, including modal maps, reconstructions, and thesis-used summary figures.")
    parser.add_argument("--plots", dest="skip_plots", action="store_false", help="Enable IEEE39 modal maps, reconstructions, and thesis-used summary figures. Matrix Pencil default: on. Ambient default: on.")
    parser.add_argument("--analysis-method", choices=["auto", "matrix-pencil", "n4sid"], default="auto", help="Select analysis backend. 'auto' uses ambient N4SID when scenario.json disturbance_type is 'ambient'; otherwise Matrix Pencil.")
    parser.add_argument("--data-dir", default=None, help="Explicit input data directory relative to IEEE39, or an absolute path. Use with exactly one --scenario.")
    parser.add_argument("--output-dir", default=None, help="Explicit output directory relative to IEEE39, or an absolute path. Use with exactly one --scenario.")
    parser.add_argument("--time-start", type=float, default=None, help="Inclusive analysis start time in seconds. Default: 0.")
    parser.add_argument("--time-end", type=float, default=None, help="Inclusive analysis end time in seconds. Default: last CSV timestamp.")
    parser.add_argument("--time-cross", choices=["global", "per-signal"], default=None, help="Start analysis after the first zero crossing of the detrended and filtered signal. 'global' uses one common start across all selected signals, while 'per-signal' resolves a separate start for each signal. When combined with --time-start, the value is treated as an offset after the detected zero crossing.")
    parser.add_argument("--time-cross-reference", default=None, help="Optional reference signal for --time-cross global, in the form g2:Current or g2:'Active Power'. If omitted, global mode uses the latest first zero crossing across all selected signals.")
    parser.add_argument("--no-reset-time", action="store_true", help="Do not shift the selected time window to start at zero.")
    parser.add_argument("--generators", nargs="+", default=None, help="Optional generator subset, e.g. g1 g2 g3.")
    parser.add_argument("--signals", nargs="+", default=None, help="Optional signal subset by label or CSV column, e.g. Voltage 'Active Power' or 's:P1 in MW'.")
    parser.add_argument("--fixed-orders", nargs="+", type=int, default=None, help="Override the fixed Matrix Pencil orders. Default: 2 4 6 8.")
    parser.add_argument("--taus", nargs="+", type=float, default=None, help="Override the tau values used for adaptive order selection. Default: 1 0.1 0.01.")
    parser.add_argument("--n4sid-orders", nargs="+", type=int, default=None, help="Ambient N4SID model orders. Default: built-in ambient order sweep.")
    parser.add_argument("--ambient-downsample-hz", type=float, default=None, help="Ambient preprocessing downsample rate in Hz. Default: 5.")
    parser.add_argument("--ambient-lpf-hz", type=float, default=None, help="Ambient preprocessing low-pass cutoff in Hz. Default: 2.")
    parser.add_argument("--ambient-no-detrend", action="store_true", help="Disable ambient detrending. Default ambient preprocessing detrends first.")
    parser.add_argument("--merge-radius", type=float, default=None, help="Ambient OPTICS pre-merge radius in standardized (Frequency, Damping) space. Default: 0.2. Only valid for ambient analysis.")
    parser.add_argument("--clustering-methods", nargs="+", choices=["kmeans", "kmedoids", "optics"], default=None, help="Ambient clustering methods. Default: kmeans kmedoids optics.")
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def early_validate_cli_args(args):
    if args.list_scenarios or args.list_analysis:
        return

    if not args.scenario:
        raise SystemExit(
            "Missing required --scenario. Examples: '--scenario load29', '--scenario all', or '--scenario Load29_Pplus2_50s'."
        )

    if args.skip_matrix_pencil and not args.analysis_dir:
        raise SystemExit(
            "--skip-matrix-pencil requires --analysis-dir so the script knows which existing analysis folder and results.csv to reuse. "
            "Example: --scenario Load29_Pplus2_50s --skip-matrix-pencil --analysis-dir analysis/Load29_Pplus2_50s_0_to_end_reset"
        )
    if args.skip_matrix_pencil and args.skip_n4sid:
        raise SystemExit("Use either --skip-matrix-pencil or --skip-n4sid, not both.")


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parse_args()

early_validate_cli_args(parse_args())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

from analysis_evaluator import update_analysis_config_with_evaluation
from ambient_n4sid_analysis import (
    AMBIENT_DEFAULT_SIGNALS,
    load_existing_ambient_results_for_scenario,
    preprocess_ambient_signal,
    run_ambient_n4sid_for_scenario,
)


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
determine_MP_orders = matrix_pencil.determine_MP_orders
filter_signal = matrix_pencil.filter_signal
prepare_matrix_pencil = matrix_pencil.prepare_matrix_pencil

from plot_style import (
    apply_thesis_style,
    style_axis,
    SIGNAL_COLORS,
)
from shared_plotting import (
    generator_display_name,
    generator_modal_label,
    plot_best_reconstruction_grid,
    plot_bubble_map,
    plot_modal_combined_map,
    plot_modal_generator_grid,
    plot_modal_signal_grid,
    plot_reconstruction_method_grid,
    save_current_figure,
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
DEFAULT_TIME_START_S = 0.0
MODE_FREQ_EPS_HZ = 1e-6
RECON_X_LIMS = (0, 50)
RECON_TICK_LABEL_SIZE = 30
RECON_AXIS_LABEL_SIZE = 34

CONTROL_AREAS = {
    "area_1": ["g1", "g8", "g9", "g10"],
    "area_2": ["g2", "g3"],
    "area_3": ["g4", "g5", "g6", "g7"],
}

IEEE39_REFERENCE_MODES = {
    "Mode 1": {"Frequency": 0.6062, "Damping": -0.0800, "Damping_Factor": 0.0210, "Generator_Involvement": "1-9 vs. 10", "relevant_generators": ["g1", "g8", "g9", "g10"], "relevant_areas": [1, 2, 3], "DRGA_Peak_Value": 17.8},
    "Mode 2": {"Frequency": 0.9497, "Damping": -0.1065, "Damping_Factor": 0.0178, "Generator_Involvement": "1,8 and 9 vs. 4,5,6 and 7", "relevant_generators": ["g1", "g4", "g5", "g6", "g7", "g8", "g9"], "relevant_areas": [1, 2], "DRGA_Peak_Value": 4.3},
    "Mode 3": {"Frequency": 1.0312, "Damping": -0.2558, "Damping_Factor": 0.0395, "Generator_Involvement": "2 and 3 vs. 4 and 5", "relevant_generators": ["g2", "g3", "g4", "g5"], "relevant_areas": [2, 3], "DRGA_Peak_Value": 2.3},
    "Mode 4": {"Frequency": 1.1211, "Damping": -0.3373, "Damping_Factor": 0.0478, "Generator_Involvement": "2 and 3 vs. 6 and 7", "relevant_generators": ["g2", "g3", "g6", "g7"], "relevant_areas": [2, 3], "DRGA_Peak_Value": 0.8},
    "Mode 5": {"Frequency": 1.3155, "Damping": -0.4033, "Damping_Factor": 0.0487, "Generator_Involvement": "2 vs. 3", "relevant_generators": ["g2", "g3"], "relevant_areas": [2], "DRGA_Peak_Value": 2.6},
    "Mode 6": {"Frequency": 1.2851, "Damping": -0.3458, "Damping_Factor": 0.0428, "Generator_Involvement": "1 vs. 8 and 9", "relevant_generators": ["g1", "g8", "g9"], "relevant_areas": [1], "DRGA_Peak_Value": 3.0},
    "Mode 7": {"Frequency": 1.4953, "Damping": -0.7033, "Damping_Factor": 0.0747, "Generator_Involvement": "4 vs. 5", "relevant_generators": ["g4", "g5"], "relevant_areas": [3], "DRGA_Peak_Value": None},
    "Mode 8": {"Frequency": 1.5202, "Damping": -0.6010, "Damping_Factor": 0.0628, "Generator_Involvement": "5 and 7 vs. 4 and 6", "relevant_generators": ["g4", "g5", "g6", "g7"], "relevant_areas": [3], "DRGA_Peak_Value": None},
    "Mode 9": {"Frequency": 1.5468, "Damping": -0.6376, "Damping_Factor": 0.0655, "Generator_Involvement": "1 vs. 8", "relevant_generators": ["g1", "g8"], "relevant_areas": [1], "DRGA_Peak_Value": None},
}


def _reference_modes_for_control_area(area_name):
    try:
        area_idx = int(str(area_name).split("_")[-1])
    except (TypeError, ValueError):
        return dict(IEEE39_REFERENCE_MODES)

    return {
        mode_name: dict(mode_data)
        for mode_name, mode_data in IEEE39_REFERENCE_MODES.items()
        if area_idx in mode_data.get("relevant_areas", [])
    }

DEFAULT_SCENARIO_PATHS = {
    "load29": {
        "data_dir": "results/Load29_Pplus2_50s",
        "output_dir": "analysis/Load29_Pplus2_50s",
    },
    "load03": {
        "data_dir": "results/Load03_Pplus2_50s",
        "output_dir": "analysis/Load03_Pplus2_50s",
    },
    "load24": {
        "data_dir": "results/Load24_Pplus2_50s",
        "output_dir": "analysis/Load24_Pplus2_50s",
    },
}


def _default_clustering_config(enabled=False, scope="areas"):
    if not enabled or scope == "none":
        return {"global": False, "by_control_area": False}
    if scope == "global":
        return {"global": True, "by_control_area": False}
    if scope == "both":
        return {"global": True, "by_control_area": True}
    return {"global": False, "by_control_area": True}


def _ambient_cli_overrides_requested(args):
    return any([
        args.n4sid_orders is not None,
        args.ambient_downsample_hz is not None,
        args.ambient_lpf_hz is not None,
        bool(args.ambient_no_detrend),
        args.merge_radius is not None,
        args.clustering_methods is not None,
    ])


def _base_scenario_defaults():
    return {
        "time_mask": {"start_inclusive": DEFAULT_TIME_START_S, "reset_time": True},
        "generators": list(IEEE39_GENERATORS),
        "columns": dict(COLUMNS),
        "fixed_orders": [2, 4, 6, 8],
        "taus": [1, 0.1, 0.01],
        "auto_order_decimation": AUTO_ORDER_DECIMATION,
        "filter": {"fc": 10, "N": 15},
        "clustering": _default_clustering_config(enabled=False),
    }


def _scenario_defaults_for_paths(data_dir, output_dir):
    scenario = _base_scenario_defaults()
    scenario.update({
        "data_dir": data_dir,
        "output_dir": output_dir,
    })
    return scenario


DEFAULT_SCENARIOS = {
    name: _scenario_defaults_for_paths(scenario["data_dir"], scenario["output_dir"])
    for name, scenario in DEFAULT_SCENARIO_PATHS.items()
}


def _resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == BASE_DIR.name:
        return REPO_DIR / path
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


def _time_cross_suffix(time_mask, time_cross):
    time_mask = time_mask or {}
    time_cross = time_cross or {}
    end = time_mask.get("end_inclusive", time_mask.get("end"))
    reset = time_mask.get("reset_time", True)
    mode = _sanitize_suffix_part(time_cross.get("mode", "global"))
    offset = _format_time_value(time_cross.get("offset_s", 0.0))
    reference = time_cross.get("reference")
    end_part = _format_time_value(end) if end is not None else "end"
    reset_part = "reset" if reset else "noreset"
    ref_part = ""
    if reference:
        ref_part = f"_ref-{_sanitize_suffix_part(reference.get('generator'))}-{_sanitize_suffix_part(reference.get('signal'))}"
    return f"tcross-{mode}{ref_part}_off{offset}_to_{end_part}_{reset_part}"


def _sanitize_suffix_part(value):
    return str(value).strip().lower().replace(" ", "_").replace(":", "").replace("/", "_")


def _selection_suffix(scenario):
    parts = []

    generators = scenario.get("generator_subset") or []
    if generators:
        parts.append("g-" + "-".join(generators))

    signal_subset = scenario.get("signal_subset") or []
    if signal_subset:
        parts.append("sig-" + "-".join(_sanitize_suffix_part(signal) for signal in signal_subset))

    return "_".join(parts)


def _analysis_output_dir(scenario):
    output_dir = _resolve_path(scenario.get("output_dir", "analysis"))
    if scenario.get("output_dir_explicit"):
        return output_dir

    if scenario.get("time_cross"):
        time_suffix = _time_cross_suffix(scenario.get("time_mask"), scenario.get("time_cross"))
    else:
        time_suffix = _time_mask_suffix(scenario.get("time_mask"))
    base_name = f"{output_dir.name}_{time_suffix}"
    selection_suffix = _selection_suffix(scenario)
    if selection_suffix:
        base_name = f"{base_name}_{selection_suffix}"
    return output_dir.parent / base_name


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


def _time_mask_bound(mask_config, inclusive_key, exclusive_key):
    mask_config = mask_config or {}
    if mask_config.get(inclusive_key) is not None:
        return float(mask_config[inclusive_key])
    if mask_config.get(exclusive_key) is not None:
        return float(mask_config[exclusive_key])
    return None


def _scenario_cache(scenario):
    return scenario.setdefault("_runtime_cache", {})


def _time_cross_config(scenario):
    return scenario.get("time_cross") or None


def _signal_cache_key(gen, column_name):
    return f"{gen}:{column_name}"


def _detect_first_zero_cross_time(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2 or y.size < 2:
        return None

    for idx in range(t.size - 1):
        y0 = float(y[idx])
        y1 = float(y[idx + 1])
        t0 = float(t[idx])
        t1 = float(t[idx + 1])

        if y0 == 0.0:
            return t0
        if y1 == 0.0:
            return t1
        if y0 * y1 < 0.0:
            frac = -y0 / (y1 - y0)
            return t0 + frac * (t1 - t0)

    return None


def _prepare_filtered_full_signal(df, column_name, scenario, gen):
    cache = _scenario_cache(scenario).setdefault("full_filtered_signals", {})
    cache_key = _signal_cache_key(gen, column_name)
    if cache_key in cache:
        return cache[cache_key]

    time_all = df.iloc[:, 0].to_numpy(dtype=float)
    signal_all = df[column_name].to_numpy(dtype=float)
    valid = np.isfinite(time_all) & np.isfinite(signal_all)
    if np.count_nonzero(valid) < 4:
        raise SystemExit(f"Not enough finite samples for {gen} {column_name} to resolve time-cross.")

    t_full = time_all[valid].copy()
    y_full = signal_all[valid].copy()
    y_detrended = detrend(y_full)
    mean_after_detrend = float(np.mean(y_detrended))
    filter_config = scenario.get("filter", {"fc": 10, "N": 15})
    y_filtered = filter_signal(
        y_detrended,
        t_full,
        fc=float(filter_config.get("fc", 10)),
        N=int(filter_config.get("N", 15)),
    )
    prepared = {
        "t": t_full,
        "y": y_filtered,
        "mean_after_detrend": mean_after_detrend,
        "mean_after_lpf": float(np.mean(y_filtered)),
    }
    cache[cache_key] = prepared
    return prepared


def _resolve_time_cross_summary(scenario):
    time_cross = _time_cross_config(scenario)
    if time_cross is None:
        return None

    cache = _scenario_cache(scenario)
    cached = cache.get("time_cross_summary")
    if cached is not None:
        return cached

    data_dir, _, _, generators, columns = _scenario_runtime_config(scenario)
    offset_s = float(time_cross.get("offset_s", 0.0))
    end_s = _time_mask_bound(scenario.get("time_mask") or {}, "end_inclusive", "end")
    reference = time_cross.get("reference")

    signal_entries = []
    for gen in generators:
        csv_path = data_dir / f"{gen}.csv"
        if not csv_path.exists():
            continue
        df = _read_numeric_csv(csv_path)
        for column_name, signal_label in columns.items():
            if column_name not in df.columns:
                continue
            prepared = _prepare_filtered_full_signal(df, column_name, scenario, gen)
            cross_time_s = _detect_first_zero_cross_time(prepared["t"], prepared["y"])
            if cross_time_s is None:
                raise SystemExit(
                    f"Could not detect first zero crossing for {gen} {signal_label} in scenario '{scenario.get('name', 'analysis')}'."
                )
            signal_entries.append({
                "gen": gen,
                "column": column_name,
                "signal": signal_label,
                "first_zero_cross_s": float(cross_time_s),
            })

    if not signal_entries:
        raise SystemExit("Could not resolve time-cross: no generator signals were available.")

    mode = str(time_cross.get("mode", "global"))
    if mode == "global":
        if reference:
            ref_matches = [
                entry for entry in signal_entries
                if entry["gen"] == reference.get("generator") and entry["signal"] == reference.get("signal")
            ]
            if not ref_matches:
                raise SystemExit(
                    f"Could not resolve time-cross reference {reference.get('generator')}:{reference.get('signal')} in the selected data."
                )
            common_cross_s = float(ref_matches[0]["first_zero_cross_s"])
        else:
            common_cross_s = max(entry["first_zero_cross_s"] for entry in signal_entries)
        common_start_s = common_cross_s + offset_s
        for entry in signal_entries:
            entry["effective_start_s"] = float(common_start_s)
    else:
        common_cross_s = None
        common_start_s = None
        for entry in signal_entries:
            entry["effective_start_s"] = float(entry["first_zero_cross_s"] + offset_s)

    per_signal = {}
    effective_starts = []
    for entry in signal_entries:
        effective_start_s = float(entry["effective_start_s"])
        prepared = _scenario_cache(scenario)["full_filtered_signals"][_signal_cache_key(entry["gen"], entry["column"])]
        mask = prepared["t"] >= effective_start_s
        if end_s is not None:
            mask &= prepared["t"] <= end_s
        selected_count = int(np.count_nonzero(mask))
        if selected_count < 4:
            raise SystemExit(
                f"Invalid time-cross for {entry['gen']} {entry['signal']}: only {selected_count} samples remain after start={effective_start_s:g}s."
            )
        per_signal.setdefault(entry["gen"], {})[entry["signal"]] = {
            "first_zero_cross_s": float(entry["first_zero_cross_s"]),
            "effective_start_s": effective_start_s,
            "selected_samples": selected_count,
        }
        effective_starts.append(effective_start_s)

    summary = {
        "mode": mode,
        "offset_s": offset_s,
        "reference": reference,
        "common_zero_cross_s": None if common_cross_s is None else float(common_cross_s),
        "common_start_s": None if common_start_s is None else float(common_start_s),
        "effective_start_range_s": {
            "min_s": float(min(effective_starts)),
            "max_s": float(max(effective_starts)),
        },
        "per_signal": per_signal,
    }
    cache["time_cross_summary"] = summary
    return summary


def _resolved_time_window_description(scenario, generators=None, columns=None):
    time_cross = _time_cross_config(scenario)
    end_s = _time_mask_bound(scenario.get("time_mask") or {}, "end_inclusive", "end")
    if time_cross is None:
        data_dir = _resolve_path(scenario["data_dir"])
        for gen in list(generators or scenario.get("generators") or []):
            csv_path = data_dir / f"{gen}.csv"
            if not csv_path.exists():
                continue
            time_values = _read_numeric_csv(csv_path).iloc[:, 0].to_numpy(dtype=float)
            return _time_window_description(time_values, scenario.get("time_mask"))
        return None

    summary = _resolve_time_cross_summary(scenario)
    if summary["mode"] == "global":
        start_s = summary["common_start_s"]
    else:
        start_s = None

    return {
        "start_s": start_s,
        "end_s": end_s,
        "start_mode": summary["mode"],
        "offset_from_zero_cross_s": float(summary["offset_s"]),
    }


def _validate_time_mask_config(mask_config, scenario_name):
    mask_config = mask_config or {}
    for key in ["start", "start_inclusive", "end", "end_inclusive"]:
        value = mask_config.get(key)
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            raise SystemExit(f"Invalid time mask for '{scenario_name}': {key} must be finite, got {value}.")

    start = _time_mask_bound(mask_config, "start_inclusive", "start")
    end = _time_mask_bound(mask_config, "end_inclusive", "end")
    if start is not None and end is not None and end <= start:
        raise SystemExit(
            f"Invalid time mask for '{scenario_name}': time_end ({end:g}) must be greater than time_start ({start:g})."
        )


def validate_scenario_time_window(name, scenario, generated_config=None, generators=None):
    _validate_time_mask_config(scenario.get("time_mask"), name)

    if _time_cross_config(scenario) is not None:
        summary = _resolve_time_cross_summary(scenario)
        mode_desc = summary["mode"]
        print(f"Resolved time-cross mode for '{name}': {mode_desc}", flush=True)
        return

    data_dir = _resolve_path(scenario["data_dir"])
    generators = list(generators or scenario.get("generators") or [])

    reference_csv = None
    finite_time = None
    for gen in generators:
        csv_path = data_dir / f"{gen}.csv"
        if not csv_path.exists():
            continue
        time_values = _read_numeric_csv(csv_path).iloc[:, 0].to_numpy(dtype=float)
        finite_time = time_values[np.isfinite(time_values)]
        reference_csv = csv_path
        break

    if reference_csv is None or finite_time is None:
        raise SystemExit(f"Could not validate time window for '{name}': no generator CSV files found in {data_dir}.")
    if finite_time.size < 4:
        raise SystemExit(
            f"Could not validate time window for '{name}': {reference_csv.name} contains fewer than 4 finite time samples."
        )
    if np.any(np.diff(finite_time) <= 0):
        raise SystemExit(
            f"Invalid time column in {reference_csv}: timestamps must be strictly increasing for scenario '{name}'."
        )

    min_time = float(np.min(finite_time))
    max_time = float(np.max(finite_time))
    mask_config = scenario.get("time_mask") or {}
    start = _time_mask_bound(mask_config, "start_inclusive", "start")
    end = _time_mask_bound(mask_config, "end_inclusive", "end")
    tol = 1e-9

    if start is not None and start < min_time - tol:
        raise SystemExit(
            f"Invalid time_start for '{name}': requested {start:g}s but available data starts at {min_time:g}s in {reference_csv.name}."
        )
    if start is not None and start > max_time + tol:
        raise SystemExit(
            f"Invalid time_start for '{name}': requested {start:g}s but available data ends at {max_time:g}s in {reference_csv.name}."
        )
    if end is not None and end < min_time - tol:
        raise SystemExit(
            f"Invalid time_end for '{name}': requested {end:g}s but available data starts at {min_time:g}s in {reference_csv.name}."
        )
    if end is not None and end > max_time + tol:
        raise SystemExit(
            f"Invalid time_end for '{name}': requested {end:g}s but available data ends at {max_time:g}s in {reference_csv.name}."
        )

    sim_stop_time = None
    if generated_config and generated_config.get("sim_stop_time_s") is not None:
        sim_stop_time = float(generated_config["sim_stop_time_s"])
        if not np.isfinite(sim_stop_time):
            raise SystemExit(f"Invalid scenario.json for '{name}': sim_stop_time_s must be finite, got {sim_stop_time}.")
        if end is not None and end > sim_stop_time + tol:
            raise SystemExit(
                f"Invalid time_end for '{name}': requested {end:g}s exceeds scenario duration {sim_stop_time:g}s from scenario.json."
            )

    mask = _time_mask(finite_time, mask_config)
    selected_count = int(np.count_nonzero(mask))
    if selected_count == 0:
        raise SystemExit(
            f"Invalid time mask for '{name}': no samples remain after applying the window [{start if start is not None else min_time:g}, {end if end is not None else max_time:g}] to {reference_csv.name}."
        )
    if selected_count < 4:
        raise SystemExit(
            f"Invalid time mask for '{name}': only {selected_count} time samples remain after masking {reference_csv.name}; at least 4 are required."
        )


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _format_duration_min_sec(seconds):
    total_seconds = max(0.0, float(seconds))
    minutes = int(total_seconds // 60)
    seconds_part = total_seconds - (minutes * 60)
    return f"{minutes:02d}:{seconds_part:04.1f}"


def _timing_entry(seconds, skipped=False):
    total_seconds = max(0.0, float(seconds))
    return {
        "seconds": round(total_seconds, 6),
        "min_sec": _format_duration_min_sec(total_seconds),
        "skipped": bool(skipped),
    }


def _build_analysis_config(
    name,
    scenario,
    data_dir,
    output_dir,
    generated_config,
    generators,
    auto_order_decimation,
    time_window,
    resolved_time_cross,
    signal_means,
    timings=None,
):
    metadata_scenario = {key: value for key, value in scenario.items() if not str(key).startswith("_")}
    for key in ("data_dir", "output_dir"):
        if metadata_scenario.get(key):
            metadata_scenario[key] = path_for_metadata(_resolve_path(metadata_scenario[key]))

    return {
        "name": name,
        **metadata_scenario,
        "data_scenario_json": path_for_metadata(data_dir / "scenario.json") if generated_config else None,
        "generators_used": generators,
        "auto_order_decimation": auto_order_decimation,
        "time_window_s": time_window,
        "resolved_time_cross": resolved_time_cross,
        "time_reset_to_zero": scenario.get("time_mask", {}).get("reset_time", True),
        "signal_means": signal_means,
        "timings": timings or {},
    }


def _run_clustering_pipeline(results_path, output_path, reference_modes=None, methods=None, include_silhouette=True):
    from clustering_analysis import (
        _load_screened_data,
        _save_reference_mad_outputs,
        run_kmeans_modal_analysis,
        run_kmedoids_modal_analysis,
        run_silhouette_analysis,
    )

    pipeline_start = time.perf_counter()
    requested_methods = list(methods or ["kmeans", "kmedoids"])

    screen_start = time.perf_counter()
    df_for_mad = _load_screened_data(str(results_path), str(output_path))
    screen_elapsed = time.perf_counter() - screen_start

    reference_elapsed = 0.0
    if df_for_mad is not None:
        reference_start = time.perf_counter()
        _save_reference_mad_outputs(df_for_mad, str(output_path), reference_modes=reference_modes)
        reference_elapsed = time.perf_counter() - reference_start

    timings = {
        "screen_and_load": _timing_entry(screen_elapsed),
        "reference_mad": _timing_entry(reference_elapsed, skipped=df_for_mad is None),
    }
    runners = {
        "kmeans": run_kmeans_modal_analysis,
        "kmedoids": run_kmedoids_modal_analysis,
    }
    for method in requested_methods:
        method_start = time.perf_counter()
        runners[method](str(results_path), str(output_path), reference_modes=reference_modes)
        timings[method] = _timing_entry(time.perf_counter() - method_start)

    silhouette_skipped = True
    silhouette_elapsed = 0.0
    if include_silhouette and {"kmeans", "kmedoids"}.issubset(set(requested_methods)):
        silhouette_start = time.perf_counter()
        run_silhouette_analysis(str(results_path), str(output_path), reference_modes=reference_modes)
        silhouette_elapsed = time.perf_counter() - silhouette_start
        silhouette_skipped = False
    timings["silhouette"] = _timing_entry(silhouette_elapsed, skipped=silhouette_skipped)
    timings["total"] = _timing_entry(time.perf_counter() - pipeline_start)
    return timings


def _load_scenario_json(data_dir):
    scenario_json = data_dir / "scenario.json"
    if not scenario_json.exists():
        return None

    return _load_json(scenario_json)


def _scenario_disturbance_type(scenario):
    data_dir = _resolve_path(scenario["data_dir"])
    generated_config = _load_scenario_json(data_dir)
    disturbance_type = None if generated_config is None else generated_config.get("disturbance_type")
    if disturbance_type is None:
        return None
    return str(disturbance_type).strip().lower()


def _resolve_analysis_method(name, scenario, args):
    disturbance_type = _scenario_disturbance_type(scenario)
    requested = str(args.analysis_method).strip().lower()
    if requested == "auto":
        method = "n4sid" if disturbance_type == "ambient" else "matrix-pencil"
    else:
        method = requested

    if method == "n4sid" and disturbance_type != "ambient":
        raise SystemExit(
            f"Scenario '{name}' does not declare disturbance_type='ambient' in scenario.json, so --analysis-method n4sid is not allowed."
        )

    return method, disturbance_type


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
    generators = scenario.get("generators") or _scenario_generators_from_json(generated_config) or IEEE39_GENERATORS
    columns = scenario.get("columns", COLUMNS)

    return data_dir, output_dir, generated_config, generators, columns


def _resolve_signal_subset(signal_values):
    if not signal_values:
        return None

    resolved = {}
    label_to_col = {label.lower(): col for col, label in COLUMNS.items()}
    col_to_label = {col.lower(): label for col, label in COLUMNS.items()}

    for raw_value in signal_values:
        key = str(raw_value).strip().lower()
        if key in label_to_col:
            col = label_to_col[key]
            resolved[col] = COLUMNS[col]
            continue
        if key in col_to_label:
            label = col_to_label[key]
            resolved[str(raw_value).strip()] = label
            continue
        available = ", ".join(sorted(COLUMNS.values()))
        raise SystemExit(f"Unknown signal '{raw_value}'. Available labels: {available}")

    return resolved


def _resolve_generator_subset(generator_values):
    if not generator_values:
        return None
    return [str(gen).strip() for gen in generator_values]


def _resolve_time_cross_reference(raw_value, scenario):
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if ":" not in text:
        raise SystemExit("--time-cross-reference must be in the form g2:Current")

    generator, signal = text.split(":", 1)
    generator = generator.strip()
    signal = signal.strip()
    if not generator or not signal:
        raise SystemExit("--time-cross-reference must be in the form g2:Current")

    available_generators = set(scenario.get("generators") or IEEE39_GENERATORS)
    if generator not in available_generators:
        available = ", ".join(sorted(available_generators))
        raise SystemExit(f"Unknown time-cross reference generator '{generator}'. Available generators: {available}")

    resolved_signals = _resolve_signal_subset([signal])
    column_name, signal_label = next(iter(resolved_signals.items()))
    return {
        "generator": generator,
        "signal": signal_label,
        "column": column_name,
    }


def _preprocess_signal(df, column_name, scenario, gen, signal_label=None):
    if scenario.get("analysis_method") == "n4sid":
        ambient_cfg = scenario.get("ambient_preprocessing", {})
        t_selected, y_selected, preprocess_meta = preprocess_ambient_signal(
            df=df,
            column_name=column_name,
            time_mask_config=scenario.get("time_mask") or {},
            detrend_enabled=bool(ambient_cfg.get("detrend", True)),
            downsample_hz=float(ambient_cfg.get("downsample_hz", 5.0)),
            lowpass_hz=float(ambient_cfg.get("low_pass_hz", 2.0)),
        )
        if t_selected is None or y_selected is None or preprocess_meta is None:
            return None, None, None
        start_abs_s = float(preprocess_meta.get("time_start_s", t_selected[0]))
        end_abs_s = float(preprocess_meta.get("time_end_s", t_selected[-1]))
        if (scenario.get("time_mask") or {}).get("reset_time", True):
            t_selected = t_selected - t_selected[0]
        return t_selected, y_selected, {
            "first_zero_cross_s": None,
            "effective_start_s": start_abs_s,
            "effective_end_s": end_abs_s,
            "selected_samples": int(preprocess_meta.get("selected_samples", t_selected.size)),
            "time_cross_mode": None,
        }

    time_cross = _time_cross_config(scenario)
    time_mask = scenario.get("time_mask") or {}
    end_s = _time_mask_bound(time_mask, "end_inclusive", "end")

    if time_cross is None:
        time_all = df.iloc[:, 0].to_numpy(dtype=float)
        signal_all = df[column_name].to_numpy(dtype=float)
        mask = _time_mask(time_all, time_mask)

        if not np.any(mask):
            return None, None, None

        t_selected = time_all[mask].copy()
        y_selected = signal_all[mask].copy()
        valid = np.isfinite(t_selected) & np.isfinite(y_selected)
        if np.count_nonzero(valid) < 4:
            return None, None, None

        t_selected = t_selected[valid]
        y_selected = y_selected[valid]
        y_selected = detrend(y_selected)
        filter_config = scenario.get("filter", {"fc": 10, "N": 15})
        y_selected = filter_signal(
            y_selected,
            t_selected,
            fc=float(filter_config.get("fc", 10)),
            N=int(filter_config.get("N", 15)),
        )
        start_abs_s = float(t_selected[0])
        end_abs_s = float(t_selected[-1])
        if time_mask.get("reset_time", True):
            t_selected = t_selected - t_selected[0]
        return t_selected, y_selected, {
            "first_zero_cross_s": None,
            "effective_start_s": start_abs_s,
            "effective_end_s": end_abs_s,
            "selected_samples": int(t_selected.size),
            "time_cross_mode": None,
        }

    prepared = _prepare_filtered_full_signal(df, column_name, scenario, gen)
    summary = _resolve_time_cross_summary(scenario)
    if signal_label is None:
        signal_label = scenario.get("columns", COLUMNS).get(column_name, column_name)
    signal_summary = summary["per_signal"][gen][signal_label]
    effective_start_s = float(signal_summary["effective_start_s"])

    mask = prepared["t"] >= effective_start_s
    if end_s is not None:
        mask &= prepared["t"] <= end_s
    if not np.any(mask):
        return None, None, None

    t_selected = prepared["t"][mask].copy()
    y_selected = prepared["y"][mask].copy()
    if time_mask.get("reset_time", True):
        t_selected = t_selected - t_selected[0]
    return t_selected, y_selected, {
        "first_zero_cross_s": float(signal_summary["first_zero_cross_s"]),
        "effective_start_s": effective_start_s,
        "effective_end_s": float(prepared["t"][mask][-1]),
        "selected_samples": int(np.count_nonzero(mask)),
        "time_cross_mode": summary["mode"],
    }


def _r2_score(y_true, y_pred):
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total == 0:
        return np.nan
    return 1.0 - residual / total


def _reconstruct_signal(t, modes):
    y_est = np.zeros_like(t)
    for _, mode in modes.iterrows():
        y_est += 2 * mode["Amplitude"] * np.exp(mode["Damping"] * t) * np.cos(
            2 * np.pi * mode["Frequency"] * t + mode["Phase"]
        )
    return y_est


def _result_diagnostic_methods(scenario):
    if scenario.get("analysis_method") == "n4sid":
        return [f"Order {int(order)}" for order in scenario.get("n4sid_orders", [])]
    fixed_orders = [f"Order {int(order)}" for order in scenario.get("fixed_orders", [])]
    taus = [f"Tau {tau}" for tau in scenario.get("taus", [])]
    return fixed_orders + taus


def _scenario_method_order(scenario):
    return _result_diagnostic_methods(scenario)


def _scenario_reconstruction_rows(scenario):
    if scenario.get("analysis_method") == "n4sid":
        return [(f"Order {int(order)}", None) for order in scenario.get("n4sid_orders", [])]
    fixed_orders = [f"Order {int(order)}" for order in scenario.get("fixed_orders", [])]
    taus = [f"Tau {tau}" for tau in scenario.get("taus", [])]
    row_count = max(len(fixed_orders), len(taus))
    return [
        (
            fixed_orders[row_idx] if row_idx < len(fixed_orders) else None,
            taus[row_idx] if row_idx < len(taus) else None,
        )
        for row_idx in range(row_count)
    ]


def _generator_display_name(gen):
    if isinstance(gen, str) and gen.startswith("g") and gen[1:].isdigit():
        return f"Generator {int(gen[1:])}"
    return str(gen)


def _collect_result_diagnostics(data_dir, scenario, generators, columns, df_results):
    missing_results = []
    methods = _result_diagnostic_methods(scenario)
    fixed_orders = [int(order) for order in scenario.get("fixed_orders", [])]
    taus_raw = list(scenario.get("taus", []))
    taus = [float(tau) for tau in taus_raw]
    auto_order_decimation = int(scenario.get("auto_order_decimation", scenario.get("order_rate", AUTO_ORDER_DECIMATION)))
    expected = [(gen, signal, method) for gen in generators for signal in columns.values() for method in methods]
    actual_set = {
        (str(row.Gen), str(row.Signal), str(row.Method))
        for row in df_results[["Gen", "Signal", "Method"]].drop_duplicates().itertuples(index=False)
    } if not df_results.empty else set()
    missing_combinations = [combo for combo in expected if combo not in actual_set]
    signal_cache = {}
    details = []

    label_to_column = {label: col for col, label in columns.items()}

    for gen, signal, method in missing_combinations:
        cache_key = (gen, signal)
        if cache_key not in signal_cache:
            csv_path = data_dir / f"{gen}.csv"
            if not csv_path.exists():
                signal_cache[cache_key] = {"status": "missing_csv"}
            else:
                df = _read_numeric_csv(csv_path)
                source_col = label_to_column[signal]
                if source_col not in df.columns:
                    signal_cache[cache_key] = {"status": "missing_signal_column"}
                else:
                    t, y, preprocess_meta = _preprocess_signal(df, source_col, scenario, gen, signal)
                    if t is None or y is None or preprocess_meta is None:
                        signal_cache[cache_key] = {"status": "not_enough_samples_after_preprocessing"}
                    else:
                        prepared_mp = prepare_matrix_pencil(y, t)
                        tau_order_map = determine_MP_orders(t, y, taus, rate=auto_order_decimation) if taus else {}
                        signal_cache[cache_key] = {
                            "status": "ok",
                            "selected_samples": int(preprocess_meta.get("selected_samples", len(t))),
                            "prepared_mp": prepared_mp,
                            "mp_fit_cache": {},
                            "tau_order_map": tau_order_map,
                        }

        state = signal_cache[cache_key]
        if state["status"] != "ok":
            reason = state["status"]
            details.append({
                "gen": gen,
                "signal": signal,
                "method": method,
                "preprocess_status": state["status"],
                "selected_samples": 0,
                "total_raw_poles": 0,
                "kept_poles": 0,
                "filtered_non_oscillatory_poles": 0,
                "missing_result_reason": reason,
            })
            missing_results.append({
                "gen": gen,
                "signal": signal,
                "method": method,
                "missing_result_reason": reason,
            })
            continue

        if method.startswith("Order "):
            order = int(method.split(" ", 1)[1])
        else:
            tau_text = method.split(" ", 1)[1]
            order = state["tau_order_map"][float(tau_text)]

        freq, _, _, _, _, _ = apply_matrix_pencil_fixed_order_prepared(
            state["prepared_mp"],
            order=order,
            fit_cache=state["mp_fit_cache"],
        )
        total_raw_poles = int(len(freq))
        kept_poles = int(sum(1 for f in freq if f > MODE_FREQ_EPS_HZ))
        filtered_non_oscillatory_poles = total_raw_poles - kept_poles
        reason = "all_poles_below_frequency_threshold"
        details.append({
            "gen": gen,
            "signal": signal,
            "method": method,
            "preprocess_status": "ok",
            "selected_samples": int(state["selected_samples"]),
            "total_raw_poles": total_raw_poles,
            "kept_poles": kept_poles,
            "filtered_non_oscillatory_poles": filtered_non_oscillatory_poles,
            "missing_result_reason": reason,
        })
        missing_results.append({
            "gen": gen,
            "signal": signal,
            "method": method,
            "missing_result_reason": reason,
        })

    actual_rows = len(actual_set)
    reason_counts = {}
    for item in missing_results:
        reason = item["missing_result_reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "result_coverage": {
            "expected_case_method_rows": int(len(expected)),
            "actual_case_method_rows": int(actual_rows),
            "missing_case_method_rows": int(len(missing_results)),
        },
        "missing_results": missing_results,
        "result_filter_diagnostics": {
            "missing_by_reason": reason_counts,
            "missing_case_method_details": details,
        },
    }


def _attach_result_diagnostics(analysis_config, data_dir, scenario, generators, columns, df_results):
    analysis_config["oscillatory_frequency_threshold_hz"] = MODE_FREQ_EPS_HZ
    diagnostics = _collect_result_diagnostics(data_dir, scenario, generators, columns, df_results)
    analysis_config.update(diagnostics)


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

            t, y_ref, _ = _preprocess_signal(df, source_col, scenario, gen, signal)
            if t is None or y_ref is None:
                continue

            for method in _scenario_method_order(scenario):
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


def _generate_ieee39_modal_grid_plots(df_results, modal_maps_dir, generators, columns):
    for gen in generators:
        gen_data = df_results[df_results["Gen"] == gen]
        if gen_data.empty:
            continue

        plot_modal_signal_grid(
            df_results=df_results,
            gen=gen,
            signals=list(columns.values()),
            output_dir=modal_maps_dir,
            filename=f"{gen}_2x2_grid",
            title=f"Modal Identification per Signal: {generator_modal_label(gen)}",
            colors=SIGNAL_COLORS.copy(),
        )

    plot_modal_generator_grid(
        df_results=df_results,
        generators=generators,
        signals=list(columns.values()),
        output_dir=modal_maps_dir,
        filename="All_Generators_Grid",
        title="System-Wide Modal Identification (All Generators)",
        colors=SIGNAL_COLORS.copy(),
    )


def _generate_ieee39_bubble_map(df_results, stats_dir):
    if df_results.empty:
        return

    source_builder = lambda row: f"{generator_display_name(row['Gen'])} | {row['Signal']}"
    plot_bubble_map(df_results, stats_dir, "5_bubble_map_single_panel", source_builder=source_builder, min_height=12.0)
    plot_bubble_map(df_results, stats_dir, "5_bubble_map", source_builder=source_builder, min_height=12.0)


def _generate_ieee39_best_reconstruction_plots(df_results, report, scenario, stats_dir, data_dir, generators, columns):
    if report.empty:
        return

    best_rows = report.loc[report.groupby(["Gen", "Signal"])["R2"].idxmax()].copy()
    inv_columns = {label: csv_col for csv_col, label in columns.items()}

    for gen in generators:
        csv_path = data_dir / f"{gen}.csv"
        if not csv_path.exists():
            continue

        df = _read_numeric_csv(csv_path)
        generator_label = _generator_display_name(gen)
        items = []

        for signal in columns.values():
            source_col = inv_columns[signal]
            if source_col not in df.columns:
                continue

            t, y_ref, _ = _preprocess_signal(df, source_col, scenario, gen, signal)
            if t is None or y_ref is None:
                continue

            y_ref = y_ref - np.mean(y_ref)
            best_row = best_rows[(best_rows["Gen"] == gen) & (best_rows["Signal"] == signal)]
            if best_row.empty:
                continue

            best_method = best_row.iloc[0]["Method"]
            best_r2 = float(best_row.iloc[0]["R2"])
            modes = df_results[
                (df_results["Gen"] == gen)
                & (df_results["Signal"] == signal)
                & (df_results["Method"] == best_method)
            ]
            if modes.empty:
                continue

            y_est = _reconstruct_signal(t, modes)
            items.append({
                "t": t,
                "y_ref": y_ref,
                "y_est": y_est,
                "title": f"{generator_label} - {signal}\nMethod: {best_method} ($R^2$: {best_r2:.4f})",
                "signal": signal,
                "show_legend": len(items) == 1,
            })

        if not items:
            continue

        plot_best_reconstruction_grid(
            items=items,
            output_dir=stats_dir,
            filename=f"10_best_reconstruction_{gen}_2x2",
            title=f"Absolute Best Signal Reconstruction (Max $R^2$) - {generator_label}",
            x_lims=RECON_X_LIMS,
        )


def generate_ieee39_plots(df_results, report, scenario):
    if df_results.empty:
        print("No Matrix Pencil results available; skipping IEEE39 plots.")
        return

    data_dir, output_dir, _, generators, columns = _scenario_runtime_config(scenario)
    plots_dir = output_dir / "plots"
    modal_maps_dir = plots_dir / "modal_maps"
    recon_dir = plots_dir / "reconstruction_grids"
    stats_dir = output_dir / "stats"

    _generate_ieee39_modal_grid_plots(df_results, modal_maps_dir, generators, columns)

    for gen in generators:
        gen_data = df_results[df_results["Gen"] == gen]
        if gen_data.empty:
            continue

        plot_modal_combined_map(
            df_results=df_results,
            output_dir=modal_maps_dir,
            filename=f"{gen}_combined_modal_map",
            title=f"Combined Modal Map: {generator_modal_label(gen)}",
            signals=list(columns.values()),
            gen=gen,
            colors=SIGNAL_COLORS.copy(),
            figsize=(10, 6),
        )

    plot_modal_combined_map(
        df_results=df_results,
        output_dir=modal_maps_dir,
        filename="system_modal_map",
        title="System-Wide Modal Map",
        signals=list(columns.values()),
        colors=SIGNAL_COLORS.copy(),
        figsize=(11, 7),
    )

    inv_columns = {label: csv_col for csv_col, label in columns.items()}
    reconstruction_rows = _scenario_reconstruction_rows(scenario)
    for gen in generators:
        csv_path = data_dir / f"{gen}.csv"
        if not csv_path.exists():
            continue

        df = _read_numeric_csv(csv_path)
        for signal in columns.values():
            source_col = inv_columns[signal]
            if source_col not in df.columns:
                continue

            t, y_ref, _ = _preprocess_signal(df, source_col, scenario, gen, signal)
            if t is None or y_ref is None:
                continue

            if not reconstruction_rows:
                continue

            plot_reconstruction_method_grid(
                t=t,
                y_ref=y_ref,
                reconstruction_rows=reconstruction_rows,
                fetch_modes=lambda method: df_results[
                    (df_results["Gen"] == gen)
                    & (df_results["Signal"] == signal)
                    & (df_results["Method"] == method)
                ],
                reconstruct_signal=_reconstruct_signal,
                output_dir=recon_dir,
                filename=f"{gen}_{signal.replace(' ', '_')}_reconstruction",
                title=f"Reconstruction Accuracy: {gen.upper()} - {signal}\nLeft: Fixed Orders | Right: Adaptive Tau",
                signal=signal,
                x_lims=RECON_X_LIMS,
            )

    _generate_ieee39_bubble_map(df_results, stats_dir)
    _generate_ieee39_best_reconstruction_plots(df_results, report, scenario, stats_dir, data_dir, generators, columns)


def run_matrix_pencil_for_scenario(name, scenario):
    mp_start = time.perf_counter()
    data_dir, output_dir, generated_config, generators, columns = _scenario_runtime_config(scenario)
    validate_scenario_time_window(name, scenario, generated_config=generated_config, generators=generators)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_orders = scenario.get("fixed_orders", [2, 4, 6, 8])
    taus = scenario.get("taus", [1, 0.1, 0.01])
    auto_order_decimation = int(
        scenario.get("auto_order_decimation", scenario.get("order_rate", AUTO_ORDER_DECIMATION))
    )
    results = []
    stats_lines = []
    signal_timings = {}
    time_window = _resolved_time_window_description(scenario, generators=generators, columns=columns)

    for gen in generators:
        csv_path = data_dir / f"{gen}.csv"
        if not csv_path.exists():
            print(f"File missing: {csv_path}")
            continue

        print(f"Generator: {gen}", flush=True)
        df = _read_numeric_csv(csv_path)

        for col, signal in columns.items():
            if col not in df.columns:
                print(f"Column {col} missing in {gen}")
                continue

            print(f"Gen: {gen}, Signal: {signal}", flush=True)
            signal_start = time.perf_counter()
            preprocess_start = time.perf_counter()
            t, y, preprocess_meta = _preprocess_signal(df, col, scenario, gen, signal)
            if t is None or y is None or preprocess_meta is None:
                print(f"Not enough samples for {gen} {signal} after preprocessing window selection")
                continue

            full_prepared = _prepare_filtered_full_signal(df, col, scenario, gen) if _time_cross_config(scenario) else None
            mean_after_detrend = float(full_prepared["mean_after_detrend"]) if full_prepared is not None else float("nan")
            mean_after_lpf = float(full_prepared["mean_after_lpf"]) if full_prepared is not None else float(np.mean(y))
            preprocess_elapsed = time.perf_counter() - preprocess_start

            prepare_start = time.perf_counter()
            prepared_mp = prepare_matrix_pencil(y, t)
            prepare_elapsed = time.perf_counter() - prepare_start

            stats_lines.append({
                "Scenario": name,
                "Gen": gen,
                "Signal": signal,
                "Mean after detrend": mean_after_detrend,
                "Mean after LPF": mean_after_lpf,
                "Effective start [s]": float(preprocess_meta["effective_start_s"]),
                "First zero cross [s]": preprocess_meta["first_zero_cross_s"],
                "Time-cross mode": preprocess_meta["time_cross_mode"],
            })

            fixed_order_elapsed = 0.0
            mp_fit_cache = {}
            fixed_order_details = {}
            auto_order_search_elapsed = 0.0
            auto_order_fit_elapsed = 0.0
            tau_details = {}

            for order in fixed_orders:
                freq, sigma, _, elapsed_time, _, amplitudes = apply_matrix_pencil_fixed_order_prepared(prepared_mp, order=order, fit_cache=mp_fit_cache)
                fixed_order_elapsed += elapsed_time
                fixed_order_details[str(order)] = {
                    "final_fit": _timing_entry(elapsed_time),
                }
                for f, s, amplitude in zip(freq, sigma, amplitudes):
                    if f > MODE_FREQ_EPS_HZ:
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

            tau_order_map, auto_order_details = determine_MP_orders(
                t,
                y,
                taus,
                rate=auto_order_decimation,
                return_details=True,
            )
            auto_order_search_elapsed = auto_order_details["elapsed_time"]

            for tau in taus:
                order = tau_order_map[tau]

                freq, sigma, _, elapsed_time, _, amplitudes = apply_matrix_pencil_fixed_order_prepared(prepared_mp, order=order, fit_cache=mp_fit_cache)
                auto_order_fit_elapsed += elapsed_time
                tau_details[str(tau)] = {
                    "selected_order": int(order),
                    "order_search_shared_across_taus": True,
                    "final_fit": _timing_entry(elapsed_time),
                }
                for f, s, amplitude in zip(freq, sigma, amplitudes):
                    if f > MODE_FREQ_EPS_HZ:
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

            signal_elapsed = time.perf_counter() - signal_start
            signal_timings.setdefault(gen, {})[signal] = {
                "preprocessing": _timing_entry(preprocess_elapsed),
                "prepare_matrix_pencil": _timing_entry(prepare_elapsed),
                "fixed_orders_total": _timing_entry(fixed_order_elapsed),
                "fixed_order_details": fixed_order_details,
                "auto_order_search_total": _timing_entry(auto_order_search_elapsed),
                "auto_order_search": {
                    "timing": _timing_entry(auto_order_details["elapsed_time"]),
                    "orders_tested": int(auto_order_details["orders_tested"]),
                },
                "auto_order_final_fit_total": _timing_entry(auto_order_fit_elapsed),
                "matrix_pencil_total": _timing_entry(prepare_elapsed + fixed_order_elapsed + auto_order_search_elapsed + auto_order_fit_elapsed),
                "total_signal": _timing_entry(signal_elapsed),
                "tau_details": tau_details,
            }

    df_results = pd.DataFrame(results)
    results_path = output_dir / "results.csv"
    df_results.to_csv(results_path, index=False)
    mp_elapsed = time.perf_counter() - mp_start
    analysis_config = _build_analysis_config(
        name=name,
        scenario=scenario,
        data_dir=data_dir,
        output_dir=output_dir,
        generated_config=generated_config,
        generators=generators,
        auto_order_decimation=auto_order_decimation,
        time_window=time_window,
        resolved_time_cross=_resolve_time_cross_summary(scenario),
        signal_means=stats_lines,
        timings={
            "matrix_pencil": _timing_entry(mp_elapsed),
            "per_generator_signal": signal_timings,
        },
    )
    _attach_result_diagnostics(analysis_config, data_dir, scenario, generators, columns, df_results)
    _save_json(output_dir / "analysis_config.json", analysis_config)

    return output_dir, results_path, df_results, analysis_config


def run_clustering_for_scenario(output_dir, results_path, df_results, scenario):
    clustering_config = scenario.get("clustering", {})
    if df_results.empty:
        print(f"No Matrix Pencil results for {output_dir}; skipping clustering.")
        return {}

    timings = {}

    if clustering_config.get("global", True):
        global_out = output_dir / "clustering" / "global"
        timings["global"] = _run_clustering_pipeline(results_path, global_out, reference_modes=IEEE39_REFERENCE_MODES)

    if clustering_config.get("by_control_area", True):
        area_root = output_dir / "clustering" / "by_control_area"
        area_timings = {}
        for area_name, gens in CONTROL_AREAS.items():
            area_out = area_root / area_name
            area_out.mkdir(parents=True, exist_ok=True)
            area_df = df_results[df_results["Gen"].isin(gens)].copy()
            area_reference_modes = _reference_modes_for_control_area(area_name)
            if area_df.empty:
                print(f"No data for {area_name}; skipping.")
                area_timings[area_name] = {"total": _timing_entry(0.0, skipped=True)}
                continue

            area_results_path = area_out / "results.csv"
            area_df.to_csv(area_results_path, index=False)
            _save_json(area_out / "control_area.json", {"name": area_name, "generators": gens})

            area_timings[area_name] = _run_clustering_pipeline(
                area_results_path,
                area_out,
                reference_modes=area_reference_modes,
            )

        _save_ieee39_combined_reference_mad_summary(area_root, IEEE39_REFERENCE_MODES)

        timings["by_control_area"] = area_timings

    return timings


def _save_ieee39_combined_reference_mad_summary(area_root, reference_modes):
    assignment_files = sorted(area_root.glob("area_*/reference_mad/mp_estimates_with_reference_assignment.csv"))
    if not assignment_files:
        return

    combined_dir = area_root / "reference_mad"
    combined_dir.mkdir(parents=True, exist_ok=True)

    assigned_df = pd.concat([pd.read_csv(path) for path in assignment_files], ignore_index=True)
    assigned_df.to_csv(combined_dir / "mp_estimates_with_reference_assignment.csv", index=False)

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

    overall = pd.DataFrame([{
        "Count": int(len(assigned_df)),
        "MAD": float(assigned_df["Distance_to_Reference"].median()),
        "Mean_Distance": float(assigned_df["Distance_to_Reference"].mean()),
        "Max_Distance": float(assigned_df["Distance_to_Reference"].max()),
    }])
    overall.to_csv(combined_dir / "reference_mad_overall.csv", index=False)


def apply_existing_analysis_config(scenario, results_path, args):
    config_path = results_path.parent / "analysis_config.json"
    if not config_path.exists():
        return None

    config = _load_json(config_path)

    if not args.data_dir and config.get("data_dir"):
        scenario["data_dir"] = config["data_dir"]

    if args.time_start is None and args.time_end is None and args.time_cross is None and not args.no_reset_time and config.get("time_mask"):
        scenario["time_mask"] = config["time_mask"]
        if "time_cross" in config:
            scenario["time_cross"] = config.get("time_cross")

    for key in ["filter", "columns", "fixed_orders", "taus", "auto_order_decimation", "generator_subset", "signal_subset"]:
        if key == "fixed_orders" and args.fixed_orders is not None:
            continue
        if key == "taus" and args.taus is not None:
            continue
        if key in config:
            scenario[key] = config[key]

    return config


def load_existing_results_for_scenario(name, scenario, args):
    if args.analysis_dir:
        output_dir = _resolve_path(args.analysis_dir)
        scenario["output_dir"] = str(output_dir)
        scenario["output_dir_explicit"] = True
    else:
        _, output_dir, _, _, _ = _scenario_runtime_config(scenario)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"

    if not results_path.exists():
        raise SystemExit(
            f"Cannot skip Matrix Pencil for '{name}' because results file does not exist: {results_path}"
        )

    config = apply_existing_analysis_config(scenario, results_path, args)

    if config is None:
        print(
            f"Warning: no analysis_config.json found next to {results_path}; "
            "using the current CLI/default time mask for reports and reconstructions.",
            flush=True,
        )

    data_dir, _, generated_config, generators, _ = _scenario_runtime_config(scenario)
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist for '{name}': {data_dir}")
    validate_scenario_time_window(name, scenario, generated_config=generated_config, generators=generators)

    df_results = pd.read_csv(results_path)
    if config is None:
        time_window = _resolved_time_window_description(scenario, generators=generators, columns=scenario.get("columns", COLUMNS))
        config = _build_analysis_config(
            name=name,
            scenario=scenario,
            data_dir=data_dir,
            output_dir=output_dir,
            generated_config=generated_config,
            generators=generators,
            auto_order_decimation=int(scenario.get("auto_order_decimation", scenario.get("order_rate", AUTO_ORDER_DECIMATION))),
            time_window=time_window,
            resolved_time_cross=_resolve_time_cross_summary(scenario),
            signal_means=[],
            timings={"matrix_pencil": _timing_entry(0.0, skipped=True), "per_generator_signal": {}},
        )
    else:
        config.setdefault("timings", {})
        config["timings"].setdefault("matrix_pencil", _timing_entry(0.0, skipped=True))
        config["timings"].setdefault("per_generator_signal", {})

    _attach_result_diagnostics(config, data_dir, scenario, generators, scenario.get("columns", COLUMNS), df_results)

    return output_dir, results_path, df_results, config


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
            time_cross = config.get("time_cross") or {}
            resolved_time_cross = config.get("resolved_time_cross") or {}
            reset = time_mask.get("reset_time", config.get("time_reset_to_zero"))
            start = time_mask.get("start_inclusive", time_window.get("start_s"))
            end = time_mask.get("end_inclusive", time_window.get("end_s", "end"))
            if time_cross and resolved_time_cross.get("mode") == "global":
                start = resolved_time_cross.get("common_start_s", start)
            elif time_cross and resolved_time_cross.get("mode") == "per-signal":
                start_range = resolved_time_cross.get("effective_start_range_s", {})
                start = f"per-signal[{start_range.get('min_s')}, {start_range.get('max_s')}]"
            details.append(f"time_start={start}")
            details.append(f"time_end={end}")
            details.append(f"reset={reset}")
            if time_cross:
                details.append(f"time_cross={time_cross.get('mode')}")
                details.append(f"offset={time_cross.get('offset_s', 0.0)}")
                reference = time_cross.get("reference") or {}
                if reference:
                    details.append(f"reference={reference.get('generator')}:{reference.get('signal')}")

        suffix = f" ({', '.join(details)})" if details else ""
        print(f"{folder.name}{suffix}")


def _scenario_from_results_folder(folder_name):
    results_root = _resolve_path("results")
    data_dir = _resolve_path(f"results/{folder_name}")

    if not data_dir.exists():
        requested = str(folder_name).strip().lower()
        matched_dirs = []
        if results_root.exists():
            for candidate in sorted(path for path in results_root.iterdir() if path.is_dir()):
                candidate_aliases = {candidate.name.lower()}
                scenario_json = candidate / "scenario.json"
                if scenario_json.exists():
                    try:
                        config = _load_json(scenario_json)
                    except Exception:
                        config = None
                    if config:
                        load_name = str(config.get("load_name", "")).strip()
                        if load_name:
                            candidate_aliases.add(load_name.replace(" ", "").lower())
                        dp = config.get("dp_percent")
                        dq = config.get("dq_percent", 0.0)
                        if load_name and dp is not None:
                            candidate_aliases.add(f"{load_name.replace(' ', '').lower()}_p{float(dp):g}_q{float(dq):g}")
                if requested in candidate_aliases:
                    matched_dirs.append(candidate)

        if not matched_dirs:
            return None

        default_alias = f"{requested}_p2_q0"
        default_like = [path for path in matched_dirs if path.name.lower() == default_alias or path.name.lower().startswith(f"{default_alias}_")]
        if len(default_like) == 1:
            data_dir = default_like[0]
        elif len(matched_dirs) == 1:
            data_dir = matched_dirs[0]
        else:
            matches = ", ".join(path.name for path in matched_dirs)
            raise SystemExit(
                f"Ambiguous scenario alias '{folder_name}'. Multiple IEEE39/results folders match this load alias: {matches}. "
                "Use the exact results folder name with --scenario."
            )

    return _scenario_defaults_for_paths(
        data_dir=path_for_metadata(data_dir),
        output_dir=f"analysis/{data_dir.name}",
    )


def select_scenarios(names, allow_custom=False):
    if names == ["all"]:
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
            selected[name] = _scenario_from_results_folder(name) or _scenario_defaults_for_paths(
                data_dir=f"results/{name}",
                output_dir=f"analysis/{name}",
            )
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

        generator_subset = _resolve_generator_subset(args.generators)
        if generator_subset is not None:
            scenario["generator_subset"] = generator_subset
            scenario["generators"] = generator_subset

        signal_subset = _resolve_signal_subset(args.signals)
        if signal_subset is not None:
            scenario["signal_subset"] = list(signal_subset.values())
            scenario["columns"] = signal_subset

        if args.fixed_orders is not None:
            scenario["fixed_orders"] = list(args.fixed_orders)

        if args.taus is not None:
            scenario["taus"] = list(args.taus)

        time_mask = dict(scenario.get("time_mask", {}))
        if args.time_cross is None:
            if args.time_start is not None:
                time_mask.pop("start", None)
                time_mask["start_inclusive"] = args.time_start
            scenario.pop("time_cross", None)
        else:
            time_mask.pop("start", None)
            time_mask.pop("start_inclusive", None)
            reference = _resolve_time_cross_reference(args.time_cross_reference, scenario)
            if args.time_cross == "per-signal" and reference is not None:
                raise SystemExit("--time-cross-reference can only be used with --time-cross global")
            scenario["time_cross"] = {
                "mode": args.time_cross,
                "offset_s": float(args.time_start if args.time_start is not None else DEFAULT_TIME_START_S),
                "signal_source": "detrended_filtered",
                "reference": reference,
            }
        if args.time_end is not None:
            time_mask.pop("end", None)
            time_mask["end_inclusive"] = args.time_end
        time_mask["reset_time"] = not args.no_reset_time
        scenario["time_mask"] = time_mask

        if args.skip_clustering is not False:
            scenario["clustering"] = _default_clustering_config(enabled=False)
        else:
            scenario["clustering"] = _default_clustering_config(enabled=True, scope=args.clustering_scope)


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
        analysis_method, disturbance_type = _resolve_analysis_method(name, scenario, args)
        effective_skip_plots = bool(args.skip_plots) if args.skip_plots is not None else False
        clustering_scope = args.clustering_scope
        if args.skip_clustering is True:
            effective_clustering = _default_clustering_config(enabled=False)
        elif args.skip_clustering is False:
            effective_clustering = _default_clustering_config(enabled=True, scope=clustering_scope)
        else:
            effective_clustering = _default_clustering_config(enabled=True, scope="areas")
        if analysis_method != "n4sid":
            scenario["clustering"] = effective_clustering

        if analysis_method == "n4sid":
            if args.skip_matrix_pencil:
                raise SystemExit("--skip-matrix-pencil cannot be used with ambient N4SID analysis.")
            if args.time_cross is not None:
                raise SystemExit("--time-cross is only supported for Matrix Pencil analysis, not ambient N4SID.")
            if args.signals is None:
                scenario["columns"] = dict(AMBIENT_DEFAULT_SIGNALS)
                scenario["signal_subset"] = list(AMBIENT_DEFAULT_SIGNALS.values())
            if not effective_skip_plots:
                print(f"Ambient N4SID will generate modal maps and reconstruction plots per sweep for '{name}'.", flush=True)

            if args.skip_n4sid:
                output_dir, results_path, df_results, analysis_config = load_existing_ambient_results_for_scenario(name, scenario)
            else:
                output_dir, results_path, df_results, analysis_config = run_ambient_n4sid_for_scenario(name, scenario, args)
            sweep_evaluations = []
            sweep_plotting_seconds = 0.0
            sweep_report_seconds = 0.0
            for sweep in analysis_config.get("sweeps", []):
                sweep_dir = _resolve_path(sweep["output_dir"])
                sweep_results_path = _resolve_path(sweep["results_csv"])
                sweep_df = pd.read_csv(sweep_results_path)
                sweep_config = _load_json(sweep_dir / "analysis_config.json")
                sweep_scenario = {
                    "analysis_method": "n4sid",
                    "data_dir": analysis_config["data_dir"],
                    "output_dir": sweep["output_dir"],
                    "output_dir_explicit": True,
                    "time_mask": sweep_config.get("time_mask", {}),
                    "columns": sweep_config.get("columns", {}),
                    "generators": sweep_config.get("generators_used", []),
                    "n4sid_orders": sweep_config.get("n4sid_orders", []),
                    "ambient_preprocessing": sweep_config.get("ambient_preprocessing", {}),
                }

                report_start = time.perf_counter()
                report = generate_ieee39_comprehensive_report(sweep_df, sweep_scenario)
                report_elapsed = time.perf_counter() - report_start
                sweep_report_seconds += report_elapsed

                plotting_elapsed = 0.0
                if not effective_skip_plots:
                    plotting_start = time.perf_counter()
                    generate_ieee39_plots(sweep_df, report, sweep_scenario)
                    plotting_elapsed = time.perf_counter() - plotting_start
                    sweep_plotting_seconds += plotting_elapsed

                sweep_config.setdefault("timings", {})["comprehensive_report"] = _timing_entry(report_elapsed)
                sweep_config["timings"]["plotting"] = _timing_entry(plotting_elapsed, skipped=effective_skip_plots)
                _save_json(sweep_dir / "analysis_config.json", sweep_config)

                evaluation_payload = update_analysis_config_with_evaluation(sweep_dir)
                sweep_config["evaluation"] = evaluation_payload
                _save_json(sweep_dir / "analysis_config.json", sweep_config)
                sweep_evaluations.append({
                    "name": sweep.get("name"),
                    "output_dir": sweep.get("output_dir"),
                    "evaluation_summary": evaluation_payload.get("summary", {}),
                })

            analysis_config.setdefault("timings", {})["comprehensive_report"] = _timing_entry(sweep_report_seconds)
            analysis_config["timings"]["plotting"] = _timing_entry(sweep_plotting_seconds, skipped=effective_skip_plots)
            analysis_config["evaluation"] = {"sweeps": sweep_evaluations}
        else:
            if args.merge_radius is not None:
                raise SystemExit("--merge-radius is only supported for ambient N4SID analysis.")
            if disturbance_type == "ambient" and _ambient_cli_overrides_requested(args):
                print(
                    f"Ignoring ambient-only CLI flags for '{name}' because --analysis-method resolved to matrix-pencil.",
                    flush=True,
                )

            if args.skip_matrix_pencil:
                output_dir, results_path, df_results, analysis_config = load_existing_results_for_scenario(name, scenario, args)
            else:
                output_dir, results_path, df_results, analysis_config = run_matrix_pencil_for_scenario(name, scenario)

            report_start = time.perf_counter()
            report = generate_ieee39_comprehensive_report(df_results, scenario)
            report_elapsed = time.perf_counter() - report_start
            analysis_config.setdefault("timings", {})["comprehensive_report"] = _timing_entry(report_elapsed)

            plotting_elapsed = 0.0
            if not effective_skip_plots:
                plotting_start = time.perf_counter()
                generate_ieee39_plots(df_results, report, scenario)
                plotting_elapsed = time.perf_counter() - plotting_start
            analysis_config["timings"]["plotting"] = _timing_entry(plotting_elapsed, skipped=effective_skip_plots)

            clustering_enabled = scenario.get("clustering", {}).get("global", False) or scenario.get("clustering", {}).get("by_control_area", False)
            clustering_elapsed = 0.0
            clustering_details = {}
            if scenario.get("clustering", {}).get("global", False) or scenario.get("clustering", {}).get("by_control_area", False):
                clustering_start = time.perf_counter()
                clustering_details = run_clustering_for_scenario(output_dir, results_path, df_results, scenario)
                clustering_elapsed = time.perf_counter() - clustering_start
            analysis_config["timings"]["clustering"] = _timing_entry(clustering_elapsed, skipped=not clustering_enabled)
            analysis_config["timings"]["clustering_details"] = clustering_details

        analysis_config["analysis_method"] = analysis_method

        scenario_elapsed = time.time() - scenario_start
        analysis_config["timings"]["scenario_total"] = _timing_entry(scenario_elapsed)
        _save_json(output_dir / "analysis_config.json", analysis_config)

        if analysis_method != "n4sid":
            evaluation_payload = update_analysis_config_with_evaluation(output_dir)
            analysis_config["evaluation"] = evaluation_payload
        _save_json(output_dir / "analysis_config.json", analysis_config)

        print(
            f"Scenario {name} finished in "
            f"{scenario_elapsed // 60:.0f} minutes and {scenario_elapsed % 60:.1f} seconds",
            flush=True,
        )

    elapsed = time.time() - start
    print("-" * 30, f"Execution Time: {elapsed // 60:.0f} minutes and {elapsed % 60:.1f} seconds", "-" * 30)


if __name__ == "__main__":
    main()
