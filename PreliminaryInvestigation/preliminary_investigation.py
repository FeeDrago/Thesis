import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend
import os
import json
from pathlib import Path
from matrix_pencil import apply_matrix_pencil_fixed_order_prepared, determine_MP_orders, filter_signal, prepare_matrix_pencil
from mp_plotter import generate_preliminary_report_plots
import time
from stats import generate_preliminary_report_stats
from clustering_analysis import (
    _load_screened_data,
    _save_reference_mad_outputs,
    run_kmeans_modal_analysis,
    run_kmedoids_modal_analysis,
    run_silhouette_analysis,
)


MODE_FREQ_EPS_HZ = 1e-6


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


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _path_for_metadata(path, base_dir):
    return os.path.relpath(str(path), str(base_dir))


def _build_analysis_config(path, generators, columns, fixed_orders, taus, signal_means, timings):
    base_dir = Path(path)
    return {
        "name": "preliminary_report",
        "data_dir": _path_for_metadata(base_dir, base_dir),
        "output_dir": _path_for_metadata(base_dir, base_dir),
        "generators_used": generators,
        "columns": columns,
        "fixed_orders": fixed_orders,
        "taus": taus,
        "filter": {"fc": 10, "N": 15},
        "time_mask": {"start_exclusive": 0.2, "reset_time": True},
        "time_window_s": {"start_s": 0.2, "end_s": None},
        "time_reset_to_zero": True,
        "signal_means": signal_means,
        "timings": timings,
    }


start_time = time.perf_counter()


path = os.path.dirname(os.path.abspath(__file__))
generators = ['g1', 'g2', 'g3', 'g4']
fixed_orders = [2, 4, 6]
taus = [1, 0.1, 0.01]
columns = {
    's:ut in p.u.': 'Voltage',
    's:cur1 in p.u.': 'Current',    
    's:P1 in MW': 'Active Power',
    's:Q1 in Mvar': 'Reactive Power'
}
results = []
stats_lines = []
signal_timings = {}
preprocessed_signals = {}


for gen in generators:

    csv_path = os.path.join(path, f"{gen}.csv")
    if not os.path.exists(csv_path):
        print(f"File missing: {csv_path}\n")
        continue
    print(f"Generator: {gen}")
    df = pd.read_csv(csv_path)

    time_col = df.iloc[:, 0].values

    # Time Mask
    mask = time_col > 0.2
    time_col = time_col[mask].copy()
    time_col = time_col - time_col[0] 


    # No Time Mask
    # time_col = time_col - time_col[0]  

    for col, signal in columns.items():
        if col not in df.columns:
            print(f"Column {col} missing in {gen}")
            continue
        print(f"Gen:{gen}, Signal: {signal}")
        signal_start = time.perf_counter()

        # Time Mask
        signal_col = df[col].values[mask].copy()

        # No Time  Mask
        # signal_col = df[col].values.copy()

        preprocess_start = time.perf_counter()
        signal_col = detrend(signal_col)
        mean_after_detrend = np.mean(signal_col)
        signal_col = filter_signal(signal_col, time_col, fc=10, N=15)
        signal_reference = signal_col.copy()
        mean_after_lpf = np.mean(signal_col)
        signal_col = signal_col - np.mean(signal_col)
        mean_after_demean = np.mean(signal_col)
        preprocess_elapsed = time.perf_counter() - preprocess_start

        preprocessed_signals.setdefault(gen, {})[signal] = {
            "t": time_col.copy(),
            "y_reference": signal_reference.copy(),
            "y_matrix_pencil": signal_col.copy(),
        }

        prepare_start = time.perf_counter()
        prepared_mp = prepare_matrix_pencil(signal_col, time_col)
        prepare_elapsed = time.perf_counter() - prepare_start

        stats_lines.append({
            "Generator": gen,
            "Signal": signal,
            "Mean after detrend": float(mean_after_detrend),
            "Mean after LPF": float(mean_after_lpf),
            "Mean after demean": float(mean_after_demean),
        })

        # Fixed Orders
        mp_fit_cache = {}
        fixed_order_elapsed = 0.0
        fixed_order_details = {}
        for order in fixed_orders:
            freq, sigma, _, elapsed_time, _, a = apply_matrix_pencil_fixed_order_prepared(prepared_mp, order=order, fit_cache=mp_fit_cache)
            fixed_order_elapsed += elapsed_time
            fixed_order_details[str(order)] = {
                "final_fit": _timing_entry(elapsed_time),
            }
            
            # Append results
            for f, s, a in zip(freq, sigma, a):
                if f > MODE_FREQ_EPS_HZ:
                    results.append({
                        'Gen': gen,
                        'Signal': signal,
                        'Method': f'Order {order}',
                        'Frequency': f,
                        'Damping': s,
                        'Amplitude': np.abs(a),
                        'Phase' : np.angle(a)
                    })

        # Automatic Order
        tau_order_search_start = time.perf_counter()
        tau_orders, tau_search_details = determine_MP_orders(time_col, signal_col, taus, rate=10, return_details=True)
        tau_order_search_elapsed = time.perf_counter() - tau_order_search_start
        tau_fit_elapsed = 0.0
        tau_details = {}
        for tau in taus:
            freq, sigma, _, elapsed_time, _, a = apply_matrix_pencil_fixed_order_prepared(prepared_mp, order=tau_orders[tau], fit_cache=mp_fit_cache)
            tau_fit_elapsed += elapsed_time
            tau_details[str(tau)] = {
                "selected_order": int(tau_orders[tau]),
                "order_search_shared_across_taus": True,
                "final_fit": _timing_entry(elapsed_time),
            }
            
            # Append results
            for f, s, a in zip(freq, sigma, a):
                if f > MODE_FREQ_EPS_HZ:
                    results.append({
                        'Gen': gen,
                        'Signal': signal,
                        'Method': f'Tau {tau}',
                        'Frequency': f,
                        'Damping': s,
                        'Amplitude': np.abs(a),
                        'Phase' : np.angle(a)
                    })

        signal_elapsed = time.perf_counter() - signal_start
        signal_timings.setdefault(gen, {})[signal] = {
            "preprocessing": _timing_entry(preprocess_elapsed),
            "prepare_matrix_pencil": _timing_entry(prepare_elapsed),
            "fixed_orders_total": _timing_entry(fixed_order_elapsed),
            "fixed_order_details": fixed_order_details,
            "auto_order_search_total": _timing_entry(tau_order_search_elapsed),
            "auto_order_search": {
                "timing": _timing_entry(tau_search_details["elapsed_time"]),
                "orders_tested": int(tau_search_details["orders_tested"]),
            },
            "auto_order_final_fit_total": _timing_entry(tau_fit_elapsed),
            "matrix_pencil_total": _timing_entry(prepare_elapsed + fixed_order_elapsed + tau_order_search_elapsed + tau_fit_elapsed),
            "total_signal": _timing_entry(signal_elapsed),
            "tau_details": tau_details,
        }
                     
# Save results to CSV
matrix_pencil_elapsed = time.perf_counter() - start_time
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(path, "results.csv"), index=False)

# Create plots
plotting_start = time.perf_counter()
generate_preliminary_report_plots(
    df_results=df_results,
    output_path=path,
    csv_path=path,
    generators=generators,
    columns=columns,
    preprocessed_signals=preprocessed_signals,
)
plotting_elapsed = time.perf_counter() - plotting_start
# Generate statistics
report_start = time.perf_counter()
generate_preliminary_report_stats(path, preprocessed_signals=preprocessed_signals)
report_elapsed = time.perf_counter() - report_start
# Clustering Analysis
res_path = os.path.join(path, "results.csv")
out_path = os.path.join(path, "clustering")
clustering_start = time.perf_counter()
screening_start = time.perf_counter()
df_for_mad = _load_screened_data(res_path, out_path)
screening_elapsed = time.perf_counter() - screening_start
reference_mad_elapsed = 0.0
if df_for_mad is not None:
    reference_mad_start = time.perf_counter()
    _save_reference_mad_outputs(df_for_mad, path)
    reference_mad_elapsed = time.perf_counter() - reference_mad_start
    
kmeans_start = time.perf_counter()
run_kmeans_modal_analysis(res_path, out_path)
kmeans_elapsed = time.perf_counter() - kmeans_start

kmedoids_start = time.perf_counter()
run_kmedoids_modal_analysis(res_path, out_path)
kmedoids_elapsed = time.perf_counter() - kmedoids_start

silhouette_start = time.perf_counter()
run_silhouette_analysis(res_path, out_path)
silhouette_elapsed = time.perf_counter() - silhouette_start
clustering_elapsed = time.perf_counter() - clustering_start

end_time = time.perf_counter()
analysis_config = _build_analysis_config(
    path=path,
    generators=generators,
    columns=columns,
    fixed_orders=fixed_orders,
    taus=taus,
    signal_means=stats_lines,
    timings={
        "matrix_pencil": _timing_entry(matrix_pencil_elapsed),
        "per_generator_signal": signal_timings,
        "plotting": _timing_entry(plotting_elapsed),
        "comprehensive_report": _timing_entry(report_elapsed),
        "clustering": _timing_entry(clustering_elapsed),
        "clustering_details": {
            "screen_and_load": _timing_entry(screening_elapsed),
            "reference_mad": _timing_entry(reference_mad_elapsed, skipped=df_for_mad is None),
            "kmeans": _timing_entry(kmeans_elapsed),
            "kmedoids": _timing_entry(kmedoids_elapsed),
            "silhouette": _timing_entry(silhouette_elapsed),
        },
        "scenario_total": _timing_entry(end_time - start_time),
    },
)
_save_json(os.path.join(path, "analysis_config.json"), analysis_config)
print("-"*30, f"Execution Time: {(end_time - start_time)//60:.0f} minutes and {(end_time - start_time)%60:.1f} seconds", "-"*30)
