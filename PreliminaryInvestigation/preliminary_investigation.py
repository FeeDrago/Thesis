import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend
import os
import json
from matrix_pencil import apply_matrix_pencil_fixed_order, determine_MP_order, filter_signal
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


def _build_analysis_config(generators, columns, fixed_orders, taus, signal_means, timings):
    return {
        "name": "preliminary_report",
        "data_dir": ".",
        "output_dir": ".",
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


start_time = time.time()


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

        # Time Mask
        signal_col = df[col].values[mask].copy()

        # No Time  Mask
        # signal_col = df[col].values.copy()
        
        signal_col = detrend(signal_col)
        mean_after_detrend = np.mean(signal_col)
        signal_col = filter_signal(signal_col, time_col, fc=10, N=15)
        mean_after_lpf = np.mean(signal_col)
        signal_col = signal_col - np.mean(signal_col)  
        mean_after_demean = np.mean(signal_col)

        stats_lines.append({
    "Generator": gen,
    "Signal": signal,
    "Mean after detrend": mean_after_detrend,
    "Mean after LPF": mean_after_lpf,
    "Mean after demean": mean_after_demean
})

        # Fixed Orders
        for order in fixed_orders:
            freq, sigma, _, _, _, a = apply_matrix_pencil_fixed_order(signal_col, time_col, order=order)
            
            # Append results
            for f, s, a in zip(freq, sigma, a):
                if f > 0:
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
        for tau in taus:
            freq, sigma, _, _, _, a = apply_matrix_pencil_fixed_order(signal_col, time_col, order=determine_MP_order(time_col, signal_col, tau, rate = 10))
            
            # Append results
            for f, s, a in zip(freq, sigma, a):
                if f > 0:
                    results.append({
                        'Gen': gen,
                        'Signal': signal,
                        'Method': f'Tau {tau}',
                        'Frequency': f,
                        'Damping': s,
                        'Amplitude': np.abs(a),
                        'Phase' : np.angle(a)
                    })
                    
# Save results to CSV
matrix_pencil_elapsed = time.time() - start_time
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(path, "results.csv"), index=False)
df_stats = pd.DataFrame(stats_lines)
df_stats.to_csv(os.path.join(path, "signal_means.csv"), index=False)

# Create plots
plotting_start = time.time()
generate_preliminary_report_plots(df_results=df_results, output_path=path, csv_path=path, generators=generators, columns=columns)
plotting_elapsed = time.time() - plotting_start
# Generate statistics
report_start = time.time()
generate_preliminary_report_stats(path)
report_elapsed = time.time() - report_start
# Clustering Analysis
res_path = os.path.join(path, "results.csv")
out_path = os.path.join(path, "clustering")
clustering_start = time.time()
df_for_mad = _load_screened_data(res_path, out_path)
if df_for_mad is not None:
    _save_reference_mad_outputs(df_for_mad, path)
run_kmeans_modal_analysis(res_path, out_path)
run_kmedoids_modal_analysis(res_path, out_path)
run_silhouette_analysis(res_path, out_path)
clustering_elapsed = time.time() - clustering_start
end_time = time.time()
analysis_config = _build_analysis_config(
    generators=generators,
    columns=columns,
    fixed_orders=fixed_orders,
    taus=taus,
    signal_means=stats_lines,
    timings={
        "matrix_pencil": _timing_entry(matrix_pencil_elapsed),
        "plotting": _timing_entry(plotting_elapsed),
        "comprehensive_report": _timing_entry(report_elapsed),
        "clustering": _timing_entry(clustering_elapsed),
        "scenario_total": _timing_entry(end_time - start_time),
    },
)
_save_json(os.path.join(path, "analysis_config.json"), analysis_config)
print("-"*30, f"Execution Time: {(end_time - start_time)//60} minutes and {(end_time - start_time)%60} seconds", "-"*30)
