import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend
import os
from matrix_pencil import apply_matrix_pencil_fixed_order_prepared, determine_MP_order, filter_signal, prepare_matrix_pencil
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
        prepared_mp = prepare_matrix_pencil(signal_col, time_col)

        stats_lines.append({
    "Generator": gen,
    "Signal": signal,
    "Mean after detrend": mean_after_detrend,
    "Mean after LPF": mean_after_lpf,
    "Mean after demean": mean_after_demean
})

        # Fixed Orders
        for order in fixed_orders:
            freq, sigma, _, _, _, a = apply_matrix_pencil_fixed_order_prepared(prepared_mp, order=order)
            
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
            freq, sigma, _, _, _, a = apply_matrix_pencil_fixed_order_prepared(prepared_mp, order=determine_MP_order(time_col, signal_col, tau, rate = 10))
            
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
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(path, "results.csv"), index=False)
df_stats = pd.DataFrame(stats_lines)
df_stats.to_csv(os.path.join(path, "signal_means.csv"), index=False)

# Create plots
generate_preliminary_report_plots(df_results=df_results, output_path=path, csv_path=path, generators=generators, columns=columns)
# Generate statistics
generate_preliminary_report_stats(path)
# Clustering Analysis
res_path = os.path.join(path, "results.csv")
out_path = os.path.join(path, "clustering")
df_for_mad = _load_screened_data(res_path, out_path)
if df_for_mad is not None:
    _save_reference_mad_outputs(df_for_mad, path)
run_kmeans_modal_analysis(res_path, out_path)
run_kmedoids_modal_analysis(res_path, out_path)
run_silhouette_analysis(res_path, out_path)
end_time = time.time()
print("-"*30, f"Execution Time: {(end_time - start_time)//60} minutes and {(end_time - start_time)%60} seconds", "-"*30)
