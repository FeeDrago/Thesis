import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend
import os
from matrix_pencil import apply_matrix_pencil_fixed_order, determine_MP_order, filter_signal
from mp_plotter import generate_preliminary_report_plots
import time
from stats import generate_preliminary_report_stats


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

        signal_col = filter_signal(detrend(signal_col), time_col, fc=10, N=15)

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
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(path, "results.csv"), index=False)
print("Results saved.")

# Create plots
generate_preliminary_report_plots(df_results=df_results, output_path=path, csv_path=path, generators=generators, columns=columns)
# Generate statistics
generate_preliminary_report_stats(path)
end_time = time.time()
print("-"*30, f"Execution Time: {(end_time - start_time)//60} minutes and {(end_time - start_time)%60} seconds", "-"*30)