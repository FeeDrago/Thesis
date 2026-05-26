import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from matrix_pencil import filter_signal
from plot_style import apply_thesis_style, save_pdf, style_axis, SIGNAL_COLORS
from shared_plotting import (
    generator_modal_label,
    plot_modal_combined_map,
    plot_modal_signal_grid,
    plot_modal_generator_grid,
    plot_reconstruction_method_grid,
)

apply_thesis_style()

RECON_X_LIMS = (0, 50)
RECON_TICK_LABEL_SIZE = 30
RECON_AXIS_LABEL_SIZE = 34

def generate_preliminary_report_plots(df_results, output_path, csv_path, generators, columns, preprocessed_signals=None):
    colors = SIGNAL_COLORS.copy()
    
    plots_path = os.path.join(output_path, "plots")
    modal_maps_path = os.path.join(plots_path, "modal_maps")
    recon_path = os.path.join(plots_path, "reconstruction_grids")
    
    # Create subdirectories for PDF and PNG
    for folder in [modal_maps_path, recon_path]:
        for sub in ["pdf", "png"]:
            d = os.path.join(folder, sub)
            if not os.path.exists(d):
                os.makedirs(d)

    # 1. Sigma vs Frequency Plots
    for gen in generators:
        for signal in columns.values():
            data = df_results[(df_results['Gen'] == gen) & (df_results['Signal'] == signal)]
            if data.empty: continue
            
            plt.figure(figsize=(8, 5))
            plt.scatter(data['Damping'], data['Frequency'], color=colors[signal], label=signal, alpha=0.6, edgecolors='k')
            plt.axvline(0, color='red', linestyle='--', alpha=0.5)
            plt.title(f"Modal Analysis: Generator {gen.upper()} - {signal}")
            plt.xlabel("Damping (Sigma) [rad/s]")
            plt.ylabel("Frequency [Hz]")
            style_axis(plt.gca())
            
            fname = f"{gen}_{signal.replace(' ', '_')}"
            save_pdf(plt, os.path.join(modal_maps_path, "pdf", f"{fname}.pdf"))
            plt.savefig(os.path.join(modal_maps_path, "png", f"{fname}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # Combined plot per generator
    for gen in generators:
        data = df_results[df_results['Gen'] == gen]
        if data.empty:
            continue

        plot_modal_combined_map(
            df_results=df_results,
            output_dir=modal_maps_path,
            filename=f"{gen}_combined",
            title=f"Combined Modal Map: {generator_modal_label(gen)}",
            signals=list(columns.values()),
            gen=gen,
            colors=colors,
            figsize=(10, 6),
        )

    # Adaptive per-generator signal grids
    for gen in generators:
        gen_data = df_results[df_results['Gen'] == gen]
        if gen_data.empty:
            continue

        plot_modal_signal_grid(
            df_results=df_results,
            gen=gen,
            signals=list(columns.values()),
            output_dir=modal_maps_path,
            filename=f"{gen}_2x2_grid",
            title=f"Modal Identification per Signal: {generator_modal_label(gen)}",
            colors=colors,
        )

    plot_modal_generator_grid(
        df_results=df_results,
        generators=generators,
        signals=list(columns.values()),
        output_dir=modal_maps_path,
        filename="All_Generators_Grid",
        title="System-Wide Modal Identification (All Generators)",
        colors=colors,
    )

    # 2. SIGNAL RECONSTRUCTION PLOTS 
    row_configs = [('Order 2', 'Tau 1'), ('Order 4', 'Tau 0.1'), ('Order 6', 'Tau 0.01')]
    inv_columns = {v: k for k, v in columns.items()}

    for gen in generators:
        for signal_label in columns.values():
            cached_signal = None
            if preprocessed_signals is not None:
                cached_signal = preprocessed_signals.get(gen, {}).get(signal_label)

            if cached_signal is not None:
                t = cached_signal["t"]
                y_ref = cached_signal["y_matrix_pencil"]
            else:
                csv_file = os.path.join(csv_path, f"{gen}.csv")
                if not os.path.exists(csv_file):
                    continue
                raw_df = pd.read_csv(csv_file)
                t_raw = raw_df.iloc[:, 0].values
                y_raw = raw_df[inv_columns[signal_label]].values

                # Time Mask
                mask = t_raw > 0.2
                t = t_raw[mask].copy()
                y_proc = y_raw[mask].copy()

                # No Time Mask
                # t = t_raw.copy()
                # y_proc = y_raw.copy()

                t = t - t[0]
                y_ref = filter_signal(detrend(y_proc), t, fc=10)


            plot_reconstruction_method_grid(
                t=t,
                y_ref=y_ref,
                reconstruction_rows=row_configs,
                fetch_modes=lambda method: df_results[(df_results['Gen'] == gen) & (df_results['Signal'] == signal_label) & (df_results['Method'] == method)],
                reconstruct_signal=lambda t_values, modes: np.sum([
                    2 * m['Amplitude'] * np.exp(m['Damping'] * t_values) * np.cos(2 * np.pi * m['Frequency'] * t_values + m['Phase'])
                    for _, m in modes.iterrows()
                ], axis=0) if not modes.empty else np.zeros_like(t_values),
                output_dir=recon_path,
                filename=f"{gen}_{signal_label.replace(' ', '_')}_Reconstruction",
                title=f"Reconstruction Accuracy: {gen.upper()} - {signal_label}\nLeft: Fixed Orders | Right: Adaptive Tau",
                signal=signal_label,
                x_lims=RECON_X_LIMS,
            )

if __name__ == "__main__":
    output_path = os.path.dirname(os.path.abspath(__file__))
    
    generators = ['g1', 'g2', 'g3', 'g4']
    cols = {
        's:ut in p.u.': 'Voltage',
        's:cur1 in p.u.': 'Current',    
        's:P1 in MW': 'Active Power',
        's:Q1 in Mvar': 'Reactive Power'
    }

    if os.path.exists(os.path.join(output_path, "results.csv")):
        df_results = pd.read_csv(os.path.join(output_path, "results.csv"))
        generate_preliminary_report_plots(df_results=df_results, output_path=output_path, csv_path=output_path, generators=generators, columns=cols)
        print("Done.")
    else:
        print(f"Error: results.csv not found.")
