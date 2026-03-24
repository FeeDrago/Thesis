import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from sklearn.metrics import r2_score, mean_squared_error
from matrix_pencil import filter_signal

# Matching LaTeX Serif Font
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["GFS Artemisia", "Times New Roman", "serif"]
})

def generate_preliminary_report_plots(df_results, output_path, csv_path, generators, columns):
    colors = {
        'Voltage': 'tab:blue',
        "Current": 'tab:green',
        "Active Power": 'tab:orange',
        "Reactive Power": 'tab:red'
    }
    
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
            plt.grid(True, linestyle=':', alpha=0.6)
            
            fname = f"{gen}_{signal.replace(' ', '_')}"
            plt.savefig(os.path.join(modal_maps_path, "pdf", f"{fname}.pdf"), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(modal_maps_path, "png", f"{fname}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # Combined plot per generator
    for gen in generators:
        data = df_results[df_results['Gen'] == gen]
        if data.empty: continue
        
        plt.figure(figsize=(10, 6))
        for signal in columns.values():
            sig_data = data[data['Signal'] == signal]
            plt.scatter(sig_data['Damping'], sig_data['Frequency'], label=signal, c=colors[signal], alpha=0.6, edgecolors='k', s=60)
        
        plt.axvline(0, color='red', linestyle='-', alpha=0.3)
        plt.title(f"Combined Modal Map: Generator {gen.upper()}")
        plt.xlabel("Damping (Sigma) [rad/s]")
        plt.ylabel("Frequency [Hz]")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        fname = f"{gen}_combined"
        plt.savefig(os.path.join(modal_maps_path, "pdf", f"{fname}.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(modal_maps_path, "png", f"{fname}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 2x2 Per Generator Plots
    for gen in generators:
        gen_data = df_results[df_results['Gen'] == gen]
        if gen_data.empty: continue
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        fig.suptitle(f"Modal Identification per Signal: Generator {gen.upper()}", fontsize=16, fontweight='bold')
        axes_flat = axes.flatten()
        
        for i, signal in enumerate(columns.values()):
            ax = axes_flat[i]
            sig_data = gen_data[gen_data['Signal'] == signal]
            
            ax.scatter(sig_data['Damping'], sig_data['Frequency'], 
                       color=colors[signal], alpha=0.6, edgecolors='k', s=50)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(signal, fontsize=12, fontweight='semibold')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            if i >= 2: ax.set_xlabel("Damping (Sigma) [rad/s]", fontsize=12)
            if i % 2 == 0: ax.set_ylabel("Frequency [Hz]", fontsize=12)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"{gen}_2x2_grid"
        plt.savefig(os.path.join(modal_maps_path, "pdf", f"{fname}.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(modal_maps_path, "png", f"{fname}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 2x2 Grid for all generators
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("System-Wide Modal Identification (All Generators)", fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()
    for i, gen in enumerate(generators):
        ax = axes_flat[i]
        gen_data = df_results[df_results['Gen'] == gen]
        for signal in columns.values():
            sig_data = gen_data[gen_data['Signal'] == signal]
            ax.scatter(sig_data['Damping'], sig_data['Frequency'], c=colors[signal], alpha=0.5, s=40, edgecolors='none')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f"Generator {gen.upper()}")
        ax.grid(True, linestyle=':', alpha=0.4)
        if i >= 2: ax.set_xlabel("Damping (Sigma)", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Frequency (Hz)", fontsize=12)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=s) for s, c in colors.items()]
    fig.legend(handles=handles, labels=colors.keys(), loc='lower center', ncol=4, title="Signals", fontsize=12, title_fontsize=13)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    fname = "All_Generators_Grid"
    plt.savefig(os.path.join(modal_maps_path, "pdf", f"{fname}.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(modal_maps_path, "png", f"{fname}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. SIGNAL RECONSTRUCTION PLOTS 
    row_configs = [('Order 2', 'Tau 1'), ('Order 4', 'Tau 0.1'), ('Order 6', 'Tau 0.01')]
    inv_columns = {v: k for k, v in columns.items()}

    for gen in generators:
        csv_file = os.path.join(csv_path, f"{gen}.csv")
        if not os.path.exists(csv_file): continue
        raw_df = pd.read_csv(csv_file)

        for signal_label in columns.values():
            t_raw = raw_df.iloc[:, 0].values
            y_raw = raw_df[inv_columns[signal_label]].values
            
            # Time Mask
            # mask = t_raw > 0.1
            # t = t_raw[mask].copy()
            # y_proc = y_raw[mask].copy()

            # No Time Mask
            t = t_raw.copy()
            y_proc = y_raw.copy()
            
            t = t - t[0]  
            y_ref = filter_signal(detrend(y_proc), t, fc=10)

            fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
            fig.suptitle(f"Reconstruction Accuracy: {gen.upper()} - {signal_label}\nLeft: Fixed Orders | Right: Adaptive Tau", 
                         fontsize=18, fontweight='bold', y=0.98)

            for row_idx, (left_meth, right_meth) in enumerate(row_configs):
                for col_idx, method in enumerate([left_meth, right_meth]):
                    ax = axes[row_idx, col_idx]
                    
                    modes = df_results[(df_results['Gen'] == gen) & 
                                       (df_results['Signal'] == signal_label) & 
                                       (df_results['Method'] == method)]
                    
                    if modes.empty:
                        ax.text(0.5, 0.5, "No Data Found", ha='center')
                        continue

                    y_est = np.zeros_like(t)
                    for _, m in modes.iterrows():
                        y_est += 2 * m['Amplitude'] * np.exp(m['Damping'] * t) * \
                                 np.cos(2 * np.pi * m['Frequency'] * t + m['Phase'])
                    
                    r2 = r2_score(y_ref, y_est)
                    rmse = np.sqrt(mean_squared_error(y_ref, y_est))
                    
                    ax.plot(t, y_ref, color='black', alpha=0.3, label='Original (Filtered)')
                    ax.plot(t, y_est, '--', color='red', label=f'MP Estimate ($R^2$={r2:.4f})')
                    ax.set_title(f"Method: {method} (RMSE: {rmse:.2e})", fontsize=11, fontweight='semibold')
                    ax.legend(loc='upper right', fontsize='small')
                    ax.grid(True, linestyle=':', alpha=0.5)
                    
                    if col_idx == 0: ax.set_ylabel("Amplitude")
                    if row_idx == 2: ax.set_xlabel("Time (s)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = f"{gen}_{signal_label.replace(' ', '_')}_Reconstruction"
            plt.savefig(os.path.join(recon_path, "pdf", f"{fname}.pdf"), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(recon_path, "png", f"{fname}.png"), dpi=300, bbox_inches='tight')
            plt.close()

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