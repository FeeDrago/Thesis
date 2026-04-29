import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy.signal import detrend
from sklearn.metrics import r2_score, mean_squared_error
from matrix_pencil import filter_signal
from matplotlib.ticker import MaxNLocator
from plot_style import apply_thesis_style, style_axis


apply_thesis_style()

RECON_X_LIMS = (0, 50)
RECON_TICK_LABEL_SIZE = 30
RECON_AXIS_LABEL_SIZE = 34


def generate_preliminary_report_stats(path):
    # Path configuration
    stats_path = os.path.join(path, "stats")
    pdf_path = os.path.join(stats_path, "pdf")
    png_path = os.path.join(stats_path, "png")

    for p in [pdf_path, png_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    # Data initialization
    df = pd.read_csv(os.path.join(path, "results.csv"))
    gen_id_map = {'g1': 'Generator 1', 'g2': 'Generator 2', 'g3': 'Generator 3', 'g4': 'Generator 4'}
    df['Gen_ID'] = df['Gen'] 
    df['Gen'] = df['Gen_ID'].map(gen_id_map)

    method_order = ['Order 2', 'Order 4', 'Order 6', 'Tau 1', 'Tau 0.1', 'Tau 0.01']
    signals_map = {
        'Voltage': 's:ut in p.u.',
        'Current': 's:cur1 in p.u.',
        'Active Power': 's:P1 in MW',
        'Reactive Power': 's:Q1 in Mvar'
    }

    # Performance analysis
    metrics = []
    for gid, glabel in gen_id_map.items():
        csv_file = os.path.join(path, f"{gid}.csv")
        if not os.path.exists(csv_file): continue
        
        raw_df = pd.read_csv(csv_file)

        # Time Mask
        t_f = raw_df.iloc[:, 0].values
        mask = t_f > 0.2
        t = t_f[mask].copy() - t_f[mask][0]

        # No Time Mask
        # t_f = raw_df.iloc[:, 0].values
        # t = t_f.copy() - t_f[0]


        for sig_l, col in signals_map.items():
            if col not in raw_df.columns: continue

            # Time Mask
            y_ref = filter_signal(detrend(raw_df[col].values[mask]), t, fc=10)
            y_ref = y_ref - np.mean(y_ref)

            # No Time Mask
            # y_ref = filter_signal(detrend(raw_df[col].values), t, fc=10)
            # y_ref = y_ref - np.mean(y_ref)
            
            for meth in method_order:
                modes = df[(df['Gen_ID'] == gid) & (df['Signal'] == sig_l) & (df['Method'] == meth)]
                if modes.empty: continue
                
                y_est = np.zeros_like(t)
                for _, m in modes.iterrows():
                    y_est += 2*m['Amplitude']*np.exp(m['Damping']*t)*np.cos(2*np.pi*m['Frequency']*t + m['Phase'])
                
                r2 = r2_score(y_ref, y_est)
                rmse = np.sqrt(mean_squared_error(y_ref, y_est))
                metrics.append({'Gen': glabel, 'Signal': sig_l, 'Method': meth, 'R2': r2, 'RMSE': rmse, 'Poles': len(modes)})

    df_m = pd.DataFrame(metrics)
    df_m.to_csv(os.path.join(stats_path, "comprehensive_report.csv"), index=False)

    # 1. Heatmap
    plt.figure(figsize=(10, 7))
    h_data = df.groupby(['Gen', 'Signal']).size().unstack(fill_value=0)
    sns.heatmap(h_data, annot=True, cmap="YlGnBu", fmt='d', cbar_kws={'label': 'Poles Count'})
    plt.title("Pole Density Heatmap", fontweight='bold')
    plt.ylabel("Generator")
    style_axis(plt.gca())
    plt.savefig(os.path.join(pdf_path, "1_heatmap.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "1_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

   # 2. Modes per signal grid
    g1 = sns.catplot(data=df, kind="count", x="Signal", hue="Method", hue_order=method_order, 
                    col="Gen", col_wrap=2, palette="muted", height=5, aspect=1.2, edgecolor="0.2", legend_out=True)
    g1.fig.suptitle("Mode Count by Signal Type", fontweight='bold', y=1.05)
    g1.set_titles("{col_name}")
    g1.set_axis_labels("Signal Type", "Mode Count")

    for ax in g1.axes.flat:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        style_axis(ax)
        
    plt.savefig(os.path.join(pdf_path, "2_bar_grid_signal.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "2_bar_grid_signal.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Modes per method grid
    g2 = sns.catplot(data=df, kind="count", x="Method", order=method_order, hue="Signal", 
                    col="Gen", col_wrap=2, palette="muted", height=5, aspect=1.2, edgecolor="0.2", legend_out=True)
    g2.fig.suptitle("Mode Count by Method", fontweight='bold', y=1.05)
    g2.set_titles("{col_name}")
    g2.set_axis_labels("Method", "Mode Count")
    
    for ax in g2.axes.flat: 
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        style_axis(ax)
        ax.tick_params(axis='x', rotation=30)
        style_axis(ax)
        
    plt.savefig(os.path.join(pdf_path, "3_bar_grid_method.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "3_bar_grid_method.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 3D projection
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    gens_u, sigs_u = list(gen_id_map.values()), list(signals_map.keys())
    df_3d = df.groupby(['Gen', 'Signal']).size().reset_index(name='Count')
    x_p = [gens_u.index(g) for g in df_3d['Gen']]
    y_p = [sigs_u.index(s) for s in df_3d['Signal']]
    dz = df_3d['Count'].values
    ax.bar3d(x_p, y_p, np.zeros(len(df_3d)), 0.5, 0.5, dz, color=plt.cm.viridis(dz/dz.max()))
    ax.set_xticks(np.arange(len(gens_u)) + 0.25)
    ax.set_xticklabels(gens_u)
    ax.set_yticks(np.arange(len(sigs_u)) + 0.25)
    ax.set_yticklabels(sigs_u)
    plt.savefig(os.path.join(pdf_path, "4_3D_overview.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "4_3D_overview.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Modal bubble map
    plt.figure(figsize=(14, 9))
    df['Src'] = df['Gen'] + " | " + df['Signal']
    counts = df.groupby('Src').size().reset_index(name='Count')
    df = df.merge(counts, on='Src')
    df['Src'] = df['Src'] + " | Poles: " + df['Count'].astype(str)
    omega = 2 * np.pi * df['Frequency']
    df['Energy'] = 0.5 * (omega**2) * (df['Amplitude']**2)
    norm_e = (df['Energy'] - df['Energy'].min()) / (df['Energy'].max() - df['Energy'].min() + 1e-12)
    plt.scatter(
        df['Frequency'],
        df['Src'],
        s=norm_e * 800 + 100,
        c=df['Damping'],
        cmap='RdYlGn',
        edgecolors='black'
    )
    plt.colorbar().set_label(r'Damping ($\sigma$)')
    plt.title("Modal Frequency/Damping/Energy Map", fontweight='bold')
    style_axis(plt.gca())
    plt.savefig(os.path.join(pdf_path, "5_bubble_map.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "5_bubble_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. R2 boxplot
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_m, x="Method", y="R2", hue="Method", order=method_order, palette="Set2", legend=False)
    plt.title("Method Reliability ($R^2$)", fontweight='bold')
    plt.ylabel("$R^2$ Accuracy Score")
    style_axis(plt.gca())
    if df_m['R2'].min() < 0.5: plt.ylim(0.0, 1.05)
    else: plt.ylim(df_m['R2'].min()*0.98, 1.02)
    plt.savefig(os.path.join(pdf_path, "6_R2_boxplot.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "6_R2_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Pareto chart
    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=df_m, x="Poles", y="R2", hue="Method", style="Gen", s=150)
    plt.title("Accuracy vs Complexity", fontweight='bold')
    plt.ylabel("$R^2$ Score")
    style_axis(plt.gca())
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.savefig(os.path.join(pdf_path, "7_pareto.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "7_pareto.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Method ranking
    best_m = df_m.loc[df_m.groupby(['Gen', 'Signal'])['R2'].idxmax()]
    plt.figure(figsize=(10, 6))
    sns.countplot(data=best_m, x="Method", hue="Method", order=method_order, palette="viridis", legend=False)
    plt.title("Best Method Ranking (Max $R^2$)", fontweight='bold')
    plt.ylabel("Frequency")
    style_axis(plt.gca())
    plt.savefig(os.path.join(pdf_path, "8_ranking.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "8_ranking.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Best Reconstruction 4x4 Grid
    fig, axes = plt.subplots(4, 4, figsize=(28, 22), sharex=True)
    fig.suptitle("Absolute Best Signal Reconstruction (Max $R^2$)", fontweight='bold', y=0.975)

    gens = list(gen_id_map.values())
    sigs = list(signals_map.keys())

    for i, glabel in enumerate(gens):
        gid = [k for k, v in gen_id_map.items() if v == glabel][0]
        csv_file = os.path.join(path, f"{gid}.csv")
        if not os.path.exists(csv_file): 
            for j in range(4): axes[i, j].axis('off')
            continue
        raw_df = pd.read_csv(csv_file)
        
        # Time Mask
        t_f = raw_df.iloc[:, 0].values
        mask = t_f > 0.2
        t = t_f[mask].copy() - t_f[mask][0]
        
        # No Time Mask
        # t_f = raw_df.iloc[:, 0].values
        # t = t_f.copy() - t_f[0]  

        for j, sig_l in enumerate(sigs):
            ax = axes[i, j]
            ax.set_xlim(*RECON_X_LIMS)
            ax.tick_params(axis='both', labelsize=RECON_TICK_LABEL_SIZE)
            col = signals_map[sig_l]
            if col not in raw_df.columns: 
                ax.axis('off')
                continue

            # Time Mask
            y_ref = filter_signal(detrend(raw_df[col].values[mask]), t, fc=10)
            y_ref = y_ref - np.mean(y_ref)
            
            # No Time Mask
            # y_ref = filter_signal(detrend(raw_df[col].values), t, fc=10)
            # y_ref = y_ref - np.mean(y_ref)

            best_row = best_m[(best_m['Gen'] == glabel) & (best_m['Signal'] == sig_l)]
            if best_row.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                continue
            best_method = best_row.iloc[0]['Method']
            best_r2 = best_row.iloc[0]['R2']
            modes = df[(df['Gen'] == glabel) & (df['Signal'] == sig_l) & (df['Method'] == best_method)]
            y_est = np.zeros_like(t)
            for _, m in modes.iterrows():
                y_est += 2 * m['Amplitude'] * np.exp(m['Damping'] * t) * np.cos(2 * np.pi * m['Frequency'] * t + m['Phase'])
            ax.plot(t, y_ref, color='black', alpha=0.3, linewidth=2, label='Original (filtered)')
            ax.plot(t, y_est, '--', color='red', linewidth=1.5, label='MP Estimate')
            ax.set_title(
                f"{glabel} - {sig_l}\nMethod: {best_method} ($R^2$: {best_r2:.4f})",
                fontweight='semibold'
            )
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 3:
                ax.set_xlabel("Time (s)", fontsize=RECON_AXIS_LABEL_SIZE)
            labels_map = {
                'Voltage': r"$\Delta V$ [p.u.]",
                'Current': r"$\Delta \mathrm{I}$ [p.u.]",
                'Active Power': r"$\Delta P$ [MW]",
                'Reactive Power': r"$\Delta Q$ [Mvar]"
            }
            ax.set_ylabel(labels_map.get(sig_l, ""), fontsize=RECON_AXIS_LABEL_SIZE)
            if i == 0 and j == 3: ax.legend(loc='upper right')
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.92, wspace=0.22, hspace=0.42)
    plt.savefig(os.path.join(pdf_path, "9_best_reconstruction_grid.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(png_path, "9_best_reconstruction_grid.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 10. Best Reconstruction 2x2 per Generator
    labels_map = {
        'Voltage': r"$\Delta V$ [p.u.]",
        'Current': r"$\Delta \mathrm{I}$ [p.u.]",
        'Active Power': r"$\Delta P$ [MW]",
        'Reactive Power': r"$\Delta Q$ [Mvar]"
    }

    for glabel in gens:
        gid = [k for k, v in gen_id_map.items() if v == glabel][0]
        csv_file = os.path.join(path, f"{gid}.csv")
        if not os.path.exists(csv_file):
            continue

        raw_df = pd.read_csv(csv_file)

        # Time Mask
        t_f = raw_df.iloc[:, 0].values
        mask = t_f > 0.2
        t = t_f[mask].copy() - t_f[mask][0]

        # No Time Mask
        # t_f = raw_df.iloc[:, 0].values
        # t = t_f.copy() - t_f[0]

        fig_g, axes_g = plt.subplots(2, 2, figsize=(24, 16), sharex=True)
        fig_g.suptitle(f"Absolute Best Signal Reconstruction (Max $R^2$) - {glabel}", fontweight='bold', y=0.99)
        axes_flat = axes_g.flatten()

        for j, sig_l in enumerate(sigs):
            ax = axes_flat[j]
            ax.set_xlim(*RECON_X_LIMS)
            ax.tick_params(axis='both', labelsize=RECON_TICK_LABEL_SIZE)
            col = signals_map[sig_l]
            if col not in raw_df.columns:
                ax.axis('off')
                continue

            # Time Mask
            y_ref = filter_signal(detrend(raw_df[col].values[mask]), t, fc=10)
            y_ref = y_ref - np.mean(y_ref)

            # No Time Mask
            # y_ref = filter_signal(detrend(raw_df[col].values), t, fc=10)
            # y_ref = y_ref - np.mean(y_ref)

            best_row = best_m[(best_m['Gen'] == glabel) & (best_m['Signal'] == sig_l)]
            if best_row.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                continue

            best_method = best_row.iloc[0]['Method']
            best_r2 = best_row.iloc[0]['R2']
            modes = df[(df['Gen'] == glabel) & (df['Signal'] == sig_l) & (df['Method'] == best_method)]
            y_est = np.zeros_like(t)
            for _, m in modes.iterrows():
                y_est += 2 * m['Amplitude'] * np.exp(m['Damping'] * t) * np.cos(2 * np.pi * m['Frequency'] * t + m['Phase'])

            ax.plot(t, y_ref, color='black', alpha=0.3, linewidth=2, label='Original (filtered)')
            ax.plot(t, y_est, '--', color='red', linewidth=1.5, label='MP Estimate')
            ax.set_title(f"{glabel} - {sig_l}\nMethod: {best_method} ($R^2$: {best_r2:.4f})", fontweight='semibold')
            if j >= 2:
                ax.set_xlabel("Time (s)", fontsize=RECON_AXIS_LABEL_SIZE)
            ax.set_ylabel(labels_map.get(sig_l, ""), fontsize=RECON_AXIS_LABEL_SIZE)
            ax.grid(True, linestyle=':', alpha=0.75, linewidth=1.3, color='gray')
            if j == 1:
                ax.legend(loc='upper right')

        fig_g.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.22, hspace=0.36)
        plt.savefig(os.path.join(pdf_path, f"10_best_reconstruction_{gid}_2x2.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(png_path, f"10_best_reconstruction_{gid}_2x2.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_g)

    

if __name__ == "__main__":
    generate_preliminary_report_stats(os.path.dirname(os.path.abspath(__file__)))
    print("Stats done")
