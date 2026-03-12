import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# --- SETUP STYLE ---
sns.set_theme(style="whitegrid") # Clean, professional background
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14})

# --- SETUP PATHS ---
path = path = os.path.dirname(os.path.abspath(__file__))
stats_path = os.path.join(path, "stats")
if not os.path.exists(stats_path):
    os.makedirs(stats_path)

# Load results
df = pd.read_csv(os.path.join(path, "results.csv"))
method_order = ['Order 2', 'Order 4', 'Order 6', 'Tau 1', 'Tau 0.1', 'Tau 0.01']

# --- 1. SAVE SUMMARY CSV ---
summary_df = df.groupby(['Gen', 'Signal', 'Method']).size().reset_index(name='Poles_Count')
summary_df.to_csv(os.path.join(stats_path, "pole_identification_summary.csv"), index=False)

# --- 2. 2D HEATMAP ---
plt.figure(figsize=(10, 7))
heatmap_data = df.groupby(['Gen', 'Signal']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt='d', cbar_kws={'label': 'Count'})
plt.title("Total Identified Poles per Signal & Generator\n(Overall Confidence Map)", fontweight='bold', pad=20)
plt.ylabel("Generator ID", fontweight='bold')
plt.xlabel("Signal Type", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(stats_path, "1_poles_heatmap.png"), dpi=300)
plt.close()

# --- 3. BAR CHART 1: 2x2 GRID (X = Signal) ---
g1 = sns.catplot(
    data=df, kind="count", x="Signal", hue="Method", 
    hue_order=method_order, col="Gen", col_wrap=2, 
    palette="muted", height=5, aspect=1.3, legend_out=True,
    edgecolor="0.2" # Subtle border on bars
)
g1.fig.suptitle("Poles Identified: Grouped by Signal Type", fontsize=18, fontweight='bold', y=1.05)
g1.set_axis_labels("Signal Type", "Pole Count")
g1.set_titles("Generator: {col_name}", fontweight='bold')
# Rotating x-labels for clarity
for ax in g1.axes.flat:
    ax.tick_params(axis='x', rotation=15)

plt.savefig(os.path.join(stats_path, "2_poles_facet_by_signal.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- 4. BAR CHART 2: 2x2 GRID (X = Method) ---
g2 = sns.catplot(
    data=df, kind="count", x="Method", order=method_order,
    hue="Signal", col="Gen", col_wrap=2, 
    palette="muted", height=5, aspect=1.3, legend_out=True,
    edgecolor="0.2"
)
g2.fig.suptitle("Poles Identified: Grouped by Matrix Pencil Method", fontsize=18, fontweight='bold', y=1.05)
g2.set_axis_labels("Method / Parameter", "Pole Count")
g2.set_titles("Generator: {col_name}", fontweight='bold')
for ax in g2.axes.flat:
    ax.tick_params(axis='x', rotation=30) # More rotation for method names

plt.savefig(os.path.join(stats_path, "3_poles_facet_by_method.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- 5. 3D BAR CHART (PRO LOOK) ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

gens = df['Gen'].unique()
sigs = df['Signal'].unique()
gen_map = {val: i for i, val in enumerate(gens)}
sig_map = {val: i for i, val in enumerate(sigs)}

df_3d = df.groupby(['Gen', 'Signal']).size().reset_index(name='Count')
x, y = df_3d['Gen'].map(gen_map).values, df_3d['Signal'].map(sig_map).values
z, dz = np.zeros_like(x), df_3d['Count'].values
dx = dy = 0.4

# Create a color map for bars based on height
colors_3d = plt.cm.viridis(dz / dz.max())
ax.bar3d(x, y, z, dx, dy, dz, color=colors_3d, alpha=0.8, edgecolor='gray', linewidth=0.5)

ax.set_xticks(np.arange(len(gens)) + 0.2)
ax.set_xticklabels(gens, fontweight='bold')
ax.set_yticks(np.arange(len(sigs)) + 0.2)
ax.set_yticklabels(sigs, fontweight='bold')
ax.set_zlabel('Count', fontweight='bold')
ax.set_title('3D Pole Identification Density', fontsize=16, fontweight='bold', pad=20)
ax.view_init(elev=25, azim=45) # Better perspective

plt.savefig(os.path.join(stats_path, "4_3D_counts_overview.png"), dpi=300)
plt.close()

# --- 6. MODAL BUBBLE MAP (FINAL BOSS PLOT) ---
plt.figure(figsize=(14, 8))
df['Gen_Signal'] = df['Gen'].str.upper() + " | " + df['Signal']
# Normalize sizes but keep them visible
min_s, max_s = 60, 1000
norm_amp = (df['Amplitude'] - df['Amplitude'].min()) / (df['Amplitude'].max() - df['Amplitude'].min() + 1e-9)
bubble_sizes = norm_amp * (max_s - min_s) + min_s

scatter = plt.scatter(
    df['Frequency'], df['Gen_Signal'], 
    s=bubble_sizes, c=df['Damping'], 
    cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.8
)

cbar = plt.colorbar(scatter)
cbar.set_label('Damping Coefficient (Sigma)', fontweight='bold')
plt.title("Modal Distribution Map\n(Size = Amplitude Strength | Color = Damping)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Oscillation Frequency [Hz]", fontweight='bold')
plt.ylabel("Data Source (Generator & Signal)", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(stats_path, "5_modal_bubble_map.png"), dpi=300)
plt.close()

print(f"\n[SUCCESS] Stats visualization complete. Files saved in: {stats_path}")