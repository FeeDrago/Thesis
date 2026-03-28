import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap 


CLUSTER_COLORS = [
    "#1f77b4",  
    "#ff7f0e",  
    "#2ca02c",  
    "#9467bd",  
    "#8c564b",  
    "#e377c2",  
    "#7f7f7f",  
    "#17becf",  
    "#808000",  
    "#393b79",  
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["GFS Artemisia", "Times New Roman", "serif"],
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 28,
    "figure.titlesize": 32,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})

def run_kmeans_modal_analysis(results_path, output_path):
    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    base_output = os.path.join(output_path, "kmeans")
    for sub in ["png", "pdf"]:
        d = os.path.join(base_output, sub)
        if not os.path.exists(d):
            os.makedirs(d)

    df = pd.read_csv(results_path)
    X = df[['Frequency', 'Damping']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    k_values = np.arange(1, 11)
    stored_results = {}
    cluster_stats = []

    cmap_to_use = ListedColormap(CLUSTER_COLORS)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        wcss.append(inertia)
        
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        stored_results[k] = (labels, centers, inertia)

        for c in range(k):
            cluster_stats.append({
                'k': k,
                'Cluster': c,
                'Frequency': centers[c, 0],
                'Damping': centers[c, 1],
                'Size': np.sum(labels == c)
            })

        plt.figure(figsize=(10, 7))
        plt.scatter(df['Damping'], df['Frequency'], c=labels, cmap=cmap_to_use, 
                    alpha=0.6, edgecolors='k', s=80) 
        
        plt.scatter(centers[:, 1], centers[:, 0], c='red', marker='x', 
                    s=250, linewidths=4, label='Centroids')
        
        plt.axvline(0, color='red', linestyle='-', alpha=0.3)
        plt.xlabel("Damping (Sigma) [rad/s]")
        plt.ylabel("Frequency [Hz]")
        plt.title(f"Modal Clustering ($k={k}$)\nWCSS: {inertia:.2f}")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        fname = f"kmeans_modal_map_k{k}"
        plt.savefig(os.path.join(base_output, "pdf", f"{fname}.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(base_output, "png", f"{fname}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Knee Detection
    wcss = np.array(wcss)
    p1 = np.array([k_values[0], wcss[0]])
    p2 = np.array([k_values[-1], wcss[-1]])
    distances = []
    for i in range(len(k_values)):
        p3 = np.array([k_values[i], wcss[i]])
        d = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
        distances.append(d)
    
    k_opt_idx = np.argmax(distances)
    k_opt = k_values[k_opt_idx]
    
    grid_ks = [max(2, k_opt-1), k_opt, k_opt+1, k_opt+2]
    grid_ks = [k for k in grid_ks if k in k_values]
    if len(grid_ks) < 4: grid_ks = [2, 3, 4, 5]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("K-Means Parameter Optimization Grid", fontweight='bold')
    axes_flat = axes.flatten()
    
    for i, k in enumerate(grid_ks[:4]):
        ax = axes_flat[i]
        labels, centers, inertia = stored_results[k]
        
        ax.scatter(df['Damping'], df['Frequency'], c=labels, cmap=cmap_to_use, alpha=0.6, s=50, edgecolors='none')
        ax.scatter(centers[:, 1], centers[:, 0], c='red', marker='x', s=120, linewidths=3)
        
        ax.axvline(0, color='red', linestyle='--', alpha=0.3)
        ax.set_title(f"K-Means Results: $k={k}$ | WCSS: {inertia:.1f}", fontweight='semibold')
        ax.grid(True, linestyle=':', alpha=0.4)
        
        if i >= 2: ax.set_xlabel("Damping (Sigma) [rad/s]")
        if i % 2 == 0: ax.set_ylabel("Frequency [Hz]")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(base_output, "pdf", "kmeans_optimization_grid.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(base_output, "png", "kmeans_optimization_grid.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, marker='o', color='tab:blue', linewidth=3, markersize=10)
    plt.scatter(k_opt, wcss[k_opt_idx], color='red', marker='o', s=200, edgecolors='k', zorder=5, 
                label='Optimal Knee Point by Maximum Chord Distance')
    plt.xticks(k_values)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for K-Means Optimization")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(os.path.join(base_output, "pdf", "elbow_method.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(base_output, "png", "elbow_method.png"), dpi=300, bbox_inches='tight')
    plt.close()

    pd.DataFrame(cluster_stats).to_csv(os.path.join(base_output, "cluster_centers_sizes.csv"), index=False)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(base_dir, "results.csv")
    out_path = os.path.join(base_dir, "clustering")
    run_kmeans_modal_analysis(res_path, out_path)