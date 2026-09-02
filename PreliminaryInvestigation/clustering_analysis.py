import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples, v_measure_score
from sklearn.preprocessing import StandardScaler
import kmedoids
from matplotlib.lines import Line2D
from plot_style import (
    apply_thesis_style,
    style_axis,
    save_pdf,
    CLUSTER_COLORS,
    ACCENT_RED,
    LINE_BLUE,
    LINE_GREEN,
    GRID_ALPHA_MAIN,
    GRID_ALPHA_SUB,
    POINT_ALPHA,
    POINT_SIZE,
    GRID_POINT_SIZE,
    REP_SIZE,
    REP_GRID_SIZE,
)

apply_thesis_style()

FREQ_MIN = 0.1
FREQ_MAX = 2.0
DAMPING_MAX = -1e-3
DAMPING_AXIS_LIMS = (-2.0, 0.0)
OPTICS_DEFAULT_SETTINGS = {
    "pm_values": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "xi_values": [round(value, 2) for value in np.arange(0.02, 0.401, 0.02)],
    "multiply_by_orders": True,
    "min_npts": 2,
    "min_assigned_ratio": 0.50,
    "render_all_parameter_maps": False,
}

DBSCAN_DEFAULT_SETTINGS = {
    "pe_values": [round(value, 3) for value in np.arange(0.01, 0.151, 0.005)],
    "pm_values": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "multiply_by_orders": True,
    "min_npts": 2,
    "min_assigned_ratio": 0.50,
}

GMM_DEFAULT_SETTINGS = {
    "covariance_type": "full",
    "init_params": "k-means++",
    "n_init": 10,
    "random_state": 42,
    "max_iter": 100,
    "reg_covar": 1e-4,
}

AGGLOMERATIVE_DEFAULT_SETTINGS = {
    "pe_values": [round(value, 3) for value in np.arange(0.01, 0.151, 0.005)],
    "linkages": ["average", "complete"],
    "metric": "euclidean",
}

HDBSCAN_DEFAULT_SETTINGS = {
    "pe_values": [round(value, 3) for value in np.arange(0.01, 0.151, 0.005)],
    "pm_values": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "multiply_by_orders": True,
    "min_npts": 2,
    "cluster_selection_methods": ["eom", "leaf"],
    "metric": "euclidean",
    "allow_single_cluster": False,
    "copy": True,
}

REFERENCE_MODES = {
    "Inter-area": {"Frequency": 0.540, "Damping": -0.127},
    "Intra-area 1": {"Frequency": 1.083, "Damping": -0.603},
    "Intra-area 2": {"Frequency": 1.119, "Damping": -0.631},
}
MAX_FULL_CLUSTER_LEGEND = 12


def _label_colors(labels):
    return [CLUSTER_COLORS[int(lbl) % len(CLUSTER_COLORS)] for lbl in labels]


def _label_colors_with_noise(labels):
    colors = []
    for label in labels:
        if int(label) < 0:
            colors.append("#9e9e9e")
        else:
            colors.append(CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)])
    return colors


def _apply_axis_style(ax, grid_alpha=GRID_ALPHA_MAIN):
    style_axis(ax, grid_alpha=grid_alpha)


def _save_figure(fig, base_output, filename):
    save_pdf(fig, os.path.join(base_output, "pdf", f"{filename}.pdf"))
    fig.savefig(os.path.join(base_output, "png", f"{filename}.png"), dpi=300)


def _prepare_output_dirs(base_output):
    for sub in ["png", "pdf"]:
        os.makedirs(os.path.join(base_output, sub), exist_ok=True)


def _cluster_legend_handles(k, representative_label=None):
    if k > MAX_FULL_CLUSTER_LEGEND:
        handles = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=CLUSTER_COLORS[0],
                markeredgecolor='k',
                markersize=10,
                label=f"{k} Clusters",
            )
        ]
        if representative_label is not None:
            handles.append(
                Line2D(
                    [0], [0],
                    marker='x',
                    color=ACCENT_RED,
                    linestyle='None',
                    markeredgewidth=3,
                    markersize=11,
                    label=representative_label,
                )
            )
        return handles

    handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            markeredgecolor='k',
            markersize=10,
            label=f"Cluster {i + 1}"
        )
        for i in range(k)
    ]
    if representative_label is not None:
        handles.append(
            Line2D(
                [0], [0],
                marker='x',
                color=ACCENT_RED,
                linestyle='None',
                markeredgewidth=3,
                markersize=11,
                label=representative_label,
            )
        )
    return handles


def _reference_mode_handles(reference_modes):
    if not reference_modes:
        return []
    return [
        Line2D(
            [0], [0],
            marker='D',
            color='k',
            markerfacecolor='white',
            markeredgecolor='k',
            linestyle='None',
            markersize=9,
            label='Reference Modes',
        )
    ]


def _noise_point_handle():
    return [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor="#9e9e9e",
            markeredgecolor='k',
            markersize=10,
            label="Noise points",
        )
    ]


def _overlay_reference_modes(ax, reference_modes):
    if reference_modes is None:
        reference_modes = REFERENCE_MODES
    if not reference_modes:
        return

    ref_names = list(reference_modes.keys())
    ref_freq = [float(reference_modes[name]["Frequency"]) for name in ref_names]
    ref_damping = [float(reference_modes[name]["Damping"]) for name in ref_names]
    ax.scatter(
        ref_damping,
        ref_freq,
        marker='D',
        s=150,
        facecolors='white',
        edgecolors='black',
        linewidths=2.2,
        zorder=6,
    )
    for name, damping, freq in zip(ref_names, ref_damping, ref_freq):
        ax.annotate(
            name,
            (damping, freq),
            xytext=(8, 6),
            textcoords='offset points',
            fontsize=12,
            fontweight='semibold',
            color='black',
        )


def _quantile_bounds(values, lower_q=0.02, upper_q=0.98):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.quantile(arr, lower_q)), float(np.quantile(arr, upper_q))


def _set_modal_axis_limits(ax, df, reference_modes=None, representatives=None, include_all_points=False):
    damping_values = list(df["Damping"].to_numpy(dtype=float))
    freq_values = list(df["Frequency"].to_numpy(dtype=float))

    if representatives is not None and len(representatives) > 0:
        reps = np.asarray(representatives, dtype=float)
        freq_values.extend(reps[:, 0].tolist())
        damping_values.extend(reps[:, 1].tolist())

    if reference_modes:
        for mode_data in reference_modes.values():
            freq_values.append(float(mode_data["Frequency"]))
            damping_values.append(float(mode_data["Damping"]))

    if include_all_points:
        damp_low = float(np.min(damping_values))
        damp_high = float(np.max(damping_values))
        freq_low = float(np.min(freq_values))
        freq_high = float(np.max(freq_values))
    else:
        damp_low, damp_high = _quantile_bounds(damping_values, lower_q=0.02, upper_q=0.98)
        freq_low, freq_high = _quantile_bounds(freq_values, lower_q=0.02, upper_q=0.98)
    if damp_low is None or freq_low is None:
        return

    ref_damping = [float(mode["Damping"]) for mode in (reference_modes or {}).values()]
    ref_freq = [float(mode["Frequency"]) for mode in (reference_modes or {}).values()]
    if ref_damping:
        damp_low = min(damp_low, min(ref_damping))
        damp_high = max(damp_high, max(ref_damping))
    if ref_freq:
        freq_low = min(freq_low, min(ref_freq))
        freq_high = max(freq_high, max(ref_freq))

    damp_span = max(0.02, damp_high - damp_low)
    freq_span = max(0.1, freq_high - freq_low)
    x_pad = max(0.015, 0.12 * damp_span)
    y_pad = max(0.05, 0.08 * freq_span)

    x_min = damp_low - x_pad
    x_max = min(0.02, damp_high + (0.5 * x_pad))
    if x_max <= x_min:
        x_max = x_min + max(0.05, damp_span)
    y_min = max(FREQ_MIN - 0.02, freq_low - y_pad)
    y_max = min(FREQ_MAX + 0.02, freq_high + y_pad)
    if y_max <= y_min:
        y_max = y_min + max(0.2, freq_span)

    ax.set_xlim(DAMPING_AXIS_LIMS[0], DAMPING_AXIS_LIMS[1])
    ax.set_ylim(y_min, y_max)


def _plot_selected_cluster_map(
    ax,
    df,
    labels,
    representatives,
    representative_label,
    title,
    reference_modes=None,
    show_legend=True,
):
    point_colors = _label_colors(labels)
    ax.scatter(
        df['Damping'], df['Frequency'], c=point_colors,
        alpha=POINT_ALPHA, edgecolors='k', linewidths=0.8, s=POINT_SIZE
    )
    ax.scatter(
        representatives[:, 1], representatives[:, 0], c=ACCENT_RED, marker='x',
        s=REP_SIZE, linewidths=4, label=representative_label
    )
    _overlay_reference_modes(ax, reference_modes)
    ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Damping (Sigma) [rad/s]")
    ax.set_ylabel("Frequency [Hz]")
    _set_modal_axis_limits(
        ax,
        df,
        reference_modes=reference_modes,
        representatives=representatives,
        include_all_points=True,
    )
    if show_legend:
        handles = _cluster_legend_handles(len(representatives), representative_label=representative_label) + _reference_mode_handles(reference_modes)
        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(4, len(handles)),
        )
    _apply_axis_style(ax, GRID_ALPHA_SUB)


def _pairwise_distances(X):
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=2))


def _resolve_optics_settings(optics_settings=None):
    settings = dict(OPTICS_DEFAULT_SETTINGS)
    if optics_settings:
        settings.update(optics_settings)
    settings["min_samples_min"] = max(2, int(settings["min_samples_min"]))
    settings["min_samples_max"] = max(settings["min_samples_min"], int(settings["min_samples_max"]))
    settings["xi"] = float(settings["xi"])
    settings["render_all_min_samples_maps"] = bool(settings.get("render_all_min_samples_maps", True))
    settings["render_parameter_sweep_plot"] = bool(settings.get("render_parameter_sweep_plot", True))
    return settings


def _pam_kmedoids(distance_matrix, n_clusters, random_state=42, max_iter=100):
    n_clusters = int(n_clusters)
    max_iter = int(max_iter)

    if n_clusters == 1:
        # FasterPAM needs k >= 2; handle k=1 directly
        totals = distance_matrix.sum(axis=1)
        medoid = int(np.argmin(totals))
        labels = np.zeros(distance_matrix.shape[0], dtype=int)
        return labels, np.array([medoid]), float(totals[medoid])

    result = kmedoids.fasterpam(
        distance_matrix.astype(np.float64),
        n_clusters,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = np.asarray(result.labels, dtype=int)
    medoid_indices = np.asarray(result.medoids, dtype=int)
    cost = float(result.loss)
    return labels, medoid_indices, cost


def _apply_frequency_screening(df, output_path=None):
    df = df.copy()
    n_initial = len(df)

    finite_mask = np.isfinite(df["Frequency"]) & np.isfinite(df["Damping"])
    df = df.loc[finite_mask].copy()
    n_after_finite = len(df)

    freq_mask = (df["Frequency"] >= FREQ_MIN) & (df["Frequency"] <= FREQ_MAX)
    df = df.loc[freq_mask].copy()
    n_after_frequency = len(df)

    damping_mask = df["Damping"] <= DAMPING_MAX
    df = df.loc[damping_mask].copy()
    n_after_damping = len(df)

    summary = pd.DataFrame([
        {"step": "initial_rows", "count": n_initial},
        {"step": "after_finite_numeric_filter", "count": n_after_finite},
        {"step": "removed_non_finite_rows", "count": n_initial - n_after_finite},
        {"step": "after_frequency_screening", "count": n_after_frequency},
        {"step": "removed_out_of_range_frequency_rows", "count": n_after_finite - n_after_frequency},
        {"step": "after_negative_damping_screening", "count": n_after_damping},
        {"step": "removed_non_negative_or_near_zero_damping_rows", "count": n_after_frequency - n_after_damping},
    ])

    if output_path is not None:
        screening_dir = os.path.join(output_path, "screening")
        os.makedirs(screening_dir, exist_ok=True)
        df.to_csv(os.path.join(screening_dir, "screened_results.csv"), index=False)
        summary.to_csv(os.path.join(screening_dir, "screening_summary.csv"), index=False)

    return df, summary


def _load_screened_data(results_path, output_path):
    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return None

    df = pd.read_csv(results_path)
    df, _ = _apply_frequency_screening(df, output_path=output_path)

    if df.empty:
        print("No data left after frequency screening.")
        return None

    return df




def _assign_reference_modes(df, reference_modes=None):
    """
    Assign each MP estimate to the nearest reference eigenvalue and compute
    the 2D distance used in Eq. (26)-style MAD evaluation.
    """
    df = df.copy()
    if reference_modes is None:
        reference_modes = REFERENCE_MODES

    reference_names = list(reference_modes.keys())
    reference_points = np.array([
        [reference_modes[name]["Frequency"], reference_modes[name]["Damping"]]
        for name in reference_names
    ], dtype=float)

    X = df[["Frequency", "Damping"]].to_numpy(dtype=float)
    diffs = X[:, None, :] - reference_points[None, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=2))
    best_idx = np.argmin(distances, axis=1)

    df["Reference_Mode"] = [reference_names[i] for i in best_idx]
    df["Reference_Frequency"] = [reference_points[i, 0] for i in best_idx]
    df["Reference_Damping"] = [reference_points[i, 1] for i in best_idx]
    df["Distance_to_Reference"] = distances[np.arange(len(df)), best_idx]
    return df


def _reference_v_measure(df, labels, reference_modes=None):
    """Compare cluster labels with nearest-PowerFactory-mode labels.

    Density-based noise points are excluded, consistently with the MAD
    aggregation. ``np.nan`` indicates that no locally relevant references
    were supplied.
    """
    if not reference_modes:
        return np.nan

    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(df):
        raise ValueError("Cluster-label count must equal the number of estimates.")
    assigned_mask = labels >= 0
    if not np.any(assigned_mask):
        return np.nan

    reference_labels = _assign_reference_modes(df, reference_modes)["Reference_Mode"].to_numpy()
    return float(v_measure_score(reference_labels[assigned_mask], labels[assigned_mask]))


def _reference_ari(df, labels, reference_modes=None):
    """Compare clustering labels with nearest-reference-mode labels using ARI.

    Density-based noise points are excluded, consistently with the V-measure
    and MAD calculations. The reference labels are a PowerFactory-derived
    proxy rather than independently observed ground-truth labels.
    """
    if not reference_modes:
        return np.nan

    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(df):
        raise ValueError("Cluster-label count must equal the number of estimates.")
    assigned_mask = labels >= 0
    if not np.any(assigned_mask):
        return np.nan

    reference_labels = _assign_reference_modes(df, reference_modes)["Reference_Mode"].to_numpy()
    return float(adjusted_rand_score(reference_labels[assigned_mask], labels[assigned_mask]))


def _complete_reference_mode_summary(summary_df, reference_modes):
    if reference_modes is None:
        return summary_df

    reference_names = list(reference_modes.keys())
    complete_df = pd.DataFrame({
        "Reference_Mode": reference_names,
        "Reference_Frequency": [float(reference_modes[name]["Frequency"]) for name in reference_names],
        "Reference_Damping": [float(reference_modes[name]["Damping"]) for name in reference_names],
    })
    complete_df = complete_df.merge(summary_df, on=["Reference_Mode", "Reference_Frequency", "Reference_Damping"], how="left")
    complete_df["Count"] = complete_df["Count"].fillna(0).astype(int)
    return complete_df


def _save_reference_mad_outputs(df, output_path, reference_modes=None):
    """
    Save MAD summaries exactly in the spirit of Eq. (26) of the reference paper:
    MAD_i = median(|lambda_hat_{i,j} - lambda_i|)

    Here lambda_hat_{i,j} are all screened MP estimates and lambda_i are the
    reference eigenvalues of the Kundur system.
    """
    ref_dir = os.path.join(output_path, "reference_mad")
    os.makedirs(ref_dir, exist_ok=True)

    assigned_df = _assign_reference_modes(df, reference_modes=reference_modes)
    assigned_df.to_csv(
        os.path.join(ref_dir, "mode_estimates_with_reference_assignment.csv"),
        index=False
    )
    assigned_df.to_csv(
        os.path.join(ref_dir, "mp_estimates_with_reference_assignment.csv"),
        index=False
    )

    overall_by_mode = (
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
    overall_by_mode = _complete_reference_mode_summary(overall_by_mode, reference_modes)
    overall_by_mode.to_csv(
        os.path.join(ref_dir, "reference_mad_summary_overall.csv"),
        index=False
    )

    by_method = (
        assigned_df.groupby(["Method", "Reference_Mode"], as_index=False)
        .agg(
            Reference_Frequency=("Reference_Frequency", "first"),
            Reference_Damping=("Reference_Damping", "first"),
            Count=("Distance_to_Reference", "size"),
            MAD=("Distance_to_Reference", "median"),
            Mean_Distance=("Distance_to_Reference", "mean"),
            Max_Distance=("Distance_to_Reference", "max"),
        )
    )
    by_method.to_csv(
        os.path.join(ref_dir, "reference_mad_summary_by_method.csv"),
        index=False
    )

    by_gen_signal = (
        assigned_df.groupby(["Gen", "Signal", "Reference_Mode"], as_index=False)
        .agg(
            Reference_Frequency=("Reference_Frequency", "first"),
            Reference_Damping=("Reference_Damping", "first"),
            Count=("Distance_to_Reference", "size"),
            MAD=("Distance_to_Reference", "median"),
            Mean_Distance=("Distance_to_Reference", "mean"),
            Max_Distance=("Distance_to_Reference", "max"),
        )
    )
    by_gen_signal.to_csv(
        os.path.join(ref_dir, "reference_mad_summary_by_gen_signal.csv"),
        index=False
    )

    pd.DataFrame([{
        "Count": int(len(assigned_df)),
        "MAD": float(assigned_df["Distance_to_Reference"].median()),
        "Mean_Distance": float(assigned_df["Distance_to_Reference"].mean()),
        "Max_Distance": float(assigned_df["Distance_to_Reference"].max()),
    }]).to_csv(
        os.path.join(ref_dir, "reference_mad_overall.csv"),
        index=False
    )

def _unique_grid_ks(k_opt, k_values):
    ordered_candidates = [k_opt - 1, k_opt, k_opt + 1, k_opt + 2]
    grid_ks = []
    valid_set = set(k_values.tolist())
    for k in ordered_candidates:
        if k in valid_set and k not in grid_ks:
            grid_ks.append(int(k))
    if len(grid_ks) < min(4, len(k_values)):
        for k in k_values:
            k = int(k)
            if k not in grid_ks:
                grid_ks.append(k)
            if len(grid_ks) == min(4, len(k_values)):
                break
    return grid_ks



def _save_metrics_summary(base_output, metrics_rows, filename):
    pd.DataFrame(metrics_rows).to_csv(os.path.join(base_output, filename), index=False)


def run_kmeans_modal_analysis(results_path, output_path, reference_modes=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "kmeans")
    _prepare_output_dirs(base_output)

    df = _load_screened_data(results_path, output_path)
    if df is None:
        return

    X = df[['Frequency', 'Damping']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_values = np.arange(1, min(11, len(df) + 1))
    stored_results = {}
    cluster_stats = []
    metrics_rows = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia = float(kmeans.inertia_)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        stored_results[int(k)] = (labels, centers, inertia)

        metrics_rows.append({
            "k": int(k),
            "WCSS": inertia,
            "Reference_V_Measure": _reference_v_measure(df, labels, reference_modes),
            "Reference_ARI": _reference_ari(df, labels, reference_modes),
        })

        for c in range(k):
            cluster_stats.append({
                'k': int(k),
                'Cluster': int(c + 1),
                'Frequency': float(centers[c, 0]),
                'Damping': float(centers[c, 1]),
                'Size': int(np.sum(labels == c))
            })

        fig, ax = plt.subplots(figsize=(11.5, 8.8))
        point_colors = _label_colors(labels)
        ax.scatter(
            df['Damping'], df['Frequency'], c=point_colors,
            alpha=POINT_ALPHA, edgecolors='k', linewidths=0.8, s=POINT_SIZE
        )
        ax.scatter(
            centers[:, 1], centers[:, 0], c=ACCENT_RED, marker='x',
            s=REP_SIZE, linewidths=4, label='Centroids'
        )
        _overlay_reference_modes(ax, reference_modes)

        ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax.set_xlabel("Damping (Sigma) [rad/s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(
            f"Modal Clustering with $k-Means$ ($k={k}$)\nWCSS: {inertia:.2f}",
            fontweight='bold'
        )
        handles = _cluster_legend_handles(k, representative_label='Centroids') + _reference_mode_handles(reference_modes)
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(5, len(handles)),
        )
        _set_modal_axis_limits(
            ax,
            df,
            reference_modes=reference_modes,
            representatives=centers,
            include_all_points=True,
        )
        _apply_axis_style(ax)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.26)
        _save_figure(fig, base_output, f"kmeans_modal_map_k{k}")
        plt.close(fig)

    metrics_df = pd.DataFrame(metrics_rows)
    wcss = metrics_df["WCSS"].to_numpy()

    if len(k_values) >= 2:
        p1 = np.array([k_values[0], wcss[0]])
        p2 = np.array([k_values[-1], wcss[-1]])
        distances = []
        for i in range(len(k_values)):
            p3 = np.array([k_values[i], wcss[i]])
            v = p2 - p1
            w = p3 - p1
            d = np.abs(v[0] * w[1] - v[1] * w[0]) / np.linalg.norm(v)
            distances.append(d)
        k_opt_idx = int(np.argmax(distances))
        k_opt = int(k_values[k_opt_idx])
    else:
        k_opt_idx = 0
        k_opt = int(k_values[0])

    grid_ks = _unique_grid_ks(k_opt, k_values)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("$k-Means$ Parameter Optimization Grid", fontweight='bold')
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        if idx >= len(grid_ks):
            ax.axis("off")
            continue

        k = grid_ks[idx]
        labels, centers, inertia = stored_results[k]

        point_colors = _label_colors(labels)
        ax.scatter(df['Damping'], df['Frequency'], c=point_colors, alpha=POINT_ALPHA, s=GRID_POINT_SIZE,
                   edgecolors='k', linewidths=0.5)
        ax.scatter(centers[:, 1], centers[:, 0], c=ACCENT_RED, marker='x', s=REP_GRID_SIZE, linewidths=3)
        _overlay_reference_modes(ax, reference_modes)

        ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax.set_title(
            f"$k-Means$ Results: $k={k}$\nWCSS: {inertia:.1f}",
            fontweight='semibold'
        )
        _apply_axis_style(ax, GRID_ALPHA_SUB)
        _set_modal_axis_limits(ax, df, reference_modes=reference_modes, representatives=centers)

        if idx >= 2:
            ax.set_xlabel("Damping (Sigma) [rad/s]")
        if idx % 2 == 0:
            ax.set_ylabel("Frequency [Hz]")

    handles = _cluster_legend_handles(max(grid_ks), representative_label="Centroids") + _reference_mode_handles(reference_modes)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=min(4, len(handles)))
    fig.tight_layout(rect=[0, 0.16, 1, 0.95])
    _save_figure(fig, base_output, "kmeans_optimization_grid")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, wcss, marker='o', color=LINE_BLUE, linewidth=3, markersize=10)
    ax.scatter(
        k_opt, wcss[k_opt_idx], color=ACCENT_RED, marker='o', s=200,
        edgecolors='k', zorder=5, label='Optimal Knee Point by Maximum Chord Distance'
    )
    ax.set_xticks(k_values)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method for $k-Means$ Optimization", fontweight='bold')
    ax.legend()
    _apply_axis_style(ax)
    _save_figure(fig, base_output, "elbow_method")
    plt.close(fig)

    labels_opt, centers_opt, inertia_opt = stored_results[k_opt]
    fig = plt.figure(figsize=(19, 9.5))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.35],
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.24,
        wspace=0.20,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1.plot(k_values, wcss, marker='o', color=LINE_BLUE, linewidth=3, markersize=10)
    ax1.scatter(
        k_opt, wcss[k_opt_idx], color=ACCENT_RED, marker='o', s=200,
        edgecolors='k', zorder=5, label='Optimal Knee Point by Maximum Chord Distance'
    )
    ax1.set_xticks(k_values)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("WCSS")
    ax1.set_title("Elbow Method for $k$-Means Optimization", fontweight='bold')
    ax1.legend(loc='upper right')
    _apply_axis_style(ax1)

    _plot_selected_cluster_map(
        ax2,
        df,
        labels_opt,
        centers_opt,
        'Centroids',
        f"$k$-Means Cluster Map ($k={k_opt}$)\nWCSS: {inertia_opt:.2f}",
        reference_modes=reference_modes,
        show_legend=False,
    )

    handles = _cluster_legend_handles(k_opt, representative_label='Centroids') + _reference_mode_handles(reference_modes)
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.07),
        ncol=min(5, len(handles)),
    )
    _save_figure(fig, base_output, "elbow_selected_kmeans")
    plt.close(fig)

    metrics_df["k_selected_by_max_chord"] = metrics_df["k"] == k_opt
    metrics_df.to_csv(os.path.join(base_output, "kmeans_metrics_summary.csv"), index=False)
    pd.DataFrame(cluster_stats).to_csv(os.path.join(base_output, "cluster_centers_sizes.csv"), index=False)
    _, selected_cluster_rows = _cluster_representatives(df, labels_opt)
    _collect_paper_mad_assignments(df, labels_opt, selected_cluster_rows, reference_modes, paper_mad_collector)


def run_kmedoids_modal_analysis(results_path, output_path, reference_modes=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "kmedoids")
    _prepare_output_dirs(base_output)

    df = _load_screened_data(results_path, output_path)
    if df is None:
        return

    X = df[['Frequency', 'Damping']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    distance_matrix = _pairwise_distances(X_scaled)

    max_k = min(10, len(df))
    k_values = np.arange(1, max_k + 1)
    stored_results = {}
    cluster_stats = []
    metrics_rows = []

    for k in k_values:
        labels, medoid_indices, cost = _pam_kmedoids(distance_matrix, n_clusters=k, random_state=42)
        cost = float(cost)
        medoids = scaler.inverse_transform(X_scaled[medoid_indices])
        stored_results[int(k)] = (labels, medoids, cost)

        metrics_rows.append({
            "k": int(k),
            "Cost": cost,
            "Reference_V_Measure": _reference_v_measure(df, labels, reference_modes),
            "Reference_ARI": _reference_ari(df, labels, reference_modes),
        })

        for c in range(k):
            cluster_stats.append({
                'k': int(k),
                'Cluster': int(c + 1),
                'Frequency': float(medoids[c, 0]),
                'Damping': float(medoids[c, 1]),
                'Size': int(np.sum(labels == c))
            })

        fig, ax = plt.subplots(figsize=(11.5, 8.8))
        point_colors = _label_colors(labels)
        ax.scatter(
            df['Damping'], df['Frequency'],
            c=point_colors, alpha=POINT_ALPHA,
            edgecolors='k', linewidths=0.8, s=POINT_SIZE
        )
        ax.scatter(
            medoids[:, 1], medoids[:, 0],
            c=ACCENT_RED, marker='x',
            s=REP_SIZE, linewidths=4, label='Medoids'
        )
        _overlay_reference_modes(ax, reference_modes)

        ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax.set_xlabel("Damping (Sigma) [rad/s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(
            f"Modal Clustering with $k-Medoids$ ($k={k}$)\nCost: {cost:.2f}",
            fontweight='bold'
        )
        handles = _cluster_legend_handles(k, representative_label='Medoids') + _reference_mode_handles(reference_modes)
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(5, len(handles)),
        )
        _set_modal_axis_limits(
            ax,
            df,
            reference_modes=reference_modes,
            representatives=medoids,
            include_all_points=True,
        )
        _apply_axis_style(ax)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.26)
        _save_figure(fig, base_output, f"kmedoids_modal_map_k{k}")
        plt.close(fig)

    metrics_df = pd.DataFrame(metrics_rows)
    costs = metrics_df["Cost"].to_numpy()

    if len(k_values) >= 2:
        p1 = np.array([k_values[0], costs[0]])
        p2 = np.array([k_values[-1], costs[-1]])
        distances = []
        for i in range(len(k_values)):
            p3 = np.array([k_values[i], costs[i]])
            v = p2 - p1
            w = p3 - p1
            d = np.abs(v[0] * w[1] - v[1] * w[0]) / np.linalg.norm(v)
            distances.append(d)
        k_opt_idx = int(np.argmax(distances))
        k_opt = int(k_values[k_opt_idx])
    else:
        k_opt_idx = 0
        k_opt = int(k_values[0])

    grid_ks = _unique_grid_ks(k_opt, k_values)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("$k-Medoids$ Parameter Optimization Grid", fontweight='bold')
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        if idx >= len(grid_ks):
            ax.axis("off")
            continue

        k = grid_ks[idx]
        labels, medoids, cost = stored_results[k]

        point_colors = _label_colors(labels)
        ax.scatter(df['Damping'], df['Frequency'], c=point_colors, alpha=POINT_ALPHA, s=GRID_POINT_SIZE,
                   edgecolors='k', linewidths=0.5)
        ax.scatter(medoids[:, 1], medoids[:, 0], c=ACCENT_RED, marker='x', s=REP_GRID_SIZE, linewidths=3)
        _overlay_reference_modes(ax, reference_modes)

        ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax.set_title(
            f"$k-Medoids$ Results: $k={k}$\nCost: {cost:.1f}",
            fontweight='semibold'
        )
        _apply_axis_style(ax, GRID_ALPHA_SUB)
        _set_modal_axis_limits(ax, df, reference_modes=reference_modes, representatives=medoids)

        if idx >= 2:
            ax.set_xlabel("Damping (Sigma) [rad/s]")
        if idx % 2 == 0:
            ax.set_ylabel("Frequency [Hz]")

    handles = _cluster_legend_handles(max(grid_ks), representative_label="Medoids") + _reference_mode_handles(reference_modes)
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, len(handles)),
    )
    fig.tight_layout(rect=[0, 0.16, 1, 0.95])
    _save_figure(fig, base_output, "kmedoids_optimization_grid")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, costs, marker='o', color=LINE_BLUE, linewidth=3, markersize=10)
    ax.scatter(
        k_opt, costs[k_opt_idx], color=ACCENT_RED, marker='o', s=200,
        edgecolors='k', zorder=5, label='Optimal Knee Point by Maximum Chord Distance'
    )
    ax.set_xticks(k_values)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Total Medoid Distance")
    ax.set_title("Elbow-Like Method for $k-Medoids$ Optimization", fontweight='bold')
    ax.legend()
    _apply_axis_style(ax)
    _save_figure(fig, base_output, "kmedoids_elbow_method")
    plt.close(fig)

    labels_opt, medoids_opt, cost_opt = stored_results[k_opt]
    fig = plt.figure(figsize=(19, 9.5))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.35],
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.24,
        wspace=0.20,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1.plot(k_values, costs, marker='o', color=LINE_BLUE, linewidth=3, markersize=10)
    ax1.scatter(
        k_opt, costs[k_opt_idx], color=ACCENT_RED, marker='o', s=200,
        edgecolors='k', zorder=5, label='Optimal Knee Point by Maximum Chord Distance'
    )
    ax1.set_xticks(k_values)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Total Medoid Distance")
    ax1.set_title("Elbow-Like Method for $k$-Medoids Optimization", fontweight='bold')
    ax1.legend(loc='upper right')
    _apply_axis_style(ax1)

    _plot_selected_cluster_map(
        ax2,
        df,
        labels_opt,
        medoids_opt,
        'Medoids',
        f"$k$-Medoids Cluster Map ($k={k_opt}$)\nCost: {cost_opt:.2f}",
        reference_modes=reference_modes,
        show_legend=False,
    )

    handles = _cluster_legend_handles(k_opt, representative_label='Medoids') + _reference_mode_handles(reference_modes)
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.07),
        ncol=min(5, len(handles)),
    )
    _save_figure(fig, base_output, "elbow_selected_kmedoids")
    plt.close(fig)

    metrics_df["k_selected_by_max_chord"] = metrics_df["k"] == k_opt
    metrics_df.to_csv(os.path.join(base_output, "kmedoids_metrics_summary.csv"), index=False)
    pd.DataFrame(cluster_stats).to_csv(os.path.join(base_output, "cluster_medoids_sizes.csv"), index=False)
    _, selected_cluster_rows = _cluster_representatives(df, labels_opt)
    _collect_paper_mad_assignments(df, labels_opt, selected_cluster_rows, reference_modes, paper_mad_collector)


def run_optics_modal_analysis(results_path, output_path, reference_modes=None, optics_settings=None):
    base_output = os.path.join(output_path, "optics")
    _prepare_output_dirs(base_output)

    df_screened = _load_screened_data(results_path, output_path)
    if df_screened is None:
        return

    resolved_optics_settings = _resolve_optics_settings(optics_settings)
    optics_df = df_screened.reset_index(drop=True)

    if len(optics_df) < 3:
        print("Not enough samples for OPTICS clustering.")
        return

    X = optics_df[["Frequency", "Damping"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    min_min_samples = int(resolved_optics_settings["min_samples_min"])
    max_min_samples = min(int(resolved_optics_settings["min_samples_max"]), len(optics_df) - 1)
    if max_min_samples < min_min_samples:
        print("Not enough samples for OPTICS clustering.")
        return

    min_samples_values = np.arange(min_min_samples, max_min_samples + 1)
    stored_results = {}
    metrics_rows = []

    for min_samples in min_samples_values:
        optics = OPTICS(
            min_samples=int(min_samples),
            cluster_method="xi",
            xi=float(resolved_optics_settings["xi"]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            labels = optics.fit_predict(X_scaled)
        unique_labels = sorted(lbl for lbl in np.unique(labels) if int(lbl) >= 0)
        noise_count = int(np.sum(labels == -1))
        n_clusters = int(len(unique_labels))

        representatives = []
        cluster_stats = []
        for cluster_label in unique_labels:
            cluster_points = optics_df.loc[labels == cluster_label, ["Frequency", "Damping"]].to_numpy(dtype=float)
            representative = np.mean(cluster_points, axis=0)
            representatives.append(representative)
            cluster_stats.append({
                "min_samples": int(min_samples),
                "Cluster": int(cluster_label + 1),
                "Frequency": float(representative[0]),
                "Damping": float(representative[1]),
                "Size": int(np.sum(labels == cluster_label)),
            })

        representatives = np.array(representatives, dtype=float) if representatives else np.empty((0, 2), dtype=float)
        stored_results[int(min_samples)] = {
            "labels": labels,
            "representatives": representatives,
            "n_clusters": n_clusters,
            "noise_count": noise_count,
            "cluster_stats": cluster_stats,
        }

        metrics_rows.append({
            "min_samples": int(min_samples),
            "Clusters": n_clusters,
            "NoisePoints": noise_count,
            "AssignedPoints": int(len(optics_df) - noise_count),
            "AssignedRatio": float((len(optics_df) - noise_count) / len(optics_df)),
            "Fragmentation": float(n_clusters / max(len(optics_df) - noise_count, 1)),
            "Xi": float(resolved_optics_settings["xi"]),
        })

        if bool(resolved_optics_settings.get("render_all_min_samples_maps", True)):
            fig, ax = plt.subplots(figsize=(11.5, 8.8))
            point_colors = _label_colors_with_noise(labels)
            ax.scatter(
                optics_df["Damping"], optics_df["Frequency"],
                c=point_colors, alpha=POINT_ALPHA,
                edgecolors='k', linewidths=0.8, s=POINT_SIZE
            )
            if len(representatives) > 0:
                ax.scatter(
                    representatives[:, 1], representatives[:, 0],
                    c=ACCENT_RED, marker='x',
                    s=REP_SIZE, linewidths=4, label='Cluster Means'
                )
            _overlay_reference_modes(ax, reference_modes)
            ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
            ax.set_xlabel("Damping (Sigma) [rad/s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_title(
                f"Modal Clustering with OPTICS ($min\\_samples={min_samples}$)\nClusters: {n_clusters} | Noise: {noise_count}",
                fontweight='bold'
            )
            handles = _noise_point_handle()
            if n_clusters > 0:
                handles += _cluster_legend_handles(n_clusters, representative_label='Cluster Means')
            handles += _reference_mode_handles(reference_modes)
            if handles:
                fig.legend(
                    handles=handles,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 0.015),
                    ncol=min(5, len(handles)),
                )
            _set_modal_axis_limits(
                ax,
                optics_df,
                reference_modes=reference_modes,
                representatives=representatives if len(representatives) > 0 else None,
                include_all_points=True,
            )
            _apply_axis_style(ax)
            fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.26)
            _save_figure(fig, base_output, f"optics_modal_map_min_samples_{min_samples}")
            plt.close(fig)

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        return

    eligible_mask = metrics_df["AssignedRatio"] >= 0.5
    candidate_df = metrics_df[eligible_mask].copy()
    if candidate_df.empty:
        candidate_df = metrics_df.copy()

    candidate_df = candidate_df.sort_values(
        ["Fragmentation", "Clusters", "NoisePoints", "min_samples"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    best_idx = int(candidate_df.index[0])
    best_min_samples = int(metrics_df.loc[best_idx, "min_samples"])
    metrics_df["selected"] = metrics_df["min_samples"] == best_min_samples
    metrics_df.to_csv(os.path.join(base_output, "optics_metrics_summary.csv"), index=False)

    selected = stored_results[best_min_samples]
    cluster_rows = selected["cluster_stats"]
    pd.DataFrame(cluster_rows).to_csv(os.path.join(base_output, "cluster_representatives_sizes.csv"), index=False)

    if bool(resolved_optics_settings.get("render_parameter_sweep_plot", True)):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_df["min_samples"], metrics_df["Clusters"], marker='o', color=LINE_BLUE, linewidth=3, markersize=10, label='Clusters')
        ax.plot(metrics_df["min_samples"], metrics_df["NoisePoints"], marker='s', color=LINE_GREEN, linewidth=3, markersize=10, label='Noise points')
        chosen_row = metrics_df[metrics_df["selected"]].iloc[0]
        ax.scatter(chosen_row["min_samples"], chosen_row["Clusters"], color=ACCENT_RED, s=180, edgecolors='k', zorder=5)
        ax.set_xlabel("OPTICS min_samples")
        ax.set_ylabel("Count")
        ax.set_title("OPTICS Parameter Sweep", fontweight='bold')
        ax.legend(loc='best')
        _apply_axis_style(ax)
        _save_figure(fig, base_output, "optics_parameter_sweep")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    point_colors = _label_colors_with_noise(selected["labels"])
    ax.scatter(
        optics_df["Damping"], optics_df["Frequency"],
        c=point_colors, alpha=POINT_ALPHA,
        edgecolors='k', linewidths=0.8, s=POINT_SIZE
    )
    if len(selected["representatives"]) > 0:
        ax.scatter(
            selected["representatives"][:, 1], selected["representatives"][:, 0],
            c=ACCENT_RED, marker='x', s=REP_SIZE, linewidths=4, label='Cluster Means'
        )
    _overlay_reference_modes(ax, reference_modes)
    ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
    ax.set_xlabel("Damping (Sigma) [rad/s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(
        f"Selected OPTICS Cluster Map ($min\\_samples={best_min_samples}$)\nClusters: {selected['n_clusters']} | Noise: {selected['noise_count']}",
        fontweight='bold'
    )
    handles = _noise_point_handle()
    if selected["n_clusters"] > 0:
        handles += _cluster_legend_handles(selected["n_clusters"], representative_label='Cluster Means')
    handles += _reference_mode_handles(reference_modes)
    if handles:
        ax.legend(handles=handles, loc='upper left')
    _set_modal_axis_limits(ax, optics_df, reference_modes=reference_modes, representatives=selected["representatives"])
    _apply_axis_style(ax)
    _save_figure(fig, base_output, "optics_selected_cluster_map")
    plt.close(fig)

def _resolve_dbscan_settings(dbscan_settings=None):
    settings = dict(DBSCAN_DEFAULT_SETTINGS)
    if dbscan_settings:
        settings.update(dbscan_settings)
    settings["pe"] = float(settings["pe"])
    settings["pm"] = float(settings["pm"])
    settings["multiply_by_orders"] = bool(settings["multiply_by_orders"])
    settings["min_npts"] = max(2, int(settings["min_npts"]))
    return settings


def _dbscan_epsilon(settings):
    sigma_min, sigma_max = DAMPING_AXIS_LIMS
    omega_min = 2.0 * np.pi * FREQ_MIN
    omega_max = 2.0 * np.pi * FREQ_MAX
    d_c = 0.5 * np.sqrt((sigma_max - sigma_min) ** 2 + (omega_max - omega_min) ** 2)
    return float(settings["pe"] * d_c)


def _dbscan_min_pts(n_signals, n_orders, settings):
    scale = float(n_signals)
    if settings["multiply_by_orders"]:
        scale *= float(max(1, int(n_orders)))
    return max(int(settings["min_npts"]), int(np.ceil(settings["pm"] * scale)))


def _dbscan_order_count(df):
    if "Order" in df.columns:
        return int(df["Order"].nunique())
    if "Method" in df.columns:
        return int(df["Method"].nunique())
    return 1


def run_dbscan_modal_analysis(results_path, output_path, reference_modes=None, dbscan_settings=None):
    base_output = os.path.join(output_path, "dbscan")
    _prepare_output_dirs(base_output)

    df_screened = _load_screened_data(results_path, output_path)
    if df_screened is None:
        return

    resolved_dbscan_settings = _resolve_dbscan_settings(dbscan_settings)
    dbscan_df = df_screened.reset_index(drop=True)

    if len(dbscan_df) < 3:
        print("Not enough samples for DBSCAN clustering.")
        return

    if {"Gen", "Signal"}.issubset(dbscan_df.columns):
        n_signals = int(dbscan_df.groupby(["Gen", "Signal"]).ngroups)
    else:
        n_signals = 1
    n_orders = _dbscan_order_count(dbscan_df)

    epsilon = _dbscan_epsilon(resolved_dbscan_settings)
    min_pts = _dbscan_min_pts(n_signals, n_orders, resolved_dbscan_settings)

    X = np.column_stack([
        dbscan_df["Damping"].to_numpy(dtype=float),
        2.0 * np.pi * dbscan_df["Frequency"].to_numpy(dtype=float),
    ])

    labels = DBSCAN(eps=epsilon, min_samples=min_pts).fit_predict(X)
    unique_labels = sorted(lbl for lbl in np.unique(labels) if int(lbl) >= 0)
    noise_count = int(np.sum(labels == -1))
    n_clusters = int(len(unique_labels))

    representatives = []
    cluster_stats = []
    for cluster_label in unique_labels:
        cluster_points = dbscan_df.loc[labels == cluster_label, ["Frequency", "Damping"]].to_numpy(dtype=float)
        representative = np.mean(cluster_points, axis=0)
        representatives.append(representative)
        cluster_stats.append({
            "Cluster": int(cluster_label + 1),
            "Frequency": float(representative[0]),
            "Damping": float(representative[1]),
            "Size": int(np.sum(labels == cluster_label)),
        })

    representatives = np.array(representatives, dtype=float) if representatives else np.empty((0, 2), dtype=float)

    pd.DataFrame(cluster_stats).to_csv(
        os.path.join(base_output, "cluster_representatives_sizes.csv"), index=False
    )
    pd.DataFrame([{
        "Epsilon": epsilon,
        "MinPts": int(min_pts),
        "pe": resolved_dbscan_settings["pe"],
        "pm": resolved_dbscan_settings["pm"],
        "Nsignals": int(n_signals),
        "NOrders": int(n_orders),
        "MultiplyByOrders": resolved_dbscan_settings["multiply_by_orders"],
        "Clusters": n_clusters,
        "NoisePoints": noise_count,
        "AssignedPoints": int(len(dbscan_df) - noise_count),
        "AssignedRatio": float((len(dbscan_df) - noise_count) / len(dbscan_df)),
        "TotalPoints": int(len(dbscan_df)),
    }]).to_csv(os.path.join(base_output, "dbscan_metrics_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(11.5, 8.8))
    point_colors = _label_colors_with_noise(labels)
    ax.scatter(
        dbscan_df["Damping"], dbscan_df["Frequency"],
        c=point_colors, alpha=POINT_ALPHA,
        edgecolors='k', linewidths=0.8, s=POINT_SIZE
    )
    if len(representatives) > 0:
        ax.scatter(
            representatives[:, 1], representatives[:, 0],
            c=ACCENT_RED, marker='x',
            s=REP_SIZE, linewidths=4, label='Cluster Means'
        )
    _overlay_reference_modes(ax, reference_modes)
    ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
    ax.set_xlabel("Damping (Sigma) [rad/s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(
        f"Modal Clustering with DBSCAN ($\\epsilon={epsilon:.3f}$, $N_{{pts}}={min_pts}$)\nClusters: {n_clusters} | Noise: {noise_count}",
        fontweight='bold'
    )
    handles = _noise_point_handle()
    if n_clusters > 0:
        handles += _cluster_legend_handles(n_clusters, representative_label='Cluster Means')
    handles += _reference_mode_handles(reference_modes)
    if handles:
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(5, len(handles)),
        )
    _set_modal_axis_limits(
        ax,
        dbscan_df,
        reference_modes=reference_modes,
        representatives=representatives if len(representatives) > 0 else None,
        include_all_points=True,
    )
    _apply_axis_style(ax)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.26)
    _save_figure(fig, base_output, "dbscan_modal_map")
    plt.close(fig)

def _resolve_tuning_settings(defaults, overrides=None):
    settings = dict(defaults)
    if overrides:
        settings.update(overrides)
    settings["pm_values"] = [float(value) for value in settings["pm_values"]]
    settings["min_npts"] = max(2, int(settings["min_npts"]))
    settings["min_assigned_ratio"] = float(settings["min_assigned_ratio"])
    settings["multiply_by_orders"] = bool(settings["multiply_by_orders"])
    return settings


def _density_inputs(df):
    if {"Gen", "Signal"}.issubset(df.columns):
        n_signals = int(df.groupby(["Gen", "Signal"]).ngroups)
    else:
        n_signals = 1
    return n_signals, _dbscan_order_count(df)


def _density_min_samples(pm, n_signals, n_orders, settings):
    scale = float(n_signals)
    if settings["multiply_by_orders"]:
        scale *= float(max(1, n_orders))
    return max(int(settings["min_npts"]), int(np.ceil(float(pm) * scale)))


def _cluster_representatives(df, labels, extra=None):
    rows = []
    representatives = []
    for label in sorted(label for label in np.unique(labels) if int(label) >= 0):
        points = df.loc[labels == label, ["Frequency", "Damping"]].to_numpy(dtype=float)
        representative = np.mean(points, axis=0)
        representatives.append(representative)
        row = {
            "Cluster": int(label + 1),
            "Frequency": float(representative[0]),
            "Damping": float(representative[1]),
            "Size": int(points.shape[0]),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    values = np.asarray(representatives, dtype=float) if representatives else np.empty((0, 2), dtype=float)
    return values, rows


def _collect_paper_mad_assignments(df, labels, cluster_rows, reference_modes, collector):
    """Collect Eq. (10)-style complex-pole distances without writing per-area output."""
    if collector is None or not reference_modes:
        return

    labels = np.asarray(labels, dtype=int)
    assigned_mask = labels >= 0
    if not np.any(assigned_mask):
        return

    cluster_df = pd.DataFrame(cluster_rows)
    if cluster_df.empty:
        return

    reference_names = list(reference_modes)
    reference_df = pd.DataFrame({
        "Reference_Mode": reference_names,
        "Reference_Frequency": [float(reference_modes[name]["Frequency"]) for name in reference_names],
        "Reference_Damping": [float(reference_modes[name]["Damping"]) for name in reference_names],
    })
    cluster_points = np.column_stack([
        cluster_df["Damping"].to_numpy(dtype=float),
        2.0 * np.pi * cluster_df["Frequency"].to_numpy(dtype=float),
    ])
    reference_points = np.column_stack([
        reference_df["Reference_Damping"].to_numpy(dtype=float),
        2.0 * np.pi * reference_df["Reference_Frequency"].to_numpy(dtype=float),
    ])
    distances = np.sqrt(np.sum((cluster_points[:, None, :] - reference_points[None, :, :]) ** 2, axis=2))
    nearest_idx = np.argmin(distances, axis=1)
    cluster_df = cluster_df.copy()
    cluster_df["Reference_Mode"] = [reference_names[idx] for idx in nearest_idx]
    cluster_df["Reference_Frequency"] = reference_df["Reference_Frequency"].to_numpy(dtype=float)[nearest_idx]
    cluster_df["Reference_Damping"] = reference_df["Reference_Damping"].to_numpy(dtype=float)[nearest_idx]
    assigned_df = df.loc[assigned_mask].copy()
    assigned_df["Cluster"] = labels[assigned_mask] + 1
    assignment_columns = [
        "Cluster",
        "Reference_Mode",
        "Reference_Frequency",
        "Reference_Damping",
    ]
    assigned_df = assigned_df.merge(cluster_df[assignment_columns], on="Cluster", how="left", validate="many_to_one")
    distances = np.sqrt(
        (assigned_df["Damping"].to_numpy(dtype=float) - assigned_df["Reference_Damping"].to_numpy(dtype=float)) ** 2
        + (2.0 * np.pi * (assigned_df["Frequency"].to_numpy(dtype=float) - assigned_df["Reference_Frequency"].to_numpy(dtype=float))) ** 2
    )
    collector.extend(
        {"Mode": mode, "Distance_rad_s": float(distance)}
        for mode, distance in zip(assigned_df["Reference_Mode"], distances)
    )


def _silhouette_for_cluster_labels(X_scaled, labels, min_assigned_ratio):
    assigned_mask = labels >= 0
    assigned_count = int(np.sum(assigned_mask))
    total_count = int(len(labels))
    assigned_ratio = float(assigned_count / total_count) if total_count else 0.0
    assigned_labels = labels[assigned_mask]
    n_clusters = int(len(np.unique(assigned_labels))) if assigned_count else 0
    silhouette = np.nan
    valid_silhouette = n_clusters >= 2 and assigned_count > n_clusters
    if valid_silhouette:
        silhouette = float(silhouette_score(X_scaled[assigned_mask], assigned_labels))
    eligible = bool(valid_silhouette and assigned_ratio >= min_assigned_ratio)
    return {
        "Clusters": n_clusters,
        "NoisePoints": int(total_count - assigned_count),
        "AssignedPoints": assigned_count,
        "AssignedRatio": assigned_ratio,
        "Silhouette": silhouette,
        "ValidSilhouette": bool(valid_silhouette),
        "Eligible": eligible,
    }


def _select_tuning_row(metrics_df):
    eligible = metrics_df[metrics_df["Eligible"]].copy()
    if not eligible.empty:
        ranked = eligible.sort_values(
            ["Silhouette", "AssignedRatio", "NoisePoints", "MinSamples"],
            ascending=[False, False, True, True],
            kind="stable",
        )
        return int(ranked.index[0]), "max_silhouette_subject_to_coverage"

    fallback = metrics_df.copy()
    fallback["_silhouette_rank"] = fallback["Silhouette"].fillna(-np.inf)
    ranked = fallback.sort_values(
        ["AssignedRatio", "_silhouette_rank", "NoisePoints", "MinSamples"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    return int(ranked.index[0]), "fallback_max_coverage"


def _save_selected_density_map(base_output, method, df, selected, reference_modes, title_parameters):
    fig, ax = plt.subplots(figsize=(10, 7))
    point_colors = _label_colors_with_noise(selected["labels"])
    ax.scatter(df["Damping"], df["Frequency"], c=point_colors, alpha=POINT_ALPHA,
               edgecolors="k", linewidths=0.8, s=POINT_SIZE)
    representatives = selected["representatives"]
    if len(representatives) > 0:
        ax.scatter(representatives[:, 1], representatives[:, 0], c=ACCENT_RED, marker="x",
                   s=REP_SIZE, linewidths=4, label="Cluster Means")
    _overlay_reference_modes(ax, reference_modes)
    ax.axvline(0, color=ACCENT_RED, linestyle="--", alpha=0.35, linewidth=2)
    ax.set_xlabel("Damping (Sigma) [rad/s]")
    ax.set_ylabel("Frequency [Hz]")
    assigned_percent = 100.0 * float(selected["metrics"]["AssignedRatio"])
    ax.set_title(
        f"Selected {method} Cluster Map ({title_parameters})\n"
        f"Clusters: {selected['metrics']['Clusters']} | Noise: {selected['metrics']['NoisePoints']} | "
        f"Assigned: {assigned_percent:.1f}% | Silhouette: {selected['metrics']['Silhouette']:.3f}" if np.isfinite(selected['metrics']['Silhouette']) else
        f"Selected {method} Cluster Map ({title_parameters})\n"
        f"Clusters: {selected['metrics']['Clusters']} | Noise: {selected['metrics']['NoisePoints']} | Assigned: {assigned_percent:.1f}% | Silhouette: n/a",
        fontweight="bold",
    )
    handles = _noise_point_handle()
    if selected["metrics"]["Clusters"] > 0:
        handles += _cluster_legend_handles(selected["metrics"]["Clusters"], representative_label="Cluster Means")
    handles += _reference_mode_handles(reference_modes)
    if handles:
        ax.legend(handles=handles, loc="upper left")
    _set_modal_axis_limits(ax, df, reference_modes=reference_modes, representatives=representatives)
    _apply_axis_style(ax)
    _save_figure(fig, base_output, f"{method.lower()}_selected_cluster_map")
    plt.close(fig)


def run_optics_modal_analysis(results_path, output_path, reference_modes=None, optics_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "optics")
    _prepare_output_dirs(base_output)
    optics_df = _load_screened_data(results_path, output_path)
    if optics_df is None or len(optics_df) < 3:
        print("Not enough samples for OPTICS clustering.")
        return None

    settings = _resolve_tuning_settings(OPTICS_DEFAULT_SETTINGS, optics_settings)
    settings["xi_values"] = [float(value) for value in settings["xi_values"]]
    n_signals, n_orders = _density_inputs(optics_df)
    X_scaled = StandardScaler().fit_transform(optics_df[["Frequency", "Damping"]].to_numpy(dtype=float))
    stored = {}
    rows = []
    for pm in settings["pm_values"]:
        min_samples = _density_min_samples(pm, n_signals, n_orders, settings)
        if min_samples >= len(optics_df):
            continue
        for xi in settings["xi_values"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                labels = OPTICS(min_samples=min_samples, cluster_method="xi", xi=xi).fit_predict(X_scaled)
            metrics = _silhouette_for_cluster_labels(X_scaled, labels, settings["min_assigned_ratio"])
            metrics["Reference_V_Measure"] = _reference_v_measure(optics_df, labels, reference_modes)
            metrics["Reference_ARI"] = _reference_ari(optics_df, labels, reference_modes)
            row = {"Pm": pm, "MinSamples": min_samples, "Xi": xi, "Nsignals": n_signals,
                   "NOrders": n_orders, "MultiplyByOrders": settings["multiply_by_orders"], **metrics}
            rows.append(row)
            representatives, cluster_rows = _cluster_representatives(
                optics_df, labels, {"Pm": pm, "MinSamples": min_samples, "Xi": xi}
            )
            stored[(pm, xi)] = {"labels": labels, "representatives": representatives,
                                "cluster_rows": cluster_rows, "metrics": metrics}
    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        print("No valid OPTICS parameter combinations.")
        return None
    best_idx, selection_reason = _select_tuning_row(metrics_df)
    metrics_df["Selected"] = metrics_df.index == best_idx
    metrics_df["SelectionReason"] = selection_reason
    metrics_df.to_csv(os.path.join(base_output, "optics_metrics_summary.csv"), index=False)
    selected_row = metrics_df.loc[best_idx]
    selected = stored[(float(selected_row["Pm"]), float(selected_row["Xi"]))]
    pd.DataFrame(selected["cluster_rows"]).to_csv(os.path.join(base_output, "cluster_representatives_sizes.csv"), index=False)
    _collect_paper_mad_assignments(
        optics_df,
        selected["labels"],
        selected["cluster_rows"],
        reference_modes=reference_modes,
        collector=paper_mad_collector,
    )
    _save_selected_density_map(base_output, "OPTICS", optics_df, selected, reference_modes,
                               f"$min\\_samples={int(selected_row['MinSamples'])}$, xi={selected_row['Xi']:.2f}")
    return {"pm": float(selected_row["Pm"]), "min_samples": int(selected_row["MinSamples"]),
            "xi": float(selected_row["Xi"]), "silhouette": None if pd.isna(selected_row["Silhouette"]) else float(selected_row["Silhouette"]),
            "assigned_ratio": float(selected_row["AssignedRatio"]), "selection_reason": selection_reason}


def run_dbscan_modal_analysis(results_path, output_path, reference_modes=None, dbscan_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "dbscan")
    _prepare_output_dirs(base_output)
    dbscan_df = _load_screened_data(results_path, output_path)
    if dbscan_df is None or len(dbscan_df) < 3:
        print("Not enough samples for DBSCAN clustering.")
        return None

    settings = _resolve_tuning_settings(DBSCAN_DEFAULT_SETTINGS, dbscan_settings)
    settings["pe_values"] = [float(value) for value in settings["pe_values"]]
    n_signals, n_orders = _density_inputs(dbscan_df)
    X_scaled = StandardScaler().fit_transform(dbscan_df[["Frequency", "Damping"]].to_numpy(dtype=float))
    X_dbscan = np.column_stack([dbscan_df["Damping"].to_numpy(dtype=float), 2.0 * np.pi * dbscan_df["Frequency"].to_numpy(dtype=float)])
    stored = {}
    rows = []
    for pe in settings["pe_values"]:
        epsilon = _dbscan_epsilon({"pe": pe})
        for pm in settings["pm_values"]:
            min_samples = _density_min_samples(pm, n_signals, n_orders, settings)
            labels = DBSCAN(eps=epsilon, min_samples=min_samples).fit_predict(X_dbscan)
            metrics = _silhouette_for_cluster_labels(X_scaled, labels, settings["min_assigned_ratio"])
            metrics["Reference_V_Measure"] = _reference_v_measure(dbscan_df, labels, reference_modes)
            metrics["Reference_ARI"] = _reference_ari(dbscan_df, labels, reference_modes)
            row = {"Pe": pe, "Pm": pm, "Epsilon": epsilon, "MinPts": min_samples, "MinSamples": min_samples,
                   "Nsignals": n_signals, "NOrders": n_orders, "MultiplyByOrders": settings["multiply_by_orders"], **metrics}
            rows.append(row)
            representatives, cluster_rows = _cluster_representatives(
                dbscan_df, labels, {"Pe": pe, "Pm": pm, "Epsilon": epsilon, "MinPts": min_samples}
            )
            stored[(pe, pm)] = {"labels": labels, "representatives": representatives,
                                "cluster_rows": cluster_rows, "metrics": metrics}
    metrics_df = pd.DataFrame(rows)
    best_idx, selection_reason = _select_tuning_row(metrics_df)
    metrics_df["Selected"] = metrics_df.index == best_idx
    metrics_df["SelectionReason"] = selection_reason
    metrics_df.to_csv(os.path.join(base_output, "dbscan_metrics_summary.csv"), index=False)
    selected_row = metrics_df.loc[best_idx]
    selected = stored[(float(selected_row["Pe"]), float(selected_row["Pm"]))]
    pd.DataFrame(selected["cluster_rows"]).to_csv(os.path.join(base_output, "cluster_representatives_sizes.csv"), index=False)
    _collect_paper_mad_assignments(
        dbscan_df,
        selected["labels"],
        selected["cluster_rows"],
        reference_modes=reference_modes,
        collector=paper_mad_collector,
    )
    _save_selected_density_map(base_output, "DBSCAN", dbscan_df, selected, reference_modes,
                               f"$\\epsilon={selected_row['Epsilon']:.3f}$, $N_{{pts}}={int(selected_row['MinPts'])}$")
    return {"pe": float(selected_row["Pe"]), "pm": float(selected_row["Pm"]), "epsilon": float(selected_row["Epsilon"]),
            "min_pts": int(selected_row["MinPts"]), "silhouette": None if pd.isna(selected_row["Silhouette"]) else float(selected_row["Silhouette"]),
            "assigned_ratio": float(selected_row["AssignedRatio"]), "selection_reason": selection_reason}


def _resolve_fixed_settings(defaults, overrides=None):
    settings = dict(defaults)
    if overrides:
        settings.update(overrides)
    return settings


def _reference_component_count(reference_modes, sample_count):
    """Use the locally relevant reference modes as the fixed partition count."""
    reference_count = len(reference_modes or {})
    return max(1, min(int(sample_count), reference_count or 1))


def _save_fixed_cluster_map(base_output, method, df, labels, representatives, reference_modes, title, include_noise=False):
    fig, ax = plt.subplots(figsize=(10, 7))
    color_fn = _label_colors_with_noise if include_noise else _label_colors
    ax.scatter(df["Damping"], df["Frequency"], c=color_fn(labels), alpha=POINT_ALPHA,
               edgecolors="k", linewidths=0.8, s=POINT_SIZE)
    if len(representatives) > 0:
        ax.scatter(representatives[:, 1], representatives[:, 0], c=ACCENT_RED, marker="x",
                   s=REP_SIZE, linewidths=4, label="Cluster Means")
    _overlay_reference_modes(ax, reference_modes)
    ax.axvline(0, color=ACCENT_RED, linestyle="--", alpha=0.35, linewidth=2)
    ax.set_xlabel("Damping (Sigma) [rad/s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title, fontweight="bold")
    handles = _noise_point_handle() if include_noise and np.any(np.asarray(labels) < 0) else []
    n_clusters = len(representatives)
    if n_clusters:
        handles += _cluster_legend_handles(n_clusters, representative_label="Cluster Means")
    handles += _reference_mode_handles(reference_modes)
    if handles:
        ax.legend(handles=handles, loc="upper left")
    _set_modal_axis_limits(ax, df, reference_modes=reference_modes, representatives=representatives)
    _apply_axis_style(ax)
    _save_figure(fig, base_output, f"{method.lower()}_selected_cluster_map")
    plt.close(fig)


def _fixed_cluster_metrics(X_scaled, df, labels, reference_modes=None):
    metrics = _silhouette_for_cluster_labels(X_scaled, np.asarray(labels, dtype=int), min_assigned_ratio=0.0)
    metrics["Reference_V_Measure"] = _reference_v_measure(df, labels, reference_modes)
    metrics["Reference_ARI"] = _reference_ari(df, labels, reference_modes)
    return metrics


def run_hdbscan_modal_analysis(results_path, output_path, reference_modes=None, hdbscan_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "hdbscan")
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 3:
        print("Not enough samples for HDBSCAN clustering.")
        return None

    settings = _resolve_fixed_settings(HDBSCAN_DEFAULT_SETTINGS, hdbscan_settings)
    min_cluster_size = min(max(2, int(settings["min_cluster_size"])), len(df))
    configured_min_samples = settings["min_samples"]
    min_samples = None if configured_min_samples is None else min(max(1, int(configured_min_samples)), len(df) - 1)
    X_scaled = StandardScaler().fit_transform(df[["Frequency", "Damping"]].to_numpy(dtype=float))
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=str(settings["cluster_selection_method"]),
        metric=str(settings["metric"]),
        allow_single_cluster=bool(settings["allow_single_cluster"]),
        copy=settings["copy"],
    ).fit_predict(X_scaled)
    representatives, cluster_rows = _cluster_representatives(df, labels)
    metrics = _fixed_cluster_metrics(X_scaled, df, labels, reference_modes)
    selection_reason = "fixed_defaults"
    metrics_row = {
        "MinClusterSize": min_cluster_size,
        "MinSamples": min_samples,
        "ClusterSelectionMethod": settings["cluster_selection_method"],
        "Metric": settings["metric"],
        "AllowSingleCluster": bool(settings["allow_single_cluster"]),
        "Copy": settings["copy"],
        "Selected": True,
        "SelectionReason": selection_reason,
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(os.path.join(base_output, "hdbscan_metrics_summary.csv"), index=False)
    pd.DataFrame(cluster_rows).to_csv(os.path.join(base_output, "cluster_representatives_sizes.csv"), index=False)
    _collect_paper_mad_assignments(df, labels, cluster_rows, reference_modes, paper_mad_collector)
    _save_fixed_cluster_map(
        base_output, "HDBSCAN", df, labels, representatives, reference_modes,
        f"HDBSCAN Cluster Map ($min\\_cluster\\_size={min_cluster_size}$, $min\\_samples={min_samples}$)\n"
        f"Clusters: {metrics['Clusters']} | Noise: {metrics['NoisePoints']}",
        include_noise=True,
    )
    return {"min_cluster_size": min_cluster_size, "min_samples": min_samples,
            "silhouette": None if pd.isna(metrics["Silhouette"]) else float(metrics["Silhouette"]),
            "assigned_ratio": float(metrics["AssignedRatio"]), "selection_reason": selection_reason}


def run_gmm_modal_analysis(results_path, output_path, reference_modes=None, gmm_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "gmm")
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 3:
        print("Not enough samples for Gaussian Mixture clustering.")
        return None

    settings = _resolve_fixed_settings(GMM_DEFAULT_SETTINGS, gmm_settings)
    n_components = _reference_component_count(reference_modes, len(df))
    X_scaled = StandardScaler().fit_transform(df[["Frequency", "Damping"]].to_numpy(dtype=float))
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=str(settings["covariance_type"]),
        init_params=str(settings["init_params"]),
        n_init=max(1, int(settings["n_init"])),
        random_state=settings["random_state"],
        max_iter=max(1, int(settings["max_iter"])),
        reg_covar=float(settings["reg_covar"]),
    )
    labels = model.fit_predict(X_scaled)
    representatives, cluster_rows = _cluster_representatives(df, labels)
    metrics = _fixed_cluster_metrics(X_scaled, df, labels, reference_modes)
    selection_reason = "fixed_reference_mode_count"
    metrics_row = {
        "SelectedK": n_components,
        "CovarianceType": settings["covariance_type"],
        "InitParams": settings["init_params"],
        "NInit": int(settings["n_init"]),
        "RandomState": settings["random_state"],
        "MaxIter": int(settings["max_iter"]),
        "RegCovar": float(settings["reg_covar"]),
        "BIC": float(model.bic(X_scaled)),
        "AIC": float(model.aic(X_scaled)),
        "LogLikelihood": float(model.lower_bound_),
        "Selected": True,
        "SelectionReason": selection_reason,
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(os.path.join(base_output, "gmm_metrics_summary.csv"), index=False)
    pd.DataFrame(cluster_rows).to_csv(os.path.join(base_output, "cluster_representatives_sizes.csv"), index=False)
    _collect_paper_mad_assignments(df, labels, cluster_rows, reference_modes, paper_mad_collector)
    _save_fixed_cluster_map(
        base_output, "GMM", df, labels, representatives, reference_modes,
        f"Gaussian Mixture Cluster Map ($k={n_components}$, full covariance)\nClusters: {metrics['Clusters']}",
    )
    return {"k": n_components, "bic": float(metrics_row["BIC"]),
            "silhouette": None if pd.isna(metrics["Silhouette"]) else float(metrics["Silhouette"]),
            "selection_reason": selection_reason}


def run_agglomerative_modal_analysis(results_path, output_path, reference_modes=None, agglomerative_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "agglomerative")
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 3:
        print("Not enough samples for Agglomerative clustering.")
        return None

    settings = _resolve_fixed_settings(AGGLOMERATIVE_DEFAULT_SETTINGS, agglomerative_settings)
    n_clusters = _reference_component_count(reference_modes, len(df))
    X_scaled = StandardScaler().fit_transform(df[["Frequency", "Damping"]].to_numpy(dtype=float))
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=str(settings["linkage"]),
        metric=str(settings["metric"]),
        compute_distances=bool(settings["compute_distances"]),
    )
    labels = model.fit_predict(X_scaled)
    representatives, cluster_rows = _cluster_representatives(df, labels)
    metrics = _fixed_cluster_metrics(X_scaled, df, labels, reference_modes)
    selection_reason = "fixed_reference_mode_count"
    metrics_row = {
        "SelectedK": n_clusters,
        "Linkage": settings["linkage"],
        "Metric": settings["metric"],
        "ComputeDistances": bool(settings["compute_distances"]),
        "Selected": True,
        "SelectionReason": selection_reason,
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(os.path.join(base_output, "agglomerative_metrics_summary.csv"), index=False)
    pd.DataFrame(cluster_rows).to_csv(os.path.join(base_output, "cluster_representatives_sizes.csv"), index=False)
    _collect_paper_mad_assignments(df, labels, cluster_rows, reference_modes, paper_mad_collector)
    _save_fixed_cluster_map(
        base_output, "Agglomerative", df, labels, representatives, reference_modes,
        f"Agglomerative Cluster Map ($k={n_clusters}$, Ward linkage)\nClusters: {metrics['Clusters']}",
    )
    return {"k": n_clusters,
            "silhouette": None if pd.isna(metrics["Silhouette"]) else float(metrics["Silhouette"]),
            "selection_reason": selection_reason}


def _paper_pole_coordinates(df):
    """Return the paper's pole coordinates [sigma, omega] in rad/s."""
    return np.column_stack([
        df["Damping"].to_numpy(dtype=float),
        2.0 * np.pi * df["Frequency"].to_numpy(dtype=float),
    ])


def _paper_silhouette_metrics(X, labels):
    labels = np.asarray(labels, dtype=int)
    assigned_mask = labels >= 0
    assigned_count = int(np.sum(assigned_mask))
    assigned_labels = labels[assigned_mask]
    clusters = int(len(np.unique(assigned_labels))) if assigned_count else 0
    valid = clusters >= 2 and assigned_count > clusters
    silhouette = float(silhouette_score(X[assigned_mask], assigned_labels)) if valid else np.nan
    return {
        "Clusters": clusters,
        "NoisePoints": int(len(labels) - assigned_count),
        "AssignedPoints": assigned_count,
        "AssignedRatio": float(assigned_count / len(labels)) if len(labels) else 0.0,
        "Silhouette": silhouette,
        "ValidSilhouette": bool(valid),
        "Eligible": bool(valid),
    }


def _select_max_silhouette(metrics_df):
    valid = metrics_df[metrics_df["ValidSilhouette"].astype(bool)].copy()
    if valid.empty:
        return None
    return int(valid.sort_values("Silhouette", ascending=False, kind="stable").index[0])


def _paper_density_settings(defaults, overrides=None, include_xi=False):
    settings = dict(defaults)
    if overrides:
        for key in ("pe_values", "pm_values", "xi_values", "linkages", "cluster_selection_methods", "multiply_by_orders", "min_npts"):
            if key in overrides:
                settings[key] = overrides[key]
    if "pe_values" in settings:
        settings["pe_values"] = [float(value) for value in settings["pe_values"]]
    if "pm_values" in settings:
        settings["pm_values"] = [float(value) for value in settings["pm_values"]]
    if include_xi:
        settings["xi_values"] = [float(value) for value in settings["xi_values"]]
    if "min_npts" in settings:
        settings["min_npts"] = max(2, int(settings["min_npts"]))
    if "multiply_by_orders" in settings:
        settings["multiply_by_orders"] = bool(settings["multiply_by_orders"])
    return settings


def _save_paper_selection(base_output, method, df, labels, reference_modes, title, collector=None):
    representatives, cluster_rows = _cluster_representatives(df, labels)
    pd.DataFrame(cluster_rows).to_csv(os.path.join(base_output, "cluster_representatives_sizes.csv"), index=False)
    _collect_paper_mad_assignments(df, labels, cluster_rows, reference_modes, collector)
    _save_fixed_cluster_map(
        base_output, method, df, labels, representatives, reference_modes, title,
        include_noise=bool(np.any(np.asarray(labels) < 0)),
    )


def _run_partitioning_paper_tuning(results_path, output_path, method, reference_modes=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, method)
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 3:
        print(f"Not enough samples for {method} clustering.")
        return None

    X = _paper_pole_coordinates(df)
    max_k = min(10, len(df) - 1)
    rows, stored = [], {}
    for k in range(2, max_k + 1):
        if method == "kmeans":
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        else:
            labels, _, _ = _pam_kmedoids(_pairwise_distances(X), n_clusters=k, random_state=42)
        metrics = _paper_silhouette_metrics(X, labels)
        metrics.update({
            "k": k,
            "Reference_V_Measure": _reference_v_measure(df, labels, reference_modes),
            "Reference_ARI": _reference_ari(df, labels, reference_modes),
        })
        rows.append(metrics)
        stored[k] = labels

    metrics_df = pd.DataFrame(rows)
    selected_idx = _select_max_silhouette(metrics_df)
    metrics_df["Selected"] = metrics_df.index == selected_idx if selected_idx is not None else False
    metrics_df["SelectionReason"] = "max_silhouette" if selected_idx is not None else "no_valid_silhouette_candidate"
    metrics_df.to_csv(os.path.join(base_output, f"{method}_metrics_summary.csv"), index=False)
    if selected_idx is None:
        return None

    selected = metrics_df.loc[selected_idx]
    labels = stored[int(selected["k"])]
    display_name = "k-Means" if method == "kmeans" else "k-Medoids"
    _save_paper_selection(
        base_output, display_name, df, labels, reference_modes,
        f"Selected {display_name} Cluster Map ($k={int(selected['k'])}$)\nSilhouette: {selected['Silhouette']:.3f}",
        paper_mad_collector,
    )
    return {"k": int(selected["k"]), "silhouette": float(selected["Silhouette"]), "selection_reason": "max_silhouette"}


def run_kmeans_modal_analysis(results_path, output_path, reference_modes=None, paper_mad_collector=None):
    return _run_partitioning_paper_tuning(results_path, output_path, "kmeans", reference_modes, paper_mad_collector)


def run_kmedoids_modal_analysis(results_path, output_path, reference_modes=None, paper_mad_collector=None):
    return _run_partitioning_paper_tuning(results_path, output_path, "kmedoids", reference_modes, paper_mad_collector)


def run_optics_modal_analysis(results_path, output_path, reference_modes=None, optics_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "optics")
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 3:
        print("Not enough samples for OPTICS clustering.")
        return None
    settings = _paper_density_settings(OPTICS_DEFAULT_SETTINGS, optics_settings, include_xi=True)
    X = _paper_pole_coordinates(df)
    n_signals, n_orders = _density_inputs(df)
    rows, stored = [], {}
    for pm in settings["pm_values"]:
        npts = _density_min_samples(pm, n_signals, n_orders, settings)
        if npts >= len(df):
            continue
        for xi in settings["xi_values"]:
            labels = OPTICS(min_samples=npts, cluster_method="xi", xi=xi).fit_predict(X)
            metrics = _paper_silhouette_metrics(X, labels)
            metrics.update({"Pm": pm, "MinSamples": npts, "Xi": xi, "Nsignals": n_signals, "NOrders": n_orders,
                            "MultiplyByOrders": settings["multiply_by_orders"],
                            "Reference_V_Measure": _reference_v_measure(df, labels, reference_modes),
                            "Reference_ARI": _reference_ari(df, labels, reference_modes)})
            rows.append(metrics)
            stored[(pm, xi)] = labels
    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        print("No valid OPTICS parameter combinations.")
        return None
    selected_idx = _select_max_silhouette(metrics_df)
    metrics_df["Selected"] = metrics_df.index == selected_idx if selected_idx is not None else False
    metrics_df["SelectionReason"] = "max_silhouette" if selected_idx is not None else "no_valid_silhouette_candidate"
    metrics_df.to_csv(os.path.join(base_output, "optics_metrics_summary.csv"), index=False)
    if selected_idx is None:
        return None
    selected = metrics_df.loc[selected_idx]
    labels = stored[(float(selected["Pm"]), float(selected["Xi"]))]
    _save_paper_selection(base_output, "OPTICS", df, labels, reference_modes,
                          f"Selected OPTICS Cluster Map ($min\\_samples={int(selected['MinSamples'])}$, xi={selected['Xi']:.2f})\nSilhouette: {selected['Silhouette']:.3f}", paper_mad_collector)
    return {"pm": float(selected["Pm"]), "min_samples": int(selected["MinSamples"]), "xi": float(selected["Xi"]),
            "silhouette": float(selected["Silhouette"]), "selection_reason": "max_silhouette"}


def run_dbscan_modal_analysis(results_path, output_path, reference_modes=None, dbscan_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "dbscan")
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 3:
        print("Not enough samples for DBSCAN clustering.")
        return None
    settings = _paper_density_settings(DBSCAN_DEFAULT_SETTINGS, dbscan_settings)
    X = _paper_pole_coordinates(df)
    n_signals, n_orders = _density_inputs(df)
    rows, stored = [], {}
    for pe in settings["pe_values"]:
        epsilon = _dbscan_epsilon({"pe": pe})
        for pm in settings["pm_values"]:
            npts = _density_min_samples(pm, n_signals, n_orders, settings)
            labels = DBSCAN(eps=epsilon, min_samples=npts).fit_predict(X)
            metrics = _paper_silhouette_metrics(X, labels)
            metrics.update({"Pe": pe, "Pm": pm, "Epsilon": epsilon, "MinPts": npts, "MinSamples": npts,
                            "Nsignals": n_signals, "NOrders": n_orders, "MultiplyByOrders": settings["multiply_by_orders"],
                            "Reference_V_Measure": _reference_v_measure(df, labels, reference_modes),
                            "Reference_ARI": _reference_ari(df, labels, reference_modes)})
            rows.append(metrics)
            stored[(pe, pm)] = labels
    metrics_df = pd.DataFrame(rows)
    selected_idx = _select_max_silhouette(metrics_df)
    metrics_df["Selected"] = metrics_df.index == selected_idx if selected_idx is not None else False
    metrics_df["SelectionReason"] = "max_silhouette" if selected_idx is not None else "no_valid_silhouette_candidate"
    metrics_df.to_csv(os.path.join(base_output, "dbscan_metrics_summary.csv"), index=False)
    if selected_idx is None:
        return None
    selected = metrics_df.loc[selected_idx]
    labels = stored[(float(selected["Pe"]), float(selected["Pm"]))]
    _save_paper_selection(base_output, "DBSCAN", df, labels, reference_modes,
                          f"Selected DBSCAN Cluster Map ($\\epsilon={selected['Epsilon']:.3f}$, $N_{{pts}}={int(selected['MinPts'])}$)\nSilhouette: {selected['Silhouette']:.3f}", paper_mad_collector)
    return {"pe": float(selected["Pe"]), "pm": float(selected["Pm"]), "epsilon": float(selected["Epsilon"]),
            "min_pts": int(selected["MinPts"]), "silhouette": float(selected["Silhouette"]), "selection_reason": "max_silhouette"}


def run_hdbscan_modal_analysis(results_path, output_path, reference_modes=None, hdbscan_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "hdbscan")
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 3:
        print("Not enough samples for HDBSCAN clustering.")
        return None
    settings = _paper_density_settings(HDBSCAN_DEFAULT_SETTINGS, hdbscan_settings)
    X = _paper_pole_coordinates(df)
    n_signals, n_orders = _density_inputs(df)
    rows, stored = [], {}
    for pe in settings["pe_values"]:
        epsilon = _dbscan_epsilon({"pe": pe})
        for pm in settings["pm_values"]:
            npts = _density_min_samples(pm, n_signals, n_orders, settings)
            if npts > len(df):
                continue
            for selection_method in settings["cluster_selection_methods"]:
                labels = HDBSCAN(min_cluster_size=npts, min_samples=None, cluster_selection_epsilon=epsilon,
                                 cluster_selection_method=selection_method, metric="euclidean",
                                 allow_single_cluster=False, copy=True).fit_predict(X)
                metrics = _paper_silhouette_metrics(X, labels)
                metrics.update({"Pe": pe, "Pm": pm, "Epsilon": epsilon, "MinClusterSize": npts, "MinSamples": None,
                                "ClusterSelectionMethod": selection_method, "Nsignals": n_signals, "NOrders": n_orders,
                                "MultiplyByOrders": settings["multiply_by_orders"],
                                "Reference_V_Measure": _reference_v_measure(df, labels, reference_modes),
                                "Reference_ARI": _reference_ari(df, labels, reference_modes)})
                rows.append(metrics)
                stored[(pe, pm, selection_method)] = labels
    metrics_df = pd.DataFrame(rows)
    selected_idx = _select_max_silhouette(metrics_df) if not metrics_df.empty else None
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=["Pe", "Pm", "Epsilon", "MinClusterSize", "MinSamples", "ClusterSelectionMethod", "Silhouette", "ValidSilhouette"])
    metrics_df["Selected"] = metrics_df.index == selected_idx if selected_idx is not None else False
    metrics_df["SelectionReason"] = "max_silhouette" if selected_idx is not None else "no_valid_silhouette_candidate"
    metrics_df.to_csv(os.path.join(base_output, "hdbscan_metrics_summary.csv"), index=False)
    if selected_idx is None:
        return None
    selected = metrics_df.loc[selected_idx]
    labels = stored[(float(selected["Pe"]), float(selected["Pm"]), selected["ClusterSelectionMethod"])]
    _save_paper_selection(base_output, "HDBSCAN", df, labels, reference_modes,
                          f"Selected HDBSCAN Cluster Map ($min\\_cluster\\_size={int(selected['MinClusterSize'])}$, $\\epsilon={selected['Epsilon']:.3f}$, {selected['ClusterSelectionMethod']})\nSilhouette: {selected['Silhouette']:.3f}", paper_mad_collector)
    return {"pe": float(selected["Pe"]), "pm": float(selected["Pm"]), "epsilon": float(selected["Epsilon"]),
            "min_cluster_size": int(selected["MinClusterSize"]), "cluster_selection_method": selected["ClusterSelectionMethod"],
            "silhouette": float(selected["Silhouette"]), "selection_reason": "max_silhouette"}


def run_gmm_modal_analysis(results_path, output_path, reference_modes=None, gmm_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "gmm")
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 2:
        print("Not enough samples for Gaussian Mixture clustering.")
        return None
    X = _paper_pole_coordinates(df)
    rows, stored = [], {}
    for k in range(1, min(10, len(df)) + 1):
        model = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4,
                                n_init=10, init_params="k-means++", random_state=42, max_iter=100)
        labels = model.fit_predict(X)
        metrics = _paper_silhouette_metrics(X, labels)
        metrics.update({"SelectedK": k, "CovarianceType": "full", "InitParams": "k-means++", "NInit": 10,
                        "RegCovar": 1e-4, "BIC": float(model.bic(X)), "AIC": float(model.aic(X)),
                        "Reference_V_Measure": _reference_v_measure(df, labels, reference_modes),
                        "Reference_ARI": _reference_ari(df, labels, reference_modes)})
        rows.append(metrics)
        stored[k] = labels
    metrics_df = pd.DataFrame(rows)
    selected_idx = int(metrics_df["BIC"].idxmin())
    metrics_df["Selected"] = metrics_df.index == selected_idx
    metrics_df["SelectionReason"] = "min_bic"
    metrics_df.to_csv(os.path.join(base_output, "gmm_metrics_summary.csv"), index=False)
    selected = metrics_df.loc[selected_idx]
    labels = stored[int(selected["SelectedK"])]
    _save_paper_selection(base_output, "GMM", df, labels, reference_modes,
                          f"Selected Gaussian Mixture Cluster Map ($k={int(selected['SelectedK'])}$, BIC={selected['BIC']:.2f})", paper_mad_collector)
    return {"k": int(selected["SelectedK"]), "bic": float(selected["BIC"]), "selection_reason": "min_bic"}


def run_agglomerative_modal_analysis(results_path, output_path, reference_modes=None, agglomerative_settings=None, paper_mad_collector=None):
    base_output = os.path.join(output_path, "agglomerative")
    _prepare_output_dirs(base_output)
    df = _load_screened_data(results_path, output_path)
    if df is None or len(df) < 3:
        print("Not enough samples for Agglomerative clustering.")
        return None
    settings = _paper_density_settings(AGGLOMERATIVE_DEFAULT_SETTINGS, agglomerative_settings)
    X = _paper_pole_coordinates(df)
    rows, stored = [], {}
    for pe in settings["pe_values"]:
        epsilon = _dbscan_epsilon({"pe": pe})
        for linkage in settings["linkages"]:
            labels = AgglomerativeClustering(n_clusters=None, distance_threshold=epsilon, metric="euclidean", linkage=linkage).fit_predict(X)
            metrics = _paper_silhouette_metrics(X, labels)
            metrics.update({"Pe": pe, "Epsilon": epsilon, "Linkage": linkage, "Metric": "euclidean",
                            "Reference_V_Measure": _reference_v_measure(df, labels, reference_modes),
                            "Reference_ARI": _reference_ari(df, labels, reference_modes)})
            rows.append(metrics)
            stored[(pe, linkage)] = labels
    metrics_df = pd.DataFrame(rows)
    selected_idx = _select_max_silhouette(metrics_df)
    metrics_df["Selected"] = metrics_df.index == selected_idx if selected_idx is not None else False
    metrics_df["SelectionReason"] = "max_silhouette" if selected_idx is not None else "no_valid_silhouette_candidate"
    metrics_df.to_csv(os.path.join(base_output, "agglomerative_metrics_summary.csv"), index=False)
    if selected_idx is None:
        return None
    selected = metrics_df.loc[selected_idx]
    labels = stored[(float(selected["Pe"]), selected["Linkage"])]
    _save_paper_selection(base_output, "Agglomerative", df, labels, reference_modes,
                          f"Selected Agglomerative Cluster Map ($\\epsilon={selected['Epsilon']:.3f}$, {selected['Linkage']} linkage)\nSilhouette: {selected['Silhouette']:.3f}", paper_mad_collector)
    return {"pe": float(selected["Pe"]), "epsilon": float(selected["Epsilon"]), "linkage": selected["Linkage"],
            "silhouette": float(selected["Silhouette"]), "selection_reason": "max_silhouette"}


def run_silhouette_analysis(results_path, output_path, reference_modes=None):
    base_output = os.path.join(output_path, "silhouette")
    _prepare_output_dirs(base_output)

    df = _load_screened_data(results_path, output_path)
    if df is None:
        return

    X = df[['Frequency', 'Damping']].values

    if len(df) < 3:
        print("Not enough samples for silhouette analysis.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    distance_matrix = _pairwise_distances(X_scaled)

    max_k = min(10, len(df) - 1)
    if max_k < 2:
        print("Not enough samples for silhouette analysis.")
        return

    k_values = np.arange(2, max_k + 1)
    kmeans_scores = []
    kmedoids_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        kmeans_scores.append(float(silhouette_score(X_scaled, kmeans_labels)))

        kmedoids_labels, kmedoids_medoid_indices, _ = _pam_kmedoids(distance_matrix, n_clusters=k, random_state=42)
        kmedoids_scores.append(float(silhouette_score(X_scaled, kmedoids_labels)))

    kmeans_scores = np.array(kmeans_scores)
    kmedoids_scores = np.array(kmedoids_scores)

    k_opt_kmeans = int(k_values[np.argmax(kmeans_scores)])
    k_opt_kmedoids = int(k_values[np.argmax(kmedoids_scores)])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(k_values, kmeans_scores, marker='o', linewidth=3, color=LINE_BLUE, label='$k-Means$')
    ax.plot(k_values, kmedoids_scores, marker='s', linewidth=3, color=LINE_GREEN, label='$k-Medoids$')
    ax.scatter(k_opt_kmeans, np.max(kmeans_scores), s=220, color=LINE_BLUE, edgecolors='k', zorder=5)
    ax.scatter(k_opt_kmedoids, np.max(kmedoids_scores), s=220, color=LINE_GREEN, edgecolors='k', zorder=5)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("Silhouette Analysis: $k$-Means vs $k$-Medoids", fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='lower left')
    _apply_axis_style(ax)
    _save_figure(fig, base_output, "silhouette_scores_comparison")
    plt.close(fig)

    kmeans_opt_model = KMeans(n_clusters=k_opt_kmeans, random_state=42, n_init=10)
    kmeans_opt_labels = kmeans_opt_model.fit_predict(X_scaled)
    kmeans_opt_centers = scaler.inverse_transform(kmeans_opt_model.cluster_centers_)
    kmeans_opt_wcss = float(kmeans_opt_model.inertia_)

    kmedoids_opt_labels, kmedoids_opt_medoid_indices, kmedoids_opt_cost = _pam_kmedoids(
        distance_matrix, n_clusters=k_opt_kmedoids, random_state=42
    )
    kmedoids_opt_medoids = scaler.inverse_transform(X_scaled[kmedoids_opt_medoid_indices])
    kmedoids_opt_cost = float(kmedoids_opt_cost)

    methods = [
        ("k-Means", k_opt_kmeans, kmeans_opt_labels, kmeans_opt_centers, "Centroids", f"WCSS: {kmeans_opt_wcss:.2f}"),
        ("k-Medoids", k_opt_kmedoids, kmedoids_opt_labels, kmedoids_opt_medoids, "Medoids", f"Cost: {kmedoids_opt_cost:.2f}")
    ]

    for method_name, k_opt, labels, representatives, rep_label, compactness_text in methods:
        sample_silhouette_values = silhouette_samples(X_scaled, labels)
        avg_score = float(silhouette_score(X_scaled, labels))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [1.05, 1.0]})

        y_lower = 10
        for i in range(k_opt):
            ith_vals = sample_silhouette_values[labels == i]
            ith_vals.sort()
            size_i = ith_vals.shape[0]
            y_upper = y_lower + size_i
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper), 0, ith_vals,
                facecolor=color, edgecolor=color, alpha=0.82
            )
            ax1.text(-0.08, y_lower + 0.5 * size_i, f"Cluster {i + 1}", va='center', fontsize=16)
            y_lower = y_upper + 10

        avg_line = ax1.axvline(
            x=avg_score, color=ACCENT_RED, linestyle='--', linewidth=2.2,
            label=f"Average silhouette = {avg_score:.3f}"
        )
        ax1.set_title(fr"${method_name}$ Silhouette Profile ($k={k_opt}$)", fontweight='bold')
        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster")
        ax1.set_yticks([])
        x_min = min(-0.1, sample_silhouette_values.min() - 0.05)
        ax1.set_xlim(x_min, 1.0)
        ticks = np.arange(-0.2, 1.01, 0.2)
        ticks = ticks[(ticks >= x_min - 1e-12) & (ticks <= 1.0 + 1e-12)]
        ax1.set_xticks(np.round(ticks, 2))
        ax1.legend(handles=[avg_line], loc='lower right')
        _apply_axis_style(ax1, 0.35)

        point_colors = _label_colors(labels)
        ax2.scatter(
            df['Damping'], df['Frequency'], c=point_colors,
            alpha=POINT_ALPHA, edgecolors='k', linewidths=0.8, s=POINT_SIZE
        )
        ax2.scatter(
            representatives[:, 1], representatives[:, 0], c=ACCENT_RED, marker='x',
            s=REP_SIZE, linewidths=4, label=rep_label
        )
        _overlay_reference_modes(ax2, reference_modes)
        ax2.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax2.set_title(
            f"${method_name}$ Cluster Map ($k={k_opt}$)\n{compactness_text}",
            fontweight='bold'
        )
        ax2.set_xlabel("Damping (Sigma) [rad/s]")
        ax2.set_ylabel("Frequency [Hz]")
        _set_modal_axis_limits(ax2, df, reference_modes=reference_modes, representatives=representatives)
        _apply_axis_style(ax2, GRID_ALPHA_SUB)

        handles = _cluster_legend_handles(k_opt, representative_label=rep_label) + _reference_mode_handles(reference_modes)
        ax2.legend(handles=handles, loc='upper left')

        fig.tight_layout()
        slug = method_name.lower().replace('-', '').replace(' ', '_')
        _save_figure(fig, base_output, f"silhouette_profile_{slug}")
        plt.close(fig)

    summary_df = pd.DataFrame({
        'k': k_values,
        'kmeans_silhouette': kmeans_scores,
        'kmedoids_silhouette': kmedoids_scores,
        'kmeans_selected_by_silhouette': k_values == k_opt_kmeans,
        'kmedoids_selected_by_silhouette': k_values == k_opt_kmedoids,
    })
    summary_df.to_csv(os.path.join(base_output, "silhouette_scores.csv"), index=False)

    optimal_summary = pd.DataFrame([
        {"Method": "k-Means", "Selection_Criterion": "Silhouette", "k_opt": k_opt_kmeans, "Silhouette": float(np.max(kmeans_scores))},
        {"Method": "k-Medoids", "Selection_Criterion": "Silhouette", "k_opt": k_opt_kmedoids, "Silhouette": float(np.max(kmedoids_scores))},
    ])
    optimal_summary.to_csv(os.path.join(base_output, "silhouette_optimal_k_summary.csv"), index=False)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(base_dir, "results.csv")
    out_path = os.path.join(base_dir, "clustering")

    df_for_mad = _load_screened_data(res_path, out_path)
    if df_for_mad is not None:
        _save_reference_mad_outputs(df_for_mad, base_dir)

    run_kmeans_modal_analysis(res_path, out_path)
    run_kmedoids_modal_analysis(res_path, out_path)
    run_silhouette_analysis(res_path, out_path)
