import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
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
    "min_samples_min": 5,
    "min_samples_max": 20,
    "xi": 0.05,
}

DBSCAN_DEFAULT_SETTINGS = {
    "pe": 0.025,
    "pm": 0.25,
    "multiply_by_orders": False,
    "min_npts": 2,
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


def run_kmeans_modal_analysis(results_path, output_path, reference_modes=None):
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


def run_kmedoids_modal_analysis(results_path, output_path, reference_modes=None):
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
