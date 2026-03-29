
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
from plot_style import (
    apply_thesis_style,
    style_axis,
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


def _label_colors(labels):
    return [CLUSTER_COLORS[int(lbl) % len(CLUSTER_COLORS)] for lbl in labels]


def _apply_axis_style(ax, grid_alpha=GRID_ALPHA_MAIN):
    style_axis(ax, grid_alpha=grid_alpha)


def _save_figure(fig, base_output, filename):
    fig.savefig(os.path.join(base_output, "pdf", f"{filename}.pdf"), format="pdf")
    fig.savefig(os.path.join(base_output, "png", f"{filename}.png"), dpi=300)


def _prepare_output_dirs(base_output):
    for sub in ["png", "pdf"]:
        os.makedirs(os.path.join(base_output, sub), exist_ok=True)


def _cluster_legend_handles(k, representative_label=None):
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


def _pairwise_distances(X):
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=2))


def _pam_kmedoids(distance_matrix, n_clusters, random_state=42, max_iter=100):
    n_samples = distance_matrix.shape[0]
    rng = np.random.default_rng(random_state)
    medoid_indices = np.sort(rng.choice(n_samples, size=n_clusters, replace=False))

    labels = np.argmin(distance_matrix[:, medoid_indices], axis=1)
    best_cost = np.sum(distance_matrix[np.arange(n_samples), medoid_indices[labels]])

    for _ in range(max_iter):
        improved = False
        current_set = set(medoid_indices.tolist())

        for medoid_pos in range(n_clusters):
            for candidate in range(n_samples):
                if candidate in current_set:
                    continue

                trial_medoids = medoid_indices.copy()
                trial_medoids[medoid_pos] = candidate
                trial_medoids.sort()

                trial_labels = np.argmin(distance_matrix[:, trial_medoids], axis=1)
                trial_cost = np.sum(distance_matrix[np.arange(n_samples), trial_medoids[trial_labels]])

                if trial_cost + 1e-12 < best_cost:
                    medoid_indices = trial_medoids
                    labels = trial_labels
                    best_cost = trial_cost
                    improved = True
                    current_set = set(medoid_indices.tolist())

        if not improved:
            break

    return labels, medoid_indices, best_cost


def _apply_frequency_screening(df, output_path=None):
    df = df.copy()
    n_initial = len(df)

    finite_mask = np.isfinite(df["Frequency"]) & np.isfinite(df["Damping"])
    df = df.loc[finite_mask].copy()
    n_after_finite = len(df)

    freq_mask = (df["Frequency"] >= FREQ_MIN) & (df["Frequency"] <= FREQ_MAX)
    df = df.loc[freq_mask].copy()
    n_after_frequency = len(df)

    summary = pd.DataFrame([
        {"step": "initial_rows", "count": n_initial},
        {"step": "after_finite_numeric_filter", "count": n_after_finite},
        {"step": "removed_non_finite_rows", "count": n_initial - n_after_finite},
        {"step": "after_frequency_screening", "count": n_after_frequency},
        {"step": "removed_out_of_range_frequency_rows", "count": n_after_finite - n_after_frequency},
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


def _compute_cluster_mad_stats(X_original, labels, representatives):
    records = []
    point_distances = []

    for cluster_idx in sorted(np.unique(labels)):
        cluster_points = X_original[labels == cluster_idx]
        rep = representatives[cluster_idx]
        distances = np.sqrt(np.sum((cluster_points - rep) ** 2, axis=1))
        point_distances.extend(distances.tolist())

        records.append({
            "Cluster": int(cluster_idx + 1),
            "Size": int(cluster_points.shape[0]),
            "Representative_Frequency": float(rep[0]),
            "Representative_Damping": float(rep[1]),
            "MAD_2D": float(np.median(distances)),
            "Mean_2D_Distance": float(np.mean(distances)),
            "Max_2D_Distance": float(np.max(distances)),
        })

    overall_mad = float(np.median(point_distances)) if point_distances else np.nan
    return pd.DataFrame(records), overall_mad


def _save_metrics_summary(base_output, metrics_rows, filename):
    pd.DataFrame(metrics_rows).to_csv(os.path.join(base_output, filename), index=False)


def run_kmeans_modal_analysis(results_path, output_path):
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
        cluster_mad_df, overall_mad = _compute_cluster_mad_stats(X, labels, centers)
        cluster_mad_df.insert(0, "k", int(k))
        cluster_mad_df.to_csv(os.path.join(base_output, f"kmeans_cluster_mad_k{k}.csv"), index=False)

        stored_results[int(k)] = (labels, centers, inertia, overall_mad)

        metrics_rows.append({
            "k": int(k),
            "WCSS": inertia,
            "MAD_2D_overall": overall_mad,
        })

        for c in range(k):
            cluster_stats.append({
                'k': int(k),
                'Cluster': int(c + 1),
                'Frequency': float(centers[c, 0]),
                'Damping': float(centers[c, 1]),
                'Size': int(np.sum(labels == c))
            })

        fig, ax = plt.subplots(figsize=(10, 7))
        point_colors = _label_colors(labels)
        ax.scatter(
            df['Damping'], df['Frequency'], c=point_colors,
            alpha=POINT_ALPHA, edgecolors='k', linewidths=0.8, s=POINT_SIZE
        )
        ax.scatter(
            centers[:, 1], centers[:, 0], c=ACCENT_RED, marker='x',
            s=REP_SIZE, linewidths=4, label='Centroids'
        )

        ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax.set_xlabel("Damping (Sigma) [rad/s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(
            f"Modal Clustering with $k-Means$ ($k={k}$)\nWCSS: {inertia:.2f} | MAD: {overall_mad:.4f}",
            fontweight='bold'
        )
        ax.legend(loc='upper left')
        _apply_axis_style(ax)
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
            d = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
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
        labels, centers, inertia, overall_mad = stored_results[k]

        point_colors = _label_colors(labels)
        ax.scatter(df['Damping'], df['Frequency'], c=point_colors, alpha=POINT_ALPHA, s=GRID_POINT_SIZE,
                   edgecolors='k', linewidths=0.5)
        ax.scatter(centers[:, 1], centers[:, 0], c=ACCENT_RED, marker='x', s=REP_GRID_SIZE, linewidths=3)

        ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax.set_title(
            f"$k-Means$ Results: $k={k}$\nWCSS: {inertia:.1f} | MAD: {overall_mad:.3f}",
            fontweight='semibold'
        )
        _apply_axis_style(ax, GRID_ALPHA_SUB)

        if idx >= 2:
            ax.set_xlabel("Damping (Sigma) [rad/s]")
        if idx % 2 == 0:
            ax.set_ylabel("Frequency [Hz]")

    handles = _cluster_legend_handles(max(grid_ks), representative_label="Centroids")
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=min(4, len(handles)))
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
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

    metrics_df["k_selected_by_max_chord"] = metrics_df["k"] == k_opt
    metrics_df.to_csv(os.path.join(base_output, "kmeans_metrics_summary.csv"), index=False)
    pd.DataFrame(cluster_stats).to_csv(os.path.join(base_output, "cluster_centers_sizes.csv"), index=False)


def run_kmedoids_modal_analysis(results_path, output_path):
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
        cluster_mad_df, overall_mad = _compute_cluster_mad_stats(X, labels, medoids)
        cluster_mad_df.insert(0, "k", int(k))
        cluster_mad_df.to_csv(os.path.join(base_output, f"kmedoids_cluster_mad_k{k}.csv"), index=False)

        stored_results[int(k)] = (labels, medoids, cost, overall_mad)

        metrics_rows.append({
            "k": int(k),
            "Cost": cost,
            "MAD_2D_overall": overall_mad,
        })

        for c in range(k):
            cluster_stats.append({
                'k': int(k),
                'Cluster': int(c + 1),
                'Frequency': float(medoids[c, 0]),
                'Damping': float(medoids[c, 1]),
                'Size': int(np.sum(labels == c))
            })

        fig, ax = plt.subplots(figsize=(10, 7))
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

        ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax.set_xlabel("Damping (Sigma) [rad/s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(
            f"Modal Clustering with $k-Medoids$ ($k={k}$)\nCost: {cost:.2f} | MAD: {overall_mad:.4f}",
            fontweight='bold'
        )
        ax.legend(loc='upper left')
        _apply_axis_style(ax)
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
            d = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
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
        labels, medoids, cost, overall_mad = stored_results[k]

        point_colors = _label_colors(labels)
        ax.scatter(df['Damping'], df['Frequency'], c=point_colors, alpha=POINT_ALPHA, s=GRID_POINT_SIZE,
                   edgecolors='k', linewidths=0.5)
        ax.scatter(medoids[:, 1], medoids[:, 0], c=ACCENT_RED, marker='x', s=REP_GRID_SIZE, linewidths=3)

        ax.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax.set_title(
            f"$k-Medoids$ Results: $k={k}$\nCost: {cost:.1f} | MAD: {overall_mad:.3f}",
            fontweight='semibold'
        )
        _apply_axis_style(ax, GRID_ALPHA_SUB)

        if idx >= 2:
            ax.set_xlabel("Damping (Sigma) [rad/s]")
        if idx % 2 == 0:
            ax.set_ylabel("Frequency [Hz]")

    handles = _cluster_legend_handles(max(grid_ks), representative_label="Medoids")
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, len(handles)),
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
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

    metrics_df["k_selected_by_max_chord"] = metrics_df["k"] == k_opt
    metrics_df.to_csv(os.path.join(base_output, "kmedoids_metrics_summary.csv"), index=False)
    pd.DataFrame(cluster_stats).to_csv(os.path.join(base_output, "cluster_medoids_sizes.csv"), index=False)


def run_silhouette_analysis(results_path, output_path):
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
    kmeans_mad = []
    kmedoids_mad = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        kmeans_scores.append(float(silhouette_score(X_scaled, kmeans_labels)))
        kmeans_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        _, overall_kmeans_mad = _compute_cluster_mad_stats(X, kmeans_labels, kmeans_centers)
        kmeans_mad.append(overall_kmeans_mad)

        kmedoids_labels, kmedoids_medoid_indices, _ = _pam_kmedoids(distance_matrix, n_clusters=k, random_state=42)
        kmedoids_scores.append(float(silhouette_score(X_scaled, kmedoids_labels)))
        kmedoids_medoids = scaler.inverse_transform(X_scaled[kmedoids_medoid_indices])
        _, overall_kmedoids_mad = _compute_cluster_mad_stats(X, kmedoids_labels, kmedoids_medoids)
        kmedoids_mad.append(overall_kmedoids_mad)

    kmeans_scores = np.array(kmeans_scores)
    kmedoids_scores = np.array(kmedoids_scores)
    kmeans_mad = np.array(kmeans_mad)
    kmedoids_mad = np.array(kmedoids_mad)

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
    kmeans_cluster_mad_df, kmeans_overall_mad = _compute_cluster_mad_stats(X, kmeans_opt_labels, kmeans_opt_centers)
    kmeans_opt_wcss = float(kmeans_opt_model.inertia_)
    kmeans_cluster_mad_df.to_csv(os.path.join(base_output, "kmeans_silhouette_selected_cluster_mad.csv"), index=False)

    kmedoids_opt_labels, kmedoids_opt_medoid_indices, kmedoids_opt_cost = _pam_kmedoids(
        distance_matrix, n_clusters=k_opt_kmedoids, random_state=42
    )
    kmedoids_opt_medoids = scaler.inverse_transform(X_scaled[kmedoids_opt_medoid_indices])
    kmedoids_cluster_mad_df, kmedoids_overall_mad = _compute_cluster_mad_stats(X, kmedoids_opt_labels, kmedoids_opt_medoids)
    kmedoids_opt_cost = float(kmedoids_opt_cost)
    kmedoids_cluster_mad_df.to_csv(os.path.join(base_output, "kmedoids_silhouette_selected_cluster_mad.csv"), index=False)

    methods = [
        ("k-Means", k_opt_kmeans, kmeans_opt_labels, kmeans_opt_centers, "Centroids", kmeans_overall_mad, f"WCSS: {kmeans_opt_wcss:.2f}"),
        ("k-Medoids", k_opt_kmedoids, kmedoids_opt_labels, kmedoids_opt_medoids, "Medoids", kmedoids_overall_mad, f"Cost: {kmedoids_opt_cost:.2f}")
    ]

    for method_name, k_opt, labels, representatives, rep_label, overall_mad, compactness_text in methods:
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
        ax2.axvline(0, color=ACCENT_RED, linestyle='--', alpha=0.35, linewidth=2)
        ax2.set_title(
            f"${method_name}$ Cluster Map ($k={k_opt}$)\n{compactness_text} | MAD: {overall_mad:.4f}",
            fontweight='bold'
        )
        ax2.set_xlabel("Damping (Sigma) [rad/s]")
        ax2.set_ylabel("Frequency [Hz]")
        _apply_axis_style(ax2, GRID_ALPHA_SUB)

        handles = _cluster_legend_handles(k_opt, representative_label=rep_label)
        ax2.legend(handles=handles, loc='upper left')

        fig.tight_layout()
        slug = method_name.lower().replace('-', '').replace(' ', '_')
        _save_figure(fig, base_output, f"silhouette_profile_{slug}")
        plt.close(fig)

    summary_df = pd.DataFrame({
        'k': k_values,
        'kmeans_silhouette': kmeans_scores,
        'kmedoids_silhouette': kmedoids_scores,
        'kmeans_mad_2d': kmeans_mad,
        'kmedoids_mad_2d': kmedoids_mad,
        'kmeans_selected_by_silhouette': k_values == k_opt_kmeans,
        'kmedoids_selected_by_silhouette': k_values == k_opt_kmedoids,
    })
    summary_df.to_csv(os.path.join(base_output, "silhouette_scores.csv"), index=False)

    optimal_summary = pd.DataFrame([
        {"Method": "k-Means", "Selection_Criterion": "Silhouette", "k_opt": k_opt_kmeans, "Silhouette": float(np.max(kmeans_scores)), "MAD_2D_overall": kmeans_overall_mad},
        {"Method": "k-Medoids", "Selection_Criterion": "Silhouette", "k_opt": k_opt_kmedoids, "Silhouette": float(np.max(kmedoids_scores)), "MAD_2D_overall": kmedoids_overall_mad},
    ])
    optimal_summary.to_csv(os.path.join(base_output, "silhouette_optimal_k_summary.csv"), index=False)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(base_dir, "results.csv")
    out_path = os.path.join(base_dir, "clustering")

    run_kmeans_modal_analysis(res_path, out_path)
    run_kmedoids_modal_analysis(res_path, out_path)
    run_silhouette_analysis(res_path, out_path)
