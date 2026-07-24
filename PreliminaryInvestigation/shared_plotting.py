from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FixedLocator

from plot_style import save_pdf, style_axis, SIGNAL_COLORS


MODAL_X_LABEL = "Damping (Sigma) [rad/s]"
MODAL_Y_LABEL = "Frequency [Hz]"
RECON_X_LABEL = "Time (s)"
RECON_DEFAULT_X_LIMS = (0, 50)
RECON_TICK_LABEL_SIZE = 30
RECON_AXIS_LABEL_SIZE = 34
SIGNAL_LABELS = {
    "Voltage": r"$\Delta V$ [p.u.]",
    "Current": r"$\Delta \mathrm{I}$ [p.u.]",
    "Active Power": r"$\Delta P$ [MW]",
    "Reactive Power": r"$\Delta Q$ [Mvar]",
}
MODAL_LOG_RATIO_THRESHOLD = 50.0
MODAL_DEFAULT_XTICKS = [-10.0, -1.0, -0.1, -0.01, 0.0, 0.002, 0.004, 0.006, 0.008]


def _format_modal_tick_value(x):
    x = float(x)
    if abs(x) < 5e-6:
        return "0"
    magnitude = abs(x)
    if magnitude >= 1.0:
        return f"{x:.0f}"
    if magnitude >= 0.1:
        return f"{x:.1f}"
    if magnitude >= 0.01:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _overlay_reference_modes(ax, reference_modes, annotate=True):
    if not reference_modes:
        return

    ref_names = list(reference_modes.keys())
    ref_freq = [float(reference_modes[name]["Frequency"]) for name in ref_names]
    ref_damping = [float(reference_modes[name]["Damping"]) for name in ref_names]
    ax.scatter(
        ref_damping,
        ref_freq,
        marker="D",
        s=120,
        facecolors="white",
        edgecolors="black",
        linewidths=1.8,
        zorder=6,
    )
    if annotate:
        for name, damping, freq in zip(ref_names, ref_damping, ref_freq):
            ax.annotate(
                name,
                (damping, freq),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="semibold",
                color="black",
            )


def _reference_modes_for_generator(reference_modes, generator_name):
    if not reference_modes:
        return None

    filtered = {}
    for mode_name, mode_data in reference_modes.items():
        relevant_generators = list(mode_data.get("relevant_generators") or [])
        if relevant_generators and generator_name not in relevant_generators:
            continue
        filtered[mode_name] = mode_data
    return filtered


MODAL_X_TICK_FORMATTER = FuncFormatter(lambda x, pos: _format_modal_tick_value(x))


def generator_display_name(gen):
    if isinstance(gen, str) and gen.startswith("g") and gen[1:].isdigit():
        return f"Generator {int(gen[1:])}"
    return str(gen)


def generator_modal_label(gen):
    return f"Generator {str(gen).upper()}"


def signal_axis_label(signal):
    return SIGNAL_LABELS.get(signal, str(signal))


def save_current_figure(path_base, filename, fig=None):
    fig = fig or plt
    path_base = Path(path_base)
    png_dir = path_base / "png"
    pdf_dir = path_base / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, pdf_dir / f"{filename}.pdf")
    fig.savefig(png_dir / f"{filename}.png", dpi=300, bbox_inches="tight")


def create_adaptive_grid(fig, item_count, ncols=2, sharex=False, sharey=False, span_last=True):
    ncols = max(1, int(ncols))
    item_count = max(1, int(item_count))
    nrows = ceil(item_count / ncols)
    grid = fig.add_gridspec(nrows, ncols)
    axes = []
    shared_axis = None

    for idx in range(item_count):
        row = idx // ncols
        col = idx % ncols
        span_full_row = span_last and ncols == 2 and item_count % ncols == 1 and idx == item_count - 1
        spec = grid[row, :] if span_full_row else grid[row, col]
        subplot_kwargs = {}
        if shared_axis is not None:
            if sharex:
                subplot_kwargs["sharex"] = shared_axis
            if sharey:
                subplot_kwargs["sharey"] = shared_axis
        ax = fig.add_subplot(spec, **subplot_kwargs)
        if shared_axis is None:
            shared_axis = ax
        axes.append(ax)

    return axes, nrows


def modal_grid_figure(item_count, ncols=2, row_height=6.0, min_height=6.0):
    nrows = ceil(max(1, item_count) / max(1, ncols))
    return plt.figure(figsize=(8 * max(1, ncols), max(min_height, row_height * nrows)))


def reconstruction_grid_figure(row_count, row_height=4.8, min_height=6.0):
    return plt.figure(figsize=(16, max(min_height, row_height * max(1, row_count))))


def _set_modal_axis_view(
    ax,
    damping_values,
    frequency_values,
    clamp_positive_max=True,
    allow_log_scales=True,
    force_symlog_x=False,
):
    damping_values = np.asarray(damping_values, dtype=float)
    frequency_values = np.asarray(frequency_values, dtype=float)
    damping_values = damping_values[np.isfinite(damping_values)]
    frequency_values = frequency_values[np.isfinite(frequency_values)]
    if damping_values.size == 0 or frequency_values.size == 0:
        return

    x_min = float(np.min(damping_values))
    x_max = float(np.max(damping_values))
    y_min = float(np.min(frequency_values))
    y_max = float(np.max(frequency_values))

    x_span = max(0.05, abs(x_min - x_max))
    y_span = max(0.1, abs(y_min - y_max))
    x_pad_left = max(0.05, 0.06 * x_span)
    x_pad_right = max(0.006, 0.012 * x_span)
    y_pad = max(0.05, 0.05 * y_span)

    x_right = max(0.02, x_max + x_pad_right) if clamp_positive_max else max(0.005, x_max)
    x_left = x_min - x_pad_left
    y_bottom = max(0.0, y_min - y_pad)
    y_top = y_max + y_pad

    nonzero_damping = np.abs(damping_values[np.abs(damping_values) > 1e-12])
    use_symlog_x = force_symlog_x or (
        allow_log_scales
        and nonzero_damping.size >= 2
        and (float(np.max(nonzero_damping)) / max(float(np.min(nonzero_damping)), 1e-12)) >= MODAL_LOG_RATIO_THRESHOLD
    )
    positive_freq = frequency_values[frequency_values > 1e-12]
    use_log_y = allow_log_scales and (
        positive_freq.size >= 2
        and (float(np.max(positive_freq)) / max(float(np.min(positive_freq)), 1e-12)) >= MODAL_LOG_RATIO_THRESHOLD
    )

    if use_symlog_x:
        linthresh = max(1e-3, min(0.05, 0.05 * max(float(np.max(nonzero_damping)), 1e-3)))
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_major_formatter(MODAL_X_TICK_FORMATTER)
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_formatter(MODAL_X_TICK_FORMATTER)

    if use_log_y:
        ax.set_yscale("log")
        y_bottom = max(1e-3, y_bottom if y_bottom > 0 else float(np.min(positive_freq)) * 0.9)
    else:
        ax.set_yscale("linear")

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)


def _dominant_frequency_guides(frequency_values, max_guides=4):
    frequency_values = np.asarray(frequency_values, dtype=float)
    frequency_values = frequency_values[np.isfinite(frequency_values) & (frequency_values > 1e-6)]
    if frequency_values.size < 12:
        return []

    log_frequency = np.log10(frequency_values)
    bin_count = max(14, min(32, int(np.sqrt(frequency_values.size) * 2)))
    hist, edges = np.histogram(log_frequency, bins=bin_count)
    if hist.size == 0 or int(np.max(hist)) <= 0:
        return []

    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_threshold = max(3, int(np.ceil(float(np.max(hist)) * 0.18)))
    min_log_sep = max(0.08, (edges[1] - edges[0]) * 1.5)

    selected_logs = []
    for idx in np.argsort(hist)[::-1]:
        if int(hist[idx]) < peak_threshold:
            continue
        center = float(centers[idx])
        if any(abs(center - selected) < min_log_sep for selected in selected_logs):
            continue
        selected_logs.append(center)
        if len(selected_logs) >= max_guides:
            break

    return sorted(float(10 ** center) for center in selected_logs)


def _add_frequency_guides(ax, frequency_guides):
    if not frequency_guides:
        return

    for guide in frequency_guides:
        ax.axhline(guide, color="#6c757d", linestyle=":", linewidth=1.2, alpha=0.35, zorder=0)


def _apply_frequency_guide_ticks(ax, frequency_guides):
    if not frequency_guides:
        return

    y_min, y_max = ax.get_ylim()
    visible_guides = [float(guide) for guide in frequency_guides if y_min <= float(guide) <= y_max]
    if not visible_guides:
        return

    top_guide = max(visible_guides)
    if y_max <= top_guide * 1.08:
        if ax.get_yscale() == "log":
            ax.set_ylim(y_min, top_guide * 1.14)
        else:
            ax.set_ylim(y_min, top_guide + 0.08 * max(top_guide - y_min, 0.1))

    ax.yaxis.set_major_locator(FixedLocator(visible_guides))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{float(y):.2f}"))


def _annotate_zero_reference(ax, fontsize=11):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if not (x_min <= 0.0 <= x_max):
        return

    y_anchor = y_min + 0.06 * (y_max - y_min)
    ax.text(
        0.0,
        y_anchor,
        "0",
        color="red",
        fontsize=fontsize,
        fontweight="semibold",
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.15},
        zorder=5,
    )


def _apply_modal_axes(
    ax,
    damping_values,
    frequency_values,
    clamp_positive_max=True,
    allow_log_scales=True,
    right_xlim=None,
    fixed_xlim=None,
    fixed_ylim=None,
    fixed_xticks=None,
    force_symlog_x=False,
    frequency_guides=None,
    frequency_guides_as_y_ticks=False,
):
    _set_modal_axis_view(
        ax,
        damping_values,
        frequency_values,
        clamp_positive_max=clamp_positive_max,
        allow_log_scales=allow_log_scales,
        force_symlog_x=force_symlog_x,
    )
    if right_xlim is not None:
        current_left, _ = ax.get_xlim()
        ax.set_xlim(current_left, float(right_xlim))
    if fixed_xlim is not None:
        ax.set_xlim(*fixed_xlim)
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    if fixed_xticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(fixed_xticks))
    if frequency_guides_as_y_ticks:
        _apply_frequency_guide_ticks(ax, frequency_guides)


def plot_modal_signal_grid(df_results, gen, signals, output_dir, filename, title, colors=None):
    colors = colors or SIGNAL_COLORS
    fig = modal_grid_figure(len(signals), ncols=2)
    axes, nrows = create_adaptive_grid(fig, len(signals), ncols=2, sharex=True, sharey=True)
    fig.suptitle(title, fontweight="bold")

    for idx, (ax, signal) in enumerate(zip(axes, signals)):
        signal_data = df_results[(df_results["Gen"] == gen) & (df_results["Signal"] == signal)]
        ax.scatter(
            signal_data["Damping"],
            signal_data["Frequency"],
            color=colors[signal],
            alpha=0.6,
            edgecolors="k",
            s=50,
        )
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(signal, fontweight="semibold")
        if idx // 2 == nrows - 1:
            ax.set_xlabel(MODAL_X_LABEL)
        if idx % 2 == 0 or len(signals) == 1:
            ax.set_ylabel(MODAL_Y_LABEL)
        _set_modal_axis_view(ax, signal_data["Damping"], signal_data["Frequency"])
        style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_current_figure(output_dir, filename, fig)
    plt.close(fig)


def plot_modal_generator_grid(
    df_results,
    generators,
    signals,
    output_dir,
    filename,
    title,
    colors=None,
    clamp_positive_max=True,
    fixed_xlim=None,
    fixed_ylim=None,
    show_zero_line=True,
    right_xlim=None,
    fixed_xticks=None,
    allow_log_scales=True,
    frequency_guides=None,
    force_symlog_x=False,
    frequency_guides_as_y_ticks=False,
    reference_modes=None,
    annotate_reference_modes=False,
):
    colors = colors or SIGNAL_COLORS
    if frequency_guides is None:
        frequency_guides = _dominant_frequency_guides(df_results["Frequency"])
    fig = modal_grid_figure(len(generators), ncols=2, row_height=4.8)
    axes, nrows = create_adaptive_grid(fig, len(generators), ncols=2, sharex=True, sharey=True)
    fig.suptitle(title, fontweight="bold")

    for idx, (ax, gen) in enumerate(zip(axes, generators)):
        gen_data = df_results[df_results["Gen"] == gen]
        gen_reference_modes = _reference_modes_for_generator(reference_modes, gen)
        _add_frequency_guides(ax, frequency_guides)
        for signal in signals:
            signal_data = gen_data[gen_data["Signal"] == signal]
            ax.scatter(
                signal_data["Damping"],
                signal_data["Frequency"],
                label=signal,
                c=colors[signal],
                alpha=0.7,
                edgecolors="white",
                linewidths=0.45,
                s=42,
                zorder=3,
            )
        _overlay_reference_modes(ax, gen_reference_modes, annotate=annotate_reference_modes)
        if show_zero_line:
            ax.axvline(0, color="red", linestyle="-", alpha=0.75, linewidth=1.8)
        ax.set_title(generator_modal_label(gen), fontweight="semibold")
        if idx // 2 == nrows - 1:
            ax.set_xlabel(MODAL_X_LABEL)
        if idx % 2 == 0 or len(generators) == 1:
            ax.set_ylabel(MODAL_Y_LABEL)
        _apply_modal_axes(
            ax,
            gen_data["Damping"],
            gen_data["Frequency"],
            clamp_positive_max=clamp_positive_max,
            allow_log_scales=allow_log_scales,
            right_xlim=right_xlim,
            fixed_xlim=fixed_xlim,
            fixed_ylim=fixed_ylim,
            fixed_xticks=fixed_xticks,
            force_symlog_x=force_symlog_x,
            frequency_guides=frequency_guides,
            frequency_guides_as_y_ticks=frequency_guides_as_y_ticks,
        )
        if show_zero_line:
            _annotate_zero_reference(ax)
        if frequency_guides_as_y_ticks:
            ax.tick_params(axis="y", labelsize=12)
        style_axis(ax)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[signal], markersize=12, label=signal)
        for signal in signals
    ]
    legend_labels = list(signals)
    if reference_modes:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="black",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.6,
                linewidth=0,
                markersize=10,
                label="Reference Modes",
            )
        )
        legend_labels.append("Reference Modes")
    fig.legend(
        handles=handles,
        labels=legend_labels,
        loc="lower center",
        ncol=min(5, max(1, len(legend_labels))),
        title="Signals / Markers",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_current_figure(output_dir, filename, fig)
    plt.close(fig)


def plot_modal_combined_map(
    df_results,
    output_dir,
    filename,
    title,
    signals,
    gen=None,
    colors=None,
    figsize=(10, 6),
    fixed_xlim=None,
    fixed_ylim=None,
    show_zero_line=True,
    fixed_xticks=None,
    right_xlim=None,
    allow_log_scales=True,
    frequency_guides=None,
    force_symlog_x=False,
    frequency_guides_as_y_ticks=False,
):
    return _plot_modal_combined_map(
        df_results=df_results,
        output_dir=output_dir,
        filename=filename,
        title=title,
        signals=signals,
        gen=gen,
        colors=colors,
        figsize=figsize,
        fixed_xlim=fixed_xlim,
        fixed_ylim=fixed_ylim,
        show_zero_line=show_zero_line,
        fixed_xticks=fixed_xticks,
        right_xlim=right_xlim,
        allow_log_scales=allow_log_scales,
        frequency_guides=frequency_guides,
        force_symlog_x=force_symlog_x,
        frequency_guides_as_y_ticks=frequency_guides_as_y_ticks,
    )


def _plot_modal_combined_map(
    df_results,
    output_dir,
    filename,
    title,
    signals,
    gen=None,
    colors=None,
    figsize=(10, 6),
    fixed_xlim=None,
    fixed_ylim=None,
    show_zero_line=True,
    fixed_xticks=None,
    right_xlim=None,
    allow_log_scales=True,
    frequency_guides=None,
    force_symlog_x=False,
    frequency_guides_as_y_ticks=False,
):
    colors = colors or SIGNAL_COLORS
    if gen is None:
        plot_df = df_results
    else:
        plot_df = df_results[df_results["Gen"] == gen]

    if plot_df.empty:
        return
    if frequency_guides is None:
        frequency_guides = _dominant_frequency_guides(plot_df["Frequency"])

    fig = plt.figure(figsize=figsize)
    _add_frequency_guides(plt.gca(), frequency_guides)
    for signal in signals:
        signal_data = plot_df[plot_df["Signal"] == signal]
        if signal_data.empty:
            continue
        plt.scatter(
            signal_data["Damping"],
            signal_data["Frequency"],
            label=signal,
            c=colors[signal],
            alpha=0.5,
            edgecolors="white",
            linewidths=0.45,
            s=46,
            zorder=3,
        )

    if show_zero_line:
        plt.axvline(0, color="red", linestyle="-", alpha=0.75, linewidth=1.8)
    plt.title(title, fontweight="bold")
    plt.xlabel(MODAL_X_LABEL)
    plt.ylabel(MODAL_Y_LABEL)
    ax = plt.gca()
    _apply_modal_axes(
        ax,
        plot_df["Damping"],
        plot_df["Frequency"],
        allow_log_scales=allow_log_scales,
        right_xlim=right_xlim,
        fixed_xlim=fixed_xlim,
        fixed_ylim=fixed_ylim,
        fixed_xticks=fixed_xticks,
        force_symlog_x=force_symlog_x,
        frequency_guides=frequency_guides,
        frequency_guides_as_y_ticks=frequency_guides_as_y_ticks,
    )
    if show_zero_line:
        _annotate_zero_reference(ax)
    if frequency_guides_as_y_ticks:
        ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="lower left", framealpha=0.9)
    style_axis(ax)
    save_current_figure(output_dir, filename, fig)
    plt.close(fig)


def plot_modal_signal_focus_map(
    df_results,
    output_dir,
    filename,
    title,
    signals,
    colors=None,
    figsize=(11, 8),
    show_zero_line=True,
    right_xlim=None,
    fixed_xticks=None,
    allow_log_scales=True,
    frequency_guides=None,
    force_symlog_x=False,
    frequency_guides_as_y_ticks=False,
    reference_modes=None,
    annotate_reference_modes=False,
):
    colors = colors or SIGNAL_COLORS
    plot_df = df_results[df_results["Signal"].isin(signals)].copy()
    if plot_df.empty:
        return
    if frequency_guides is None:
        frequency_guides = _dominant_frequency_guides(plot_df["Frequency"])

    fig, axes = plt.subplots(len(signals), 1, figsize=figsize, sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    fig.suptitle(title, fontweight="bold")

    for idx, (ax, signal) in enumerate(zip(axes, signals)):
        signal_data = plot_df[plot_df["Signal"] == signal]
        _add_frequency_guides(ax, frequency_guides)
        ax.scatter(
            signal_data["Damping"],
            signal_data["Frequency"],
            c=colors[signal],
            alpha=0.62,
            edgecolors="white",
            linewidths=0.45,
            s=48,
            zorder=3,
        )
        _overlay_reference_modes(ax, reference_modes, annotate=annotate_reference_modes)
        if show_zero_line:
            ax.axvline(0, color="red", linestyle="-", alpha=0.75, linewidth=1.8)
        ax.set_title(f"{signal} Modes", fontweight="semibold")
        ax.set_ylabel(MODAL_Y_LABEL)
        ax.text(
            0.02,
            0.08,
            f"{len(signal_data)} modes",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#5c6370",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.25},
        )
        _apply_modal_axes(
            ax,
            plot_df["Damping"],
            plot_df["Frequency"],
            allow_log_scales=allow_log_scales,
            right_xlim=right_xlim,
            fixed_xticks=fixed_xticks,
            force_symlog_x=force_symlog_x,
            frequency_guides=frequency_guides,
            frequency_guides_as_y_ticks=frequency_guides_as_y_ticks,
        )
        if show_zero_line:
            _annotate_zero_reference(ax)
        if frequency_guides_as_y_ticks:
            ax.tick_params(axis="y", labelsize=11)
        style_axis(ax)

    axes[-1].set_xlabel(MODAL_X_LABEL)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_current_figure(output_dir, filename, fig)
    plt.close(fig)


def plot_reconstruction_method_grid(
    t,
    y_ref,
    reconstruction_rows,
    fetch_modes,
    reconstruct_signal,
    output_dir,
    filename,
    title,
    signal,
    x_lims=RECON_DEFAULT_X_LIMS,
):
    fig = reconstruction_grid_figure(len(reconstruction_rows))
    grid = fig.add_gridspec(max(1, len(reconstruction_rows)), 2)
    fig.suptitle(title, fontweight="bold", y=0.98)

    for row_idx, row_methods in enumerate(reconstruction_rows):
        methods = [method for method in row_methods if method is not None]
        if not methods:
            continue

        if len(methods) == 1:
            axis_method_pairs = [(fig.add_subplot(grid[row_idx, :]), methods[0], True)]
        else:
            axis_method_pairs = [
                (fig.add_subplot(grid[row_idx, 0]), methods[0], True),
                (fig.add_subplot(grid[row_idx, 1]), methods[1], False),
            ]

        for ax, method, show_ylabel in axis_method_pairs:
            ax.set_xlim(*x_lims)
            ax.tick_params(axis="both", labelsize=RECON_TICK_LABEL_SIZE)
            modes = fetch_modes(method)
            if modes is None or modes.empty:
                ax.text(0.5, 0.5, "No Data Found", ha="center", va="center", transform=ax.transAxes)
                continue

            y_est = reconstruct_signal(t, modes)
            rmse = float((((y_ref - y_est) ** 2).mean()) ** 0.5)
            total = float(((y_ref - y_ref.mean()) ** 2).sum())
            r2 = float("nan") if total == 0 else 1.0 - float(((y_ref - y_est) ** 2).sum()) / total

            ax.plot(t, y_ref, color="black", alpha=0.3, linewidth=2, label="Original (Filtered)")
            ax.plot(t, y_est, "--", color="red", linewidth=1.5, label=f"MP Estimate ($R^2$={r2:.4f})")
            ax.set_title(f"Method: {method} (RMSE: {rmse:.2e})", fontweight="semibold")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.75, linewidth=1.3, color="gray")
            if show_ylabel:
                ax.set_ylabel(signal_axis_label(signal), fontsize=RECON_AXIS_LABEL_SIZE)
            if row_idx == len(reconstruction_rows) - 1:
                ax.set_xlabel(RECON_X_LABEL, fontsize=RECON_AXIS_LABEL_SIZE)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_current_figure(output_dir, filename, fig)
    plt.close(fig)


def plot_best_reconstruction_grid(items, output_dir, filename, title, x_lims=RECON_DEFAULT_X_LIMS):
    fig = modal_grid_figure(len(items), ncols=2, row_height=8.0, min_height=8.0)
    axes, nrows = create_adaptive_grid(fig, len(items), ncols=2, sharex=True, span_last=True)
    fig.suptitle(title, fontweight="bold", y=0.99)

    for idx, (ax, item) in enumerate(zip(axes, items)):
        ax.set_xlim(*x_lims)
        ax.tick_params(axis="both", labelsize=RECON_TICK_LABEL_SIZE)
        if item.get("empty"):
            ax.text(0.5, 0.5, item.get("message", "No Data"), ha="center", va="center", transform=ax.transAxes)
            continue

        ax.plot(item["t"], item["y_ref"], color="black", alpha=0.3, linewidth=2, label="Original (filtered)")
        ax.plot(item["t"], item["y_est"], "--", color="red", linewidth=1.5, label="MP Estimate")
        ax.set_title(item["title"], fontweight="semibold")
        ax.set_ylabel(signal_axis_label(item["signal"]), fontsize=RECON_AXIS_LABEL_SIZE)
        if idx // 2 == nrows - 1:
            ax.set_xlabel(RECON_X_LABEL, fontsize=RECON_AXIS_LABEL_SIZE)
        ax.grid(True, linestyle=":", alpha=0.75, linewidth=1.3, color="gray")
        if item.get("show_legend"):
            ax.legend(loc="upper right")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.22, hspace=0.36)
    save_current_figure(output_dir, filename, fig)
    plt.close(fig)


def plot_bubble_map(df_results, output_dir, filename, source_builder=None, min_height=9.0):
    if df_results is None or df_results.empty:
        return

    plot_df = df_results.copy()
    if source_builder is None:
        plot_df["Src"] = plot_df["Gen"].astype(str) + " | " + plot_df["Signal"].astype(str)
    else:
        plot_df["Src"] = plot_df.apply(source_builder, axis=1)

    counts = plot_df.groupby("Src").size().reset_index(name="Count")
    plot_df = plot_df.merge(counts, on="Src")
    plot_df["Src"] = plot_df["Src"] + " | Poles: " + plot_df["Count"].astype(str)
    omega = 2 * np.pi * plot_df["Frequency"]
    plot_df["Energy"] = 0.5 * (omega ** 2) * (plot_df["Amplitude"] ** 2)
    norm_energy = (plot_df["Energy"] - plot_df["Energy"].min()) / (
        plot_df["Energy"].max() - plot_df["Energy"].min() + 1e-12
    )

    unique_sources = int(plot_df["Src"].nunique())
    # Bubble maps need much more vertical room than the modal grids because each
    # generator-signal label occupies a full categorical row.
    fig_height = max(min_height, min(0.62 * max(1, unique_sources) + 1.5, 42.0))
    fig_width = 15 if unique_sources <= 12 else 16
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(
        plot_df["Frequency"],
        plot_df["Src"],
        s=norm_energy * 800 + 100,
        c=plot_df["Damping"],
        cmap="RdYlGn",
        edgecolors="black",
    )
    plt.colorbar().set_label(r"Damping ($\sigma$)")
    plt.title("Modal Frequency/Damping/Energy Map", fontweight="bold")
    style_axis(plt.gca())
    save_current_figure(output_dir, filename, fig)
    plt.close(fig)
