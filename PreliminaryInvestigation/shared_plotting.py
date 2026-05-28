from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
MODAL_SYMLOG_LINTHRESH = 0.1
MODAL_SYMLOG_LINSCALE = 1.0


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


def _set_modal_axis_view(ax, damping_values, frequency_values):
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
    x_pad_right = max(0.02, 0.02 * x_span)
    y_pad = max(0.05, 0.05 * y_span)

    ax.set_xscale("symlog", linthresh=MODAL_SYMLOG_LINTHRESH, linscale=MODAL_SYMLOG_LINSCALE)
    ax.set_xlim(x_min - x_pad_left, max(0.02, x_max + x_pad_right))
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)


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


def plot_modal_generator_grid(df_results, generators, signals, output_dir, filename, title, colors=None):
    colors = colors or SIGNAL_COLORS
    fig = modal_grid_figure(len(generators), ncols=2, row_height=4.8)
    axes, nrows = create_adaptive_grid(fig, len(generators), ncols=2, sharex=True, sharey=True)
    fig.suptitle(title, fontweight="bold")

    for idx, (ax, gen) in enumerate(zip(axes, generators)):
        gen_data = df_results[df_results["Gen"] == gen]
        for signal in signals:
            signal_data = gen_data[gen_data["Signal"] == signal]
            ax.scatter(
                signal_data["Damping"],
                signal_data["Frequency"],
                label=signal,
                c=colors[signal],
                alpha=0.6,
                edgecolors="k",
                s=60,
            )
        ax.axvline(0, color="red", linestyle="-", alpha=0.3)
        ax.set_title(generator_modal_label(gen), fontweight="semibold")
        if idx // 2 == nrows - 1:
            ax.set_xlabel(MODAL_X_LABEL)
        if idx % 2 == 0 or len(generators) == 1:
            ax.set_ylabel(MODAL_Y_LABEL)
        _set_modal_axis_view(ax, gen_data["Damping"], gen_data["Frequency"])
        style_axis(ax)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[signal], markersize=12, label=signal)
        for signal in signals
    ]
    fig.legend(handles=handles, labels=signals, loc="lower center", ncol=min(4, max(1, len(signals))), title="Signals")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_current_figure(output_dir, filename, fig)
    plt.close(fig)


def plot_modal_combined_map(df_results, output_dir, filename, title, signals, gen=None, colors=None, figsize=(10, 6)):
    colors = colors or SIGNAL_COLORS
    if gen is None:
        plot_df = df_results
    else:
        plot_df = df_results[df_results["Gen"] == gen]

    if plot_df.empty:
        return

    fig = plt.figure(figsize=figsize)
    for signal in signals:
        signal_data = plot_df[plot_df["Signal"] == signal]
        if signal_data.empty:
            continue
        plt.scatter(
            signal_data["Damping"],
            signal_data["Frequency"],
            label=signal,
            c=colors[signal],
            alpha=0.6,
            edgecolors="k",
            s=60,
        )

    plt.axvline(0, color="red", linestyle="-", alpha=0.3)
    plt.title(title, fontweight="bold")
    plt.xlabel(MODAL_X_LABEL)
    plt.ylabel(MODAL_Y_LABEL)
    plt.legend()
    _set_modal_axis_view(plt.gca(), plot_df["Damping"], plot_df["Frequency"])
    style_axis(plt.gca())
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
