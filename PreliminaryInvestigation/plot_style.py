import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import json
import os
import re
from pathlib import Path
from matplotlib import font_manager
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
import textwrap
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator, MaxNLocator

THESIS_SERIF_FONTS = ["Times New Roman", "GFS Artemisia", "serif"]

SIGNAL_COLORS = {
    "Voltage": "tab:blue",
    "Current": "tab:green",
    "Active Power": "tab:orange",
    "Reactive Power": "tab:red",
}

CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#ffd700",
    "#17becf",
    "#808000",
    "#393b79",
]

ACCENT_RED = "red"
LINE_BLUE = "tab:blue"
LINE_GREEN = "tab:green"
GRID_ALPHA_MAIN = 0.82
GRID_ALPHA_SUB = 0.68
POINT_ALPHA = 0.72
POINT_SIZE = 90
GRID_POINT_SIZE = 60
REP_SIZE = 260
REP_GRID_SIZE = 140

# The ambient selected-map renderer is the reference for ALL report exports.
REPORT_FIGSIZE = (11.5, 8.8)
REPORT_MARGINS = dict(left=0.11, right=0.97, top=0.88, bottom=0.30)
REPORT_LEGEND = dict(loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4, fontsize=10)
# One common envelope covers both ambient sweeps and both ringdown systems.
REPORT_MODAL_XLIM = (-6.5, 0.005)
REPORT_MODAL_YLIM = (0.0, 2.0)
REPORT_MODAL_YTICKS = (0.5, 1.0, 1.5, 2.0)
_RINGDOWN_STYLE = ContextVar("ringdown_plot_style", default=False)


PLOT_TRANSLATION_OVERRIDES = {
    # Preserve complete names deliberately omitted from the approved CSV.
    # Longest-match replacement stops generic words such as "Clustering"
    # from producing mixed-language method names.
    "Agglomerative Hierarchical Clustering": "Agglomerative Hierarchical Clustering",
    "Hierarchical Clustering": "Hierarchical Clustering",
    "Ordering Points To Identify the Clustering Structure": "Ordering Points To Identify the Clustering Structure",
    "Hierarchical Density-Based Spatial Clustering of Applications with Noise": "Hierarchical Density-Based Spatial Clustering of Applications with Noise",
    "soft clustering": "soft clustering",
    "Doubly-Fed Induction Generator": "Doubly-Fed Induction Generator",
    "Kundur Two-Area 4-Generator Benchmark System": "Kundur Two-Area 4-Generator Benchmark System",
    "System-Wide Modal Identification (All Generators)": "Χάρτης Ρυθμών (Όλες οι γεννήτριες)",
    "Modal Identification per Signal": "Ταυτοποίηση Ρυθμών ανά Σήμα",
    "Modal Analysis": "Ανάλυση Ρυθμών",
    "Combined Modal Map": "Συνδυασμένος Χάρτης Ρυθμών",
    "System-Wide Modal Map": "Χάρτης Ρυθμών σε Επίπεδο Συστήματος",
    "Voltage Modes": "Ρυθμοί Τάσης",
    "Current Modes": "Ρυθμοί Ρεύματος",
    "Active Power Modes": "Ρυθμοί Ενεργού Ισχύος",
    "Reactive Power Modes": "Ρυθμοί Άεργου Ισχύος",
    "Voltage": "Τάση",
    "Current": "Ρεύμα",
    "Active Power": "Ενεργός Ισχύς",
    "Reactive Power": "Άεργος Ισχύς",
    "Signals / Markers": "Σήματα / Δείκτες",
    "Absolute Best Signal Reconstruction (Max $R^2$)": "Βέλτιστη Ανακατασκευή Σήματος (Μέγιστο $R^2$)",
    "Reconstruction Accuracy": "Ακρίβεια Ανακατασκευής",
    "Left: Fixed Orders | Right: Adaptive Tau": "Αριστερά: Σταθερές Τάξεις Μοντέλου | Δεξιά: Προσαρμοστικό Tau",
    "Original (Filtered)": "Αρχικό (Φιλτραρισμένο)",
    "MP Estimate": "Εκτίμηση MP",
    "Time (s)": "Χρόνος (s)",
    "No Data Found": "Δεν Βρέθηκαν Δεδομένα",
    "No Data": "Χωρίς Δεδομένα",
    "Modal Frequency/Damping/Energy Map": "Διάγραμμα Συχνότητας, Απόσβεσης και Ενέργειας Ρυθμών",
    "Screened N4SID Estimates by Order Sweep and Control Area": "Διατηρούμενες Εκτιμήσεις N4SID ανά Σάρωση Τάξεων Μοντέλου και Περιοχή Ελέγχου",
    "Poles": "Πόλοι",
    "Pole count": "Πλήθος Πόλων",
    "$k-Means$ Parameter Optimization Grid": "Πλέγμα Βελτιστοποίησης της Παραμέτρου $k$ για k-Means",
    "$k-Medoids$ Parameter Optimization Grid": "Πλέγμα Βελτιστοποίησης της Παραμέτρου $k$ για k-Medoids",
    "$k-Means$ Results": "Αποτελέσματα k-Means",
    "$k-Medoids$ Results": "Αποτελέσματα k-Medoids",
    "Elbow Method for $k-Means$ Optimization": "Μέθοδος του Αγκώνα για Βελτιστοποίηση k-Means",
    "Elbow Method for $k$-Means Optimization": "Μέθοδος του Αγκώνα για Βελτιστοποίηση k-Means",
    "Elbow-Like Method for $k-Medoids$ Optimization": "Μέθοδος Αγκώνα για Βελτιστοποίηση k-Medoids",
    "Elbow-Like Method for $k$-Medoids Optimization": "Μέθοδος Αγκώνα για Βελτιστοποίηση k-Medoids",
    "Optimal Knee Point by Maximum Chord Distance": "Βέλτιστο Σημείο Καμπής βάσει Μέγιστης Απόστασης Χορδής",
    "Number of clusters (k)": "Αριθμός Συστάδων (k)",
    "Total Medoid Distance": "Συνολική Απόσταση από Medoids",
    "OPTICS Parameter Sweep": "Σάρωση Παραμέτρων OPTICS",
    "Modal Clustering with $k-Means$": "Συσταδοποίηση Ρυθμών με k-Means",
    "Modal Clustering with $k-Medoids$": "Συσταδοποίηση Ρυθμών με k-Medoids",
    "Modal Clustering with OPTICS": "Συσταδοποίηση Ρυθμών με OPTICS",
    "Modal Clustering with DBSCAN": "Συσταδοποίηση Ρυθμών με DBSCAN",
    "$k$-Means Cluster Map": "Χάρτης Συσταδοποίησης k-Means",
    "$k$-Medoids Cluster Map": "Χάρτης Συσταδοποίησης k-Medoids",
    "HDBSCAN Cluster Map": "Χάρτης Συσταδοποίησης HDBSCAN",
    "Gaussian Mixture Cluster Map": "Χάρτης Συσταδοποίησης Gaussian Mixture",
    "Agglomerative Cluster Map": "Χάρτης Συσταδοποίησης Agglomerative",
    "Cluster Map": "Χάρτης Συσταδοποίησης",
    "Cluster Maps": "Χάρτες Συσταδοποίησης",
    "Cost": "Κόστος",
    "full covariance": "πλήρης συνδιακύμανση",
    "Ward linkage": "σύνδεση Ward",
    "Reference Modes": "Ρυθμοί αναφοράς",
    "Reference Mode": "Ρυθμός αναφοράς",
    "Mode Count": "Πλήθος ρυθμών",
    "Cluster Means": "Εκπρόσωποι συστάδων",
    "Cluster Centers": "Εκπρόσωποι συστάδων",
    "Signal Type": "Τύπος σήματος",
    "Order Sweeps": "Σαρώσεις τάξεων μοντέλου",
    "Orders1": "Ομάδα τάξεων 1",
    "Orders2": "Ομάδα τάξεων 2",
    "Real Part vs Frequency": "Πραγματικό μέρος έναντι συχνότητας",
    "Mode Count by Signal Type": "Πλήθος ρυθμών ανά τύπο σήματος",
    "Mode Count by Method": "Πλήθος ρυθμών ανά μέθοδο",
    "Noise / filtered estimates": "Θόρυβος / φιλτραρισμένες εκτιμήσεις",
    "Centroids (cluster colours)": "Centroids (χρώματα συστάδων)",
    "Medoids (cluster colours)": "Medoids (χρώματα συστάδων)",
    "Clustered estimates (colours identify clusters)": "Ομαδοποιημένες εκτιμήσεις (τα χρώματα διακρίνουν τις συστάδες)",
    "Fixed Orders": "Σταθερές τάξεις μοντέλου",
    "No selected clustering": "Δεν επιλέχθηκε συσταδοποίηση",
    "Selected cluster maps": "Επιλεγμένοι χάρτες συσταδοποίησης",
    "Assigned": "Αντιστοίχιση",
}


@lru_cache(maxsize=1)
def _plot_translation_rules():
    """Load the terminology approved for visible thesis and plot text."""
    terminology_path = Path(__file__).resolve().parents[1] / "english_terms_translations.csv"
    mapping = {}
    if terminology_path.is_file():
        with terminology_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream, delimiter=";")
            next(rows, None)
            mapping.update({
                term: replacement[:1].upper() + replacement[1:]
                for term, replacement in rows
                if term and replacement
            })
    mapping.update(PLOT_TRANSLATION_OVERRIDES)
    terms = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z_-])(?:" + "|".join(re.escape(term) for term in terms) + r")(?![A-Za-z_-])",
        re.IGNORECASE,
    )
    return pattern, {term.casefold(): replacement for term, replacement in mapping.items()}


def translate_plot_text(value):
    """Translate approved terms without touching internal data-column names."""
    if not isinstance(value, str) or not value:
        return value
    pattern, mapping = _plot_translation_rules()
    translated = pattern.sub(lambda match: mapping[match.group(0).casefold()], value)
    translated = re.sub(
        r"\bεπιλεγμένο\s+(.*?)χάρτης συσταδοποίησης\b",
        lambda match: "Επιλεγμένος Χάρτης Συσταδοποίησης"
        + (f" {match.group(1).strip()}" if match.group(1).strip() else ""),
        translated,
        flags=re.IGNORECASE,
    )
    return translated


def translate_figure_text(fig):
    """Translate every visible Matplotlib text artist immediately before export."""
    for text in fig.findobj(Text):
        translated = translate_plot_text(text.get_text())
        if translated != text.get_text():
            text.set_text(translated)


@contextmanager
def ringdown_plot_style():
    """Use the ambient report's visual conventions without changing analysis."""
    token = _RINGDOWN_STYLE.set(True)
    try:
        yield
    finally:
        _RINGDOWN_STYLE.reset(token)


def ringdown_plotting(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with ringdown_plot_style():
            return function(*args, **kwargs)
    return wrapped


def using_ringdown_style():
    return _RINGDOWN_STYLE.get()


def set_report_modal_axes(ax):
    ax._report_kind = "modal"
    set_signed_symlog_damping_axis(ax, *REPORT_MODAL_XLIM)
    ax.set_yscale("linear")
    ax.set_ylim(*REPORT_MODAL_YLIM)
    ax.set_yticks(REPORT_MODAL_YTICKS)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def _report_legend(fig, axes):
    """Consolidate all legends, including manually constructed cluster keys."""
    entries = {}
    for legend in [*fig.legends, *(ax.get_legend() for ax in axes)]:
        if legend is None:
            continue
        for handle, label in zip(legend.legend_handles, legend.get_texts()):
            entries.setdefault(translate_plot_text(label.get_text()), handle)
        legend.remove()
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            translated_label = translate_plot_text(label)
            if translate_plot_text(f"{label} (cluster colours)") in entries:
                continue
            entries.setdefault(translated_label, handle)
    if entries:
        # Four columns are also used in the ambient selected cluster maps.
        return fig.legend(list(entries.values()), list(entries), **REPORT_LEGEND)
    return None


def finish_ringdown_figure(fig):
    """Final export pass: catches local font/legend/layout overrides everywhere.

    Modal axes share limits AND locators. Time and statistical axes retain
    their physical quantities; multi-panel figures share one panel geometry.
    PDF and PNG are subsequently saved from this exact same fixed canvas.
    """
    if not using_ringdown_style() or getattr(fig, "_report_finished", False):
        return
    axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>" and ax.axison]
    if not axes:
        return
    colorbars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    nrows = ncols = 1
    for ax in axes:
        spec = ax.get_subplotspec()
        if spec is not None:
            nr, nc = spec.get_gridspec().get_geometry()
            nrows, ncols = max(nrows, nr), max(ncols, nc)
    # Preserve the ambient standalone size; use an identical panel size for
    # every grid of the same shape, regardless of which system produced it.
    multi = len(axes) > 1
    if not multi:
        nrows = ncols = 1
    fig.set_size_inches(REPORT_FIGSIZE[0], REPORT_FIGSIZE[1] if not multi else 4.1 * nrows + 2.0)
    labelsize, ticksize, titlesize = (13, 12, 14) if multi else (26, 24, 28)
    if ncols > 2:
        labelsize, ticksize, titlesize = (size * 2 / ncols for size in (13, 12, 14))
    if getattr(axes[0], "_report_kind", None) == "bubble":
        fig.set_size_inches(REPORT_FIGSIZE[0], max(REPORT_FIGSIZE[1], 0.30 * len(axes[0].get_yticks()) + 3))
    for text in fig.findobj(Text):
        text.set_fontfamily("serif")
    for ax in axes:
        if getattr(ax, "_report_kind", None) == "modal":
            set_report_modal_axes(ax)
            for line in ax.lines:
                if np.array_equal(np.asarray(line.get_xdata()), [0, 0]):
                    line.set(color=ACCENT_RED, linestyle="--", alpha=0.35, linewidth=2)
        elif getattr(ax, "_report_kind", None) == "reconstruction":
            ax.set_xlim(0, 50)
            ax.set_xticks(np.arange(0, 51, 10))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        elif getattr(ax, "_report_kind", None) == "bubble":
            ax.set_xlim(0, REPORT_MODAL_YLIM[1])
            ax.set_xticks((0, *REPORT_MODAL_YTICKS))
        ax.tick_params(axis="both", labelsize=ticksize)
        ax.xaxis.label.set_size(labelsize)
        ax.yaxis.label.set_size(labelsize)
        ax.title.set_size(titlesize)
        ax.title.set_weight("bold")
        ax.title.set_text("\n".join(textwrap.fill(line, width=46 if multi else 58,
            break_long_words=False, break_on_hyphens=False) for line in ax.title.get_text().splitlines()))
        if getattr(ax, "_report_kind", None) == "bubble":
            ax.tick_params(axis="y", labelsize=12)
        for text in ax.texts:
            text.set_fontsize(10 if multi else 12)
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set_fontsize(ticksize)
        style_axis(ax)
        ax.grid(True, color=plt.rcParams["grid.color"])
        for spine in ax.spines.values():
            spine.set_visible(True)
    for ax in colorbars:
        ax.tick_params(labelsize=ticksize)
        ax.yaxis.label.set_size(labelsize)
    if fig._suptitle:
        fig._suptitle.set_fontsize(18 if multi else 28)
        fig._suptitle.set_y(0.985)
    legend = _report_legend(fig, axes)
    # Explicit positions avoid tight_layout and bbox='tight' changing plot
    # sizes with the number/length of legend entries.
    left, right = REPORT_MARGINS["left"], REPORT_MARGINS["right"]
    bottom, top = ((0.30 if nrows == 1 else 0.14), 0.90) if multi else (0.30, 0.88)
    if colorbars:
        left, right = (0.30, 0.82) if getattr(axes[0], "_report_kind", None) == "bubble" else (0.16, 0.80)
    gapx, gapy = (0.16 * min(1, 2 / ncols) if multi else 0), (0.075 if multi else 0)
    width = (right - left - (ncols - 1) * gapx) / ncols
    height = (top - bottom - (nrows - 1) * gapy) / nrows
    for ax in axes:
        spec = ax.get_subplotspec()
        row, col, rowspan, colspan = 0, 0, 1, 1
        if multi and spec is not None:
            row, col = spec.rowspan.start, spec.colspan.start
            rowspan, colspan = len(spec.rowspan), len(spec.colspan)
        ax.set_position([left + col * (width + gapx),
                         top - row * (height + gapy) - rowspan * height - (rowspan - 1) * gapy,
                         colspan * width + (colspan - 1) * gapx,
                         rowspan * height + (rowspan - 1) * gapy])
    for ax in colorbars:
        ax.set_position([right + 0.035, bottom, 0.025, top - bottom])

    # Greek labels can be wider than their English originals.  If the shared
    # bottom legend touches any axis label, raise the axes just enough to keep
    # the two bounding boxes separate while preserving the common top edge.
    if legend is not None and multi:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer)
        overlap_px = max(
            (
                legend_bbox.y1 - ax.get_tightbbox(renderer).y0
                if legend_bbox.x1 > ax.get_tightbbox(renderer).x0
                and legend_bbox.x0 < ax.get_tightbbox(renderer).x1
                and legend_bbox.y1 > ax.get_tightbbox(renderer).y0
                else 0
            )
            for ax in axes
        )
        if overlap_px > 0:
            bottom += overlap_px / fig.bbox.height + 0.012
            height = (top - bottom - (nrows - 1) * gapy) / nrows
            for ax in axes:
                spec = ax.get_subplotspec()
                row, col, rowspan, colspan = 0, 0, 1, 1
                if spec is not None:
                    row, col = spec.rowspan.start, spec.colspan.start
                    rowspan, colspan = len(spec.rowspan), len(spec.colspan)
                ax.set_position([left + col * (width + gapx),
                                 top - row * (height + gapy) - rowspan * height - (rowspan - 1) * gapy,
                                 colspan * width + (colspan - 1) * gapx,
                                 rowspan * height + (rowspan - 1) * gapy])
            for ax in colorbars:
                ax.set_position([right + 0.035, bottom, 0.025, top - bottom])
    fig._report_finished = True


def save_figure_pair(fig_or_plt, path_base, filename, fixed_canvas=False):
    """One export route for every plot, including diagnostics and statistics."""
    from pathlib import Path
    fig = fig_or_plt.gcf() if fig_or_plt is plt else fig_or_plt
    translate_figure_text(fig)
    finish_ringdown_figure(fig)
    fixed_canvas = fixed_canvas or using_ringdown_style()
    root = Path(path_base)
    for extension in ("pdf", "png"):
        (root / extension).mkdir(parents=True, exist_ok=True)
    save_pdf(fig, root / "pdf" / f"{filename}.pdf", tight=not fixed_canvas)
    fig.savefig(root / "png" / f"{filename}.png", dpi=300,
                bbox_inches=fig.bbox_inches if fixed_canvas else "tight")
    audit_path = os.environ.get("RINGDOWN_STYLE_AUDIT")
    if audit_path and using_ringdown_style():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        record = {
            "path": str(root / "pdf" / f"{filename}.pdf"),
            "size_inches": fig.get_size_inches().tolist(),
            "axes": [{
                "kind": getattr(ax, "_report_kind", "statistical"),
                "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                "xticks": np.asarray(ax.get_xticks()).tolist(), "yticks": np.asarray(ax.get_yticks()).tolist(),
                "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
                "xlabel_size": ax.xaxis.label.get_fontsize(),
                "tick_size": ax.xaxis.get_ticklabels()[0].get_fontsize() if ax.xaxis.get_ticklabels() else None,
                "internal_legend": ax.get_legend() is not None,
                "bounds": list(ax.get_position().bounds),
            } for ax in fig.axes if ax.get_label() != "<colorbar>" and ax.axison],
            "legends_inside_canvas": all(fig.bbox.contains(*corner)
                for legend in fig.legends for corner in legend.get_window_extent(renderer).get_points()),
            "legends_outside_axes": all(not legend.get_window_extent(renderer).overlaps(ax.get_tightbbox(renderer))
                for legend in fig.legends for ax in fig.axes if ax.get_label() != "<colorbar>" and ax.axison),
        }
        with open(audit_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

PDF_METADATA = {
    "CreationDate": None,
    "ModDate": None,
}


def apply_thesis_style():
    # Windows ambient PDFs embed Times New Roman. Register those same fonts
    # under WSL, where Matplotlib would otherwise silently choose Artemisia.
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        font_path = Path("/mnt/c/Windows/Fonts") / name
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    sns.set_theme(style="whitegrid", font="serif")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": THESIS_SERIF_FONTS,
        "font.size": 20,
        "axes.labelsize": 26,
        "axes.titlesize": 28,
        "figure.titlesize": 32,
        "legend.fontsize": 20,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.titleweight": "bold",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.1,
        "grid.linestyle": ":",
        "grid.linewidth": 1.2,
        "grid.alpha": GRID_ALPHA_MAIN,
        "lines.linewidth": 2.6,
        "lines.markersize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fancybox": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def style_axis(ax, grid_alpha=GRID_ALPHA_MAIN):
    ax.grid(True, linestyle=":", linewidth=1.2, alpha=grid_alpha)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def set_signed_symlog_damping_axis(ax, x_min, x_max=0.005, linthresh=0.05):
    """Format a signed logarithmic damping axis with readable decimal ticks."""
    ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
    ax.set_xlim(x_min, x_max)
    tick_candidates = (-10, -5, -3, -2, -1, -0.5, -0.2, -0.1, -0.05, -0.01, 0)
    ticks = [tick for tick in tick_candidates if x_min <= tick <= x_max]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:g}")
    )


def save_pdf(fig_or_plt, path, tight=True):
    """Save a PDF, optionally preserving the declared figure canvas.

    Most diagnostic plots benefit from cropping their unused margins.  The
    selected clustering maps are placed side-by-side in the thesis, however,
    so their PDF media boxes must remain identical regardless of the number of
    cluster labels in an individual legend.
    """
    save_kwargs = {"format": "pdf", "metadata": PDF_METADATA}
    # Explicitly override the global ``savefig.bbox`` setting.  Otherwise the
    # thesis style's default tight crop changes the physical PDF size even for
    # figures that require a fixed canvas.
    # Passing ``None`` defers to ``rcParams['savefig.bbox']`` (currently
    # ``tight``).  Use the figure's declared bounding box explicitly for
    # selected maps so their PDF media boxes are truly identical.
    save_kwargs["bbox_inches"] = "tight" if tight else fig_or_plt.bbox_inches
    fig_or_plt.savefig(path, **save_kwargs)
