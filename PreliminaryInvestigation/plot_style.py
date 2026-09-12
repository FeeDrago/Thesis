import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

THESIS_SERIF_FONTS = ["GFS Artemisia", "Times New Roman", "serif"]

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

PDF_METADATA = {
    "CreationDate": None,
    "ModDate": None,
}


def apply_thesis_style():
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
