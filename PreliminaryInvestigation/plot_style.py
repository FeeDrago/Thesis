import matplotlib.pyplot as plt
import seaborn as sns

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
    "#7f7f7f",
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


def save_pdf(fig_or_plt, path):
    fig_or_plt.savefig(path, format="pdf", bbox_inches="tight", metadata=PDF_METADATA)
