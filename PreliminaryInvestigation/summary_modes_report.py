import csv
import math
from pathlib import Path
from statistics import mean


RESULTS_CSV = "results.csv"
OUTPUT_FILE = "summary_modes_report.txt"
INTER_AREA_BAND_HZ = (0.4, 0.6)
INTRA_AREA_BAND_HZ = (1.0, 1.1)
GENERATOR_ORDER = ["g1", "g2", "g3", "g4"]
SIGNAL_ORDER = ["Active Power", "Reactive Power", "Voltage", "Current"]
SIGNAL_SHORT = {
    "Active Power": "P",
    "Reactive Power": "Q",
    "Voltage": "V",
    "Current": "I",
}


def _energy(row):
    omega = 2.0 * math.pi * float(row["Frequency"])
    amplitude = float(row["Amplitude"])
    return 0.5 * (omega ** 2) * (amplitude ** 2)


def _read_results(base_dir):
    results_path = base_dir / RESULTS_CSV
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["Frequency"] = float(row["Frequency"])
        row["Damping"] = float(row["Damping"])
        row["Amplitude"] = float(row["Amplitude"])
        row["Phase"] = float(row["Phase"])
        row["Energy"] = _energy(row)
    return rows


def _best_mode(rows, gen, signal, band):
    lo, hi = band
    candidates = [
        row for row in rows
        if row["Gen"] == gen and row["Signal"] == signal and lo <= row["Frequency"] <= hi
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["Energy"])


def _fmt_num(value, decimals=3):
    return f"{value:.{decimals}f}"


def _fmt_or_dash(value, decimals=3):
    if value is None:
        return "--"
    return _fmt_num(value, decimals)


def _selected_entries(rows):
    selected = []
    for gen in GENERATOR_ORDER:
        for signal in SIGNAL_ORDER:
            inter = _best_mode(rows, gen, signal, INTER_AREA_BAND_HZ)
            intra = _best_mode(rows, gen, signal, INTRA_AREA_BAND_HZ)
            selected.append({
                "Gen": gen,
                "Signal": signal,
                "SignalShort": SIGNAL_SHORT[signal],
                "Inter": inter,
                "Intra": intra,
            })
    return selected


def _aggregate(selected, key):
    values = [entry[key] for entry in selected if entry[key] is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "mean_frequency": mean(row["Frequency"] for row in values),
        "mean_damping": mean(row["Damping"] for row in values),
        "min_frequency": min(row["Frequency"] for row in values),
        "max_frequency": max(row["Frequency"] for row in values),
        "min_damping": min(row["Damping"] for row in values),
        "max_damping": max(row["Damping"] for row in values),
    }


def _latex_rows(selected):
    lines = []
    for idx, gen in enumerate(GENERATOR_ORDER):
        gen_entries = [entry for entry in selected if entry["Gen"] == gen]
        for row_idx, entry in enumerate(gen_entries):
            gen_label = gen.upper() if row_idx == 0 else ""
            inter = entry["Inter"]
            intra = entry["Intra"]
            lines.append(
                f"{gen_label} & {entry['SignalShort']} & "
                f"{_fmt_or_dash(inter['Frequency'] if inter else None)} & "
                f"{_fmt_or_dash(inter['Damping'] if inter else None)} & "
                f"{entry['SignalShort'] if intra else '--'} & "
                f"{_fmt_or_dash(intra['Frequency'] if intra else None)} & "
                f"{_fmt_or_dash(intra['Damping'] if intra else None)} \\\\"
            )
        if idx < len(GENERATOR_ORDER) - 1:
            lines.append("\\midrule")
    return lines


def _write_report(base_dir, selected, inter_summary, intra_summary):
    path = base_dir / OUTPUT_FILE
    lines = []
    lines.append("TABLE_ROWS")
    lines.extend(_latex_rows(selected))
    lines.append("")
    lines.append("PARAGRAPH_NUMBERS")
    lines = [
        *lines,
        (
            f"Inter-area: count={inter_summary['count']}, "
            f"mean f={inter_summary['mean_frequency']:.4f} Hz, "
            f"mean sigma={inter_summary['mean_damping']:.4f} rad/s, "
            f"f-range=[{inter_summary['min_frequency']:.4f}, {inter_summary['max_frequency']:.4f}] Hz, "
            f"sigma-range=[{inter_summary['min_damping']:.4f}, {inter_summary['max_damping']:.4f}] rad/s"
        ),
        (
            f"Intra-area: count={intra_summary['count']}, "
            f"mean f={intra_summary['mean_frequency']:.4f} Hz, "
            f"mean sigma={intra_summary['mean_damping']:.4f} rad/s, "
            f"f-range=[{intra_summary['min_frequency']:.4f}, {intra_summary['max_frequency']:.4f}] Hz, "
            f"sigma-range=[{intra_summary['min_damping']:.4f}, {intra_summary['max_damping']:.4f}] rad/s"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    base_dir = Path(__file__).resolve().parent

    rows = _read_results(base_dir)
    selected = _selected_entries(rows)
    inter_summary = _aggregate(selected, "Inter")
    intra_summary = _aggregate(selected, "Intra")

    if inter_summary is None or intra_summary is None:
        raise RuntimeError("Could not compute summary modes from results.csv")

    report_path = _write_report(base_dir, selected, inter_summary, intra_summary)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
