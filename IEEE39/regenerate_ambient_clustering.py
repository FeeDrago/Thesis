from pathlib import Path

import pandas as pd

from ambient_n4sid_analysis import (
    CONTROL_AREAS,
    _load_json,
    _load_reference_modes,
    _reference_modes_for_control_area,
    _resolve_path,
    _save_combined_reference_mad_summary,
    _save_json,
)
from clustering_analysis import (
    _load_screened_data,
    _save_reference_mad_outputs,
    run_kmeans_modal_analysis,
    run_kmedoids_modal_analysis,
    run_optics_modal_analysis,
    run_silhouette_analysis,
)


def main():
    scenario_name = "Ambient_Mag0.1_T600s_dt10ms_seed1997"
    base_output_dir = _resolve_path(f"analysis/{scenario_name}")
    config_path = base_output_dir / "analysis_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing ambient analysis config: {config_path}")

    analysis_config = _load_json(config_path)
    reference_source, reference_modes = _load_reference_modes(analysis_config["data_dir"])

    for sweep in analysis_config.get("sweeps", []):
        sweep_dir = _resolve_path(sweep["output_dir"])
        sweep_results_path = _resolve_path(sweep["results_csv"])
        if not sweep_results_path.exists():
            raise SystemExit(f"Missing sweep results: {sweep_results_path}")

        sweep_df = pd.read_csv(sweep_results_path)
        sweep_config_path = sweep_dir / "analysis_config.json"
        sweep_config = _load_json(sweep_config_path) if sweep_config_path.exists() else {}
        methods = sweep_config.get("clustering_methods", analysis_config.get("clustering_methods", ["kmeans", "kmedoids", "optics"]))
        base_optics_settings = dict(sweep_config.get("optics_settings", analysis_config.get("optics_settings", {})))

        area_root = sweep_dir / "clustering" / "by_control_area"
        area_root.mkdir(parents=True, exist_ok=True)

        for area_name, generators in CONTROL_AREAS.items():
            area_dir = area_root / area_name
            area_dir.mkdir(parents=True, exist_ok=True)
            area_results_path = area_dir / "results.csv"
            area_df = sweep_df[sweep_df["Gen"].isin(generators)].copy()
            area_df.to_csv(area_results_path, index=False)
            _save_json(area_dir / "control_area.json", {"name": area_name, "generators": generators})

            area_reference_modes = _reference_modes_for_control_area(reference_modes, area_name)
            df_screened = _load_screened_data(str(area_results_path), str(area_dir))
            if df_screened is not None:
                _save_reference_mad_outputs(df_screened, str(area_dir), reference_modes=area_reference_modes)

            if "kmeans" in methods:
                run_kmeans_modal_analysis(str(area_results_path), str(area_dir), reference_modes=area_reference_modes)
            if "kmedoids" in methods:
                run_kmedoids_modal_analysis(str(area_results_path), str(area_dir), reference_modes=area_reference_modes)
            if {"kmeans", "kmedoids"}.issubset(set(methods)):
                run_silhouette_analysis(str(area_results_path), str(area_dir), reference_modes=area_reference_modes)
            if "optics" in methods:
                optics_settings = _selected_optics_settings_for_area(area_dir, base_optics_settings)
                run_optics_modal_analysis(
                    str(area_results_path),
                    str(area_dir),
                    reference_modes=area_reference_modes,
                    optics_settings=optics_settings,
                )

        _save_combined_reference_mad_summary(area_root, reference_modes)

        sweep_config["reference_modes_source"] = reference_source
        sweep_config["reference_modes"] = reference_modes
        _save_json(sweep_config_path, sweep_config)

    analysis_config["reference_modes_source"] = reference_source
    analysis_config["reference_modes"] = reference_modes
    _save_json(config_path, analysis_config)


def _selected_optics_settings_for_area(area_dir, base_optics_settings):
    settings = dict(base_optics_settings or {})
    settings["render_all_min_samples_maps"] = False
    settings["render_parameter_sweep_plot"] = False
    metrics_path = area_dir / "optics" / "optics_metrics_summary.csv"
    if not metrics_path.exists():
        return settings

    df = pd.read_csv(metrics_path)
    selected = df[df["selected"] == True]
    if selected.empty:
        return settings

    selected_min_samples = int(selected.iloc[0]["min_samples"])
    settings["min_samples_min"] = selected_min_samples
    settings["min_samples_max"] = selected_min_samples
    return settings


if __name__ == "__main__":
    main()
