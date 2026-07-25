from pathlib import Path

import pandas as pd

from ambient_n4sid_analysis import (
    AMBIENT_DEFAULT_DBSCAN_SETTINGS,
    AMBIENT_DEFAULT_OPTICS_SETTINGS,
    CONTROL_AREAS,
    _load_json,
    _load_reference_modes,
    _reference_modes_for_control_area,
    _resolve_path,
    save_ambient_clustering_selection_summary,
    _save_combined_reference_mad_summary,
    _save_json,
)
from clustering_analysis import (
    _load_screened_data,
    _save_reference_mad_outputs,
    run_dbscan_modal_analysis,
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
        base_optics_settings = _extend_pm_settings(
            sweep_config.get("optics_settings", analysis_config.get("optics_settings", {})),
            AMBIENT_DEFAULT_OPTICS_SETTINGS,
        )
        base_dbscan_settings = _extend_pm_settings(
            sweep_config.get("dbscan_settings", analysis_config.get("dbscan_settings", {})),
            AMBIENT_DEFAULT_DBSCAN_SETTINGS,
        )

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
            if "dbscan" in methods:
                run_dbscan_modal_analysis(
                    str(area_results_path),
                    str(area_dir),
                    reference_modes=area_reference_modes,
                    dbscan_settings=base_dbscan_settings,
                )
            if "optics" in methods:
                run_optics_modal_analysis(
                    str(area_results_path),
                    str(area_dir),
                    reference_modes=area_reference_modes,
                    optics_settings=base_optics_settings,
                )

        _save_combined_reference_mad_summary(area_root, reference_modes)

        sweep_config["reference_modes_source"] = reference_source
        sweep_config["reference_modes"] = reference_modes
        sweep_config["optics_settings"] = base_optics_settings
        sweep_config["dbscan_settings"] = base_dbscan_settings
        _save_json(sweep_config_path, sweep_config)

    analysis_config["reference_modes_source"] = reference_source
    analysis_config["reference_modes"] = reference_modes
    analysis_config["optics_settings"] = _extend_pm_settings(
        analysis_config.get("optics_settings", {}), AMBIENT_DEFAULT_OPTICS_SETTINGS
    )
    analysis_config["dbscan_settings"] = _extend_pm_settings(
        analysis_config.get("dbscan_settings", {}), AMBIENT_DEFAULT_DBSCAN_SETTINGS
    )
    _save_json(config_path, analysis_config)
    save_ambient_clustering_selection_summary(base_output_dir, analysis_config)


def _extend_pm_settings(settings, defaults):
    merged = dict(defaults)
    merged.update(dict(settings or {}))
    merged["pm_values"] = sorted(
        {float(value) for value in defaults["pm_values"]}
        | {float(value) for value in (settings or {}).get("pm_values", [])}
    )
    merged["render_all_parameter_maps"] = False
    return merged


if __name__ == "__main__":
    main()
