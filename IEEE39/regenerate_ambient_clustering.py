from pathlib import Path

import pandas as pd

from ambient_n4sid_analysis import (
    AMBIENT_DEFAULT_AGGLOMERATIVE_SETTINGS,
    AMBIENT_DEFAULT_DBSCAN_SETTINGS,
    AMBIENT_DEFAULT_GMM_SETTINGS,
    AMBIENT_DEFAULT_HDBSCAN_SETTINGS,
    AMBIENT_DEFAULT_CLUSTERING_METHODS,
    AMBIENT_DEFAULT_OPTICS_SETTINGS,
    AMBIENT_PAPER_MAD_SETTINGS,
    CONTROL_AREAS,
    _load_json,
    _load_reference_modes,
    _reference_modes_for_control_area,
    _resolve_path,
    _save_aggregated_paper_mad,
    save_ambient_clustering_selection_summary,
    _save_json,
)

from clustering_analysis import (
    run_agglomerative_modal_analysis,
    run_dbscan_modal_analysis,
    run_gmm_modal_analysis,
    run_hdbscan_modal_analysis,
    run_kmeans_modal_analysis,
    run_kmedoids_modal_analysis,
    run_optics_modal_analysis,
    run_silhouette_analysis,
)


LEGACY_DEFAULT_CLUSTERING_METHODS = ["kmeans", "kmedoids", "optics", "dbscan"]


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
        methods = _extend_clustering_methods(
            sweep_config.get("clustering_methods", analysis_config.get("clustering_methods"))
        )
        base_optics_settings = _extend_pm_settings(
            sweep_config.get("optics_settings", analysis_config.get("optics_settings", {})),
            AMBIENT_DEFAULT_OPTICS_SETTINGS,
        )
        base_dbscan_settings = _extend_pm_settings(
            sweep_config.get("dbscan_settings", analysis_config.get("dbscan_settings", {})),
            AMBIENT_DEFAULT_DBSCAN_SETTINGS,
        )
        base_hdbscan_settings = _extend_settings(
            sweep_config.get("hdbscan_settings", analysis_config.get("hdbscan_settings", {})),
            AMBIENT_DEFAULT_HDBSCAN_SETTINGS,
        )
        base_gmm_settings = _extend_settings(
            sweep_config.get("gmm_settings", analysis_config.get("gmm_settings", {})),
            AMBIENT_DEFAULT_GMM_SETTINGS,
        )
        base_agglomerative_settings = _extend_settings(
            sweep_config.get("agglomerative_settings", analysis_config.get("agglomerative_settings", {})),
            AMBIENT_DEFAULT_AGGLOMERATIVE_SETTINGS,
        )

        area_root = sweep_dir / "clustering" / "by_control_area"
        area_root.mkdir(parents=True, exist_ok=True)
        paper_mad_collectors = {method: [] for method in methods}

        for area_name, generators in CONTROL_AREAS.items():
            area_dir = area_root / area_name
            area_dir.mkdir(parents=True, exist_ok=True)
            area_results_path = area_dir / "results.csv"
            area_df = sweep_df[sweep_df["Gen"].isin(generators)].copy()
            area_df.to_csv(area_results_path, index=False)
            _save_json(area_dir / "control_area.json", {"name": area_name, "generators": generators})

            area_reference_modes = _reference_modes_for_control_area(reference_modes, area_name)
            if "kmeans" in methods:
                run_kmeans_modal_analysis(str(area_results_path), str(area_dir), reference_modes=area_reference_modes, paper_mad_collector=paper_mad_collectors["kmeans"])
            if "kmedoids" in methods:
                run_kmedoids_modal_analysis(str(area_results_path), str(area_dir), reference_modes=area_reference_modes, paper_mad_collector=paper_mad_collectors["kmedoids"])
            if {"kmeans", "kmedoids"}.issubset(set(methods)):
                run_silhouette_analysis(str(area_results_path), str(area_dir), reference_modes=area_reference_modes)
            if "dbscan" in methods:
                run_dbscan_modal_analysis(
                    str(area_results_path),
                    str(area_dir),
                    reference_modes=area_reference_modes,
                    dbscan_settings=base_dbscan_settings,
                    paper_mad_collector=paper_mad_collectors["dbscan"],
                )
            if "optics" in methods:
                run_optics_modal_analysis(
                    str(area_results_path),
                    str(area_dir),
                    reference_modes=area_reference_modes,
                    optics_settings=base_optics_settings,
                    paper_mad_collector=paper_mad_collectors["optics"],
                )
            if "hdbscan" in methods:
                run_hdbscan_modal_analysis(
                    str(area_results_path), str(area_dir), reference_modes=area_reference_modes,
                    hdbscan_settings=base_hdbscan_settings,
                    paper_mad_collector=paper_mad_collectors["hdbscan"],
                )
            if "gmm" in methods:
                run_gmm_modal_analysis(
                    str(area_results_path), str(area_dir), reference_modes=area_reference_modes,
                    gmm_settings=base_gmm_settings,
                    paper_mad_collector=paper_mad_collectors["gmm"],
                )
            if "agglomerative" in methods:
                run_agglomerative_modal_analysis(
                    str(area_results_path), str(area_dir), reference_modes=area_reference_modes,
                    agglomerative_settings=base_agglomerative_settings,
                    paper_mad_collector=paper_mad_collectors["agglomerative"],
                )

        _save_aggregated_paper_mad(sweep_dir, reference_modes, paper_mad_collectors)

        sweep_config["reference_modes_source"] = reference_source
        sweep_config["reference_modes"] = reference_modes
        sweep_config["clustering_methods"] = methods
        sweep_config["optics_settings"] = base_optics_settings
        sweep_config["dbscan_settings"] = base_dbscan_settings
        sweep_config["hdbscan_settings"] = base_hdbscan_settings
        sweep_config["gmm_settings"] = base_gmm_settings
        sweep_config["agglomerative_settings"] = base_agglomerative_settings
        sweep_config["paper_mad"] = dict(AMBIENT_PAPER_MAD_SETTINGS)
        _save_json(sweep_config_path, sweep_config)

    analysis_config["reference_modes_source"] = reference_source
    analysis_config["reference_modes"] = reference_modes
    analysis_config["clustering_methods"] = _extend_clustering_methods(analysis_config.get("clustering_methods"))
    analysis_config["optics_settings"] = _extend_pm_settings(
        analysis_config.get("optics_settings", {}), AMBIENT_DEFAULT_OPTICS_SETTINGS
    )
    analysis_config["dbscan_settings"] = _extend_pm_settings(
        analysis_config.get("dbscan_settings", {}), AMBIENT_DEFAULT_DBSCAN_SETTINGS
    )
    analysis_config["hdbscan_settings"] = _extend_settings(
        analysis_config.get("hdbscan_settings", {}), AMBIENT_DEFAULT_HDBSCAN_SETTINGS
    )
    analysis_config["gmm_settings"] = _extend_settings(
        analysis_config.get("gmm_settings", {}), AMBIENT_DEFAULT_GMM_SETTINGS
    )
    analysis_config["agglomerative_settings"] = _extend_settings(
        analysis_config.get("agglomerative_settings", {}), AMBIENT_DEFAULT_AGGLOMERATIVE_SETTINGS
    )
    analysis_config["paper_mad"] = dict(AMBIENT_PAPER_MAD_SETTINGS)
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


def _extend_clustering_methods(methods):
    """Migrate only the historical default method set; preserve explicit subsets."""
    if methods is None:
        return list(AMBIENT_DEFAULT_CLUSTERING_METHODS)
    resolved = list(methods)
    if set(resolved) == set(LEGACY_DEFAULT_CLUSTERING_METHODS) and len(resolved) == len(LEGACY_DEFAULT_CLUSTERING_METHODS):
        return [*resolved, *[method for method in AMBIENT_DEFAULT_CLUSTERING_METHODS if method not in resolved]]
    return resolved


def _extend_settings(settings, defaults):
    merged = dict(defaults)
    merged.update(dict(settings or {}))
    return merged


if __name__ == "__main__":
    main()
