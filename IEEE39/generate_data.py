from pathlib import Path
import argparse
import csv
import json
import re
import time
from textwrap import dedent

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

try:
    import powerfactory as pf
except ImportError:
    pf = None


# ============================================================
# USER SETTINGS
# ============================================================

PROJECT_NAME = "39 Bus New England System"
STUDY_CASE_NAME = "RMS mine"
GRID_NAME = "Grid"

AMBIENT_PROJECT_NAME = "39 Bus New England System TEST"
AMBIENT_STUDY_CASE_NAME = "RMS mine"
AMBIENT_GRID_NAME = None
AMBIENT_DEFAULT_NAME = "Ambient"
AMBIENT_DIST_MAG_PERCENT = 0.1
AMBIENT_LOW_PASS_HZ = 5.0
AMBIENT_RANDOM_SEED = 1997
AMBIENT_EXPORT_MODAL_CSVS = True

MIN_LOAD_MW = 100.0

EVENT_TIME_S = 0.0
SIM_STOP_TIME_S = 50.0
SIM_STEP_MS = 10.0
AMBIENT_SIM_STOP_TIME_S = 600.0

GENERATOR_NAMES = None

SCENARIOS = [
    {"name": None, "key": "load29", "load_name": "Load 29", "dp_percent": 2.0, "dq_percent": 0.0},
    {"name": None, "key": "load03", "load_name": "Load 03", "dp_percent": 2.0, "dq_percent": 0.0},
    {"name": None, "key": "load24", "load_name": "Load 24", "dp_percent": 2.0, "dq_percent": 0.0},
]

STEP_EVENT_RESULT_SCHEMA = {
    "variables": ["s:ut", "s:cur1", "s:Q1", "s:P1"],
    "headers": [
        "b:tnow in s",
        "s:ut in p.u.",
        "s:cur1 in p.u.",
        "s:Q1 in Mvar",
        "s:P1 in MW",
    ],
}

AMBIENT_RESULT_SCHEMA = {
    "variables": ["s:ut", "s:cur1"],
    "headers": [
        "b:tnow in s",
        "s:ut in p.u.",
        "s:cur1 in p.u.",
    ],
}


# ============================================================
# NAMING / PATHS
# ============================================================

def make_scenario_key(load_name, dp_percent, dq_percent):
    return f"{load_name.replace(' ', '').lower()}_p{float(dp_percent):g}_q{float(dq_percent):g}"


def make_load_alias(load_name):
    return load_name.replace(" ", "").lower()


def normalize_load_name(load_name):
    load_name = str(load_name).strip()
    match = re.fullmatch(r"load\s*(\d+)", load_name, flags=re.IGNORECASE)
    if match:
        return f"Load {int(match.group(1)):02d}"
    return load_name


def event_time_suffix(event_time_s):
    if abs(float(event_time_s) - float(EVENT_TIME_S)) < 1e-12:
        return ""
    return f"_evt{float(event_time_s):g}s"


def make_scenario_folder_alias(load_name, dp_percent, dq_percent, sim_stop_time, event_time_s=EVENT_TIME_S):
    load_part = re.sub(r"[^\w\-.+]+", "_", load_name.strip()).replace("_", "")
    p_part = f"Pplus{abs(float(dp_percent)):g}" if float(dp_percent) >= 0 else f"Pminus{abs(float(dp_percent)):g}"
    evt_part = event_time_suffix(event_time_s)

    if dq_percent is None or abs(float(dq_percent)) < 1e-12:
        return f"{load_part}_{p_part}_{sim_stop_time:g}s{evt_part}"

    q_part = f"Qplus{abs(float(dq_percent)):g}" if float(dq_percent) >= 0 else f"Qminus{abs(float(dq_percent)):g}"
    return f"{load_part}_{p_part}_{q_part}_{sim_stop_time:g}s{evt_part}"


def build_scenario_lookup(scenarios):
    lookup = {}
    for scenario in scenarios:
        dq_percent = scenario.get("dq_percent", 0.0)
        aliases = [
            make_load_alias(scenario["load_name"]),
            make_scenario_key(scenario["load_name"], scenario["dp_percent"], dq_percent),
            make_scenario_folder_alias(scenario["load_name"], scenario["dp_percent"], dq_percent, SIM_STOP_TIME_S),
        ]
        if scenario.get("key"):
            aliases.append(scenario["key"])
        if scenario.get("name"):
            aliases.append(scenario["name"])
        for alias in aliases:
            lookup[alias] = scenario
    return lookup


SCENARIOS_BY_NAME = build_scenario_lookup(SCENARIOS)


def get_base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def resolve_results_root(output_dir=None) -> Path:
    if output_dir is None:
        return get_base_dir() / "results"
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return get_base_dir() / path


def path_for_metadata(path: Path) -> str:
    try:
        return path.relative_to(get_base_dir()).as_posix()
    except ValueError:
        return path.name


def safe_name(text: str) -> str:
    return re.sub(r"[^\w\-.+]+", "_", str(text).strip())


def make_step_scenario_name(load, dp_percent, dq_percent, sim_stop_time, custom_name=None, event_time_s=EVENT_TIME_S):
    evt_part = event_time_suffix(event_time_s)
    if custom_name:
        return f"{safe_name(custom_name)}{evt_part}"

    load_part = safe_name(load.loc_name).replace("_", "")
    p_part = f"Pplus{abs(dp_percent):g}" if dp_percent >= 0 else f"Pminus{abs(dp_percent):g}"

    if dq_percent is None or abs(dq_percent) < 1e-12:
        return f"{load_part}_{p_part}_{sim_stop_time:g}s{evt_part}"

    q_part = f"Qplus{abs(dq_percent):g}" if dq_percent >= 0 else f"Qminus{abs(dq_percent):g}"
    return f"{load_part}_{p_part}_{q_part}_{sim_stop_time:g}s{evt_part}"


def make_ambient_scenario_name(sim_stop_time_s, sim_step_ms, magnitude_percent, random_seed, custom_name=None):
    if custom_name:
        return safe_name(custom_name)
    return (
        f"Ambient_Mag{abs(float(magnitude_percent)):g}"
        f"_T{float(sim_stop_time_s):g}s"
        f"_dt{float(sim_step_ms):g}ms"
        f"_seed{int(random_seed)}"
    )


# ============================================================
# POWERFACTORY CONNECTION / ACTIVATION
# ============================================================

def get_app():
    if pf is None:
        raise RuntimeError("PowerFactory Python module is not available in this environment.")

    app = pf.GetApplication()
    if app is None:
        app = pf.GetApplicationExt()
    if app is None:
        raise RuntimeError("Could not connect to PowerFactory.")

    try:
        app.Show()
    except Exception:
        pass
    try:
        app.ClearOutputWindow()
    except Exception:
        pass
    return app


def activate_project(app, project_name):
    project = app.GetActiveProject()
    if project is not None and project.loc_name == project_name:
        app.PrintPlain(f"Project already active: {project.loc_name}")
        return project

    app.PrintPlain(f"Activating project: {project_name}")
    ret = app.ActivateProject(project_name)
    project = app.GetActiveProject()
    if project is None:
        raise RuntimeError(f"Could not activate project '{project_name}'. ActivateProject returned: {ret}")
    app.PrintPlain(f"Active project: {project.loc_name}")
    return project


def find_study_case(app, study_case_name):
    study_folder = app.GetProjectFolder("study")
    if study_folder is None:
        raise RuntimeError("Could not find Study Cases folder.")

    try:
        all_cases = study_folder.GetContents("*.IntCase", 1)
    except Exception:
        all_cases = study_folder.GetContents()

    for sc in all_cases:
        if sc.loc_name == study_case_name:
            return sc

    available = [sc.loc_name for sc in all_cases]
    raise RuntimeError(f"Study case '{study_case_name}' not found.\nAvailable study cases:\n" + "\n".join(available))


def activate_study_case(app, study_case_name):
    study_case = find_study_case(app, study_case_name)
    app.PrintPlain(f"Activating study case: {study_case.loc_name}")
    study_case.Activate()
    active = app.GetActiveStudyCase()
    if active is None:
        raise RuntimeError("Study case activation failed.")
    app.PrintPlain(f"Active study case: {active.loc_name}")
    return active


def activate_grid_if_needed(app, grid_name=None):
    if grid_name is None:
        return None

    grids = app.GetCalcRelevantObjects("*.ElmNet")
    for grid in grids:
        if grid.loc_name == grid_name:
            app.PrintPlain(f"Activating grid: {grid.loc_name}")
            grid.Activate()
            return grid

    available = [g.loc_name for g in grids]
    raise RuntimeError(f"Grid '{grid_name}' not found.\nAvailable grids:\n" + "\n".join(available))


def activate_context(app, project_name, study_case_name, grid_name):
    project = activate_project(app, project_name)
    study_case = activate_study_case(app, study_case_name)
    grid = activate_grid_if_needed(app, grid_name)

    app.PrintPlain("PowerFactory context activated.")
    app.PrintPlain(f"Project: {project.loc_name}")
    app.PrintPlain(f"Study case: {study_case.loc_name}")
    if grid is not None:
        app.PrintPlain(f"Grid: {grid.loc_name}")
    return project, study_case, grid


def get_from_study_case(app, class_name: str):
    obj = app.GetFromStudyCase(class_name)
    if obj is None:
        raise RuntimeError(f"Could not find {class_name} in active Study Case.")
    return obj


# ============================================================
# LOAD / GENERATOR FINDERS
# ============================================================

def get_load_p_mw(load):
    for attr in ["plini", "plini_a", "pgini", "m:P:bus1"]:
        try:
            value = load.GetAttribute(attr)
            if value is not None:
                return float(value)
        except Exception:
            pass
    return None


def get_load_q_mvar(load):
    for attr in ["qlini", "qlini_a", "qgini", "m:Q:bus1"]:
        try:
            value = load.GetAttribute(attr)
            if value is not None:
                return float(value)
        except Exception:
            pass
    return None


def find_loads(app):
    loads = []
    for pattern in ["*.ElmLod", "*.ElmLodlv", "*.ElmLodmv"]:
        try:
            found = app.GetCalcRelevantObjects(pattern)
            if found:
                loads.extend(found)
        except Exception:
            pass

    unique = []
    seen = set()
    for load in loads:
        key = id(load)
        if key not in seen:
            unique.append(load)
            seen.add(key)

    if not unique:
        raise RuntimeError("No load objects found. Check active project/study case/grid.")
    return unique


def find_load(app, load_name=None, min_load_mw=100.0):
    loads = find_loads(app)
    if load_name is not None:
        normalized_load_name = normalize_load_name(load_name)
        for load in loads:
            if load.loc_name == load_name or normalize_load_name(load.loc_name) == normalized_load_name:
                return load
        available = [load.loc_name for load in loads]
        raise RuntimeError(f"Load '{load_name}' not found.\nAvailable loads:\n" + "\n".join(available))

    candidates = []
    for load in loads:
        p_mw = get_load_p_mw(load)
        if p_mw is not None and p_mw >= min_load_mw:
            candidates.append((p_mw, load))

    if not candidates:
        info = [f"{load.loc_name}: P={get_load_p_mw(load)} MW" for load in loads]
        raise RuntimeError(f"No load found with P >= {min_load_mw} MW.\nLoads:\n" + "\n".join(info))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_generators(app):
    gens = app.GetCalcRelevantObjects("*.ElmSym")
    if not gens:
        raise RuntimeError("No synchronous generators found: *.ElmSym")

    gens = sorted(gens, key=lambda g: g.loc_name)
    if GENERATOR_NAMES is None:
        return gens

    selected = []
    missing = []
    for name in GENERATOR_NAMES:
        match = next((gen for gen in gens if gen.loc_name == name), None)
        if match is None:
            missing.append(name)
        else:
            selected.append(match)

    if missing:
        available = [g.loc_name for g in gens]
        raise RuntimeError(f"Missing generators: {missing}\nAvailable generators:\n" + "\n".join(available))
    return selected


# ============================================================
# EVENTS / SIMULATION SETUP
# ============================================================

def clean_old_events(app):
    evt_folder = get_from_study_case(app, "IntEvt")
    for obj in list(evt_folder.GetContents()):
        try:
            obj.Delete()
        except Exception:
            pass


def set_first_existing_attribute(obj, candidates, value):
    last_error = None
    for attr in candidates:
        try:
            obj.SetAttribute(attr, value)
            return attr
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Could not set any of these attributes on {obj.loc_name}: {candidates}\nLast error: {last_error}"
    )


def create_load_event(app, load, time_s, dp_percent, dq_percent):
    evt_folder = get_from_study_case(app, "IntEvt")
    event = evt_folder.CreateObject("EvtLod", f"load_event_{safe_name(load.loc_name)}")

    set_first_existing_attribute(event, ["time"], time_s)
    target_attr = set_first_existing_attribute(event, ["p_target", "pTarget", "target", "p_object", "pObj"], load)
    p_attr = set_first_existing_attribute(event, ["dP", "dp", "P", "p", "dplini", "plini", "deltaP", "DeltaP"], dp_percent)

    q_attr = None
    if dq_percent is not None:
        try:
            q_attr = set_first_existing_attribute(
                event,
                ["dQ", "dq", "Q", "q", "dqlini", "qlini", "deltaQ", "DeltaQ"],
                dq_percent,
            )
        except Exception:
            q_attr = "not_set"

    return event, {
        "target_attribute": target_attr,
        "active_power_attribute": p_attr,
        "reactive_power_attribute": q_attr,
    }


def setup_result_variables(app, generators, schema):
    elmres = get_from_study_case(app, "ElmRes")
    try:
        elmres.Clear()
    except Exception:
        try:
            elmres.DeleteVars()
        except Exception:
            pass

    for gen in generators:
        for var in schema["variables"]:
            elmres.AddVars(gen, var)
    return elmres


def try_set_attr(obj, attr_names, value):
    for attr in attr_names:
        try:
            obj.SetAttribute(attr, value)
            return attr
        except Exception:
            pass
    return None


def configure_ambient_rms(app, step_ms):
    inc = get_from_study_case(app, "ComInc")
    try_set_attr(inc, ["iopt_sim"], "rms")
    try_set_attr(inc, ["tstart"], 0.0)
    try_set_attr(inc, ["dtgrd", "dt", "tstep", "dtemt", "dtout"], float(step_ms))
    try_set_attr(inc, ["iopt_sync"], 1)
    try_set_attr(inc, ["syncperiod"], float(step_ms))
    try_set_attr(inc, ["ciopt_sample"], 2)


def run_load_flow_initial_conditions_and_rms(app, tstop, step_ms):
    ldf = get_from_study_case(app, "ComLdf")
    inc = get_from_study_case(app, "ComInc")
    sim = get_from_study_case(app, "ComSim")

    app.PrintPlain("Running Load Flow...")
    err = ldf.Execute()
    if err:
        raise RuntimeError(f"Load Flow failed with error code {err}")

    inc_step_attr = try_set_attr(inc, ["dtgrd", "dt", "tstep", "dtemt", "dtout"], float(step_ms))
    sim_step_attr = try_set_attr(sim, ["dtgrd", "dt", "tstep", "dtemt", "dtout"], float(step_ms))
    sim_stop_attr = try_set_attr(sim, ["tstop", "tmax", "t_end"], float(tstop))

    app.PrintPlain(f"Initial Conditions step attr used: {inc_step_attr}")
    app.PrintPlain(f"Simulation step attr used: {sim_step_attr}")
    app.PrintPlain(f"Simulation stop attr used: {sim_stop_attr}")

    app.PrintPlain("Running RMS Initial Conditions...")
    err = inc.Execute()
    if err:
        raise RuntimeError(f"Initial Conditions failed with error code {err}")

    app.PrintPlain("Running RMS Simulation...")
    err = sim.Execute()
    if err:
        raise RuntimeError(f"RMS Simulation failed with error code {err}")


# ============================================================
# AMBIENT EXCITATION
# ============================================================

def require_ambient_dependencies():
    missing = []
    if np is None:
        missing.append("numpy")
    if scipy_signal is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError("Ambient mode requires these Python packages: " + ", ".join(missing))


def find_grid_file(app):
    netdat = app.GetProjectFolder("netdat")
    if not netdat:
        raise RuntimeError("Could not find netdat folder.")

    grids = netdat.GetContents("*.ElmNet")
    if grids:
        return grids[0]

    named = netdat.GetContents("Grid")
    if named:
        return named[0]

    raise RuntimeError("Could not find a grid object under netdat.")


def clear_old_ambient_load_models(grid):
    for old_comp in list(grid.GetContents("*_ExtLoad.ElmComp")):
        try:
            old_comp.Delete()
        except Exception:
            pass


def generate_filtered_noise(rng, fs_hz, n_samples, cutoff_hz):
    white_noise = rng.normal(0.0, 1.0, int(n_samples))
    nyquist = 0.5 * fs_hz
    if nyquist <= 0:
        raise RuntimeError("Ambient sampling frequency must be positive.")

    norm_cutoff = min(float(cutoff_hz) / nyquist, 0.999999)
    if norm_cutoff <= 0:
        raise RuntimeError("Ambient low-pass cutoff must be positive.")

    b, a = scipy_signal.butter(N=4, Wn=norm_cutoff, btype="low", analog=False)
    filtered = scipy_signal.lfilter(b, a, white_noise)
    max_abs = float(np.max(np.abs(filtered)))
    if max_abs > 0:
        filtered = filtered / max_abs
    return filtered


def set_attr_if_exists(obj, attr, value):
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        try:
            obj.SetAttribute(attr, value)
            return True
        except Exception:
            return False


def reset_calculation_if_possible(app):
    try:
        app.ResetCalculation()
    except Exception:
        pass


def convert_text_matrix_to_csv(src_path, dest_path):
    src_path = Path(src_path)
    dest_path = Path(dest_path)

    with src_path.open("r", encoding="utf-8", errors="replace") as src_handle, dest_path.open("w", newline="", encoding="utf-8") as dest_handle:
        writer = csv.writer(dest_handle)
        for raw_line in src_handle:
            line = raw_line.strip()
            if not line:
                continue

            if ";" in line:
                fields = [field.strip() for field in line.split(";")]
            elif "," in line:
                fields = [field.strip() for field in line.split(",")]
            else:
                fields = re.split(r"\s{2,}", line)
                if len(fields) <= 1:
                    fields = re.split(r"\s+", line)

            if fields:
                writer.writerow(fields)


def export_ambient_modal_analysis_csvs(app, scenario_dir):
    modal_dir = scenario_dir / "modal"
    modal_dir.mkdir(parents=True, exist_ok=True)

    reset_calculation_if_possible(app)

    com_mod = get_from_study_case(app, "ComMod")
    com_mod.dirMatl = str(modal_dir.resolve())
    com_mod.iEValMatl = 1
    com_mod.iPart = 1
    com_mod.iPartMatl = 1
    com_mod.isResOscModesOnly = 1
    com_mod.outputType = 1
    com_mod.iSysMatsMatl = 1

    err = com_mod.Execute()
    if err:
        raise RuntimeError(f"Ambient modal analysis export failed with error code {err}")

    raw_to_csv = {
        "EVals.mtl": "eigenvalues.csv",
        "PartFacs.mtl": "participation_factors.csv",
        "VariableToIdx_Amat.txt": "state_index.csv",
    }
    exported = []

    for raw_name, csv_name in raw_to_csv.items():
        raw_path = modal_dir / raw_name
        if not raw_path.exists():
            raise RuntimeError(f"Expected ambient modal artifact was not created: {raw_path}")
        csv_path = modal_dir / csv_name
        convert_text_matrix_to_csv(raw_path, csv_path)
        try:
            raw_path.unlink()
        except OSError:
            pass
        exported.append({"name": csv_name, "file": path_for_metadata(csv_path)})

    for unused_name in ["Amat.mtl", "Jacobian.mtl", "M.mtl", "VariableToIdx_Jacobian.txt"]:
        unused_path = modal_dir / unused_name
        if unused_path.exists():
            try:
                unused_path.unlink()
            except OSError:
                pass

    reset_calculation_if_possible(app)
    return exported


def create_ambient_load_profiles(app, scenario_dir, time_step_s, time_end_s, magnitude_percent, low_pass_hz, random_seed):
    require_ambient_dependencies()

    ambient_dir = scenario_dir / "ambient_load_profiles"
    ambient_dir.mkdir(parents=True, exist_ok=True)

    grid = find_grid_file(app)
    clear_old_ambient_load_models(grid)

    lib_folder = app.GetProjectFolder("lib")
    if lib_folder is None:
        raise RuntimeError("Could not find PowerFactory library folder.")
    udm_folder = lib_folder.GetContents("User Defined Models.IntPrjFolder")
    if not udm_folder:
        raise RuntimeError("Could not find 'User Defined Models' folder.")
    composite_types = udm_folder[0].GetContents("Composite Type Load.BlkDef")
    if not composite_types:
        raise RuntimeError("Could not find 'Composite Type Load.BlkDef'.")
    composite_type = composite_types[0]

    loads = app.GetCalcRelevantObjects("*.ElmLod")
    if not loads:
        raise RuntimeError("Ambient mode could not find any '*.ElmLod' loads.")
    time_vector = np.arange(-time_step_s, time_end_s + (2 * time_step_s), time_step_s)
    sample_count = len(time_vector)
    fs_hz = 1.0 / time_step_s
    rng = np.random.RandomState(int(random_seed))

    profile_rows = []
    for load in loads:
        base_p = get_load_p_mw(load)
        base_q = get_load_q_mvar(load)
        if base_p is None:
            continue
        if base_q is None:
            base_q = 0.0

        p_dist = generate_filtered_noise(rng, fs_hz, sample_count, low_pass_hz)
        q_dist = generate_filtered_noise(rng, fs_hz, sample_count, low_pass_hz)

        p_series = (p_dist * magnitude_percent / 100.0 + 1.0) * base_p
        q_series = (q_dist * magnitude_percent / 100.0 + 1.0) * base_q

        load_file = ambient_dir / f"{safe_name(load.loc_name)}.txt"
        with load_file.open("w", newline="") as handle:
            handle.write("2\n")
            for t_now, p_now, q_now in zip(time_vector, p_series, q_series):
                handle.write(f"{t_now:.4f}\t{p_now:.4f}\t{q_now:.4f}\n")

        new_comp = grid.CreateObject("ElmComp", f"{load.loc_name}_ExtLoad")
        new_comp.typ_id = composite_type

        current_load_type = getattr(load, "typ_id", None)
        if current_load_type is not None:
            for attr, value in [("systp", 0), ("phtech", 2), ("lodst", 0), ("loddy", 100), ("aP", 0), ("aQ", 0), ("bP", 0), ("bQ", 0)]:
                set_attr_if_exists(current_load_type, attr, value)

        for slot in getattr(new_comp, "pblk", []):
            if slot.loc_name == "load slot":
                slot.loc_name = "load_slot"
            elif slot.loc_name == "load measurement":
                slot.loc_name = "load_measurement"

        set_attr_if_exists(new_comp, "load_slot", load)
        measurement_file = new_comp.CreateObject("ElmFile", "Measurement File")
        measurement_file.f_name = str(load_file.resolve())
        set_attr_if_exists(new_comp, "load_measurement", measurement_file)

        profile_rows.append({
            "load_name": load.loc_name,
            "file": path_for_metadata(load_file),
            "samples": sample_count,
            "base_p_mw": base_p,
            "base_q_mvar": base_q,
        })

    if not profile_rows:
        raise RuntimeError("Ambient mode did not create any external load profiles.")
    return profile_rows


# ============================================================
# EXPORT / CSV VALIDATION
# ============================================================

def set_comres_attr(obj, attr, value):
    try:
        obj.SetAttribute(attr, value)
        return True
    except Exception:
        try:
            setattr(obj, attr, value)
            return True
        except Exception:
            return False


def export_raw_results_fast_comres(app, elmres, scenario_dir):
    comres = get_from_study_case(app, "ComRes")
    raw_csv = scenario_dir / "raw_all_generators.csv"

    app.PrintPlain(f"Fast exporting raw results to: {raw_csv}")
    print(f"Fast exporting raw results to: {raw_csv}", flush=True)

    set_comres_attr(comres, "pResult", elmres)
    set_comres_attr(comres, "f_name", str(raw_csv))
    set_comres_attr(comres, "iopt_exp", 6)
    set_comres_attr(comres, "iopt_csel", 0)
    set_comres_attr(comres, "iopt_tsel", 0)
    set_comres_attr(comres, "iopt_locn", 2)
    set_comres_attr(comres, "ciopt_head", 1)

    err = comres.Execute()
    if err:
        raise RuntimeError(f"ComRes export failed with error code {err}")

    app.PrintPlain("Fast raw ComRes export done.")
    print("Fast raw ComRes export done.", flush=True)
    return raw_csv


def read_comres_csv_flexible(raw_csv):
    if pd is None:
        raise RuntimeError("pandas is required to split ComRes CSV results.")

    attempts = [
        {"sep": ";", "header": [0, 1]},
        {"sep": ",", "header": [0, 1]},
        {"sep": None, "header": [0, 1], "engine": "python"},
        {"sep": ";", "header": 0},
        {"sep": ",", "header": 0},
        {"sep": None, "header": 0, "engine": "python"},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(raw_csv, **kwargs)
            if df.shape[0] > 0 and df.shape[1] > 1:
                return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read raw ComRes CSV: {last_error}")


def normalize_col_name(col):
    if isinstance(col, tuple):
        parts = []
        for item in col:
            text = str(item)
            if text.lower() != "nan" and "unnamed" not in text.lower():
                parts.append(text)
        return " ".join(parts)
    return str(col)


def compact_text(text):
    return str(text).replace(" ", "").replace("_", "").replace("-", "").lower()


def find_time_column_pandas(df):
    for col in df.columns:
        compact = compact_text(normalize_col_name(col))
        if "tnow" in compact or "time" in compact or "b:tnow" in compact:
            return col
    return df.columns[0]


def find_generator_variable_column(df, gen_name, variable):
    gen_key = compact_text(gen_name)
    var_key = compact_text(variable)

    for col in df.columns:
        compact = compact_text(normalize_col_name(col))
        if gen_key in compact and var_key in compact:
            return col

    gen_digits = "".join(ch for ch in gen_name if ch.isdigit())
    if gen_digits:
        for col in df.columns:
            compact = compact_text(normalize_col_name(col))
            if var_key in compact and gen_digits in compact:
                return col

    sample_cols = [normalize_col_name(c) for c in list(df.columns)[:20]]
    raise RuntimeError(
        f"Could not find column for generator '{gen_name}', variable '{variable}'.\nFirst columns seen:\n"
        + "\n".join(sample_cols)
    )


def parse_numeric_text(value):
    value = str(value).strip().replace(",", ".")
    if not value or value.lower() in ("nan", "none"):
        return ""
    try:
        return f"{float(value):.10g}"
    except ValueError:
        return ""


def find_generator_variable_index(object_headers, variable_headers, gen_name, variable):
    gen_key = compact_text(gen_name)
    var_key = compact_text(variable)
    gen_digits = "".join(ch for ch in gen_name if ch.isdigit())

    for idx, (object_header, variable_header) in enumerate(zip(object_headers, variable_headers)):
        compact_object = compact_text(object_header)
        compact_variable = compact_text(variable_header)
        if var_key not in compact_variable:
            continue
        if gen_key in compact_object or (gen_digits and gen_digits in compact_object):
            return idx

    raise RuntimeError(f"Could not find raw CSV column for generator '{gen_name}', variable '{variable}'.")


def split_raw_comres_standard_csv(raw_csv, generators, scenario_dir, schema):
    with open(raw_csv, newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        try:
            object_headers = next(reader)
            variable_headers = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"Raw ComRes CSV is missing headers: {raw_csv}") from exc

        if len(object_headers) <= 1 or len(variable_headers) <= 1:
            raise RuntimeError("Raw ComRes CSV does not look like a semicolon-separated two-header export.")

        time_idx = next(
            (
                idx
                for idx, header in enumerate(variable_headers)
                if "tnow" in compact_text(header) or "time" in compact_text(header)
            ),
            0,
        )

        generator_columns = []
        for gen in generators:
            indices = [find_generator_variable_index(object_headers, variable_headers, gen.loc_name, var) for var in schema["variables"]]
            generator_columns.append(indices)

        outputs = []
        writers = []
        try:
            for idx in range(1, len(generators) + 1):
                out_csv = scenario_dir / f"g{idx}.csv"
                out_file = open(out_csv, "w", newline="")
                writer = csv.writer(out_file)
                writer.writerow(schema["headers"])
                outputs.append((out_csv, out_file))
                writers.append(writer)

            row_counts = [0] * len(generators)
            for row in reader:
                if not row:
                    continue
                for gen_idx, column_indices in enumerate(generator_columns):
                    values = [parse_numeric_text(row[time_idx] if time_idx < len(row) else "")]
                    values.extend(parse_numeric_text(row[col_idx] if col_idx < len(row) else "") for col_idx in column_indices)
                    if any(value != "" for value in values):
                        row_counts[gen_idx] += 1
                    writers[gen_idx].writerow(values)

            for gen_idx, (gen, (out_csv, _)) in enumerate(zip(generators, outputs)):
                if row_counts[gen_idx] == 0:
                    raise RuntimeError(f"Split produced no numeric rows for {gen.loc_name} -> {out_csv}")
                print(f"Saved {out_csv}", flush=True)
        finally:
            for _, out_file in outputs:
                out_file.close()


def to_numeric_dot_decimal(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    missing = series.str.lower().isin(["", "nan", "none"])
    series = series.mask(missing)
    return pd.to_numeric(series, errors="coerce")


def split_raw_comres_to_generator_csvs(raw_csv, generators, scenario_dir, schema):
    print("Splitting raw CSV into one file per generator...", flush=True)

    try:
        split_raw_comres_standard_csv(raw_csv, generators, scenario_dir, schema)
        return
    except Exception as exc:
        print(f"Standard CSV split failed, trying pandas fallback: {exc}", flush=True)

    df = read_comres_csv_flexible(raw_csv)
    time_col = find_time_column_pandas(df)

    for idx, gen in enumerate(generators, start=1):
        output = pd.DataFrame()
        output[schema["headers"][0]] = to_numeric_dot_decimal(df[time_col])
        for variable, clean_header in zip(schema["variables"], schema["headers"][1:]):
            raw_col = find_generator_variable_column(df, gen.loc_name, variable)
            output[clean_header] = to_numeric_dot_decimal(df[raw_col])

        out_csv = scenario_dir / f"g{idx}.csv"
        if output.empty or output.dropna(how="all").empty:
            raise RuntimeError(f"Split produced no numeric rows for {gen.loc_name} -> {out_csv}")
        output.to_csv(out_csv, index=False, float_format="%.10g")
        print(f"Saved {out_csv}", flush=True)


def parse_csv_float(value, csv_path, row_number, column_name):
    text = str(value).strip().replace(",", ".")
    if not text:
        raise RuntimeError(f"Empty value in {csv_path}, row {row_number}, column '{column_name}'")
    try:
        return float(text)
    except ValueError as exc:
        raise RuntimeError(
            f"Non-numeric value in {csv_path}, row {row_number}, column '{column_name}': {value}"
        ) from exc


def validate_generator_csvs(scenario_dir, generators, schema):
    csv_files = []
    headers = schema["headers"]

    for idx, gen in enumerate(generators, start=1):
        csv_path = scenario_dir / f"g{idx}.csv"
        if not csv_path.exists():
            raise RuntimeError(f"Missing generated CSV: {csv_path}")

        with open(csv_path, newline="") as handle:
            reader = csv.reader(handle)
            try:
                found_headers = next(reader)
            except StopIteration as exc:
                raise RuntimeError(f"Generated CSV is empty: {csv_path}") from exc

            if found_headers != headers:
                raise RuntimeError(f"Unexpected headers in {csv_path}.\nExpected: {headers}\nFound: {found_headers}")

            row_count = 0
            previous_time = None
            for row_number, row in enumerate(reader, start=2):
                if not row or all(str(value).strip() == "" for value in row):
                    continue
                if len(row) != len(headers):
                    raise RuntimeError(
                        f"Wrong number of columns in {csv_path}, row {row_number}. Expected {len(headers)}, found {len(row)}"
                    )

                values = [parse_csv_float(value, csv_path, row_number, column_name) for value, column_name in zip(row, headers)]
                current_time = values[0]
                if previous_time is not None and current_time < previous_time:
                    raise RuntimeError(f"Time column is not monotonic in {csv_path}, row {row_number}")
                previous_time = current_time
                row_count += 1

        if row_count == 0:
            raise RuntimeError(f"Generated CSV has no numeric data rows: {csv_path}")

        csv_files.append({"generator": gen.loc_name, "file": path_for_metadata(csv_path), "rows": row_count})

    return csv_files


def export_results_fast_and_split(app, elmres, generators, scenario_dir, schema):
    raw_csv = export_raw_results_fast_comres(app, elmres, scenario_dir)
    split_raw_comres_to_generator_csvs(raw_csv, generators, scenario_dir, schema)
    return validate_generator_csvs(scenario_dir, generators, schema)


# ============================================================
# DEBUG
# ============================================================

def print_debug_context(app):
    project = app.GetActiveProject()
    study_case = app.GetActiveStudyCase()
    loads = app.GetCalcRelevantObjects("*.ElmLod")
    gens = app.GetCalcRelevantObjects("*.ElmSym")

    print("Active project:", project.loc_name if project else None, flush=True)
    print("Active study case:", study_case.loc_name if study_case else None, flush=True)

    app.PrintPlain(f"Active project: {project.loc_name if project else None}")
    app.PrintPlain(f"Active study case: {study_case.loc_name if study_case else None}")
    app.PrintPlain(f"Number of ElmLod: {len(loads)}")
    app.PrintPlain(f"Number of ElmSym: {len(gens)}")

    if project is None:
        raise RuntimeError("No active project.")
    if study_case is None:
        raise RuntimeError("No active study case.")
    if not loads:
        raise RuntimeError("No ElmLod loads found.")
    if not gens:
        raise RuntimeError("No ElmSym generators found.")


# ============================================================
# SCENARIO RUNNERS
# ============================================================

def write_scenario_json(scenario_dir, config):
    with open(scenario_dir / "scenario.json", "w") as handle:
        json.dump(config, handle, indent=2)


def run_step_scenario(app, scenario, results_root, context_settings):
    load_name = scenario.get("load_name")
    dp_percent = float(scenario.get("dp_percent", 2.0))
    dq_percent = float(scenario.get("dq_percent", 0.0))
    sim_stop_time_s = float(scenario.get("sim_stop_time_s", SIM_STOP_TIME_S))
    event_time_s = float(scenario.get("event_time_s", EVENT_TIME_S))
    custom_name = scenario.get("name")
    schema = STEP_EVENT_RESULT_SCHEMA

    load = find_load(app, load_name, MIN_LOAD_MW)
    p_mw = get_load_p_mw(load)
    scenario_name = make_step_scenario_name(load, dp_percent, dq_percent, sim_stop_time_s, custom_name, event_time_s)
    scenario_dir = results_root / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "scenario_name": scenario_name,
        "disturbance_type": "step_event",
        "project_name": context_settings["project_name"],
        "study_case_name": context_settings["study_case_name"],
        "grid_name": context_settings.get("grid_name"),
        "load_name": load.loc_name,
        "load_initial_p_mw": p_mw,
        "min_load_mw": MIN_LOAD_MW,
        "dp_percent": dp_percent,
        "dq_percent": dq_percent,
        "event_time_s": event_time_s,
        "sim_stop_time_s": sim_stop_time_s,
        "sim_step_ms": context_settings["sim_step_ms"],
        "csv_headers": schema["headers"],
        "generator_names_setting": GENERATOR_NAMES,
    }

    try:
        clean_old_events(app)
        event, event_attrs = create_load_event(app, load, event_time_s, dp_percent, dq_percent)
        config["event_name"] = event.loc_name
        config["event_attributes_used"] = event_attrs

        generators = find_generators(app)
        config["generators"] = [g.loc_name for g in generators]
        elmres = setup_result_variables(app, generators, schema)
        run_load_flow_initial_conditions_and_rms(app, sim_stop_time_s, context_settings["sim_step_ms"])
        config["csv_files"] = export_results_fast_and_split(app, elmres, generators, scenario_dir, schema)
        config["status"] = "OK"
        write_scenario_json(scenario_dir, config)
        return config
    except Exception as exc:
        config["status"] = "FAILED"
        config["error"] = str(exc)
        write_scenario_json(scenario_dir, config)
        return config


def parse_ambient_scenario_name(scenario_names):
    if not scenario_names:
        return None
    if len(scenario_names) != 1:
        raise SystemExit("--ambient accepts at most one optional --scenario value, used only as the ambient run folder label.")
    value = scenario_names[0].strip()
    if not value:
        raise SystemExit("Ambient scenario label cannot be empty.")
    return value


def run_ambient_scenario(app, results_root, context_settings, ambient_name, sim_stop_time_s, sim_step_ms, magnitude_percent, low_pass_hz, random_seed):
    schema = AMBIENT_RESULT_SCHEMA
    scenario_name = make_ambient_scenario_name(sim_stop_time_s, sim_step_ms, magnitude_percent, random_seed, ambient_name)
    scenario_dir = results_root / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "scenario_name": scenario_name,
        "disturbance_type": "ambient",
        "project_name": context_settings["project_name"],
        "study_case_name": context_settings["study_case_name"],
        "grid_name": context_settings.get("grid_name"),
        "sim_stop_time_s": sim_stop_time_s,
        "sim_step_ms": sim_step_ms,
        "csv_headers": schema["headers"],
        "generator_names_setting": GENERATOR_NAMES,
        "ambient_seed_effective": int(random_seed),
        "ambient_settings": {
            "magnitude_percent": magnitude_percent,
            "low_pass_hz": low_pass_hz,
            "random_seed": int(random_seed),
            "export_modal_csvs": AMBIENT_EXPORT_MODAL_CSVS,
        },
    }

    try:
        clean_old_events(app)
        config["ambient_load_profiles"] = create_ambient_load_profiles(
            app=app,
            scenario_dir=scenario_dir,
            time_step_s=float(sim_step_ms) * 1e-3,
            time_end_s=sim_stop_time_s,
            magnitude_percent=magnitude_percent,
            low_pass_hz=low_pass_hz,
            random_seed=random_seed,
        )

        generators = find_generators(app)
        config["generators"] = [g.loc_name for g in generators]
        elmres = setup_result_variables(app, generators, schema)
        configure_ambient_rms(app, sim_step_ms)
        run_load_flow_initial_conditions_and_rms(app, sim_stop_time_s, sim_step_ms)
        config["csv_files"] = export_results_fast_and_split(app, elmres, generators, scenario_dir, schema)
        if AMBIENT_EXPORT_MODAL_CSVS:
            config["modal_analysis_csvs"] = export_ambient_modal_analysis_csvs(app, scenario_dir)
        config["status"] = "OK"
        write_scenario_json(scenario_dir, config)
        return config
    except Exception as exc:
        config["status"] = "FAILED"
        config["error"] = str(exc)
        write_scenario_json(scenario_dir, config)
        return config


# ============================================================
# CLI SCENARIO RESOLUTION
# ============================================================

def parse_inline_scenario(spec):
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) not in (2, 3, 4, 5, 6) or not parts[0]:
        raise SystemExit(
            f"Invalid scenario spec '{spec}'. Use load_name:dp[:dq[:duration[:event_time[:name]]]], for example 'Load 29:2:0:60:0.5'."
        )

    try:
        return {
            "name": parts[5] if len(parts) == 6 and parts[5] else None,
            "load_name": normalize_load_name(parts[0]),
            "dp_percent": float(parts[1]),
            "dq_percent": float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0,
            "sim_stop_time_s": float(parts[3]) if len(parts) >= 4 and parts[3] else SIM_STOP_TIME_S,
            "event_time_s": float(parts[4]) if len(parts) >= 5 and parts[4] else EVENT_TIME_S,
        }
    except ValueError as exc:
        raise SystemExit(f"Invalid numeric value in scenario spec '{spec}': {exc}") from exc


def parse_defaulted_load_scenario(load_name):
    load_name = normalize_load_name(load_name)
    if not load_name:
        raise SystemExit("Empty load name is not allowed.")
    return {
        "name": None,
        "load_name": load_name,
        "dp_percent": 2.0,
        "dq_percent": 0.0,
        "sim_stop_time_s": SIM_STOP_TIME_S,
        "event_time_s": EVENT_TIME_S,
    }


def select_step_scenarios(names):
    selected = []

    if not names:
        selected.extend(SCENARIOS)
    elif names == ["all"]:
        selected.extend(SCENARIOS)
    elif "all" in names:
        if len(names) > 1:
            raise SystemExit("Use either --scenario all or specific scenario names/specs, not both.")
        selected.extend(SCENARIOS)
    else:
        for name in names:
            if ":" in name:
                selected.append(parse_inline_scenario(name))
            elif name not in SCENARIOS_BY_NAME:
                selected.append(parse_defaulted_load_scenario(name))
            else:
                selected.append(SCENARIOS_BY_NAME[name])

    if not selected:
        raise SystemExit("No scenarios selected.")
    return selected


def list_scenarios():
    seen = set()
    for scenario in SCENARIOS:
        key = make_scenario_key(scenario["load_name"], scenario["dp_percent"], scenario.get("dq_percent", 0.0))
        if key in seen:
            continue
        seen.add(key)
        folder_alias = make_scenario_folder_alias(
            scenario["load_name"],
            scenario["dp_percent"],
            scenario.get("dq_percent", 0.0),
            SIM_STOP_TIME_S,
        )
        aliases = [alias for alias in [scenario.get("key"), key, folder_alias] if alias]
        print(
            f"{aliases[0]} ({', '.join(aliases[1:])}): "
            f"load={scenario['load_name']}, dp={scenario['dp_percent']}, dq={scenario.get('dq_percent', 0.0)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate IEEE39 CSV data from PowerFactory runs.\n\n"
            "Default mode creates step-event scenarios. Use --ambient for ambient excitation data."
        ),
        epilog=dedent(
            """
            Step-event scenario input forms:
              1. Preset key: load29
              2. Multiple preset keys: load03 load24
              3. All presets: all
              4. Bare load name with defaults: "Load 20"
              5. Inline custom spec: "Load 20:2[:dq[:duration[:event_time[:name]]]]"

            Examples:
              python IEEE39/generate_data.py --scenario load29
              python IEEE39/generate_data.py --scenario load03 load24
              python IEEE39/generate_data.py --scenario "Load 20:2" --duration 60 --event-time 0.5
              python IEEE39/generate_data.py --ambient
              python IEEE39/generate_data.py --ambient --scenario ambient_test
              python IEEE39/generate_data.py --ambient --duration 900 --ambient-magnitude-percent 0.2
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scenario", nargs="+", default=None, help="Step-event scenarios, or a single ambient run label when used with --ambient.")
    parser.add_argument("--ambient", action="store_true", help="Generate ambient excitation data instead of load-step event data.")
    parser.add_argument("--output-dir", default=None, help="Results directory relative to IEEE39, or an absolute path. Default: results.")
    parser.add_argument("--project-name", default=None, help="PowerFactory project name override.")
    parser.add_argument("--study-case", default=None, help="PowerFactory study case name override.")
    parser.add_argument("--grid-name", default=None, help="PowerFactory grid name override. Pass 'none' to disable grid activation.")
    parser.add_argument("--duration", type=float, default=None, help="Simulation stop time in seconds.")
    parser.add_argument("--event-time", type=float, default=EVENT_TIME_S, help=f"Load event time in seconds. Default: {EVENT_TIME_S:g}.")
    parser.add_argument("--sim-step-ms", type=float, default=SIM_STEP_MS, help=f"Simulation step in milliseconds. Default: {SIM_STEP_MS:g}.")
    parser.add_argument("--ambient-magnitude-percent", type=float, default=AMBIENT_DIST_MAG_PERCENT, help=f"Ambient load fluctuation magnitude in percent. Default: {AMBIENT_DIST_MAG_PERCENT:g}.")
    parser.add_argument("--ambient-lowpass-hz", type=float, default=AMBIENT_LOW_PASS_HZ, help=f"Ambient low-pass cutoff in Hz. Default: {AMBIENT_LOW_PASS_HZ:g}.")
    parser.add_argument("--ambient-seed", type=int, default=AMBIENT_RANDOM_SEED, help=f"Ambient random seed. Default: {AMBIENT_RANDOM_SEED}.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print the available preset step-event scenario keys and aliases, then exit.")
    return parser.parse_args()


def resolve_optional_grid_name(raw_value, default_value):
    if raw_value is None:
        return default_value
    if str(raw_value).strip().lower() == "none":
        return None
    return raw_value


def resolve_context_from_args(args):
    if args.ambient:
        project_name = args.project_name or AMBIENT_PROJECT_NAME
        study_case_name = args.study_case or AMBIENT_STUDY_CASE_NAME
        grid_name = resolve_optional_grid_name(args.grid_name, AMBIENT_GRID_NAME)
        duration = float(args.duration) if args.duration is not None else AMBIENT_SIM_STOP_TIME_S
    else:
        project_name = args.project_name or PROJECT_NAME
        study_case_name = args.study_case or STUDY_CASE_NAME
        grid_name = resolve_optional_grid_name(args.grid_name, GRID_NAME)
        duration = float(args.duration) if args.duration is not None else SIM_STOP_TIME_S

    return {
        "project_name": project_name,
        "study_case_name": study_case_name,
        "grid_name": grid_name,
        "duration": duration,
        "sim_step_ms": float(args.sim_step_ms),
    }


def run_all_scenarios(args):
    resolved = resolve_context_from_args(args)
    app = get_app()
    project, study_case, grid = activate_context(app, resolved["project_name"], resolved["study_case_name"], resolved["grid_name"])
    print_debug_context(app)

    context_settings = {
        "project_name": project.loc_name,
        "study_case_name": study_case.loc_name,
        "grid_name": grid.loc_name if grid is not None else None,
        "sim_step_ms": resolved["sim_step_ms"],
    }

    results_root = resolve_results_root(args.output_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    if args.ambient:
        ambient_name = parse_ambient_scenario_name(args.scenario)
        result = run_ambient_scenario(
            app=app,
            results_root=results_root,
            context_settings=context_settings,
            ambient_name=ambient_name,
            sim_stop_time_s=resolved["duration"],
            sim_step_ms=resolved["sim_step_ms"],
            magnitude_percent=float(args.ambient_magnitude_percent),
            low_pass_hz=float(args.ambient_lowpass_hz),
            random_seed=int(args.ambient_seed),
        )
        results = [result]
    else:
        selected_scenarios = [dict(scenario) for scenario in select_step_scenarios(args.scenario)]
        for scenario in selected_scenarios:
            scenario.setdefault("sim_stop_time_s", resolved["duration"])
            scenario.setdefault("event_time_s", float(args.event_time))

        results = []
        for index, scenario in enumerate(selected_scenarios, start=1):
            print(f"\nStarting scenario {index}/{len(selected_scenarios)}", flush=True)
            app.PrintPlain(f"Starting scenario {index}/{len(selected_scenarios)}")
            scenario_start = time.time()
            result = run_step_scenario(app, scenario, results_root, context_settings)
            elapsed = time.time() - scenario_start
            print(f"Scenario {index}/{len(selected_scenarios)} finished in {elapsed // 60:.0f} min {elapsed % 60:.1f} sec", flush=True)
            results.append(result)

    ok_count = sum(1 for result in results if result.get("status") == "OK")
    fail_count = len(results) - ok_count
    total_elapsed = time.time() - total_start

    print("=" * 80, flush=True)
    print("All scenarios finished.", flush=True)
    print(f"OK: {ok_count}", flush=True)
    print(f"FAILED: {fail_count}", flush=True)
    print(f"Total execution time: {total_elapsed // 60:.0f} min {total_elapsed % 60:.1f} sec", flush=True)

    app.PrintPlain("=" * 80)
    app.PrintPlain("All scenarios finished.")
    app.PrintPlain(f"OK: {ok_count}")
    app.PrintPlain(f"FAILED: {fail_count}")


if __name__ == "__main__":
    args = parse_args()
    if args.list_scenarios:
        list_scenarios()
        raise SystemExit(0)

    start_time = time.time()
    run_all_scenarios(args)
    end_time = time.time()
    print("-" * 30, f"Execution Time: {(end_time - start_time) // 60} minutes and {(end_time - start_time) % 60} seconds", "-" * 30)
