from pathlib import Path
import argparse
import csv
import json
import re
import time

try:
    import pandas as pd
except ImportError:
    pd = None

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

MIN_LOAD_MW = 100.0

# RMS simulation settings
EVENT_TIME_S = 0
SIM_STOP_TIME_S = 50.0
SIM_STEP_MS = 10

# Generator selection.
# None = all generators
# ["G 01"] = only G1
# ["G 01", "G 02", "G 03", "G 04"] = selected generators
GENERATOR_NAMES = None

# Each dictionary = one independent scenario.
# One load event per scenario folder.
SCENARIOS = [
    # Zone 1
    {
        "name": None,              # None = auto folder name
        "key": "load29",
        "load_name": "Load 29",
        "dp_percent": 2.0,
        "dq_percent": 0.0,
    },
    # Zone 2
    {
        "name": None,
        "key": "load03",
        "load_name": "Load 03",
        "dp_percent": 2.0,
        "dq_percent": 0.0,
    },
    # Zone 3
    {
        "name": None,
        "key": "load24",
        "load_name": "Load 24",
        "dp_percent": 2.0,
        "dq_percent": 0.0,
    },
]

def make_scenario_key(load_name, dp_percent, dq_percent):
    return f"{load_name.replace(' ', '').lower()}_p{float(dp_percent):g}_q{float(dq_percent):g}"


def make_load_alias(load_name):
    return load_name.replace(" ", "").lower()


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

GEN_VARIABLES = [
    "s:ut",
    "s:cur1",
    "s:Q1",
    "s:P1",
]

CSV_HEADERS = [
    "b:tnow in s",
    "s:ut in p.u.",
    "s:cur1 in p.u.",
    "s:Q1 in Mvar",
    "s:P1 in MW",
]


# ============================================================
# PATHS / NAMING
# ============================================================

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
    text = text.strip()
    text = re.sub(r"[^\w\-.+]+", "_", text)
    return text


def make_scenario_name(load, dp_percent, dq_percent, sim_stop_time, custom_name=None, event_time_s=EVENT_TIME_S):
    """
    Examples:
    Load03_Pplus2_50s
    Load39_Pminus5_Qplus2_50s
    """
    evt_part = event_time_suffix(event_time_s)
    if custom_name:
        return f"{safe_name(custom_name)}{evt_part}"

    load_part = safe_name(load.loc_name).replace("_", "")

    if dp_percent >= 0:
        p_part = f"Pplus{abs(dp_percent):g}"
    else:
        p_part = f"Pminus{abs(dp_percent):g}"

    if dq_percent is None or abs(dq_percent) < 1e-12:
        return f"{load_part}_{p_part}_{sim_stop_time:g}s{evt_part}"

    if dq_percent >= 0:
        q_part = f"Qplus{abs(dq_percent):g}"
    else:
        q_part = f"Qminus{abs(dq_percent):g}"

    return f"{load_part}_{p_part}_{q_part}_{sim_stop_time:g}s{evt_part}"


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
        raise RuntimeError(
            f"Could not activate project '{project_name}'. "
            f"ActivateProject returned: {ret}"
        )

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
    raise RuntimeError(
        f"Study case '{study_case_name}' not found.\n"
        f"Available study cases:\n" + "\n".join(available)
    )


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
    raise RuntimeError(
        f"Grid '{grid_name}' not found.\n"
        f"Available grids:\n" + "\n".join(available)
    )


def activate_context(app):
    project = activate_project(app, PROJECT_NAME)
    study_case = activate_study_case(app, STUDY_CASE_NAME)
    activate_grid_if_needed(app, GRID_NAME)

    app.PrintPlain("PowerFactory context activated.")
    app.PrintPlain(f"Project: {project.loc_name}")
    app.PrintPlain(f"Study case: {study_case.loc_name}")

    return project, study_case


def get_from_study_case(app, class_name: str):
    obj = app.GetFromStudyCase(class_name)
    if obj is None:
        raise RuntimeError(f"Could not find {class_name} in active Study Case.")
    return obj


# ============================================================
# LOAD / GENERATOR FINDERS
# ============================================================

def get_load_p_mw(load):
    candidates = ["plini", "plini_a", "pgini", "m:P:bus1"]

    for attr in candidates:
        try:
            val = load.GetAttribute(attr)
            if val is not None:
                return float(val)
        except Exception:
            pass

    return None


def find_load(app, load_name=None, min_load_mw=100.0):
    loads = []

    for pattern in ["*.ElmLod", "*.ElmLodlv", "*.ElmLodmv"]:
        try:
            found = app.GetCalcRelevantObjects(pattern)
            if found:
                loads.extend(found)
        except Exception:
            pass

    unique_loads = []
    seen = set()

    for load in loads:
        key = id(load)
        if key not in seen:
            unique_loads.append(load)
            seen.add(key)

    loads = unique_loads

    if not loads:
        raise RuntimeError("No load objects found. Check active project/study case/grid.")

    if load_name is not None:
        for load in loads:
            if load.loc_name == load_name:
                return load

        available = [load.loc_name for load in loads]
        raise RuntimeError(
            f"Load '{load_name}' not found.\n"
            f"Available loads:\n" + "\n".join(available)
        )

    candidates = []

    for load in loads:
        p_mw = get_load_p_mw(load)
        if p_mw is not None and p_mw >= min_load_mw:
            candidates.append((p_mw, load))

    if not candidates:
        info = [f"{load.loc_name}: P={get_load_p_mw(load)} MW" for load in loads]
        raise RuntimeError(
            f"No load found with P >= {min_load_mw} MW.\n"
            f"Loads:\n" + "\n".join(info)
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
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
        match = None

        for gen in gens:
            if gen.loc_name == name:
                match = gen
                break

        if match is None:
            missing.append(name)
        else:
            selected.append(match)

    if missing:
        available = [g.loc_name for g in gens]
        raise RuntimeError(
            f"Missing generators: {missing}\n"
            f"Available generators:\n" + "\n".join(available)
        )

    return selected


# ============================================================
# EVENTS / RESULTS
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
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Could not set any of these attributes on {obj.loc_name}: {candidates}\n"
        f"Last error: {last_error}"
    )


def create_load_event(app, load, time_s, dp_percent, dq_percent):
    evt_folder = get_from_study_case(app, "IntEvt")
    event = evt_folder.CreateObject("EvtLod", f"load_event_{safe_name(load.loc_name)}")

    set_first_existing_attribute(event, ["time"], time_s)

    target_attr = set_first_existing_attribute(
        event,
        ["p_target", "pTarget", "target", "p_object", "pObj"],
        load,
    )

    p_attr = set_first_existing_attribute(
        event,
        ["dP", "dp", "P", "p", "dplini", "plini", "deltaP", "DeltaP"],
        dp_percent,
    )

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


def setup_result_variables(app, generators):
    elmres = get_from_study_case(app, "ElmRes")

    try:
        elmres.Clear()
    except Exception:
        try:
            elmres.DeleteVars()
        except Exception:
            pass

    for gen in generators:
        for var in GEN_VARIABLES:
            elmres.AddVars(gen, var)

    return elmres


# ============================================================
# SIMULATION
# ============================================================

def try_set_attr(obj, attr_names, value):
    for attr in attr_names:
        try:
            obj.SetAttribute(attr, value)
            return attr
        except Exception:
            pass

    return None


def run_load_flow_initial_conditions_and_rms(app, tstop, step):
    ldf = get_from_study_case(app, "ComLdf")
    inc = get_from_study_case(app, "ComInc")
    sim = get_from_study_case(app, "ComSim")

    app.PrintPlain("Running Load Flow...")
    err = ldf.Execute()

    if err:
        raise RuntimeError(f"Load Flow failed with error code {err}")

    app.PrintPlain("Setting RMS time step / simulation options...")

    inc_step_attr = try_set_attr(
        inc,
        ["dtgrd", "dt", "tstep", "dtemt", "dtout"],
        float(step),
    )

    sim_step_attr = try_set_attr(
        sim,
        ["dtgrd", "dt", "tstep", "dtemt", "dtout"],
        float(step),
    )

    sim_stop_attr = try_set_attr(
        sim,
        ["tstop", "tmax", "t_end"],
        float(tstop),
    )

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
# FAST EXPORT WITH COMRES + PANDAS SPLIT
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
    """
    Fast PowerFactory export of all selected ElmRes variables.
    Creates raw_all_generators.csv.
    """
    comres = get_from_study_case(app, "ComRes")
    raw_csv = scenario_dir / "raw_all_generators.csv"

    app.PrintPlain(f"Fast exporting raw results to: {raw_csv}")
    print(f"Fast exporting raw results to: {raw_csv}", flush=True)

    set_comres_attr(comres, "pResult", elmres)
    set_comres_attr(comres, "f_name", str(raw_csv))

    # Common ComRes options
    set_comres_attr(comres, "iopt_exp", 6)      # CSV
    set_comres_attr(comres, "iopt_csel", 0)     # all selected variables
    set_comres_attr(comres, "iopt_tsel", 0)     # all time steps
    set_comres_attr(comres, "iopt_locn", 2)     # full path
    set_comres_attr(comres, "ciopt_head", 1)    # include headers

    err = comres.Execute()

    if err:
        raise RuntimeError(f"ComRes export failed with error code {err}")

    app.PrintPlain("Fast raw ComRes export done.")
    print("Fast raw ComRes export done.", flush=True)

    return raw_csv


def read_comres_csv_flexible(raw_csv):
    """
    Tries common ComRes CSV formats.
    """
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

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read raw ComRes CSV: {last_error}")


def normalize_col_name(col):
    """
    Converts normal or MultiIndex pandas column name to searchable string.
    """
    if isinstance(col, tuple):
        parts = []
        for x in col:
            sx = str(x)
            if sx.lower() != "nan" and "unnamed" not in sx.lower():
                parts.append(sx)
        return " ".join(parts)

    return str(col)


def compact_text(text):
    return str(text).replace(" ", "").replace("_", "").replace("-", "").lower()


def find_time_column_pandas(df):
    for col in df.columns:
        name = normalize_col_name(col).lower()
        compact = compact_text(name)

        if "tnow" in compact or "time" in compact or "b:tnow" in compact:
            return col

    return df.columns[0]


def find_generator_variable_column(df, gen_name, variable):
    """
    Finds raw CSV column for one generator and one variable.
    Handles common ComRes header styles.
    """
    gen_key = compact_text(gen_name)
    var_key = compact_text(variable)

    candidates = []

    for col in df.columns:
        name = normalize_col_name(col)
        compact = compact_text(name)

        if gen_key in compact and var_key in compact:
            candidates.append(col)

    if candidates:
        return candidates[0]

    # Fallback: variable appears and generator number appears separately.
    # Example G 01 -> 01
    gen_digits = "".join(ch for ch in gen_name if ch.isdigit())

    if gen_digits:
        for col in df.columns:
            name = normalize_col_name(col)
            compact = compact_text(name)

            if var_key in compact and gen_digits in compact:
                return col

    # Debug help
    sample_cols = [normalize_col_name(c) for c in list(df.columns)[:20]]

    raise RuntimeError(
        f"Could not find column for generator '{gen_name}', variable '{variable}'.\n"
        f"First columns seen:\n" + "\n".join(sample_cols)
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


def split_raw_comres_standard_csv(raw_csv, generators, scenario_dir):
    with open(raw_csv, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        try:
            object_headers = next(reader)
            variable_headers = next(reader)
        except StopIteration as e:
            raise RuntimeError(f"Raw ComRes CSV is missing headers: {raw_csv}") from e

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
            generator_columns.append(
                [find_generator_variable_index(object_headers, variable_headers, gen.loc_name, var) for var in GEN_VARIABLES]
            )

        outputs = []
        writers = []

        try:
            for idx in range(1, len(generators) + 1):
                out_csv = scenario_dir / f"g{idx}.csv"
                out_file = open(out_csv, "w", newline="")
                writer = csv.writer(out_file)
                writer.writerow(CSV_HEADERS)
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
    """
    Converts ComRes values to numeric floats regardless of comma/dot decimal export.
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.astype(str).str.strip().str.replace(",", ".", regex=False)

    missing = series.str.lower().isin(["", "nan", "none"])
    series = series.mask(missing)

    return pd.to_numeric(series, errors="coerce")


def split_raw_comres_to_generator_csvs(raw_csv, generators, scenario_dir):
    """
    Splits raw_all_generators.csv into g1.csv, g2.csv, ...
    with clean headers requested by the user.
    """
    print("Splitting raw CSV into one file per generator...", flush=True)

    try:
        split_raw_comres_standard_csv(raw_csv, generators, scenario_dir)
        return
    except Exception as e:
        print(f"Standard CSV split failed, trying pandas fallback: {e}", flush=True)

    df = read_comres_csv_flexible(raw_csv)
    time_col = find_time_column_pandas(df)

    for idx, gen in enumerate(generators, start=1):
        print(f"Splitting {gen.loc_name} -> g{idx}.csv", flush=True)

        output = pd.DataFrame()
        output[CSV_HEADERS[0]] = to_numeric_dot_decimal(df[time_col])

        for variable, clean_header in zip(GEN_VARIABLES, CSV_HEADERS[1:]):
            raw_col = find_generator_variable_column(df, gen.loc_name, variable)
            output[clean_header] = to_numeric_dot_decimal(df[raw_col])

        out_csv = scenario_dir / f"g{idx}.csv"

        if output.empty or output.dropna(how="all").empty:
            raise RuntimeError(f"Split produced no numeric rows for {gen.loc_name} -> {out_csv}")

        output.to_csv(out_csv, index=False, float_format="%.10g")

        print(f"Saved {out_csv}", flush=True)


def export_results_fast_and_split(app, elmres, generators, scenario_dir):
    raw_csv = export_raw_results_fast_comres(app, elmres, scenario_dir)
    split_raw_comres_to_generator_csvs(raw_csv, generators, scenario_dir)
    return validate_generator_csvs(scenario_dir, generators)


def parse_csv_float(value, csv_path, row_number, column_name):
    text = str(value).strip().replace(",", ".")
    if not text:
        raise RuntimeError(f"Empty value in {csv_path}, row {row_number}, column '{column_name}'")

    try:
        return float(text)
    except ValueError as e:
        raise RuntimeError(
            f"Non-numeric value in {csv_path}, row {row_number}, column '{column_name}': {value}"
        ) from e


def validate_generator_csvs(scenario_dir, generators):
    csv_files = []

    for idx, gen in enumerate(generators, start=1):
        csv_path = scenario_dir / f"g{idx}.csv"

        if not csv_path.exists():
            raise RuntimeError(f"Missing generated CSV: {csv_path}")

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration as e:
                raise RuntimeError(f"Generated CSV is empty: {csv_path}") from e

            if headers != CSV_HEADERS:
                raise RuntimeError(
                    f"Unexpected headers in {csv_path}.\n"
                    f"Expected: {CSV_HEADERS}\n"
                    f"Found: {headers}"
                )

            row_count = 0
            previous_time = None

            for row_number, row in enumerate(reader, start=2):
                if not row or all(str(value).strip() == "" for value in row):
                    continue

                if len(row) != len(CSV_HEADERS):
                    raise RuntimeError(
                        f"Wrong number of columns in {csv_path}, row {row_number}. "
                        f"Expected {len(CSV_HEADERS)}, found {len(row)}"
                    )

                values = [
                    parse_csv_float(value, csv_path, row_number, column_name)
                    for value, column_name in zip(row, CSV_HEADERS)
                ]

                current_time = values[0]
                if previous_time is not None and current_time < previous_time:
                    raise RuntimeError(f"Time column is not monotonic in {csv_path}, row {row_number}")

                previous_time = current_time
                row_count += 1

        if row_count == 0:
            raise RuntimeError(f"Generated CSV has no numeric data rows: {csv_path}")

        csv_files.append({
            "generator": gen.loc_name,
            "file": path_for_metadata(csv_path),
            "rows": row_count,
        })

    return csv_files


# ============================================================
# DEBUG
# ============================================================

def print_debug_context(app):
    project = app.GetActiveProject()
    study_case = app.GetActiveStudyCase()

    print("Active project:", project.loc_name if project else None, flush=True)
    print("Active study case:", study_case.loc_name if study_case else None, flush=True)

    loads = app.GetCalcRelevantObjects("*.ElmLod")
    gens = app.GetCalcRelevantObjects("*.ElmSym")

    # print("Number of ElmLod:", len(loads), flush=True)
    # print("First loads:", [l.loc_name for l in loads[:10]], flush=True)

    # print("Number of ElmSym:", len(gens), flush=True)
    # print("First generators:", [g.loc_name for g in gens[:10]], flush=True)

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
# MAIN
# ============================================================

def run_single_scenario(app, scenario, results_root):
    """
    Runs exactly one scenario.
    Guarantees one event per scenario because it calls clean_old_events()
    before creating the scenario's load event.
    """
    load_name = scenario.get("load_name")
    dp_percent = float(scenario.get("dp_percent", 2.0))
    dq_percent = float(scenario.get("dq_percent", 0.0))
    sim_stop_time_s = float(scenario.get("sim_stop_time_s", SIM_STOP_TIME_S))
    event_time_s = float(scenario.get("event_time_s", EVENT_TIME_S))
    custom_name = scenario.get("name")

    load = find_load(app, load_name, MIN_LOAD_MW)
    p_mw = get_load_p_mw(load)

    scenario_name = make_scenario_name(
        load=load,
        dp_percent=dp_percent,
        dq_percent=dq_percent,
        sim_stop_time=sim_stop_time_s,
        custom_name=custom_name,
        event_time_s=event_time_s,
    )

    scenario_dir = results_root / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    app.PrintPlain("=" * 80)
    app.PrintPlain(f"Running scenario: {scenario_name}")
    app.PrintPlain(f"Selected load: {load.loc_name}, P={p_mw} MW")
    app.PrintPlain(f"Scenario folder: {scenario_dir}")

    print("=" * 80, flush=True)
    print(f"Running scenario: {scenario_name}", flush=True)
    print(f"Selected load: {load.loc_name}, P={p_mw} MW", flush=True)
    print(f"Scenario folder: {scenario_dir}", flush=True)

    config = {
        "scenario_name": scenario_name,
        "project_name": PROJECT_NAME,
        "study_case_name": STUDY_CASE_NAME,
        "grid_name": GRID_NAME,
        "load_name": load.loc_name,
        "load_initial_p_mw": p_mw,
        "min_load_mw": MIN_LOAD_MW,
        "dp_percent": dp_percent,
        "dq_percent": dq_percent,
        "event_time_s": event_time_s,
        "sim_stop_time_s": sim_stop_time_s,
        "sim_step_ms": SIM_STEP_MS,
        "csv_headers": CSV_HEADERS,
        "generator_names_setting": GENERATOR_NAMES,
    }

    try:
        # Critical: remove previous events so each scenario has exactly one event.
        clean_old_events(app)

        event, event_attrs = create_load_event(
            app=app,
            load=load,
            time_s=event_time_s,
            dp_percent=dp_percent,
            dq_percent=dq_percent,
        )

        config["event_name"] = event.loc_name
        config["event_attributes_used"] = event_attrs

        generators = find_generators(app)
        config["generators"] = [g.loc_name for g in generators]

        elmres = setup_result_variables(app, generators)

        run_load_flow_initial_conditions_and_rms(
            app=app,
            tstop=sim_stop_time_s,
            step=SIM_STEP_MS,
        )

        config["csv_files"] = export_results_fast_and_split(app, elmres, generators, scenario_dir)

        config["status"] = "OK"

        with open(scenario_dir / "scenario.json", "w") as f:
            json.dump(config, f, indent=2)

        app.PrintPlain(f"Scenario done: {scenario_name}")
        print(f"Scenario done: {scenario_name}", flush=True)

        return config

    except Exception as e:
        config["status"] = "FAILED"
        config["error"] = str(e)

        with open(scenario_dir / "scenario.json", "w") as f:
            json.dump(config, f, indent=2)

        app.PrintPlain(f"FAILED scenario {scenario_name}: {e}")
        print(f"FAILED scenario {scenario_name}: {e}", flush=True)

        return config


def parse_inline_scenario(spec):
    """
    Parses CLI specs in the form load_name:dp[:dq[:duration[:event_time[:name]]]].
    Examples: "Load 29:2", "Load 24:-5:2:60:0.5:my_case".
    """
    parts = [part.strip() for part in spec.split(":")]

    if len(parts) not in (2, 3, 4, 5, 6) or not parts[0]:
        raise SystemExit(
            f"Invalid scenario spec '{spec}'. Use load_name:dp[:dq[:duration[:event_time[:name]]]], "
            "for example 'Load 29:2:0:60:0.5'."
        )

    try:
        scenario = {
            "name": parts[5] if len(parts) == 6 and parts[5] else None,
            "load_name": parts[0],
            "dp_percent": float(parts[1]),
            "dq_percent": float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0,
            "sim_stop_time_s": float(parts[3]) if len(parts) >= 4 and parts[3] else SIM_STOP_TIME_S,
            "event_time_s": float(parts[4]) if len(parts) >= 5 and parts[4] else EVENT_TIME_S,
        }
    except ValueError as e:
        raise SystemExit(f"Invalid numeric value in scenario spec '{spec}': {e}") from e

    return scenario


def parse_case_args(case_args):
    scenarios = []

    for values in case_args or []:
        if len(values) == 1 and ":" in values[0]:
            scenarios.append(parse_inline_scenario(values[0]))
            continue

        if len(values) not in (2, 3, 4):
            raise SystemExit(
                "Each --case must be either a quoted spec "
                "load_name:dp[:dq[:duration[:event_time[:name]]]] or the legacy form "
                "LOAD_NAME DP_PERCENT [DQ_PERCENT] [NAME]"
            )

        try:
            scenarios.append(
                {
                    "name": values[3] if len(values) == 4 else None,
                    "load_name": values[0],
                    "dp_percent": float(values[1]),
                    "dq_percent": float(values[2]) if len(values) >= 3 else 0.0,
                    "sim_stop_time_s": SIM_STOP_TIME_S,
                    "event_time_s": EVENT_TIME_S,
                }
            )
        except ValueError as e:
            raise SystemExit(f"Invalid numeric value in --case {' '.join(values)}: {e}") from e

    return scenarios


def select_scenarios(names, cli_cases=None):
    selected = []

    if not names:
        if not cli_cases:
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
                continue

            if name not in SCENARIOS_BY_NAME:
                available = ", ".join(SCENARIOS_BY_NAME.keys())
                raise SystemExit(f"Unknown scenario '{name}'. Available: {available}")

            selected.append(SCENARIOS_BY_NAME[name])

    selected.extend(parse_case_args(cli_cases))

    if not selected:
        raise SystemExit("No scenarios selected.")

    return selected


def list_scenarios():
    seen = set()

    for scenario in SCENARIOS:
        key = make_scenario_key(
            scenario["load_name"],
            scenario["dp_percent"],
            scenario.get("dq_percent", 0.0),
        )

        if key in seen:
            continue

        seen.add(key)
        folder_alias = make_scenario_folder_alias(
            scenario["load_name"],
            scenario["dp_percent"],
            scenario.get("dq_percent", 0.0),
            SIM_STOP_TIME_S,
        )
        aliases = [scenario.get("key"), key, folder_alias]
        aliases = [alias for alias in aliases if alias]

        print(
            f"{aliases[0]} ({', '.join(aliases[1:])}): "
            f"load={scenario['load_name']}, "
            f"dp={scenario['dp_percent']}, "
            f"dq={scenario.get('dq_percent', 0.0)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate IEEE39 PowerFactory scenario CSV data.")
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=None,
        help="Scenario keys/folder aliases to run, 'all', or inline specs load_name:dp[:dq[:duration[:event_time[:name]]]].",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs="+",
        default=[],
        metavar="VALUE",
        help="Add an ad-hoc scenario. Use either --case LOAD_NAME DP_PERCENT [DQ_PERCENT] [NAME] or a quoted spec like --case 'Load 24:2:0:60:0.5'. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Results directory relative to IEEE39, or an absolute path. Default: results.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=SIM_STOP_TIME_S,
        help=f"Simulation stop time in seconds. Default: {SIM_STOP_TIME_S:g}.",
    )
    parser.add_argument(
        "--event-time",
        type=float,
        default=EVENT_TIME_S,
        help=f"Load event time in seconds. Default: {EVENT_TIME_S:g}.",
    )
    parser.add_argument("--list-scenarios", action="store_true", help="Print available scenario keys and exit.")
    return parser.parse_args()


def run_all_scenarios(scenarios=None, output_dir=None):
    if scenarios is None:
        scenarios = SCENARIOS

    app = get_app()

    activate_context(app)
    print_debug_context(app)

    results_root = resolve_results_root(output_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    ok_count = 0
    fail_count = 0

    for i, scenario in enumerate(scenarios, start=1):
        print(f"\nStarting scenario {i}/{len(scenarios)}", flush=True)
        app.PrintPlain(f"Starting scenario {i}/{len(scenarios)}")

        scenario_start = time.time()

        result = run_single_scenario(
            app=app,
            scenario=scenario,
            results_root=results_root,
        )

        scenario_end = time.time()
        elapsed = scenario_end - scenario_start

        print(
            f"Scenario {i}/{len(scenarios)} finished in "
            f"{elapsed // 60:.0f} min {elapsed % 60:.1f} sec",
            flush=True,
        )

        if result.get("status") == "OK":
            ok_count += 1
        else:
            fail_count += 1

    total_end = time.time()
    total_elapsed = total_end - total_start

    print("=" * 80, flush=True)
    print(f"All scenarios finished.", flush=True)
    print(f"OK: {ok_count}", flush=True)
    print(f"FAILED: {fail_count}", flush=True)
    print(
        f"Total execution time: {total_elapsed // 60:.0f} min {total_elapsed % 60:.1f} sec",
        flush=True,
    )

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
    selected_scenarios = [dict(scenario) for scenario in select_scenarios(args.scenario, args.case)]
    for scenario in selected_scenarios:
        scenario.setdefault("sim_stop_time_s", float(args.duration))
        scenario.setdefault("event_time_s", float(args.event_time))
    run_all_scenarios(selected_scenarios, args.output_dir)
    end_time = time.time()
    print("-"*30, f"Execution Time: {(end_time - start_time)//60} minutes and {(end_time - start_time)%60} seconds", "-"*30)
