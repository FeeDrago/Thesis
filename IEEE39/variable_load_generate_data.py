

import warnings
import time
import os
import numpy as np
import math
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import savemat
import csv

# Add powerfactory.pyd module to system PATH and import it
import sys
sys.path.append(
    r"C:\Program Files\DIgSILENT\PowerFactory 2025\Python\3.13")
import powerfactory as pf


PROJECT_NAME = '39 Bus New England System TEST'
STUDY_CASE1 = 'RMS mine'

STEP_SIZE = 10 # in ms!
SIM_TIME = 600 # in s!

INERTIA_LIMS = [3, 8]
INERTIA_DEC_PRECISION = 2  # Inertia constant decimal precision

DIST_MAG = 0.1  # Disturbance magnitude (%)


OUTPUT_PATH ="Results"  # Absolute path
# to save output data.

np.random.seed(1997)  # to get the same results each run

def activate_project_and_case(app, project_name, study_case):
    '''
    Activate powerfactory project and case study passed as arguments.
    Throw relevant exceptions if needed
    '''
    # Activate project
    project = app.ActivateProject(project_name)
    if (project > 0):
        raise Exception("Project can not be found or activated!")
    # Activate Study Case
    study_case_folder = app.GetProjectFolder('study')
    study_case = study_case_folder.GetContents(study_case)
    if (len(study_case) == 0):
        raise Exception("Study case can not be found or activated!")
    study_case[0].Activate()





def prepare_dynamic_sim(
        app, sim_type, start_time, step_size, sync
    ):
    '''
    Set transient simulation type, start time, stop time and step size. Then,
    calculate initial conditions.
    By setting sync = True, the 'Enforced synchronisation' option is activated
    with Period = step_size
    '''
    inc = app.GetFromStudyCase('ComInc')
    # set simulation type: 'rms' or 'ins' (for EMT)
    inc.iopt_sim = sim_type
    # set start time, step size and end time
    inc.tstart = start_time
    inc.dtgrd = step_size
    inc.iopt_sync = sync
    if (sync):
        inc.syncperiod = step_size
        inc.ciopt_sample = 2
    # set initial conditions
    inc.Execute()


def write_results(idx, res, inertias, path, file):
    '''
    Write results to output files:
        1) Inertia data will be written to inertia_data.txt
        2) The actual output matrix will be saved in a sim{idx}.mat MATLAB
        file
    '''
    res.Load()
    M = res.GetNumberOfRows()
    N = res.GetNumberOfColumns()
    results = np.zeros((M, N+1))  # N+1 columns to include time
    for j in range(-1, N):
        for i in range(0, M):  # Start from -1 to include time
            results[i, j+1] = res.GetValue(i, j)[1]

    if (idx == 0):
        file.write("sim{i}.mat files contain results for: \n")
        file2 = open(r"{}/output_list.csv".format(OUTPUT_PATH), 'w',
                     newline='', encoding='utf-8')
        f2_writer = csv.writer(file2)

        for j in range(-1, N):
            txt = res.GetObject(j).loc_name + "->" + res.GetVariable(j)
            file.write(txt + "\n")
            f2_writer.writerow([txt])
        file.write("\nPrinting information for each simulation: \n")
        file2.close()
    savemat(r"{}/sim{}.mat".format(path, idx), {'r': results})
    res.Release()

def rename_file(path, src, dest):
    old_path = os.path.join(path, src)
    new_path = os.path.join(path, dest)
    try:
        os.remove(new_path)
    except OSError:
        pass
    os.rename(old_path, new_path)


def clear_evt_folder(fold):
    for evt in fold:
        evt.Delete()

def generate_filtered_noise(fs, n_samples, cutoff=5):
    white_noise = np.random.normal(0, 1, n_samples)

    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq

    b, a = signal.butter(N=4, Wn=norm_cutoff, btype='low', analog=False)
    filtered_noise = signal.lfilter(b, a, white_noise)

    max_abs = np.max(np.abs(filtered_noise))
    if max_abs != 0:
        filtered_noise = (filtered_noise / max_abs)



    return filtered_noise


def find_grid_file(app):
    netdat = app.GetProjectFolder("netdat")
    if not netdat:
        raise Exception("Could not find netdat folder.")

    # Access the 'Grid' folder inside netdat
    grid_folder = netdat.GetContents("*.ElmNet")  # ElmNet is a typical grid object (network)
    if not grid_folder:
        # If 'Grid' is a user-created folder under netdat, search for it
        grid_folder = netdat.GetContents("Grid")[0]
    else:
        grid_folder = grid_folder[0]
    return grid_folder

def create_ext_lods(app, time_step, time_end, idx, res_folder):

    folder_path = os.path.join(res_folder, "loads", f"sim{str(idx)}")
    os.makedirs(folder_path, exist_ok=True)


    grid = find_grid_file(app)
    lib_fold = app.GetProjectFolder("lib")
    udm_fold = lib_fold.GetContents('User Defined Models.IntPrjFolder')[0]

    for old_lods in grid.GetContents("*_ExtLoad.ElmComp"):
        old_lods.Delete()


    time_vector = list(np.arange(-time_step, time_end + 2*time_step, time_step))
    nsamp = len(time_vector)
    # Uncomment if you want all loads to have the same behaviour! and comment bellow!!
    # p_dist_0 = generate_filtered_noise(1/time_step, nsamp, cutoff=5)
    # q_dist_0 = generate_filtered_noise(1/time_step, nsamp, cutoff=5)

    lods = app.GetCalcRelevantObjects("*.ElmLod")

    for i in range(len(lods)):
        current_lod = lods[i]
        p_dist_0 = generate_filtered_noise(1/time_step, nsamp, cutoff=5)
        q_dist_0 = generate_filtered_noise(1/time_step, nsamp, cutoff=5)
        p_dist = (p_dist_0 * DIST_MAG/100 + 1) * current_lod.plini
        q_dist = (q_dist_0 * DIST_MAG/100 + 1) * current_lod.qlini

        data = np.column_stack((time_vector, p_dist, q_dist))

        file_path = os.path.join(folder_path, f"{current_lod.loc_name}.txt")
        with open(file_path, "w") as f:
            f.write("2\n")  # First line with the number 2
            np.savetxt(f, data, fmt='%.4f', delimiter='\t')


        new_comp = grid.CreateObject('ElmComp', lods[i].loc_name+'_ExtLoad')
        new_comp.typ_id = udm_fold.GetContents("Composite Type Load.BlkDef")[0]


        current_lod_typ = current_lod.typ_id
        current_lod_typ.systp = 0
        current_lod_typ.phtech = 2
        current_lod_typ.lodst = 0
        current_lod_typ.loddy = 100
        current_lod_typ.aP = 0
        current_lod_typ.aQ = 0
        current_lod_typ.bP = 0
        current_lod_typ.bQ = 0
        for j in range(len(new_comp.pblk)):
            slot = new_comp.pblk[j]
            if slot.loc_name == 'load slot':
                slot.loc_name = "load_slot"
            elif slot.loc_name == 'load measurement':
                slot.loc_name = "load_measurement"
        new_comp.load_slot = current_lod
        mf = new_comp.CreateObject('ElmFile', 'Measurement File')
        mf.f_name = os.path.abspath(file_path)
        # a = mf.afac
        # a[1]= current_lod.plini*DIST_MAG/100 * 0
        # a[2] = current_lod.qlini*DIST_MAG/100 * 0
        # mf.afac = a
        # b = mf.bfac
        # b[1]= current_lod.plini
        # b[2] = current_lod.qlini
        # mf.bfac = b
        new_comp.load_measurement = mf


def main():
    t0 = time.time()
    print("Beginning simulations..")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    app = pf.GetApplication()
    activate_project_and_case(app, PROJECT_NAME, STUDY_CASE1)

    generators = app.GetCalcRelevantObjects("*.ElmSym")

    file = open(r"{}/info.txt".format(OUTPUT_PATH), "w")
    res = app.GetFromStudyCase('*.ElmRes')




    for i in range(1):
        t1 = time.time()

        create_ext_lods(app, STEP_SIZE*(1e-3), SIM_TIME, i, OUTPUT_PATH)

        sim = app.GetFromStudyCase('ComSim')
        prepare_dynamic_sim(app, sim_type='rms', start_time=0.,
                            step_size=STEP_SIZE, sync = True
        )

        sim.tstop=SIM_TIME
        res.Clear()
        print("\t Executing simulation..")
        sim.Execute()
        print("\t Simulation finished, writing results..")

        t2 = time.time()


        app.ResetCalculation()

        sim = app.GetFromStudyCase('ComMod')
        sim.dirMatl = os.path.abspath(OUTPUT_PATH)
        sim.iEValMatl = 1
        sim.iPart = 1
        sim.iPartMatl = 1
        sim.isResOscModesOnly = 1
        sim.outputType = 1
        sim.iSysMatsMatl = 1

        #modal_res = app.GetFromStudyCase('*.ComModres')
        #modal_res.Clear()
        sim.Execute()
        #modal_res.Execute()

        # For this to work, one has to choose Output to MATLAB files from
        # Modal analysis settings!
        rename_file(OUTPUT_PATH, 'EVals.mtl', r"sim{}_modes.mtl".format(i))
        rename_file(OUTPUT_PATH, 'PartFacs.mtl', r"sim{}_partFacs.mtl".format(i))
        rename_file(OUTPUT_PATH, 'VariableToIdx_Amat.txt', r"sim{}_states.txt".format(i))

        files_to_remove = [
            "Amat.mtl",
            "Jacobian.mtl",
            "M.mtl",
            "VariableToIdx_Jacobian.txt"
        ]

        for filename in files_to_remove:
            file_path = os.path.join(OUTPUT_PATH, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

        app.ResetCalculation()

    del app


    print("Finished execution!")
    # input("Press the <ENTER> key to continue...")

    file.close()


if __name__ == "__main__":
    main()
