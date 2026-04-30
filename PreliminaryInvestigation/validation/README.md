# Ground-Truth Validation

Ο παρών φάκελος περιέχει βοηθητικά scripts για τον έλεγχο της υλοποίησης του Matrix Pencil με synthetic signals γνωστού ground truth.

## Περιεχόμενα

**Κύριο Validation Script:** Το `groundtruth_validation.py` τρέχει synthetic cases με γνωστές modal παραμέτρους και γράφει τα αποτελέσματα σε ξεχωριστό output folder για κάθε run.

**Σύγκριση Runs:** Το `compare_groundtruth_runs.py` συγκρίνει δύο ολοκληρωμένα validation runs, για παράδειγμα ένα από Windows και ένα από WSL.

**Legacy Validation:** Το `legacy_groundtruth_validation.py` τρέχει τα ίδια synthetic tests με local legacy professor-style implementation, ώστε να μπορεί να γίνει σύγκριση current vs legacy.

## Synthetic Cases

Τα built-in synthetic cases είναι τα εξής:

- `two_mode_clean`
- `two_mode_noisy`
- `three_mode_noisy_close`

Το `three_mode_noisy_close` είναι σκόπιμα πιο δύσκολο, επειδή περιέχει δύο κοντινά modes γύρω από `1.08-1.12 Hz`.

## Δομή Αποτελεσμάτων

Κάθε validation run γράφει τα αποτελέσματα σε ξεχωριστό φάκελο:

`validation_outputs/<run-label>/`

ή, στην περίπτωση του legacy validation:

`validation_outputs_legacy/<run-label>/`

Τα βασικά αρχεία που παράγονται είναι:

- `run_config.json`
- `case_metrics.csv`
- `matched_modes.csv`
- `raw_modes.csv`
- `summary.json`

Η λογική αυτή επιτρέπει να κρατιούνται ξεχωριστά τα runs από διαφορετικά environments χωρίς overwrite.

## Προτεινόμενα Run Labels

Για ευκολία στη σύγκριση προτείνονται labels όπως:

- Windows current: `windows_gt_01`
- WSL current: `wsl_gt_01`
- Windows legacy: `windows_legacy_gt_01`
- WSL legacy: `wsl_legacy_gt_01`

## Εκτέλεση Current Validation

**Όλα τα synthetic cases:**

```bash
python PreliminaryInvestigation/validation/groundtruth_validation.py --run-label windows_gt_01
python PreliminaryInvestigation/validation/groundtruth_validation.py --run-label wsl_gt_01
```

**Μόνο επιλεγμένα cases:**

```bash
python PreliminaryInvestigation/validation/groundtruth_validation.py --run-label windows_quick --cases two_mode_clean two_mode_noisy
```

## Εκτέλεση Legacy Validation

```bash
python PreliminaryInvestigation/validation/legacy_groundtruth_validation.py --run-label windows_legacy_gt_01
python PreliminaryInvestigation/validation/legacy_groundtruth_validation.py --run-label wsl_legacy_gt_01
```

## Σύγκριση Δύο Runs

**Current Windows vs Current WSL:**

```bash
python PreliminaryInvestigation/validation/compare_groundtruth_runs.py \
  --run-a PreliminaryInvestigation/validation/validation_outputs/windows_gt_01 \
  --run-b PreliminaryInvestigation/validation/validation_outputs/wsl_gt_01 \
  --comparison-label windows_vs_wsl_gt_01
```

**Legacy Windows vs Legacy WSL:**

```bash
python PreliminaryInvestigation/validation/compare_groundtruth_runs.py \
  --run-a PreliminaryInvestigation/validation/validation_outputs_legacy/windows_legacy_gt_01 \
  --run-b PreliminaryInvestigation/validation/validation_outputs_legacy/wsl_legacy_gt_01 \
  --comparison-label legacy_windows_vs_wsl
```

**Current vs Legacy στο ίδιο environment:**

```bash
python PreliminaryInvestigation/validation/compare_groundtruth_runs.py \
  --run-a PreliminaryInvestigation/validation/validation_outputs/windows_gt_01 \
  --run-b PreliminaryInvestigation/validation/validation_outputs_legacy/windows_legacy_gt_01 \
  --comparison-label current_vs_legacy_windows
```

## Ερμηνεία Αρχείων

**`case_metrics.csv`:** Περιέχει τα βασικά metrics κάθε case και κάθε method. Τα πιο χρήσιμα πεδία είναι τα `R2`, `RMSE`, `Selected_Order`, `Mean_Freq_Error_Hz`, `Max_Freq_Error_Hz`, `Mean_Damping_Error`, `Max_Damping_Error` και `Matched_All_Truth_Modes`.

**`matched_modes.csv`:** Περιέχει την αντιστοίχιση ground-truth mode προς estimated mode, ώστε να φανεί αν οι estimated frequencies και dampings είναι κοντά στις σωστές τιμές.

**`raw_modes.csv`:** Περιέχει τα oscillatory modes όπως βγήκαν από τον αλγόριθμο μετά το frequency thresholding.

**`summary.json`:** Περιέχει μια σύντομη σύνοψη του run ή της σύγκρισης.

## Πρακτική Ερμηνεία

Στα `two_mode_clean` και `two_mode_noisy` cases αυτό που θέλουμε να δούμε είναι:

- `Matched_All_Truth_Modes = True`
- μικρά frequency errors
- μικρά damping errors
- υψηλό `R2`
- χαμηλό `RMSE`

Στο `three_mode_noisy_close` αυτό που μας ενδιαφέρει περισσότερο είναι:

- να παραμένει σταθερή η αναγνώριση του mode κοντά στα `0.54 Hz`
- να υπάρχει λογική συμπεριφορά στη separation των δύο κοντινών modes
- να μην αλλάζει ουσιαστικά το αποτέλεσμα μεταξύ Windows και WSL

Το συγκεκριμένο case είναι stress test δυσκολίας και όχι το βασικό proof correctness.

## Σημείωση για το Πρόσημο του Damping

Τα synthetic signals ορίζονται ως αποσβενόμενα, άρα οι σωστές estimated damping τιμές στο output είναι αρνητικές.
