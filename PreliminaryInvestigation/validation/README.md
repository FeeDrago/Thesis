# Ground-Truth Validation

Ο παρών φάκελος περιέχει βοηθητικά scripts για τον έλεγχο της υλοποίησης του Matrix Pencil με synthetic signals γνωστού ground truth.

## Περιεχόμενα

**Κύριο Validation Script:** Το `groundtruth_validation.py` τρέχει synthetic cases με γνωστές modal παραμέτρους και γράφει τα αποτελέσματα σε ξεχωριστό output folder για κάθε run.

**Σύγκριση Runs:** Το `compare_groundtruth_runs.py` συγκρίνει δύο ολοκληρωμένα validation runs, για παράδειγμα ένα από Windows και ένα από WSL.

**Legacy Validation:** Το `legacy_groundtruth_validation.py` τρέχει τα ίδια synthetic tests με local legacy implementation, ώστε να μπορεί να γίνει σύγκριση current vs legacy. 

## Synthetic Cases

Τα built-in synthetic cases είναι τα εξής:

- `two_mode_clean`
- `two_mode_noisy`
- `three_mode_noisy_close`


## Δομή Αποτελεσμάτων

Κάθε validation run γράφει τα αποτελέσματα σε ξεχωριστό φάκελο:

- current validation: `validation_outputs/<run-label>/`
- legacy validation: `validation_outputs_legacy/<run-label>/`

Τα comparison runs γράφονται by default εδώ:

- `validation_outputs/comparisons/<comparison-label>/`

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
python PreliminaryInvestigation/validation/groundtruth_validation.py --run-label windows_gt_02
python PreliminaryInvestigation/validation/groundtruth_validation.py --run-label wsl_gt_02
```

**Μόνο επιλεγμένα cases:**

```bash
python PreliminaryInvestigation/validation/groundtruth_validation.py --run-label windows_quick --cases two_mode_clean two_mode_noisy
```

## Εκτέλεση Legacy Validation

```bash
python PreliminaryInvestigation/validation/legacy_groundtruth_validation.py --run-label windows_legacy_gt_02
python PreliminaryInvestigation/validation/legacy_groundtruth_validation.py --run-label wsl_legacy_gt_02
```

## Σύγκριση Δύο Runs

**Current Windows vs Current WSL:**

```bash
python PreliminaryInvestigation/validation/compare_groundtruth_runs.py \
  --run-a PreliminaryInvestigation/validation/validation_outputs/windows_gt_02 \
  --run-b PreliminaryInvestigation/validation/validation_outputs/wsl_gt_02 \
  --comparison-label windows_vs_wsl_gt_02
```

**Legacy Windows vs Legacy WSL:**

```bash
python PreliminaryInvestigation/validation/compare_groundtruth_runs.py \
  --run-a PreliminaryInvestigation/validation/validation_outputs_legacy/windows_legacy_gt_02 \
  --run-b PreliminaryInvestigation/validation/validation_outputs_legacy/wsl_legacy_gt_02 \
  --comparison-label legacy_windows_vs_wsl_gt_02
```

**Current vs Legacy στο ίδιο environment:**

```bash
python PreliminaryInvestigation/validation/compare_groundtruth_runs.py \
  --run-a PreliminaryInvestigation/validation/validation_outputs/windows_gt_02 \
  --run-b PreliminaryInvestigation/validation/validation_outputs_legacy/windows_legacy_gt_02 \
  --comparison-label current_vs_legacy_windows_gt_02
```

## Ερμηνεία Αρχείων

### Αρχεία ενός validation run

**`run_config.json`:** Περιέχει τα βασικά settings του run, όπως το `run_label`, το `seed`, ποια `cases` έτρεξαν και το `mode_freq_eps_hz`. Το `mode_freq_eps_hz` είναι απλώς ένα πολύ μικρό κατώφλι συχνότητας. Modes με σχεδόν μηδενική συχνότητα δεν θεωρούνται κανονικά oscillatory modes και δεν μπαίνουν στο matching με το ground truth.

**`case_metrics.csv`:** Ένα row για κάθε `case` και κάθε `method`. Τα βασικά πεδία είναι τα `R2`, `RMSE`, `Selected_Order`, `Estimated_Mode_Count`, `Mean_Freq_Error_Hz`, `Max_Freq_Error_Hz`, `Mean_Damping_Error`, `Max_Damping_Error`, `Mean_2D_Error` και `Matched_All_Truth_Modes`.

**`matched_modes.csv`:** Περιέχει την αντιστοίχιση των ground-truth modes με τα estimated modes του αλγορίθμου. Πρώτα αγνοούνται όσα estimated modes έχουν πρακτικά μηδενική συχνότητα. Μετά, το validation δοκιμάζει τις δυνατές αντιστοιχίσεις και κρατά αυτή που ελαχιστοποιεί το συνολικό `distance_2d`. Για κάθε truth-estimated ζεύγος, το `distance_2d` υπολογίζεται από το error στη συχνότητα και το error στο damping ως `sqrt(frequency_error_hz^2 + damping_error^2)`. Αν τα estimated modes είναι λιγότερα από τα truth modes, δεν μπορεί να υπάρξει πλήρες match.

**`raw_modes.csv`:** Όλα τα oscillatory estimated modes που κράτησε ο αλγόριθμος μετά το frequency thresholding, με `frequency_hz`, `damping`, `amplitude` και `phase_rad`. Χρήσιμο όταν θέλουμε να δούμε τι έβγαλε πραγματικά ο αλγόριθμος, ακόμα και για modes που δεν ταίριαξαν με truth mode.

**`summary.json`:** Σύντομη σύνοψη του run. Περιέχει πόσα methods δοκιμάστηκαν συνολικά, πόσα πέτυχαν πλήρες match (`methods_with_full_truth_match`) και ποιο method ήταν το καλύτερο ανά case μέσα από το `best_methods_by_case`.

Το `methods_with_full_truth_match` δείχνει σε πόσα `(case, method)` ζεύγη βρέθηκε αντιστοίχιση για όλα τα ground-truth modes του case. Αυτό δεν σημαίνει απαραίτητα ότι οι εκτιμήσεις ήταν πολύ ακριβείς. Σημαίνει μόνο ότι κανένα truth mode δεν έμεινε χωρίς αντίστοιχο estimated mode.

### Αρχεία ενός comparison run

**`case_metrics_comparison.csv`:** Side-by-side σύγκριση δύο runs σε επίπεδο case/method. Περιέχει τις δύο τιμές για `R2`, `RMSE`, mean frequency/damping errors και `Mean_2D_Error`, μαζί με winner columns όπως `Better_R2`, `Better_RMSE`, `Better_Mean_2D_Error` και `Overall_Better_Run`.

**`matched_modes_comparison.csv`:** Side-by-side σύγκριση των matched truth modes. Για κάθε truth mode δείχνει τα absolute errors των δύο runs απέναντι στο ground truth (`frequency_error_hz`, `damping_error`, `distance_2d`) και winner columns όπως `Better_Frequency_Error`, `Better_Damping_Error` και `Better_2D_Distance`.

**`summary.json`:** Σύνοψη της σύγκρισης. Περιέχει τα δύο input runs, labels για `run_a` και `run_b`, snapshot των `run_config.json` και `summary.json` των δύο runs, καθώς και winner counts στα βασικά metrics. Τα `total_case_method_comparisons` είναι πόσες γραμμές case/method συγκρίθηκαν, ενώ τα `total_truth_mode_comparisons` είναι πόσες γραμμές truth-mode συγκρίθηκαν.

Τα comparison outputs δεν εστιάζουν μόνο στο πόσο διαφέρουν τα δύο runs μεταξύ τους, αλλά κυρίως στο ποιος είναι πιο κοντά στο ground truth σε κάθε metric και σε κάθε truth mode.
