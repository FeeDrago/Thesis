# Διπλωματική Εργασία

Το παρόν αποθετήριο περιέχει το υλικό της διπλωματικής μου εργασίας.

## Φάκελος: Preliminary Investigation
Εδώ βρίσκεται συγκεντρωμένο όλο το υλικό της προκαταρκτικής μελέτης:

**Αναφορά:** Το κείμενο της προκαταρκτικής αναφοράς είναι διαθέσιμο σε έτοιμο αρχείο PDF για άμεση ανάγνωση, μαζί με τα πηγαία αρχεία του LaTeX σε περίπτωση που χρειαστούν.

**Κώδικας & Αποτελέσματα:** Περιλαμβάνονται τα αρχεία Python της ανάλυσης, καθώς και τα διαγράμματα (plots) που έχουν παραχθεί. Τα διαγράμματα είναι ήδη αποθηκευμένα, ώστε να έχετε άμεση εικόνα των αποτελεσμάτων χωρίς να χρειαστεί να εκτελέσετε τα scripts.

## Φάκελος: IEEE39
Εδώ βρίσκεται το υλικό για την παραγωγή και την ανάλυση δεδομένων του συστήματος IEEE 39.

**Παραγωγή Δεδομένων:** Το αρχείο `IEEE39/generate_data.py` εκτελεί σενάρια στο PowerFactory και αποθηκεύει τα αποτελέσματα στον φάκελο `IEEE39/results`. Κάθε σενάριο γράφεται σε ξεχωριστό υποφάκελο και περιέχει τα αρχεία `g1.csv`, `g2.csv`, κλπ., μαζί με ένα αρχείο `scenario.json` που περιλαμβάνει το configuration και την κατάσταση εκτέλεσης του σεναρίου.

**Ρύθμιση PowerFactory Python:** Για να τρέξει το data generation, το Python environment πρέπει να μπορεί να φορτώσει το Python API του PowerFactory. Στο virtual environment δημιουργήστε ένα αρχείο `powerfactory.pth` μέσα στο `.venv/Lib/site-packages` με μοναδική γραμμή το path προς τον φάκελο Python της εγκατάστασης του PowerFactory, π.χ. `C:\Program Files\DIgSILENT\PowerFactory <version>\Python\<python-version>`. Προσαρμόστε το `<version>` και το `<python-version>` ανάλογα με την εγκατάστασή σας.

**Εκτέλεση Generate Data:** Πριν εκτελεστεί το `IEEE39/generate_data.py`, πρέπει να είναι ενεργό το VPN και το PowerFactory να είναι κλειστό. Αν το PowerFactory είναι ήδη ανοιχτό, κλείστε το πρώτα και μετά ξεκινήστε το script.

**CLI Help:** Τα scripts που δέχονται παραμέτρους από command line εμφανίζουν όλες τις διαθέσιμες επιλογές με `--help`, π.χ. `python IEEE39/generate_data.py --help` και `python IEEE39/analyze_ieee39.py --help`.

**PowerFactory Context:** Αν τα default ονόματα του PowerFactory δεν ταιριάζουν στο μηχάνημα που τρέχει το script, μπορούν να γίνουν override από command line με `--project-name`, `--study-case` και `--grid-name`, για παράδειγμα `python IEEE39/generate_data.py --scenario load29 --project-name "39 Bus New England System" --study-case "RMS" --grid-name "Grid"`.

**Επιλογή Σεναρίων:** Τα διαθέσιμα preset scenarios εμφανίζονται με την εντολή `python IEEE39/generate_data.py --list-scenarios`. Το βασικό interface είναι το `--scenario` και δέχεται preset keys όπως `load29`, πολλαπλά preset keys όπως `load03 load24`, το ειδικό `all`, αλλά και bare custom load names όπως `"Load 20"`. Για παράδειγμα: `python IEEE39/generate_data.py --scenario load29`, `python IEEE39/generate_data.py --scenario load03 load24`, `python IEEE39/generate_data.py --scenario all`, ή `python IEEE39/generate_data.py --scenario "Load 20"`.

**Προσαρμοσμένα Σενάρια:** Για ad-hoc loads που δεν είναι hard-coded, ο τρόπος είναι το `--scenario`. Αν δοθεί μόνο όνομα load, π.χ. `python IEEE39/generate_data.py --scenario "Load 20"`, το script το ερμηνεύει αυτόματα ως custom scenario με defaults `dp=2`, `dq=0`, `duration=50s`, `event_time=0s`. Αν χρειάζονται ρητές τιμές, μπορεί να χρησιμοποιηθεί inline spec μορφής `load_name:dp[:dq[:duration[:event_time[:name]]]]`, για παράδειγμα `python IEEE39/generate_data.py --scenario "Load 20:2"`, `python IEEE39/generate_data.py --scenario "Load 24:2:0:60:0.5"`, ή `python IEEE39/generate_data.py --scenario "Load 20:2:0:60:0.5:load20_test"`. Το προαιρετικό τελευταίο πεδίο `name` είναι μόνο custom label για το τελικό scenario run folder κάτω από το `IEEE39/results`. Δεν αλλάζει το parent output root και δεν είναι το ίδιο πράγμα με το `--output-dir`.

**Χρόνος Προσομοίωσης και Load Event:** Από προεπιλογή η προσομοίωση τρέχει μέχρι `50s` και το load event τοποθετείται στο `t=0`, αλλά μπορούν να αλλάξουν από command line με `--duration` και `--event-time`. Για παράδειγμα, `python IEEE39/generate_data.py --scenario load03 --duration 60 --event-time 0.5` δημιουργεί τα δεδομένα με stop time `60s` και event στο `0.5s`. Αν στο inline spec δοθούν ήδη duration ή event time, π.χ. `python IEEE39/generate_data.py --scenario "Load 20:2:0:60:0.5"`, τότε αυτά υπερισχύουν των global defaults. Όταν το event time είναι διαφορετικό από το default, προστίθεται suffix τύπου `_evt0.5s` στο όνομα του scenario folder, ώστε να ξεχωρίζουν τα runs.

**Φάκελος Αποτελεσμάτων:** Από προεπιλογή τα αποτελέσματα γράφονται στο `IEEE39/results`. Αν χρειαστεί διαφορετικός φάκελος, μπορεί να δοθεί `--output-dir`, για παράδειγμα `python IEEE39/generate_data.py --scenario load29 --output-dir results_test`. Στο `generate_data.py`, κάθε relative path δίνεται ως relative προς τον φάκελο `IEEE39`, ενώ μπορεί να δοθεί και absolute path.

**Ανάλυση Δεδομένων:** Το αρχείο `IEEE39/analyze_ieee39.py` διαβάζει τα `g*.csv` από το `IEEE39/results` και γράφει τα αποτελέσματα στο `IEEE39/analysis`. Το `--scenario` είναι υποχρεωτικό για κανονικό run. Το `--scenario` δέχεται τρεις μορφές input: preset aliases όπως `load29`, πολλαπλά aliases όπως `load03 load24`, ή το ειδικό `all`, ακριβές folder name από το `IEEE39/results`, π.χ. `Load29_Pplus2_50s`, και custom run label όταν χρησιμοποιείται μαζί με `--data-dir`. Τα preset keys όπως `load29` δείχνουν στα προκαθορισμένα `Pplus2` source folders, άρα το `load29` αντιστοιχεί στο input `IEEE39/results/Load29_Pplus2_50s`. Για διαφορετικά paths μπορεί να δοθεί ρητά input και output, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario load29_p4 --data-dir results/Load29_Pplus4_50s --output-dir analysis/Load29_Pplus4_50s`. Όταν χρησιμοποιείται `--data-dir`, το όνομα που δίνεται στο `--scenario` είναι μόνο label για το run. Στο `analyze_ieee39.py`, τα `--data-dir`, `--output-dir` και `--analysis-dir` δέχονται relative paths relative προς τον φάκελο `IEEE39`, ή absolute paths.

**Scenario Alias Resolution:** Όταν δίνεται απλό alias τύπου `load07`, το `analyze_ieee39.py` προσπαθεί πρώτα να το αντιστοιχίσει στο default scenario variant του συγκεκριμένου load, δηλαδή στο run με `Pplus2`, `Q=0`, default duration και default event time, αν υπάρχει μοναδικό matching results folder. Αν αυτό δεν υπάρχει, αλλά υπάρχει μόνο ένα results folder για το ίδιο `load_name`, χρησιμοποιεί αυτό. Αν υπάρχουν πολλαπλά matching results folders για το ίδιο load, το script σταματά με error και ζητά να δοθεί το ακριβές folder name, π.χ. `Load07_Pplus2_50s`.

**Common Analyze Defaults:** Είτε το run προκύπτει από preset alias, είτε από ακριβές `results` folder name, είτε από custom label με `--data-dir`, η βασική default ανάλυση είναι η ίδια: start στο `0s`, end στο τελευταίο διαθέσιμο sample, `reset_time=True`, generators `g1..g10`, σήματα `Voltage`, `Current`, `Active Power`, `Reactive Power`, fixed orders `[2, 4, 6]`, taus `[1, 0.1, 0.01]`, `auto_order_decimation=10` και low-pass filter `fc=10`, `N=15`. Οι μόνες διαφορές που αλλάζουν χωρίς CLI override είναι τα paths και το run label.

**Ονόματα Analysis Folders:** Αν δεν δοθεί ρητά `--output-dir`, το output folder περιλαμβάνει και το time window mode. Για το default fixed window, το `load29` γράφει σε φάκελο τύπου `IEEE39/analysis/Load29_Pplus2_50s_0_to_end_reset`. Αν δοθεί `--time-end 20`, γράφει σε `..._0_to_20_reset`. Αν δοθεί `--time-cross global`, το suffix αλλάζει σε μορφή `..._tcross-global_off0_to_end_reset`. Αν δοθεί και reference signal, π.χ. `--time-cross global --time-cross-reference g2:Current`, το suffix γίνεται `..._tcross-global_ref-g2-current_off0_to_end_reset`. Για `--time-cross per-signal --time-start 0.2` γίνεται `..._tcross-per-signal_off0.2_to_end_reset`. Αν δοθεί `--no-reset-time`, το suffix τελειώνει σε `noreset`. Όταν χρησιμοποιούνται subsets με `--generators` ή `--signals`, προστίθεται και αντίστοιχο suffix στο folder name. Το `analysis_config.json` της ανάλυσης περιέχει το `time_mask`, το `time_cross`, το resolved time window, τα resolved zero-cross starts και τα subsets που χρησιμοποιήθηκαν, ώστε το `--skip-matrix-pencil` να μπορεί να αναπαράγει το ίδιο setup.

**Επιλογές Ανάλυσης:** Από προεπιλογή η ανάλυση τρέχει χωρίς plots και χωρίς clustering, ώστε να είναι πιο γρήγορα τα επαναλαμβανόμενα runs για window tuning. Για να ενεργοποιηθούν ρητά, χρησιμοποιούνται τα `--plots` και `--clustering`. Το `--clustering` από μόνο του ενεργοποιεί by-control-area clustering. Αν δεν θέλετε clustering, μπορείτε να αφήσετε το default ή να δώσετε και `--skip-clustering`. Αν δεν θέλετε plots, μπορείτε να αφήσετε το default ή να δώσετε και `--skip-plots`. Αν θέλετε γρήγορα δοκιμαστικά runs μόνο για συγκεκριμένες γεννήτριες ή σήματα, μπορούν να χρησιμοποιηθούν τα `--generators` και `--signals`, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario load03 --time-start 0.4 --generators g2 g3 g6 --signals Voltage "Reactive Power"`. Η εντολή `python IEEE39/analyze_ieee39.py --scenario load29` κάνει full analysis run: διαβάζει το default source folder `IEEE39/results/Load29_Pplus2_50s`, ξανατρέχει Matrix Pencil, και γράφει σε default analysis folder όπως `IEEE39/analysis/Load29_Pplus2_50s_0_to_end_reset`. Αν οι πόλοι έχουν ήδη υπολογιστεί και υπάρχει `results.csv`, μπορεί να παρακαμφθεί ο Matrix Pencil με `--skip-matrix-pencil`, αλλά πλέον αυτό απαιτεί ρητά και `--analysis-dir`. Για παράδειγμα: `python IEEE39/analyze_ieee39.py --scenario Load29_Pplus2_50s --skip-matrix-pencil --analysis-dir analysis/Load29_Pplus2_50s_0_to_end_reset`. Σημαντική διευκρίνιση: στο `--skip-matrix-pencil` με `--analysis-dir`, το `--scenario` δεν πρέπει να είναι το όνομα του analysis folder, αλλά το source scenario alias ή το source folder μέσα στο `IEEE39/results`. Δηλαδή σωστό είναι `python IEEE39/analyze_ieee39.py --scenario Load29_Pplus2_50s --skip-matrix-pencil --plots --analysis-dir analysis/Load29_Pplus2_50s_tcross-global_off0_to_end_reset`, ενώ λάθος είναι να δοθεί `--scenario Load29_Pplus2_50s_tcross-global_off0_to_end_reset`. Τα `--data-dir`, `--output-dir` και `--analysis-dir` δέχονται είτε absolute paths είτε relative paths relative προς τον φάκελο `IEEE39`. 

**Clustering:** Το clustering είναι default off σε όλα τα analyze modes. Όταν ενεργοποιείται με `--clustering`, από προεπιλογή τρέχει μόνο ανά περιοχή ελέγχου, δηλαδή πρακτικά σαν `--clustering --clustering-scope areas`. Η επιλογή γίνεται με `--clustering-scope areas`, `--clustering-scope both`, `--clustering-scope global`, ή `--clustering-scope none`. Για παράδειγμα, `python IEEE39/analyze_ieee39.py --scenario load29 --skip-matrix-pencil --clustering --clustering-scope both` χρησιμοποιεί υπάρχον `results.csv` και παράγει clustering και συνολικά και ανά περιοχή ελέγχου.

**Στατιστικά & Διαγράμματα:** Το `comprehensive_report.csv` παράγεται πάντα στο `IEEE39/analysis/<scenario>/stats/comprehensive_report.csv`, ακόμη και όταν χρησιμοποιείται `--skip-matrix-pencil` με ήδη υπάρχον `results.csv`. Τα modal maps και τα reconstructions του IEEE39 δημιουργούνται μόνο όταν ενεργοποιείται το `--plots`. Για το `load29`, με fixed default window, τα modal maps θα βρίσκονται στο `IEEE39/analysis/Load29_Pplus2_50s_0_to_end_reset/plots/modal_maps` και τα reconstruction grids στο `IEEE39/analysis/Load29_Pplus2_50s_0_to_end_reset/plots/reconstruction_grids`. Αν χρησιμοποιούνται subsets με `--generators` ή `--signals`, το `comprehensive_report.csv` περιέχει μόνο το subset που ζητήθηκε.

**Αυτόματη Αξιολόγηση Analysis Runs:** Κάθε φορά που ολοκληρώνεται το `IEEE39/analyze_ieee39.py`, προστίθεται στο `analysis_config.json` του αντίστοιχου analysis folder ένα πεδίο `evaluation`. Εκεί αποθηκεύονται:

- συνοπτικά reconstruction metrics (`mean_R2`, `best_mean_R2`, `negative_R2_count` κλπ.)
- modal identification metrics σε σχέση με τα γνωστά literature modes του IEEE39
- οι καλύτερες mode matches για κάθε literature mode
- το καλύτερο reconstruction ανά generator/signal
- ένα μικρό subset με τα χειρότερα best-case reconstructions για γρήγορη επισκόπηση
- modal identification summary: πόσα literature modes βρέθηκαν στα `loose`, `mid` και `strong` thresholds, ποια ήταν αυτά τα modes, και ποιο identified mode ήταν το πιο κοντινό σε κάθε reference mode της βιβλιογραφίας

Επιπλέον, στο root του ίδιου `analysis_config.json` αποθηκεύονται πλέον και diagnostics για το raw Matrix Pencil output coverage, όπως:

- `oscillatory_frequency_threshold_hz`
- `result_coverage`
- `missing_results`
- `result_filter_diagnostics`

Έτσι φαίνεται ρητά ποια `(generator, signal, method)` combinations δεν έδωσαν τελικά oscillatory poles στο `results.csv`, καθώς και αν αυτό οφείλεται π.χ. σε `missing_csv`, `missing_signal_column`, `not_enough_samples_after_preprocessing` ή `all_poles_below_frequency_threshold`.

Άρα, για κάθε μεμονωμένο analysis folder, η βασική πηγή αλήθειας είναι πλέον το ίδιο το `analysis_config.json`.

Αν θέλετε σύγκριση πολλών runs μαζί, το πιο γρήγορο αρχείο είναι το `run_summary.csv` που γράφει το `summarize_analysis_runs.py`, όπου υπάρχουν ήδη συγκεντρωμένα τα counts και τα recovered modes για κάθε run.

**Standalone Evaluation for Existing Folders:** Αν θέλετε να ενημερώσετε το `analysis_config.json` παλιότερων analysis folders χωρίς να ξανατρέξετε Matrix Pencil, μπορείτε να χρησιμοποιήσετε το `python IEEE39/evaluate_analysis_folder.py --analysis-dir <folder> [<folder> ...]`. Το `--analysis-dir` δέχεται ένα ή περισσότερα analysis folder paths και, όταν δοθεί relative path, θεωρείται relative προς τον φάκελο `IEEE39`. Για παράδειγμα, `python IEEE39/evaluate_analysis_folder.py --analysis-dir analysis/Load03_Pplus2_50s_0.4_to_end_reset` θα ξαναϋπολογίσει την αξιολόγηση και θα τη γράψει μέσα στο `analysis_config.json` αυτού του folder. Το script δεν δημιουργεί ξεχωριστά output files. Ενημερώνει μόνο το πεδίο `evaluation` μέσα στο υπάρχον `analysis_config.json`.

**Συγκεντρωτική Σύγκριση Πολλών Runs:** Αν θέλετε aggregate σύγκριση πολλών analysis folders μαζί, μπορείτε να χρησιμοποιήσετε το `python IEEE39/summarize_analysis_runs.py`. Το script αυτό δεν βασίζεται στα folder names, αλλά μόνο στα metadata του `analysis_config.json` και του συνδεδεμένου `scenario.json` μέσα σε κάθε analysis folder. Υπάρχουν δύο modes και πρέπει να επιλέγεται ακριβώς ένα από αυτά: είτε `--load`, είτε `--analysis-dir`. Στο metadata scan mode, το script σαρώνει όλα τα subfolders κάτω από το `IEEE39/analysis`, απαιτεί `--load`, και χρησιμοποιεί από προεπιλογή τα standard IEEE39 values `dp=2`, `dq=0`, `event_time=0`, `duration=50`, εκτός αν δοθούν ρητά άλλα φίλτρα. Όταν δεν δοθεί `--output-dir` σε αυτό το mode, το output folder ονομάζεται αυτόματα κάτω από το `IEEE39/analysis/summaries`, π.χ. `analysis/summaries/summary_load03_dp2_dq0_evt0_dur50`. Στο explicit-folder mode, μπορείτε να δώσετε ρητά folders με `--analysis-dir`, αλλά τότε πρέπει να δώσετε και custom `--output-dir`, ώστε να είναι ξεκάθαρο ότι η σύνοψη αφορά μόνο αυτά τα συγκεκριμένα folders. Το `--load` είναι convenience φίλτρο και δέχεται είτε analysis scenario name τύπου `load03`, είτε source load name τύπου `Load 03`. Το `--output-dir` δέχεται relative ή absolute path και, όταν είναι relative, θεωρείται relative προς τον φάκελο `IEEE39`. Παραδείγματα:

- `python IEEE39/summarize_analysis_runs.py --load load03`
- `python IEEE39/summarize_analysis_runs.py --load "Load 03" --dp-percent 2 --dq-percent 0 --event-time 0 --duration 50`
- `python IEEE39/summarize_analysis_runs.py --analysis-dir analysis/Load03_Pplus2_50s_0_to_end_reset analysis/Load03_Pplus2_50s_0.4_to_end_reset --output-dir analysis/summaries/summary_load03_manual_compare`

Το script γράφει:

- `run_summary.csv`: compact table για γρήγορο compare/ranking των runs
- `run_summary_full.csv`: αναλυτικό table με όλα τα διαθέσιμα summary metrics και τα derived ranking columns
- `summary.json`: compact περιγραφή του τι φίλτρα εφαρμόστηκαν, πόσα runs βρέθηκαν και ποιο run βγήκε πρώτο σε modal και reconstruction ranking

**Νικητές στο aggregate summary:** Το `summary.json` του script γράφει πλέον ξεχωριστά:

- `top_modal_run`
- `top_reconstruction_run`
- `top_unweighted_run`
- `top_weighted_run`

Ο `top_unweighted_run` προκύπτει από απλό άθροισμα `modal_rank + reconstruction_rank`, δηλαδή χωρίς extra βάρη. Ο `top_weighted_run` προκύπτει από το `weighted_overall_score`, όπου μπορείτε να δώσετε μεγαλύτερη σημασία στο modal identification ή στο reconstruction.

Το `modal_rank` προκύπτει με modal-first λογική: πρώτα ταξινομούνται τα runs με βάση τα `modal_mid_modes`, μετά τα `modal_strong_modes`, μετά τα `modal_loose_modes`, και σε ισοβαθμίες χρησιμοποιούνται ως tie-breakers τα `best_mean_R2`, `mean_R2` και `negative_R2_count`. Τα `modal_loose/mid/strong_modes` δεν μετρούν μόνο το πόσο κοντά είναι ένα identified mode σε συχνότητα και απόσβεση, αλλά ελέγχουν και αν εμφανίζεται σε γεννήτριες που συμφωνούν με τη βιβλιογραφία για το συγκεκριμένο literature mode. Το `reconstruction_rank` προκύπτει από τα `best_mean_R2`, `best_min_R2`, `mean_R2` και `negative_R2_count`. Το `best_mean_R2` είναι ο μέσος όρος του καλύτερου `R2` ανά `(generator, signal)`, δηλαδή για κάθε generator/signal κρατιέται πρώτα η μέθοδος με το μεγαλύτερο `R2` και μετά υπολογίζεται ο μέσος όρος αυτών των best-case reconstructions. Το clustering δεν συμμετέχει σε αυτό το ranking. Χρησιμοποιείται μόνο ως ξεχωριστό exploratory/post-processing output.

Τα thresholds για το modal identification είναι τα εξής:

| Level | Tolerance στη συχνότητα | Tolerance στην απόσβεση |
| :---: | :---: | :---: |
| `loose` | `0.08 Hz` | `0.15` |
| `mid` | `0.05 Hz` | `0.10` |
| `strong` | `0.03 Hz` | `0.05` |

Άρα το `mid` είναι το βασικό πρακτικό threshold του ranking, το `strong` είναι πιο αυστηρό, ενώ το `loose` είναι πιο permissive.

**Βάρη στο aggregate ranking:** Στο `summarize_analysis_runs.py` μπορείτε να δώσετε μεγαλύτερο βάρος στο modal identification με:

- `--modal-weight <value>`
- `--reconstruction-weight <value>`

Από προεπιλογή το aggregate script χρησιμοποιεί:

- `modal_weight = 3`
- `reconstruction_weight = 1`

ώστε το τελικό `weighted_overall_score` να δίνει μεγαλύτερη σημασία στο modal identification απ' ό,τι στο reconstruction. Αν θέλετε ακόμη πιο αυστηρό modal-first ranking, μπορείτε π.χ. να τρέξετε:

`python IEEE39/summarize_analysis_runs.py --load load03 --modal-weight 5 --reconstruction-weight 1 --output-dir analysis/summaries/summary_load03_modalheavy`

**Χρόνος Σημάτων:** Από προεπιλογή η ανάλυση διατηρεί όλα τα samples από `0s` και μετά μέχρι την τελευταία χρονική μέτρηση του CSV. Μετά μετατοπίζει τον χρόνο του επιλεγμένου παραθύρου ώστε το πρώτο κρατημένο sample να γίνει `t=0`.

**Fixed Time Mask:** Αν χρειάζεται σταθερό παράθυρο, δίνεται `--time-start` ή/και `--time-end`. Όταν δεν χρησιμοποιείται `--time-cross`, το `--time-start` σημαίνει απόλυτο χρόνο έναρξης πάνω στον αρχικό άξονα του CSV. Για παράδειγμα, `python IEEE39/analyze_ieee39.py --scenario load29 --time-start 0.4 --time-end 20` κρατά μόνο τις γραμμές από `0.4s` έως `20s`. Αν πρέπει να διατηρηθούν οι αρχικοί χρόνοι του CSV χωρίς μετατόπιση, δίνεται `--no-reset-time`.

**Time Cross:** Εναλλακτικά, μπορεί να δοθεί `--time-cross global` ή `--time-cross per-signal`. Και στις δύο περιπτώσεις η ανίχνευση του πρώτου zero crossing γίνεται πάντα πάνω στο ίδιο σήμα που θα αναλυθεί, αφού πρώτα γίνει `detrend` και μετά `low-pass filtering`. Με το `global`, αν δεν δοθεί reference signal, υπολογίζεται ο πρώτος zero crossing για κάθε επιλεγμένο generator/signal και τελικά χρησιμοποιείται ένας κοινός χρόνος έναρξης για όλο το run, ίσος με το μέγιστο από αυτούς τους χρόνους. Αν δοθεί `--time-cross-reference`, τότε το global mode χρησιμοποιεί αποκλειστικά τον zero crossing του συγκεκριμένου reference signal και τον εφαρμόζει ως κοινό start σε όλο το run. Με το `per-signal`, κάθε generator/signal κρατά τον δικό του πρώτο zero crossing και αναλύεται με διαφορετικό χρόνο έναρξης.

**Time Cross Reference:** Το `--time-cross-reference` χρησιμοποιείται μόνο με `--time-cross global` και δέχεται μορφή όπως `g2:Current`, `g3:Voltage` ή `g2:s:cur1 in p.u.`. Με αυτόν τον τρόπο μπορείτε να δοκιμάσετε global κοινό start που προέρχεται από ένα modal-relevant reference signal, αντί να αφήσετε το script να πάρει τον πιο αργό πρώτο μηδενισμό από όλα τα επιλεγμένα signals.

**Time Start with Time Cross:** Όταν χρησιμοποιείται `--time-cross`, το `--time-start` δεν είναι απόλυτος χρόνος έναρξης, αλλά offset μετά τον ανιχνευμένο zero crossing. Για παράδειγμα, `python IEEE39/analyze_ieee39.py --scenario load03 --time-cross global` ξεκινά ακριβώς από τον μέγιστο χρόνο zero crossing, ενώ `python IEEE39/analyze_ieee39.py --scenario load03 --time-cross global --time-cross-reference g2:Current --time-start 0.2` ξεκινά `0.2s` μετά από τον zero crossing του `g2:Current` και χρησιμοποιεί αυτόν τον κοινό μετατοπισμένο χρόνο σε όλα τα signals. Αντίστοιχα, `python IEEE39/analyze_ieee39.py --scenario load03 --time-cross per-signal --time-start 0.1` ξεκινά κάθε signal `0.1s` μετά από τον δικό του πρώτο zero crossing.
