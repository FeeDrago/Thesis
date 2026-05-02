# Διπλωματική Εργασία

Το παρόν αποθετήριο περιέχει το υλικό της διπλωματικής μου εργασίας.

## Φάκελος: Preliminary Investigation
Εδώ βρίσκεται συγκεντρωμένο όλο το υλικό της προκαταρκτικής μελέτης:

**Αναφορά:** Το κείμενο της προκαταρκτικής αναφοράς είναι διαθέσιμο σε έτοιμο αρχείο PDF για άμεση ανάγνωση, μαζί με τα πηγαία αρχεία του LaTeX σε περίπτωση που χρειαστούν.

**Κώδικας & Αποτελέσματα:** Περιλαμβάνονται τα αρχεία Python της ανάλυσης, καθώς και τα διαγράμματα (plots) που έχουν παραχθεί. Τα διαγράμματα είναι ήδη αποθηκευμένα, ώστε να έχετε άμεση εικόνα των αποτελεσμάτων χωρίς να χρειαστεί να εκτελέσετε τα scripts.

## Φάκελος: IEEE39
Εδώ βρίσκεται το υλικό για την παραγωγή και την ανάλυση δεδομένων του συστήματος IEEE 39-bus.

**Παραγωγή Δεδομένων:** Το αρχείο `IEEE39/generate_data.py` εκτελεί σενάρια στο PowerFactory και αποθηκεύει τα αποτελέσματα στον φάκελο `IEEE39/results`. Κάθε σενάριο γράφεται σε ξεχωριστό υποφάκελο και περιέχει τα αρχεία `g1.csv`, `g2.csv`, κλπ., μαζί με ένα αρχείο `scenario.json` που περιλαμβάνει το configuration και την κατάσταση εκτέλεσης του σεναρίου.

**Ρύθμιση PowerFactory Python:** Για να τρέξει το data generation, το Python environment πρέπει να μπορεί να φορτώσει το Python API του PowerFactory. Στο virtual environment δημιουργήστε ένα αρχείο `powerfactory.pth` μέσα στο `.venv/Lib/site-packages` με μοναδική γραμμή το path προς τον φάκελο Python της εγκατάστασης του PowerFactory, π.χ. `C:\Program Files\DIgSILENT\PowerFactory <version>\Python\<python-version>`. Προσαρμόστε το `<version>` και το `<python-version>` ανάλογα με την εγκατάστασή σας.

**Εκτέλεση Generate Data:** Πριν εκτελεστεί το `IEEE39/generate_data.py`, πρέπει να είναι ενεργό το VPN και το PowerFactory να είναι κλειστό. Αν το PowerFactory είναι ήδη ανοιχτό, κλείστε το πρώτα και μετά ξεκινήστε το script.

**CLI Help:** Τα scripts που δέχονται παραμέτρους από command line εμφανίζουν όλες τις διαθέσιμες επιλογές με `--help`, π.χ. `python IEEE39/generate_data.py --help` και `python IEEE39/analyze_ieee39.py --help`.

**PowerFactory Context:** Αν τα default ονόματα του PowerFactory δεν ταιριάζουν στο μηχάνημα που τρέχει το script, μπορούν να γίνουν override από command line με `--project-name`, `--study-case` και `--grid-name`, για παράδειγμα `python IEEE39/generate_data.py --scenario load29 --project-name "39 Bus New England System" --study-case "RMS" --grid-name "Grid"`.

**Επιλογή Σεναρίων:** Τα διαθέσιμα σενάρια εμφανίζονται με την εντολή `python IEEE39/generate_data.py --list-scenarios`. Η εκτέλεση γίνεται με `--scenario`, για παράδειγμα `python IEEE39/generate_data.py --scenario load29`, `python IEEE39/generate_data.py --scenario load03 load24`, ή `python IEEE39/generate_data.py --scenario all`.

**Προσαρμοσμένα Σενάρια:** Μπορεί να δοθεί σενάριο απευθείας από το command line με μορφή `load_name:dp[:dq[:duration[:event_time[:name]]]]`, για παράδειγμα `python IEEE39/generate_data.py --scenario "Load 29:2:0"` ή `python IEEE39/generate_data.py --scenario "Load 24:2:0:60:0.5"`. Εναλλακτικά μπορεί να χρησιμοποιηθεί το `--case`, είτε στην παλιά μορφή `python IEEE39/generate_data.py --case "Load 24" 2 0`, είτε ως quoted spec, για παράδειγμα `python IEEE39/generate_data.py --case "Load 24:2:0:60:0.5"`.

**Χρόνος Προσομοίωσης και Load Event:** Από προεπιλογή η προσομοίωση τρέχει μέχρι `50s` και το load event τοποθετείται στο `t=0`, αλλά μπορούν να αλλάξουν από command line με `--duration` και `--event-time`. Για παράδειγμα, `python IEEE39/generate_data.py --scenario load03 --duration 60 --event-time 0.5` δημιουργεί τα δεδομένα με stop time `60s` και event στο `0.5s`. Τα quoted inline specs ή `--case` specs μπορούν επίσης να ορίσουν δικό τους duration και event time, π.χ. `python IEEE39/generate_data.py --case "Load 24:2:0:60:0.5"`. Όταν το event time είναι διαφορετικό από το default, προστίθεται suffix τύπου `_evt0.5s` στο όνομα του scenario folder, ώστε να ξεχωρίζουν τα runs.

**Φάκελος Αποτελεσμάτων:** Από προεπιλογή τα αποτελέσματα γράφονται στο `IEEE39/results`. Αν χρειαστεί διαφορετικός φάκελος, μπορεί να δοθεί `--output-dir`, για παράδειγμα `python IEEE39/generate_data.py --scenario load29 --output-dir results_test`. Στο `generate_data.py`, κάθε relative path δίνεται ως relative προς τον φάκελο `IEEE39`, ενώ μπορεί να δοθεί και absolute path.

**Ανάλυση Δεδομένων:** Το αρχείο `IEEE39/analyze_ieee39.py` διαβάζει τα `g*.csv` από το `IEEE39/results` και γράφει τα αποτελέσματα στο `IEEE39/analysis`. Η εκτέλεση γίνεται με `python IEEE39/analyze_ieee39.py --scenario load29`, `python IEEE39/analyze_ieee39.py --scenario load03 load24`, ή `python IEEE39/analyze_ieee39.py --scenario all`. Τα default keys όπως `load29` δείχνουν στα προκαθορισμένα σενάρια `Pplus2`, άρα το `load29` αντιστοιχεί στο input `IEEE39/results/Load29_Pplus2_50s`. Για διαφορετικά paths μπορεί να δοθεί ρητά input και output, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario load29_p4 --data-dir results/Load29_Pplus4_50s --output-dir analysis/Load29_Pplus4_50s`. Το όνομα που δίνεται στο `--scenario` σε αυτήν την περίπτωση είναι απλώς label για το run και δεν χρειάζεται να είναι προκαθορισμένο key. Στο `analyze_ieee39.py`, τα `--data-dir`, `--output-dir` και `--analysis-dir` δέχονται relative paths relative προς τον φάκελο `IEEE39`, ή absolute paths.

**Scenario Alias Resolution:** Όταν δίνεται απλό alias τύπου `load07`, το `analyze_ieee39.py` προσπαθεί πρώτα να το αντιστοιχίσει στο default scenario variant του συγκεκριμένου load, δηλαδή στο run με `Pplus2`, `Q=0`, default duration και default event time, αν υπάρχει μοναδικό matching results folder. Αν αυτό δεν υπάρχει, αλλά υπάρχει μόνο ένα results folder για το ίδιο `load_name`, χρησιμοποιεί αυτό. Αν υπάρχουν πολλαπλά matching results folders για το ίδιο load, το script σταματά με error και ζητά να δοθεί το ακριβές folder name, π.χ. `Load07_Pplus2_50s`.

**Ονόματα Analysis Folders:** Αν δεν δοθεί ρητά `--output-dir`, το output folder περιλαμβάνει και το time window mode. Για το default fixed window, το `load29` γράφει σε φάκελο τύπου `IEEE39/analysis/Load29_Pplus2_50s_0_to_end_reset`. Αν δοθεί `--time-end 20`, γράφει σε `..._0_to_20_reset`. Αν δοθεί `--time-cross global`, το suffix αλλάζει σε μορφή `..._tcross-global_off0_to_end_reset`. Αν δοθεί και reference signal, π.χ. `--time-cross global --time-cross-reference g2:Current`, το suffix γίνεται `..._tcross-global_ref-g2-current_off0_to_end_reset`. Για `--time-cross per-signal --time-start 0.2` γίνεται `..._tcross-per-signal_off0.2_to_end_reset`. Αν δοθεί `--no-reset-time`, το suffix τελειώνει σε `noreset`. Όταν χρησιμοποιούνται subsets με `--generators` ή `--signals`, προστίθεται και αντίστοιχο suffix στο folder name. Το `analysis_config.json` της ανάλυσης περιέχει το `time_mask`, το `time_cross`, το resolved time window, τα resolved zero-cross starts και τα subsets που χρησιμοποιήθηκαν, ώστε το `--skip-matrix-pencil` να μπορεί να αναπαράγει το ίδιο setup.

**Επιλογές Ανάλυσης:** Από προεπιλογή η ανάλυση τρέχει χωρίς plots και χωρίς clustering, ώστε να είναι πιο γρήγορα τα επαναλαμβανόμενα runs για window tuning. Για να ενεργοποιηθούν ρητά, χρησιμοποιούνται τα `--plots` και `--clustering`. Αν δεν θέλετε clustering, μπορείτε να αφήσετε το default ή να δώσετε και `--skip-clustering`. Αν δεν θέλετε plots, μπορείτε να αφήσετε το default ή να δώσετε και `--skip-plots`. Αν θέλετε γρήγορα δοκιμαστικά runs μόνο για συγκεκριμένες γεννήτριες ή σήματα, μπορούν να χρησιμοποιηθούν τα `--generators` και `--signals`, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario load03 --time-start 0.4 --generators g2 g3 g6 --signals Voltage "Reactive Power"`. Αν οι πόλοι έχουν ήδη υπολογιστεί και υπάρχει `results.csv`, μπορεί να παρακαμφθεί ο Matrix Pencil με `--skip-matrix-pencil`. Η εντολή `python IEEE39/analyze_ieee39.py --scenario load29 --skip-matrix-pencil` επιλέγει το default analysis folder με το default fixed time mask, δηλαδή `IEEE39/analysis/Load29_Pplus2_50s_0_to_end_reset`. Για άλλο ήδη υπάρχον analysis folder, πρώτα εμφανίζονται οι επιλογές με `python IEEE39/analyze_ieee39.py --list-analysis` και μετά δίνεται ρητά φάκελος, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario load29 --skip-matrix-pencil --analysis-dir analysis/Load29_Pplus2_50s_tcross-global_off0_to_end_reset`. Το `--analysis-dir` είναι relative προς `IEEE39` όταν δίνεται relative path.

**Clustering:** Όταν ενεργοποιείται με `--clustering`, από προεπιλογή το clustering γίνεται μόνο ανά περιοχή ελέγχου. Η επιλογή γίνεται με `--clustering-scope areas`, `--clustering-scope both`, `--clustering-scope global`, ή `--clustering-scope none`. Για παράδειγμα, `python IEEE39/analyze_ieee39.py --scenario load29 --skip-matrix-pencil --clustering --clustering-scope both` χρησιμοποιεί υπάρχον `results.csv` και παράγει clustering και συνολικά και ανά περιοχή ελέγχου.

**Στατιστικά & Διαγράμματα:** Το `comprehensive_report.csv` παράγεται πάντα στο `IEEE39/analysis/<scenario>/stats/comprehensive_report.csv`, ακόμη και όταν χρησιμοποιείται `--skip-matrix-pencil` με ήδη υπάρχον `results.csv`. Τα modal maps και τα reconstructions του IEEE39 δημιουργούνται μόνο όταν ενεργοποιείται το `--plots`. Για το `load29`, με fixed default window, τα modal maps θα βρίσκονται στο `IEEE39/analysis/Load29_Pplus2_50s_0_to_end_reset/plots/modal_maps` και τα reconstruction grids στο `IEEE39/analysis/Load29_Pplus2_50s_0_to_end_reset/plots/reconstruction_grids`. Αν χρησιμοποιούνται subsets με `--generators` ή `--signals`, το `comprehensive_report.csv` περιέχει μόνο το subset που ζητήθηκε.

**Αυτόματη Αξιολόγηση Analysis Runs:** Κάθε φορά που ολοκληρώνεται το `IEEE39/analyze_ieee39.py`, ενημερώνεται αυτόματα το `analysis_config.json` του αντίστοιχου analysis folder με ένα πεδίο `evaluation`. Εκεί αποθηκεύονται:

- συνοπτικά reconstruction metrics (`mean_R2`, `best_mean_R2`, `negative_R2_count` κλπ.)
- modal identification metrics σε σχέση με τα γνωστά literature modes του IEEE39
- οι καλύτερες mode matches για κάθε literature mode
- το καλύτερο reconstruction ανά generator/signal
- ένα μικρό subset με τα χειρότερα best-case reconstructions για γρήγορη επισκόπηση

Με αυτόν τον τρόπο κάθε analysis folder παραμένει self-contained, χωρίς να χρειάζεται να δημιουργούνται πολλά extra output files ανά run.

**Τι περιέχει το evaluation:** Το `evaluation` section του `analysis_config.json` γράφει δύο ειδών πληροφορία:

- reconstruction summary: `mean_R2`, `best_mean_R2`, `negative_R2_count`, `best_*_R2`
- modal identification summary: πόσα literature modes βρέθηκαν στα `loose`, `mid` και `strong` thresholds, ποια ήταν αυτά τα modes, και ποιο identified mode ήταν το πιο κοντινό σε κάθε reference mode της βιβλιογραφίας

Άρα, για κάθε μεμονωμένο analysis folder, η βασική πηγή αλήθειας είναι πλέον το ίδιο το `analysis_config.json`.

**Standalone Evaluation for Existing Folders:** Αν θέλετε να ενημερώσετε το `analysis_config.json` παλιότερων analysis folders χωρίς να ξανατρέξετε Matrix Pencil, μπορείτε να χρησιμοποιήσετε το `python IEEE39/evaluate_analysis_folder.py --analysis-dir <folder> [<folder> ...]`. Το `--analysis-dir` δέχεται ένα ή περισσότερα analysis folder paths και, όταν δοθεί relative path, θεωρείται relative προς τον φάκελο `IEEE39`. Για παράδειγμα, `python IEEE39/evaluate_analysis_folder.py --analysis-dir analysis/Load03_Pplus2_50s_0.4_to_end_reset` θα ξαναϋπολογίσει την αξιολόγηση και θα τη γράψει μέσα στο `analysis_config.json` αυτού του folder. Το script δεν δημιουργεί ξεχωριστά output files. Ενημερώνει μόνο το πεδίο `evaluation` μέσα στο υπάρχον `analysis_config.json`.

**Συγκεντρωτική Σύγκριση Πολλών Runs:** Αν θέλετε aggregate σύγκριση πολλών analysis folders μαζί, μπορείτε να χρησιμοποιήσετε το `python IEEE39/summarize_analysis_runs.py`. Το script αυτό δεν βασίζεται στα folder names, αλλά στα metadata του `analysis_config.json` και του συνδεδεμένου `scenario.json`. Μπορείτε είτε να δώσετε ρητά folders με `--analysis-dir`, είτε να αφήσετε το script να ψάξει κάτω από το `IEEE39/analysis` και να φιλτράρετε με metadata όπως `--load`, `--scenario-name`, `--load-name`, `--dp-percent`, `--dq-percent`, `--event-time`, `--duration`. Το `--load` είναι convenience φίλτρο και δέχεται είτε analysis scenario name τύπου `load03`, είτε source load name τύπου `Load 03`. Το `--output-dir` δέχεται relative ή absolute path και, όταν είναι relative, θεωρείται relative προς τον φάκελο `IEEE39`. Για παράδειγμα, `python IEEE39/summarize_analysis_runs.py --load load03 --dp-percent 2 --dq-percent 0 --event-time 0 --duration 50 --output-dir analysis/summary_load03_dp2_dq0_evt0_dur50` βρίσκει όλα τα σχετικά analysis folders και γράφει:

- `run_summary.csv`: ένα row ανά analysis run με reconstruction και modal-identification metrics
- `summary.json`: compact περιγραφή του τι φίλτρα εφαρμόστηκαν, πόσα runs βρέθηκαν και ποιο run βγήκε πρώτο σε modal και reconstruction ranking

Το aggregate summary script δεν αλλάζει κανένα `analysis_config.json`. Γράφει μόνο τα δικά του συγκεντρωτικά outputs στον φάκελο που δώσατε στο `--output-dir`.

**Νικητές στο aggregate summary:** Το `summary.json` του aggregate script γράφει πλέον ξεχωριστά:

- `top_modal_run`
- `top_reconstruction_run`
- `top_unweighted_run`
- `top_weighted_run`

Ο `top_unweighted_run` προκύπτει από απλό άθροισμα `modal_rank + reconstruction_rank`, δηλαδή χωρίς extra βάρη. Ο `top_weighted_run` προκύπτει από το `weighted_overall_score`, όπου μπορείτε να δώσετε μεγαλύτερη σημασία στο modal identification ή στο reconstruction.

**Βάρη στο aggregate ranking:** Στο `summarize_analysis_runs.py` μπορείτε να δώσετε μεγαλύτερο βάρος στο modal identification με:

- `--modal-weight <value>`
- `--reconstruction-weight <value>`

Από προεπιλογή το aggregate script χρησιμοποιεί:

- `modal_weight = 3`
- `reconstruction_weight = 1`

ώστε το τελικό `weighted_overall_score` να δίνει μεγαλύτερη σημασία στο modal identification απ' ό,τι στο reconstruction. Αν θέλετε ακόμη πιο αυστηρό modal-first ranking, μπορείτε π.χ. να τρέξετε:

`python IEEE39/summarize_analysis_runs.py --load load03 --dp-percent 2 --dq-percent 0 --event-time 0 --duration 50 --modal-weight 5 --reconstruction-weight 1 --output-dir analysis/summary_load03_modalheavy`

Σε αυτή την περίπτωση, το `run_summary.csv` θα περιέχει και τα πεδία:

- `modal_rank`
- `reconstruction_rank`
- `unweighted_overall_score`
- `modal_weight`
- `reconstruction_weight`
- `weighted_overall_score`

ώστε να φαίνεται ξεκάθαρα πώς προέκυψε τόσο η unweighted όσο και η weighted τελική κατάταξη.

**Χρόνος Σημάτων:** Από προεπιλογή η ανάλυση διατηρεί όλα τα samples από `0s` και μετά μέχρι την τελευταία χρονική μέτρηση του CSV. Μετά μετατοπίζει τον χρόνο του επιλεγμένου παραθύρου ώστε το πρώτο κρατημένο sample να γίνει `t=0`.

**Fixed Time Mask:** Αν χρειάζεται σταθερό παράθυρο, δίνεται `--time-start` ή/και `--time-end`. Όταν δεν χρησιμοποιείται `--time-cross`, το `--time-start` σημαίνει απόλυτο χρόνο έναρξης πάνω στον αρχικό άξονα του CSV. Για παράδειγμα, `python IEEE39/analyze_ieee39.py --scenario load29 --time-start 0.4 --time-end 20` κρατά μόνο τις γραμμές από `0.4s` έως `20s`. Αν πρέπει να διατηρηθούν οι αρχικοί χρόνοι του CSV χωρίς μετατόπιση, δίνεται `--no-reset-time`.

**Time Cross:** Εναλλακτικά, μπορεί να δοθεί `--time-cross global` ή `--time-cross per-signal`. Και στις δύο περιπτώσεις η ανίχνευση του πρώτου zero crossing γίνεται πάντα πάνω στο ίδιο σήμα που θα αναλυθεί, αφού πρώτα γίνει `detrend` και μετά `low-pass filtering`. Με το `global`, αν δεν δοθεί reference signal, υπολογίζεται ο πρώτος zero crossing για κάθε επιλεγμένο generator/signal και τελικά χρησιμοποιείται ένας κοινός χρόνος έναρξης για όλο το run, ίσος με το μέγιστο από αυτά τα first-cross times. Αν δοθεί `--time-cross-reference`, τότε το global mode χρησιμοποιεί αποκλειστικά τον zero crossing του συγκεκριμένου reference signal και τον εφαρμόζει ως κοινό start σε όλο το run. Με το `per-signal`, κάθε generator/signal κρατά τον δικό του πρώτο zero crossing και αναλύεται με διαφορετικό effective start.

**Time Cross Reference:** Το `--time-cross-reference` χρησιμοποιείται μόνο με `--time-cross global` και δέχεται μορφή όπως `g2:Current`, `g3:Voltage` ή `g2:s:cur1 in p.u.`. Με αυτόν τον τρόπο μπορείτε να δοκιμάσετε global κοινό start που προέρχεται από ένα modal-informative reference signal, αντί να αφήσετε το script να πάρει το πιο αργό first-cross από όλα τα επιλεγμένα signals.

**Time Start with Time Cross:** Όταν χρησιμοποιείται `--time-cross`, το `--time-start` δεν είναι απόλυτος χρόνος έναρξης, αλλά offset μετά τον ανιχνευμένο zero crossing. Για παράδειγμα, `python IEEE39/analyze_ieee39.py --scenario load03 --time-cross global` ξεκινά ακριβώς από τον κοινό πρώτο zero crossing, ενώ `python IEEE39/analyze_ieee39.py --scenario load03 --time-cross global --time-cross-reference g2:Current --time-start 0.2` ξεκινά `0.2s` μετά από τον zero crossing του `g2:Current` και χρησιμοποιεί αυτόν τον κοινό shifted χρόνο σε όλα τα signals. Αντίστοιχα, `python IEEE39/analyze_ieee39.py --scenario load03 --time-cross per-signal --time-start 0.1` ξεκινά κάθε signal `0.1s` μετά από τον δικό του πρώτο zero crossing.

**Time Cross Metadata:** Όταν χρησιμοποιείται `--time-cross`, το `analysis_config.json` αποθηκεύει τόσο το ζητούμενο mode (`global` ή `per-signal`) όσο και το resolved αποτέλεσμα: common zero-cross time για το global mode, resolved effective starts ανά signal, και το offset που εφαρμόστηκε από το `--time-start`.
