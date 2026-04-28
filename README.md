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

**Χρόνος Προσομοίωσης και Load Event:** Από προεπιλογή η προσομοίωση τρέχει μέχρι `50s` και το load event τοποθετείται στο `t=0`, αλλά πλέον μπορούν να αλλάξουν από command line με `--duration` και `--event-time`. Για παράδειγμα, `python IEEE39/generate_data.py --scenario load03 --duration 60 --event-time 0.5` δημιουργεί τα δεδομένα με stop time `60s` και event στο `0.5s`. Τα quoted inline specs ή `--case` specs μπορούν επίσης να ορίσουν δικό τους duration και event time, π.χ. `python IEEE39/generate_data.py --case "Load 24:2:0:60:0.5"`. Όταν το event time είναι διαφορετικό από το default, προστίθεται suffix τύπου `_evt0.5s` στο όνομα του scenario folder, ώστε να ξεχωρίζουν τα runs.

**Φάκελος Αποτελεσμάτων:** Από προεπιλογή τα αποτελέσματα γράφονται στο `IEEE39/results`. Αν χρειαστεί διαφορετικός φάκελος, μπορεί να δοθεί `--output-dir`, για παράδειγμα `python IEEE39/generate_data.py --scenario load29 --output-dir results_test`.

**Ανάλυση Δεδομένων:** Το αρχείο `IEEE39/analyze_ieee39.py` διαβάζει τα `g*.csv` από το `IEEE39/results` και γράφει τα αποτελέσματα στο `IEEE39/analysis`. Η εκτέλεση γίνεται με `python IEEE39/analyze_ieee39.py --scenario load29`, `python IEEE39/analyze_ieee39.py --scenario load03 load24`, ή `python IEEE39/analyze_ieee39.py --scenario all`. Τα default keys όπως `load29` δείχνουν στα προκαθορισμένα σενάρια `Pplus2`, άρα το `load29` αντιστοιχεί στο input `IEEE39/results/Load29_Pplus2_50s`. Για διαφορετικά paths μπορεί να δοθεί ρητά input και output, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario custom --data-dir results/Load29_Pplus4_50s --output-dir analysis/Load29_Pplus4_50s`.

**Ονόματα Analysis Folders:** Αν δεν δοθεί ρητά `--output-dir`, το output folder περιλαμβάνει και το time mask. Για παράδειγμα, το default `load29` γράφει σε φάκελο τύπου `IEEE39/analysis/Load29_Pplus2_50s_0.2_to_end_reset`. Αν δοθεί `--time-end 20`, γράφει σε `..._0.2_to_20_reset`. Αν δοθεί `--no-reset-time`, το suffix τελειώνει σε `noreset`. Όταν χρησιμοποιούνται subsets με `--generators` ή `--signals`, προστίθεται και αντίστοιχο suffix στο folder name. Το `analysis_config.json` της ανάλυσης περιέχει το time mask, το πραγματικό time window, τα `signal_means` και τα subsets που χρησιμοποιήθηκαν, ώστε το `--skip-matrix-pencil` να χρησιμοποιεί το ίδιο setup με αυτό που δημιούργησε τους πόλους.

**Επιλογές Ανάλυσης:** Αν χρειάζεται μόνο η εξαγωγή Matrix Pencil χωρίς clustering, δίνεται `--skip-clustering`, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario load29 --skip-clustering`. Αν δεν χρειάζονται διαγράμματα, δίνεται `--skip-plots`. Αν θέλετε γρήγορα δοκιμαστικά runs μόνο για συγκεκριμένες γεννήτριες ή σήματα, μπορούν να χρησιμοποιηθούν τα `--generators` και `--signals`, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario load03 --time-start 1.2 --generators g2 g3 g6 --signals Voltage "Reactive Power" --skip-clustering --skip-plots`. Αν οι πόλοι έχουν ήδη υπολογιστεί και υπάρχει `results.csv`, μπορεί να παρακαμφθεί ο Matrix Pencil με `--skip-matrix-pencil`. Η εντολή `python IEEE39/analyze_ieee39.py --scenario load29 --skip-matrix-pencil` επιλέγει το default analysis folder με το default time mask, δηλαδή `IEEE39/analysis/Load29_Pplus2_50s_0.2_to_end_reset`. Για άλλο ήδη υπάρχον analysis folder, πρώτα εμφανίζονται οι επιλογές με `python IEEE39/analyze_ieee39.py --list-analysis` και μετά δίνεται ρητά φάκελος, για παράδειγμα `python IEEE39/analyze_ieee39.py --scenario load29 --skip-matrix-pencil --analysis-dir analysis/Load29_Pplus2_50s_0.2_to_20_reset`. Εναλλακτικά μπορεί να δοθεί συγκεκριμένο αρχείο με `--results-file`.

**Clustering:** Από προεπιλογή το clustering γίνεται και συνολικά για όλο το σύστημα και ανά περιοχή ελέγχου. Η επιλογή γίνεται με `--clustering-scope both`, `--clustering-scope global`, `--clustering-scope areas`, ή `--clustering-scope none`. Για παράδειγμα, `python IEEE39/analyze_ieee39.py --scenario load29 --skip-matrix-pencil --clustering-scope areas` χρησιμοποιεί υπάρχον `results.csv` και παράγει clustering μόνο ανά περιοχή ελέγχου.

**Στατιστικά & Διαγράμματα:** Το `comprehensive_report.csv` παράγεται πάντα στο `IEEE39/analysis/<scenario>/stats/comprehensive_report.csv`, ακόμη και όταν χρησιμοποιείται `--skip-matrix-pencil` με ήδη υπάρχον `results.csv`. Τα modal maps και τα reconstructions του IEEE39 δημιουργούνται μέσα στο output folder της ανάλυσης. Για το `load29`, τα modal maps βρίσκονται στο `IEEE39/analysis/Load29_Pplus2_50s_0.2_to_end_reset/plots/modal_maps` και τα reconstruction grids στο `IEEE39/analysis/Load29_Pplus2_50s_0.2_to_end_reset/plots/reconstruction_grids`. Αν χρησιμοποιούνται subsets με `--generators` ή `--signals`, το `comprehensive_report.csv` περιέχει μόνο το subset που ζητήθηκε.

**Χρόνος Σημάτων:** Από προεπιλογή η ανάλυση δεν διατηρεί τα samples πριν από `0.2s` και κρατά όλα τα υπόλοιπα μέχρι την τελευταία χρονική μέτρηση του CSV. Μετά μετατοπίζει τον χρόνο του επιλεγμένου παραθύρου ώστε το πρώτο κρατημένο sample να γίνει `t=0`.

**Time Mask:** Αν χρειάζεται διαφορετικό παράθυρο, δίνεται `--time-start` ή/και `--time-end`. Για παράδειγμα, `python IEEE39/analyze_ieee39.py --scenario load29 --time-start 0.2 --time-end 20` κρατά μόνο τις γραμμές από `0.2s` έως `20s`. Αν πρέπει να διατηρηθούν οι αρχικοί χρόνοι του CSV χωρίς μετατόπιση, δίνεται `--no-reset-time`.
