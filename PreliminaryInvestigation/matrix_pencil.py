import numpy as np
import time

from scipy.signal import firwin, filtfilt
from scipy.signal.windows import chebwin

def filter_signal(noisy, t, fc, N=15):
    """
    Εφαρμόζει ένα ψηφιακό φίλτρο χαμηλής διέλευσης (Low-pass) τύπου FIR 
    με παράθυρο Chebyshev για την αφαίρεση του υψίσυχνου θορύβου.
    """
    dt = t[1] - t[0] # Υπολογισμός χρονικού βήματος
    fs = 1 / dt      # Συχνότητα δειγματοληψίας
    
    Fnorm = fc * 2 / fs # Κανονικοποιημένη συχνότητα (Nyquist)
    
    # Σχεδίαση συντελεστών φίλτρου με παράθυρο Chebyshev (50dB εξασθένηση)
    b = firwin(N+1, Fnorm, window=('chebwin', 50)) 
    
    # Εφαρμογή φιλτραρίσματος zero-phase (δεν μετατοπίζει το σήμα χρονικά)
    y_filt = filtfilt(b, 1, noisy) 
    
    return y_filt

def determine_MP_order(t, y, tau, rate=1, max_order=50):
    """
    Βρίσκει αυτόματα την ιδανική τάξη (αριθμό modes) του συστήματος.
    Αυξάνει την τάξη μέχρι η βελτίωση του R-squared να είναι μικρότερη από tau.
    """
    t_decimated = t
    y_decimated = y

    # Αν το rate > 1, κάνει υποδειγματοληψία (decimation) για να τρέξει πιο γρήγορα
    if rate > 1:
        t_decimated = t[::rate]
        y_decimated = y[::rate]

    prevRsq = float('-inf')
    cond = True
    order = 0

    # Επαναληπτική διαδικασία εύρεσης τάξης
    while cond:
        order = order + 1

        if order > max_order: # Ασφάλεια για να μην τρέχει επ' άπειρον
            break

        # Δοκιμαστική εκτέλεση του Matrix Pencil για την τρέχουσα τάξη
        _, _, y_est, _, _, _ = apply_matrix_pencil_fixed_order(
            y_decimated, t_decimated, order
        )

        y_decimated_arr = np.asarray(y_decimated)
        y_est_arr = np.asarray(y_est)

        # Υπολογισμός του συντελεστή προσδιορισμού R^2 (πόσο καλά ταιριάζει το μοντέλο)
        ss_res = np.sum((y_decimated_arr - y_est_arr) ** 2) # Άθροισμα τετραγώνων σφαλμάτων
        ss_tot = np.sum((y_decimated_arr - np.mean(y_decimated_arr)) ** 2)
        Rsq = (1 - ss_res / ss_tot ) * 100

        # Αν η βελτίωση του R^2 είναι μικρότερη από tau, σταματάμε
        cond = abs(Rsq - prevRsq) > tau
        prevRsq = Rsq

    order = max(1, order - 1) # Επιστρέφουμε στην τελευταία σταθερή τάξη
    return order

def apply_matrix_pencil_fixed_order(y, t, order):
    """
    Η κύρια υλοποίηση της μεθόδου Matrix Pencil (Sarkar & Pereira).
    Αναλύει το σήμα σε ένα άθροισμα μιγαδικών εκθετικών όρων.
    """
    y = np.asarray(y).reshape(-1)
    t = np.asarray(t).reshape(-1)

    N = len(y)
    # Επιλογή της παραμέτρου L (Pencil Parameter). Συνήθως μεταξύ N/3 και N/2
    L = int(np.ceil(0.5 * (np.ceil(N / 3) + np.floor(N / 2))))

    y_col = y.reshape(-1, 1)
    x_col = t.reshape(-1, 1)

    start = time.perf_counter()

    # --- Έλεγχοι διαστάσεων και παραμέτρων ---
    N = y_col.shape[0]
    if x_col.shape[0] != N: raise ValueError('length(Y) should be length(X)')
    
    tol = order
    M_given = np.round(tol) == tol # Έλεγχος αν η τάξη είναι ακέραιος

    # --- Κατασκευή του Hankel Matrix (Y) ---
    T = np.diff(x_col[:2, 0])[0] # Χρονική απόσταση δειγμάτων
    Y = np.zeros((N - L, L + 1), dtype=np.complex128)
    ind = np.arange(0, N - L)
    for j in range(L + 1):
        Y[:, j] = y_col[ind + j, 0]

    # --- Singular Value Decomposition (SVD) --- 
    # Χρησιμοποιείται για να "καθαρίσει" τον θόρυβο και να βρει τους κυρίαρχους χώρους
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh.conj().T
    S = np.diag(s)

    # Καθορισμός αριθμού εκθετικών όρων (M)
    if M_given:
        M = int(tol)
    else:
        # Αν το order < 1, λειτουργεί ως κατώφλι (threshold) επί των ιδιοτιμών
        D = np.diag(S)
        M = len(D)
        for k in range(len(D) - 1):
            M = k + 1
            if abs(D[k + 1] / D[0]) <= tol:
                break

    # --- Υπολογισμός των Πόλων (z) --- 
    # Χωρίζουμε τους πίνακες για να δημιουργήσουμε το "Pencil"
    SM = S[:, :M]
    VM = V[:, :M]
    V1 = VM[:L, :]   # Ο πίνακας χωρίς την τελευταία σειρά
    V2 = VM[1:L + 1, :] # Ο πίνακας χωρίς την πρώτη σειρά

    Y1 = U @ SM @ V1.conj().T
    Y2 = U @ SM @ V2.conj().T

    # Επίλυση του προβλήματος γενικευμένων ιδιοτιμών
    A = np.linalg.pinv(Y1) @ Y2
    z = np.linalg.eigvals(A)
    z = z[:M]
    
    # Μετατροπή από το επίπεδο z στο επίπεδο s (συνεχής χρόνος)
    poles_MP = (1 / T) * np.log(z)

    # --- Υπολογισμός Πλατών (Amplitudes) ---
    # Επίλυση ελαχίστων τετραγώνων για να βρούμε τα βάρη κάθε mode
    Z = np.exp(x_col @ poles_MP.reshape(1, -1))
    a, _, _, _ = np.linalg.lstsq(Z, y_col, rcond=None)
    y_est = np.real((Z @ a).reshape(-1)) # Η τελική ανακατασκευή του σήματος

    elapsed_time = time.perf_counter() - start

    # Εξαγωγή συχνότητας (Hz) και απόσβεσης (sigma)
    freq = np.imag(poles_MP / (2 * np.pi)) # Φανταστικό μέρος
    sigma = np.real(poles_MP)             # Πραγματικό μέρος

    return freq, sigma, y_est, elapsed_time, poles_MP, a.reshape(-1)