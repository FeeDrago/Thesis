import numpy as np
import time

from scipy.signal import firwin, filtfilt
from scipy.signal.windows import chebwin


def prepare_matrix_pencil(y, t):
    y = np.asarray(y).reshape(-1)
    t = np.asarray(t).reshape(-1)

    N = len(y)
    # L controls the Hankel/window size used to build the pencil.
    L = int(np.ceil(0.5 * (np.ceil(N / 3) + np.floor(N / 2))))

    y_col = y.reshape(-1, 1)
    x_col = t.reshape(-1, 1)

    if x_col.shape[0] != y_col.shape[0]:
        raise ValueError('length(Y) should be length(X)')

    T = np.diff(x_col[:2, 0])[0]
    # Y is the sampled data matrix whose dominant singular subspace carries the modes.
    Y = np.lib.stride_tricks.sliding_window_view(y, L + 1).astype(np.complex128, copy=False)

    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh.conj().T

    return {
        "y": y,
        "t": t,
        "y_col": y_col,
        "x_col": x_col,
        "N": N,
        "L": L,
        "T": T,
        "U": U,
        "singular_values": s,
        "V": V,
    }


def _resolve_model_order(singular_values, order):
    tol = order
    M_given = np.round(tol) == tol

    if M_given:
        return int(tol)

    D = np.asarray(singular_values).reshape(-1)
    M = len(D)
    for k in range(len(D) - 1):
        M = k + 1
        if abs(D[k + 1] / D[0]) <= tol:
            break
    return M


def apply_matrix_pencil_fixed_order_prepared(prepared, order, fit_cache=None):
    if fit_cache is not None and order in fit_cache:
        freq, sigma, y_est, _, poles_MP, amplitudes = fit_cache[order]
        return freq, sigma, y_est, 0.0, poles_MP, amplitudes

    start = time.perf_counter()

    y_col = prepared["y_col"]
    x_col = prepared["x_col"]
    U = prepared["U"]
    singular_values = prepared["singular_values"]
    V = prepared["V"]
    L = prepared["L"]
    T = prepared["T"]

    M = _resolve_model_order(singular_values, order)

    # M is the reduced model order, so the pencil must also be solved at size M.
    max_rank = min(U.shape[1], V.shape[1])
    if M < 1 or M > max_rank:
        raise ValueError(f"Model order M must be between 1 and {max_rank}, got {M}")
    if M > L:
        raise ValueError(f"Model order M must not exceed pencil parameter L={L}, got {M}")

    VM = V[:, :M]
    V1 = VM[:L, :]
    V2 = VM[1:L + 1, :]

    # Solve the reduced pencil directly in the M-dimensional signal subspace.
    # This gives exactly M poles instead of solving a larger L x L problem first.
    A = np.linalg.pinv(V1) @ V2
    z = np.linalg.eigvals(A)

    # Guard the log against poles that are numerically too close to zero.
    eps = np.finfo(float).tiny
    z = np.where(np.abs(z) < eps, eps + 0j, z)

    # Convert discrete-time poles z into continuous-time poles p using z = exp(pT).
    poles_MP = (1 / T) * np.log(z)

    # Fit the modal amplitudes after the poles are known.
    Z = np.exp(x_col @ poles_MP.reshape(1, -1))
    a, _, _, _ = np.linalg.lstsq(Z, y_col, rcond=None)
    a = a.reshape(-1)

    # Order the modes by energy so reports stay stable across environments.
    omega = np.imag(poles_MP)
    modal_energy = 0.5 * (omega ** 2) * (np.abs(a) ** 2)
    mode_order = np.argsort(-modal_energy, kind="stable")

    poles_MP = poles_MP[mode_order]
    a = a[mode_order]

    # Rebuild the estimate with the sorted poles and amplitudes.
    Z = np.exp(x_col @ poles_MP.reshape(1, -1))
    y_est = np.real((Z @ a.reshape(-1, 1)).reshape(-1))

    elapsed_time = time.perf_counter() - start

    # The imaginary part gives oscillation frequency, the real part gives damping.
    freq = np.imag(poles_MP / (2 * np.pi))
    sigma = np.real(poles_MP)

    result = (freq, sigma, y_est, elapsed_time, poles_MP, a)
    if fit_cache is not None:
        fit_cache[order] = result
    return result

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


def _prepare_decimated_signal(t, y, rate):
    if rate > 1:
        return t[::rate], y[::rate]
    return t, y


def _compute_rsq_curve(prepared, y_reference, max_order, fit_cache=None):
    y_reference_arr = np.asarray(y_reference)
    ss_tot = np.sum((y_reference_arr - np.mean(y_reference_arr)) ** 2)
    rsq_curve = []

    for order in range(1, max_order + 1):
        _, _, y_est, _, _, _ = apply_matrix_pencil_fixed_order_prepared(prepared, order, fit_cache=fit_cache)
        y_est_arr = np.asarray(y_est)
        ss_res = np.sum((y_reference_arr - y_est_arr) ** 2)
        rsq_curve.append((order, (1 - ss_res / ss_tot) * 100))

    return rsq_curve


def _select_order_from_rsq_curve(rsq_curve, tau):
    prev_rsq = float('-inf')
    for order, rsq in rsq_curve:
        if abs(rsq - prev_rsq) <= tau:
            return max(1, order - 1)
        prev_rsq = rsq

    return max(1, rsq_curve[-1][0]) if rsq_curve else 1


def determine_MP_orders(t, y, taus, rate=1, max_order=50, return_details=False, fit_cache=None):
    start = time.perf_counter()

    t_decimated, y_decimated = _prepare_decimated_signal(t, y, rate)
    prepared = prepare_matrix_pencil(y_decimated, t_decimated)
    rsq_curve = _compute_rsq_curve(prepared, y_decimated, max_order, fit_cache=fit_cache)

    tau_values = list(taus)
    orders = {tau: _select_order_from_rsq_curve(rsq_curve, tau) for tau in tau_values}

    if not return_details:
        return orders

    details = {
        "elapsed_time": time.perf_counter() - start,
        "orders_tested": len(rsq_curve),
    }
    return orders, details

def determine_MP_order(t, y, tau, rate=1, max_order=50):
    """
    Βρίσκει αυτόματα την ιδανική τάξη (αριθμό modes) του συστήματος.
    Αυξάνει την τάξη μέχρι η βελτίωση του R-squared να είναι μικρότερη από tau.
    """
    orders = determine_MP_orders(t, y, [tau], rate=rate, max_order=max_order)
    return orders[tau]

def apply_matrix_pencil_fixed_order(y, t, order):
    """
    Η κύρια υλοποίηση της μεθόδου Matrix Pencil (Sarkar & Pereira).
    Αναλύει το σήμα σε ένα άθροισμα μιγαδικών εκθετικών όρων.
    """
    prepared = prepare_matrix_pencil(y, t)
    return apply_matrix_pencil_fixed_order_prepared(prepared, order)
