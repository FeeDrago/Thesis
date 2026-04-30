import numpy as np
import time

from scipy.signal import firwin, filtfilt


def filter_signal(noisy, t, fc, N=15):
    dt = t[1] - t[0]
    fs = 1 / dt
    fnorm = fc * 2 / fs
    b = firwin(N + 1, fnorm, window=("chebwin", 50))
    y_filt = filtfilt(b, 1, noisy)
    return y_filt


def determine_MP_order(t, y, tau, rate=1, max_order=50):
    t_decimated = t
    y_decimated = y

    if rate > 1:
        t_decimated = t[::rate]
        y_decimated = y[::rate]

    prev_rsq = float("-inf")
    cond = True
    order = 0

    while cond:
        order = order + 1
        if order > max_order:
            break

        _, _, y_est, _, _, _ = apply_matrix_pencil_fixed_order(y_decimated, t_decimated, order)

        y_decimated_arr = np.asarray(y_decimated)
        y_est_arr = np.asarray(y_est)

        ss_res = np.sum((y_decimated_arr - y_est_arr) ** 2)
        ss_tot = np.sum((y_decimated_arr - np.mean(y_decimated_arr)) ** 2)
        rsq = (1 - ss_res / ss_tot) * 100

        cond = abs(rsq - prev_rsq) > tau
        prev_rsq = rsq

    order = max(1, order - 1)
    return order


def apply_matrix_pencil_fixed_order(y, t, order):
    y = np.asarray(y).reshape(-1)
    t = np.asarray(t).reshape(-1)

    n = len(y)
    l = int(np.ceil(0.5 * (np.ceil(n / 3) + np.floor(n / 2))))

    y_col = y.reshape(-1, 1)
    x_col = t.reshape(-1, 1)

    start = time.perf_counter()

    n = y_col.shape[0]
    if x_col.shape[0] != n:
        raise ValueError("length(Y) should be length(X)")

    if y_col.shape[1] != 1:
        raise ValueError("Y should be column vector")

    if x_col.shape[1] != 1:
        raise ValueError("X should be column vector")

    tol = order
    m_given = np.round(tol) == tol
    if (tol < 0) or ((not m_given) and (tol > 1)):
        raise ValueError("TOL should be either >= 1 and an integer or < 1 and > 0")

    if l > n / 2:
        raise ValueError("L shoud be < N/2")

    if m_given and (l < tol):
        raise ValueError("TOL should be <= L")

    sample_period = np.diff(x_col[:2, 0])[0]

    y_matrix = np.zeros((n - l, l + 1), dtype=np.complex128)
    ind = np.arange(0, n - l)
    for j in range(l + 1):
        y_matrix[:, j] = y_col[ind + j, 0]

    u, s, vh = np.linalg.svd(y_matrix, full_matrices=False)
    v = vh.conj().T
    s_matrix = np.diag(s)

    if m_given:
        m = int(tol)
    else:
        d = np.diag(s_matrix)
        m = len(d)
        for k in range(len(d) - 1):
            m = k + 1
            if abs(d[k + 1] / d[0]) <= tol:
                break

    sm = s_matrix[:, :m]
    vm = v[:, :m]
    v1 = vm[:l, :]
    v2 = vm[1:l + 1, :]

    y1 = u @ sm @ v1.conj().T
    y2 = u @ sm @ v2.conj().T

    a_matrix = np.linalg.pinv(y1) @ y2
    z = np.linalg.eigvals(a_matrix)
    z = z[:m]
    poles_mp = (1 / sample_period) * np.log(z)

    z_matrix = np.exp(x_col @ poles_mp.reshape(1, -1))
    amplitudes, _, _, _ = np.linalg.lstsq(z_matrix, y_col, rcond=None)
    y_est = np.real((z_matrix @ amplitudes).reshape(-1))

    elapsed_time = time.perf_counter() - start

    freq = np.imag(poles_mp / (2 * np.pi))
    sigma = np.real(poles_mp)

    return freq, sigma, y_est, elapsed_time, poles_mp, amplitudes.reshape(-1)
