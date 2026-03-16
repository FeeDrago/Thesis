import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend
from matrix_pencil import apply_matrix_pencil_fixed_order, determine_MP_order, filter_signal

# Εξασφαλίζει ότι τα τυχαία νούμερα (θόρυβος) θα είναι τα ίδια κάθε φορά που τρέχεις το script
np.random.seed(0)

# --- 1. ΔΗΜΙΟΥΡΓΙΑ ΤΕΧΝΗΤΟΥ ΣΗΜΑΤΟΣ (GROUND TRUTH) ---
dt = 0.01  # Χρονικό βήμα δειγματοληψίας (100 Hz)
t_max = 30 # Συνολική διάρκεια σήματος σε δευτερόλεπτα
t = np.linspace(0, t_max, int(t_max/dt) + 1) # Δημιουργία χρονικού άξονα

# Παράμετροι δύο εκθετικά αποσβενόμενων ημιτόνων (αυτό που ψάχνουμε να βρούμε)
A1, A2 = 2.0, 2.0             # Πλάτη (Amplitudes)
sigma1, sigma2 = 0.1102, 0.1596 # Συντελεστές απόσβεσης (Damping)
freq1, freq2 = 0.25, 0.39     # Συχνότητες σε Hz
phi1, phi2 = 1.5*np.pi, 0.5*np.pi # Φάσεις

# Σύνθεση του καθαρού σήματος y
y = (
    A1 * np.exp(-sigma1*t) * np.cos(2*np.pi*freq1*t + phi1)
    + A2 * np.exp(-sigma2*t) * np.cos(2*np.pi*freq2*t + phi2)
)

# --- 2. ΠΡΟΣΘΗΚΗ ΘΟΡΥΒΟΥ ΚΑΙ ΠΡΟ-ΕΠΕΞΕΡΓΑΣΙΑ ---
noise_std = 0.1 # Ένταση του λευκού θορύβου (AWGN)
epsilon = noise_std * np.random.randn(len(t)) # Δημιουργία τυχαίων τιμών
y = y + epsilon # Προσθήκη του θορύβου στο καθαρό σήμα

# Αφαίρεση των πρώτων δειγμάτων (0.1s) για να αποφύγουμε αρχικές μεταβατικές καταστάσεις
mask = t > 0.1
t = t[mask].copy()
y = y[mask].copy()

t = t - t[0]  # Επαναφορά του χρόνου ώστε να ξεκινάει από το 0
y = detrend(y) # Αφαίρεση σταθερής τιμής (DC offset) ή γραμμικής τάσης

# Εφαρμογή χαμηλοπερατού φίλτρου (Low-pass) για την αφαίρεση του υψηλσυχνου θορύβου
fc = 10 # Συχνότητα αποκοπής στα 10 Hz
N = 15  # Τάξη του φίλτρου
y = filter_signal(y, t, fc, N)

# --- 3. ΕΦΑΡΜΟΓΗ ΑΛΓΟΡΙΘΜΟΥ MATRIX PENCIL ---
# Αυτόματη εύρεση της τάσης (πόσα modes υπάρχουν). 
# tau=0.01: ακρίβεια, rate=10: υποδειγματοληψία για ταχύτητα
MP_order = determine_MP_order(t, y, 0.01, 10) 

# Κύριος αλγόριθμος για τον υπολογισμό των παραμέτρων
freq, sigma, y_est, exec_time, poles_MP, a = apply_matrix_pencil_fixed_order(y, t, order=MP_order)

# Εκτύπωση αποτελεσμάτων
print("Matrix pencil order:", MP_order)             # Πόσα modes βρήκε
print("Estimated frequencies (Hz):", freq)          # Οι συχνότητες που αναγνώρισε
print("Estimated damping (sigma):", sigma)          # Η απόσβεση που αναγνώρισε
print("Execution time:", exec_time)                 # Χρόνος υπολογισμού

# --- 4. ΟΠΤΙΚΟΠΟΙΗΣΗ ---
plt.figure()
plt.plot(t, y, label="signal") # Το σήμα με τον θόρυβο και το φίλτρο
plt.plot(t, y_est, '--', label="matrix pencil estimate") # Η μαθηματική προσέγγιση
plt.legend()
plt.xlabel("time")
plt.ylabel("signal")
plt.title("Matrix Pencil Test")
plt.show()