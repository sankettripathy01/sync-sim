# piqs_utils/measurement_utils.py
"""
Measurement & analysis utilities.

Contains:
- qfi_from_rho_and_drho(rho, drho, tol=1e-8, pure_thresh=1-1e-8)
- qfi_from_state_triplet(states_minus, states_plus, states_center, E, ...)
- crude_lambda(t, y)
- extract_lambda(t, y)
- damped_sine(t, A, lam, omega, phi, offset)

Notes:
- These functions are agnostic to how the states / expectation traces were produced
  (PIQS, full-Hilbert, experiment, etc.). They expect numpy arrays / qutip.Qobj inputs.
"""

from typing import List
import numpy as np
import qutip as qt
from math import isfinite
from scipy.optimize import curve_fit


# ------------------- Quantum Fisher Information -------------------
def qfi_from_rho_and_drho(rho, drho, tol: float = 1e-8, pure_thresh: float = 1.0 - 1e-8) -> float:
    """
    Compute quantum Fisher information (QFI) for a parameter encoded in rho with derivative drho.

    Parameters
    ----------
    rho : qutip.Qobj or ndarray
        Density matrix at the parameter point.
    drho : qutip.Qobj or ndarray
        Derivative of density matrix w.r.t parameter (or finite-difference approximation).
    tol : float
        Denominator cutoff for the spectral formula to avoid dividing by near-zero (p_i + p_j).
    pure_thresh : float
        Threshold to treat the state as pure (largest eigenvalue > pure_thresh).

    Returns
    -------
    float
        Quantum Fisher Information (non-negative).
    """
    # Convert to numpy arrays
    if isinstance(rho, qt.Qobj):
        rho_mat = rho.full()
    else:
        rho_mat = np.asarray(rho, dtype=complex)

    if isinstance(drho, qt.Qobj):
        drho_mat = drho.full()
    else:
        drho_mat = np.asarray(drho, dtype=complex)

    # Symmetrize to remove small numerical asymmetries
    rho_mat = 0.5 * (rho_mat + rho_mat.conj().T)
    drho_mat = 0.5 * (drho_mat + drho_mat.conj().T)

    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(rho_mat)
    evals = np.real(evals)
    evals_clipped = np.clip(evals, 0.0, None)
    tr = np.sum(evals_clipped)
    if tr <= 0:
        return 0.0
    evals_clipped /= tr

    # If nearly pure, use simplified pure-state formula
    pmax = np.max(evals_clipped)
    if pmax > pure_thresh:
        i0 = int(np.argmax(evals_clipped))
        psi = evecs[:, i0]
        psi = psi / np.linalg.norm(psi)
        psi_prime = drho_mat @ psi
        psi_prime -= psi * np.vdot(psi, psi_prime)
        F_pure = 4.0 * np.real(np.vdot(psi_prime, psi_prime))
        if isfinite(F_pure):
            return float(max(F_pure, 0.0))

    # General mixed-state spectral formula
    V = evecs
    M = V.conj().T @ drho_mat @ V
    F = 0.0
    n = len(evals_clipped)
    for i in range(n):
        p_i = float(evals_clipped[i])
        for j in range(n):
            p_j = float(evals_clipped[j])
            denom = p_i + p_j
            if denom > tol:
                F += 2.0 * (abs(M[i, j]) ** 2) / denom

    if not isfinite(F):
        return 0.0
    return float(np.real_if_close(max(F, 0.0)))


def qfi_from_state_triplet(
    states_minus: List[qt.Qobj],
    states_plus: List[qt.Qobj],
    states_center: List[qt.Qobj],
    E: float,
    tol: float = 1e-6,
    pure_thresh: float = 1.0 - 1e-6,
) -> np.ndarray:
    """
    Compute QFI(t) given three lists of states evaluated at parameter values (v-E, v+E, v).

    Parameters
    ----------
    states_minus, states_plus, states_center : lists of qutip.Qobj
        Stored density matrices (should be sorted by time).
    E : float
        Finite-difference half-step so that drho ≈ (rho_plus - rho_minus) / (2E).
    tol, pure_thresh : floats
        Passed to qfi_from_rho_and_drho.

    Returns
    -------
    np.ndarray
        QFI at each time index (length = min(len(states_minus), len(states_plus), len(states_center))).
    """
    L = min(len(states_minus), len(states_plus), len(states_center))
    if L == 0:
        return np.array([], dtype=float)
    F_t = np.zeros(L, dtype=float)
    for i in range(L):
        rho_p = states_plus[i].full()
        rho_m = states_minus[i].full()
        drho_mat = (rho_p - rho_m) / (2.0 * E)
        rho_center_mat = 0.5 * (states_center[i].full() + states_center[i].full().conj().T)
        F_t[i] = qfi_from_rho_and_drho(rho_center_mat, drho_mat, tol=tol, pure_thresh=pure_thresh)
    return F_t


# ------------------- Decay Extraction -------------------
def damped_sine(t, A, lam, omega, phi, offset):
    """Model function: A * exp(-lam * t) * sin(omega * t + phi) + offset"""
    return A * np.exp(-lam * t) * np.sin(omega * t + phi) + offset


def crude_lambda(t, y):
    """
    Crude envelope-based estimate of decay rate:
    - Locate peaks of |y|
    - Fit log(amplitude) vs t (linear fit), slope = -lambda
    Returns lambda >= 0 or 0.0 if fit fails.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    if len(y) < 5:
        return 0.0

    dy = np.diff(y)
    peaks = np.where((np.hstack([dy, 0]) < 0) & (np.hstack([0, dy]) > 0))[0]
    if len(peaks) < 3:
        return 0.0

    ts = t[peaks]
    amps = np.abs(y[peaks])
    mask = amps > 1e-8
    ts = ts[mask]
    amps = amps[mask]
    if len(ts) < 3:
        return 0.0

    logA = np.log(amps)
    p = np.polyfit(ts, logA, 1)
    lam = -p[0]
    return float(max(lam, 0.0))


def extract_lambda(t, y):
    """
    Fit a damped sine model to y(t) and extract the decay rate lambda.

    If fitting fails or returns a negative value, falls back to crude_lambda().
    """
    t = np.asarray(t)
    y = np.asarray(y)
    if len(y) < 5:
        return 0.0

    A0 = (np.max(y) - np.min(y)) / 2.0
    lam0 = 0.5
    duration = max(t[-1] - t[0], 1.0)
    omega0 = 2 * np.pi / (duration / 2.0)
    phi0 = 0.0
    offset0 = np.mean(y)

    try:
        popt, _ = curve_fit(
            damped_sine, t, y,
            p0=[A0, lam0, omega0, phi0, offset0],
            bounds=([0, 0, 0, -10, -np.inf], [np.inf, np.inf, np.inf, 10, np.inf]),
            maxfev=20000,
        )
        lam = popt[1]
        if (not isfinite(lam)) or lam < 0:
            return crude_lambda(t, y)
        return float(lam)
    except Exception:
        return crude_lambda(t, y)
