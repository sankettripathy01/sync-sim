"""
PIQS / Dicke utilities.

Minimal, focused helpers to run PIQS simulations and to extract expectations
from stored density matrices.

Public API
----------
simulate_piqs_dynamics(...)
    Run a PIQS (Dicke) model evolution and return dict with states, Jx, Jz, res.
expectations_from_states(states, op)
    Compute expectation values of an operator for each stored density matrix.
"""


# compatibility import for PIQS (tries several likely locations)
try:
    # preferred: qutip >=5 style
    from qutip.piqs import Dicke, jspin, dicke
    _PIQS_SOURCE = "qutip.piqs"
except Exception:
    try:
        # some installs put the code under qutip.piqs.piqs
        from qutip.piqs.piqs import Dicke, jspin, dicke
        _PIQS_SOURCE = "qutip.piqs.piqs"
    except Exception:
        try:
            # older standalone package (if installed)
            from piqs import Dicke, jspin, dicke
            _PIQS_SOURCE = "piqs"
        except Exception as e:
            raise ImportError(
                "Unable to import PIQS ('Dicke','jspin','dicke').\n"
                "Detected qutip version: {}\n".format(getattr(__import__('qutip'), '__version__', 'unknown')) +
                "Tried import paths: 'qutip.piqs', 'qutip.piqs.piqs', 'piqs'.\n\n"
                "Quick fixes:\n"
                "  1) Ensure your notebook kernel uses the same venv where qutip is installed.\n"
                "  2) Reinstall qutip in that venv (pip install --force-reinstall qutip).\n"
                "  3) If you absolutely need the separate package, install PIQS from source.\n\n"
                f"Original import error: {e}"
            )


import numpy as np
import qutip as qt
#from qutip.piqs import Dicke, jspin, dicke
from typing import List, Dict, Any, Optional


def _prepare_dicke_system(N, T1, T2, gamma_c, w, v, dephasing_factor=1.0):
    """
    Internal helper: build Dicke system and collective operators.
    Returns (system, jx, jy, jz).
    """
    jx, jy, jz = jspin(N)
    system = Dicke(N=N)
    system.emission = 1.0 / T1
    system.dephasing = dephasing_factor * (1.0 / T2)
    system.pumping = float(w)
    system.collective_emission = float(gamma_c)
    system.hamiltonian = (v) * jz
    return system, jx, jy, jz


def simulate_piqs_dynamics(
    N: int,
    T1: float,
    T2: float,
    gamma_c: float,
    w: float,
    v: float,
    tlist: np.ndarray,
    rotation_sign: str = "minus",
    dephasing_factor: float = 1.0,
    return_states: bool = True,
    options: Optional[qt.Options] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Simulate the PIQS/Dicke model and return results needed for analysis.

    Parameters
    ----------
    N : int
        Number of two-level atoms (Dicke spin N/2).
    T1, T2 : float
        Relaxation and dephasing times.
    gamma_c : float
        Collective decay rate (absolute).
    w : float
        Pump rate.
    v : float
        Detuning / Hamiltonian coefficient.
    tlist : array_like
        Time grid for evolution.
    rotation_sign : {'minus','plus'}
        Initial collective Rx direction; 'minus' corresponds to Rx(-pi/2).
    dephasing_factor : float
        Multiplier applied to system.dephasing (flexibility for conventions).
    return_states : bool
        If True, request and return stored states (may use more memory).
    options : qutip.Options, optional
        Optional solver options; `store_states` will be forced True if return_states=True.
    verbose : bool
        If True, print a concise one-line summary.

    Returns
    -------
    result : dict
        {
            'tlist': np.array,
            'states': list[qutip.Qobj] (may be empty if solver did not store),
            'Jx': np.ndarray,   # normalized per spin: <Jx>/(N/2)
            'Jz': np.ndarray,   # normalized per spin: <Jz>/(N/2)
            'res': qutip.Result  # raw result object from mesolve
        }
    """
    # Build initial collective Dicke ground and rotate by collective Rx
    # Prepare system and collective operators
    system, jx, jy, jz = _prepare_dicke_system(N, T1, T2, gamma_c, w, v, dephasing_factor)

    # Build initial collective Dicke ground robustly from jz eigenstates
    # (ensures exact matching dimensions with jx/jz operators)
    evals, evecs = jz.eigenstates()
    idx_min = int(np.argmin([float(np.real(ev)) for ev in evals]))
    psi_g = evecs[idx_min]

    # Apply collective Rx rotation (choose sign as before)
    if rotation_sign == "minus":
        Rx = (-1j * (np.pi / 2.0) * jx).expm()
    else:
        Rx = (1j * (np.pi / 2.0) * jx).expm()

    psi0 = Rx * psi_g
    rho0 = psi0 * psi0.dag()

    # Ensure options is a qutip.Options with store_states as requested
    if options is None:
        options = qt.Options(store_states=bool(return_states))
    else:
        # try to set attribute if possible; otherwise rebuild a basic Options preserving common fields
        try:
            options.store_states = bool(return_states) or getattr(options, "store_states", False)
        except Exception:
            options = qt.Options(
                nsteps=getattr(options, "nsteps", None),
                atol=getattr(options, "atol", None),
                rtol=getattr(options, "rtol", None),
                store_states=bool(return_states),
            )

    # Liouvillian and evolution. Request Jx and Jz expectation values.
    L = system.liouvillian()
    res = qt.mesolve(L, rho0, tlist, [], [jx, jz], options=options)

    # Extract and sanitize stored states if available
    states_out: List[qt.Qobj] = []
    if hasattr(res, "states") and len(res.states) > 0:
        for s in res.states:
            # enforce Hermiticity and normalize trace to 1 (protect against tiny numerics)
            s_h = 0.5 * (s + s.dag())
            s_mat = s_h.full()
            s_mat = 0.5 * (s_mat + s_mat.conj().T)
            tr = np.trace(s_mat)
            if abs(tr) > 0:
                s_mat = s_mat / tr
            states_out.append(qt.Qobj(s_mat))

    # Expectations: normalize per spin (Jx and Jz are collective operators with eigenvalues ±N/2)
    if hasattr(res, "expect") and len(res.expect) >= 2:
        Jx = np.real(res.expect[0]) / (N / 2.0)
        Jz = np.real(res.expect[1]) / (N / 2.0)
        Jx = np.asarray(Jx, dtype=float)
        Jz = np.asarray(Jz, dtype=float)
    else:
        # Fallback: fill NaNs
        Jx = np.full(len(tlist), np.nan, dtype=float)
        Jz = np.full(len(tlist), np.nan, dtype=float)

    if verbose:
        print(f"PIQS: N={N}, len(tlist)={len(tlist)}, γc={gamma_c:.4g}, w={w:.4g}, v={v}, rot={rotation_sign}", flush=True)

    return {"tlist": np.asarray(tlist), "states": states_out, "Jx": Jx, "Jz": Jz, "res": res}


def expectations_from_states(states: List[qt.Qobj], op) -> np.ndarray:
    """
    Compute expectation values of operator `op` for each density matrix in `states`.

    Parameters
    ----------
    states : list of qutip.Qobj
        Density matrices (can be empty).
    op : qutip.Qobj or array-like
        Operator to measure.

    Returns
    -------
    np.ndarray
        Real-valued expectation array with length == len(states).
    """
    if len(states) == 0:
        return np.array([], dtype=float)

    if not isinstance(op, qt.Qobj):
        op = qt.Qobj(op)

    vals = np.array([np.real((s * op).tr()) for s in states], dtype=float)
    return vals
