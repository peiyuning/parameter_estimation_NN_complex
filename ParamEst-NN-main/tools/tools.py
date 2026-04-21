import os
import math
from functools import lru_cache

import numpy as np
import qutip as qt
import torch
from scipy.integrate import cumulative_trapezoid, trapezoid as scipy_trapezoid
from scipy.linalg import expm


# =============================================================================
# basic helpers
# =============================================================================

def create_directory(filepath: str):
    if not os.path.exists(filepath):
        print(f"Directory {filepath} not found: creating...")
        os.makedirs(filepath)
    else:
        print("Folder already exists.")


def trapz_integral(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return scipy_trapezoid(y, x, axis=axis)


def to_time_delay(array: np.ndarray):
    array = np.asarray(array, dtype=float).reshape(-1)
    if array.size == 0:
        return np.asarray([], dtype=float)
    return np.concatenate(([array[0]], np.diff(array)))


def to_time_delay_matrix(matrix: np.ndarray):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if matrix.shape[1] == 0:
        return np.zeros_like(matrix, dtype=float)
    first_column = matrix[:, [0]]
    return np.concatenate((first_column, np.diff(matrix, axis=1)), axis=1)


def _trace_row(dim: int):
    ident = np.eye(dim, dtype=np.complex128)
    return ident.reshape(1, dim * dim, order="F")


def _vectorized_density_from_ket(psi: qt.Qobj):
    return qt.operator_to_vector(qt.ket2dm(psi)).full()


def _ensure_ket(psi0, dims):
    if not isinstance(psi0, qt.Qobj):
        psi0 = qt.Qobj(psi0, dims=dims)

    if psi0.isoper:
        raise TypeError("psi0 must be a ket, not a density operator.")
    if not psi0.isket:
        raise TypeError(f"psi0 must be a ket with shape (N,1). Got shape={psi0.shape}.")
    return psi0.unit()


def _traj_state_lists(statestraj):
    if statestraj is None or len(statestraj) == 0:
        raise RuntimeError("mcsolve returned empty states.")
    if isinstance(statestraj[0], qt.Qobj):
        return [statestraj]
    return statestraj


def _final_ket_from_solution(sol, dims):
    statestraj = _traj_state_lists(getattr(sol, "states", None))
    traj_states = statestraj[0]
    psi = traj_states[-1]
    if not isinstance(psi, qt.Qobj):
        psi = qt.Qobj(psi, dims=dims)

    if psi.isoper:
        evals, evecs = psi.eigenstates()
        evals = np.real(np.asarray(evals))
        idx = int(np.argmax(evals))
        if np.isclose(evals[idx], 1.0, atol=1e-8):
            psi = evecs[idx]
        else:
            raise TypeError("mcsolve returned a mixed density matrix; expected pure-state trajectory.")
    if not psi.isket:
        raise TypeError(f"expected ket after conversion, got shape={psi.shape}")
    return psi.unit()


def _extract_jump_record(sol, traj_idx=0):
    col_times = getattr(sol, "col_times", None)
    col_which = getattr(sol, "col_which", None)

    if col_times is None:
        raise RuntimeError("mcsolve solution has no col_times.")

    times = np.asarray(col_times[traj_idx], dtype=float).reshape(-1)
    if col_which is None:
        which = np.zeros(times.shape, dtype=int)
    else:
        which = np.asarray(col_which[traj_idx], dtype=int).reshape(-1)

    if times.shape != which.shape:
        raise RuntimeError("col_times and col_which shapes do not match.")
    return times, which


def _filter_detected_times(times, which, detected_indices):
    detected_indices = tuple(int(k) for k in detected_indices)
    mask = np.isin(which, np.asarray(detected_indices, dtype=int))
    return np.asarray(times[mask], dtype=float), np.asarray(which[mask], dtype=int)


def _resolve_channel_selection(channel_names, selection):
    n = len(channel_names)
    if selection is None:
        return tuple(range(n))

    out = []
    for item in selection:
        if isinstance(item, str):
            if item not in channel_names:
                raise ValueError(f"unknown channel name {item!r}. available: {channel_names}")
            out.append(channel_names.index(item))
        else:
            idx = int(item)
            if not (0 <= idx < n):
                raise ValueError(f"channel index {idx} out of range 0..{n-1}")
            out.append(idx)

    out = tuple(dict.fromkeys(out))
    if len(out) == 0:
        raise ValueError("at least one channel must be selected.")
    return out


# =============================================================================
# Quantum model for Monte Carlo generation and single-symbol likelihood recursion
# =============================================================================

class QuantumModel:
    """
    Generic finite-dimensional Markovian open quantum model.

    Important:
    - For trajectory generation, detected_indices controls which collapse channels
      are kept in the observed click record.
    - computeLikelihood(data) is the exact recursive likelihood for a *single observed
      symbol* generated by J = sum_{k in detected_indices} J_k.
      This is the right object for TLS and for any one-symbol observed process.
      For genuinely multi-symbol non-iid models, use the kernel functions below.
    """

    def __init__(
        self,
        H,
        c_ops,
        psi0,
        J=None,
        detected_indices=None,
        with_eigvals=True,
        channel_names=None,
    ):
        self.H = H
        self.c_ops = list(c_ops)
        self.psi0 = _ensure_ket(psi0, dims=H.dims[0])
        self.dim = self.psi0.shape[0]
        self.rho0 = _vectorized_density_from_ket(self.psi0)
        self.trace_row = _trace_row(self.dim)

        if channel_names is None:
            channel_names = [f"ch{k}" for k in range(len(self.c_ops))]
        if len(channel_names) != len(self.c_ops):
            raise ValueError("channel_names must match c_ops.")
        self.channel_names = list(channel_names)

        self.detected_indices = (
            tuple(range(len(self.c_ops)))
            if detected_indices is None
            else tuple(int(k) for k in detected_indices)
        )
        if len(self.detected_indices) == 0:
            raise ValueError("detected_indices must be non-empty.")

        self.L = qt.liouvillian(H, self.c_ops)

        if J is None:
            J_super = None
            for k in self.detected_indices:
                term = qt.sprepost(self.c_ops[k], self.c_ops[k].dag())
                J_super = term if J_super is None else (J_super + term)
            self.J_super = J_super
        else:
            if isinstance(J, qt.Qobj):
                self.J_super = J
            else:
                self.J_super = qt.Qobj(J, dims=self.L.dims)

        self.Lhat = self.L - self.J_super
        self.J = self.J_super.full()
        self.Lhat_np = self.Lhat.full()

        if with_eigvals:
            eigs, E = np.linalg.eig(self.Lhat_np)
            Einv = np.linalg.inv(E)
            self.eigs_hat = np.asarray(eigs, dtype=np.complex128)
            self.Ehat = np.asarray(E, dtype=np.complex128)
            self.Einvhat = np.asarray(Einv, dtype=np.complex128)
        else:
            self.eigs_hat = None
            self.Ehat = None
            self.Einvhat = None

    def _propagator(self, tau: float, method="spectral"):
        tau = float(tau)
        if method == "spectral":
            if self.eigs_hat is None:
                raise RuntimeError("spectral decomposition not precomputed.")
            D = np.exp(self.eigs_hat * tau)
            return self.Ehat @ (D[:, None] * self.Einvhat)
        elif method == "direct":
            return expm(tau * self.Lhat_np)
        else:
            raise ValueError("method must be 'spectral' or 'direct'.")

    def _estimate_detected_rate(self):
        rho_ss = qt.steadystate(self.H, self.c_ops)
        rate = 0.0
        for k in self.detected_indices:
            ck = self.c_ops[k]
            rate += float(np.real(qt.expect(ck.dag() * ck, rho_ss)))
        if rate > 1e-12:
            return rate

        total = 0.0
        for ck in self.c_ops:
            total += float(np.real(qt.expect(ck.dag() * ck, rho_ss)))
        return max(total, 1.0)

    def simulateTrajectories(self, tfin: float, ntraj: int, tlistexp=None, seed=None):
        H = self.H
        c_ops = self.c_ops
        psi0 = self.psi0

        mc_kwargs = {}
        if seed is not None:
            np.random.seed(int(seed))
            mc_kwargs["seeds"] = int(seed)

        tlist = [0.0, float(tfin)] if tlistexp is None else list(tlistexp)

        sol = qt.mcsolve(
            H,
            psi0,
            tlist,
            c_ops,
            e_ops=[],
            ntraj=ntraj,
            **mc_kwargs,
        )

        taus = []
        for traj_idx in range(ntraj):
            times, which = _extract_jump_record(sol, traj_idx=traj_idx)
            det_times, _ = _filter_detected_times(times, which, self.detected_indices)
            taus.append(to_time_delay(det_times))

        if tlistexp is None:
            return taus

        statestraj = _traj_state_lists(sol.states)
        first_det = self.detected_indices[0]
        op = self.c_ops[first_det].dag() * self.c_ops[first_det]
        pop_array = np.asarray([qt.expect(op, traj) for traj in statestraj], dtype=float)
        return taus, pop_array

    def simulateTrajectoryFixedTime(self, tfin: float, seed=None, return_channels=False):
        if tfin <= 0:
            raise ValueError(f"tfin must be > 0, got {tfin}")

        mc_kwargs = {}
        if seed is not None:
            mc_kwargs["seeds"] = int(seed)

        sol = qt.mcsolve(
            self.H,
            self.psi0,
            [0.0, float(tfin)],
            self.c_ops,
            e_ops=[],
            ntraj=1,
            options={"progress_bar": False},
            **mc_kwargs,
        )

        times, which = _extract_jump_record(sol, traj_idx=0)
        det_times, det_which = _filter_detected_times(times, which, self.detected_indices)
        taus = to_time_delay(det_times)
        if return_channels:
            return taus, det_which
        return taus

    def simulateTrajectoryFixedJumps(self, njumpsMC=48, seed=None, return_channels=False):
        if njumpsMC <= 0:
            return (np.asarray([], dtype=float), np.asarray([], dtype=int)) if return_channels else np.asarray([], dtype=float)

        H = self.H
        c_ops = self.c_ops
        psi = self.psi0
        dims = H.dims[0]

        rng = np.random.default_rng(int(seed)) if seed is not None else None
        detected_rate = self._estimate_detected_rate()

        abs_detected_times = []
        detected_labels = []
        tshift = 0.0
        max_chunks = 10000
        chunk_counter = 0

        while len(abs_detected_times) < njumpsMC:
            chunk_counter += 1
            if chunk_counter > max_chunks:
                raise RuntimeError("simulateTrajectoryFixedJumps exceeded max_chunks.")

            remaining = njumpsMC - len(abs_detected_times)
            tf = max(1.0, 1.5 * remaining / max(detected_rate, 1e-12))

            mc_kwargs = {}
            if rng is not None:
                mc_kwargs["seeds"] = int(rng.integers(0, 2**31 - 1))

            sol = qt.mcsolve(
                H,
                psi,
                [0.0, tf],
                c_ops,
                e_ops=[],
                ntraj=1,
                options={"progress_bar": False},
                **mc_kwargs,
            )

            times, which = _extract_jump_record(sol, traj_idx=0)
            det_times, det_which = _filter_detected_times(times, which, self.detected_indices)

            if det_times.size > 0:
                abs_detected_times.extend(list(tshift + det_times))
                detected_labels.extend(list(det_which))

            psi = _final_ket_from_solution(sol, dims=dims)
            tshift += tf

        abs_detected_times = np.asarray(abs_detected_times[:njumpsMC], dtype=float)
        detected_labels = np.asarray(detected_labels[:njumpsMC], dtype=int)
        taus = to_time_delay(abs_detected_times)

        if return_channels:
            return taus, detected_labels
        return taus

    def computeLikelihood(self, data: np.ndarray, r: float = 5.0, tfin: float = -1, method: str = "spectral") -> np.ndarray:
        """
        Exact recursive likelihood trace for a one-symbol observed record with waiting times data.

        This is appropriate for:
        - TLS
        - any model where the observed record is a single aggregated symbol

        For multi-symbol non-iid models, use the kernel matrix functions below.
        """
        del r  # legacy argument; continuous waiting-time likelihood has no tau*r factor.
        tau_list = np.asarray(data, dtype=float).reshape(-1)
        if np.any(tau_list < 0):
            raise ValueError("waiting times must be nonnegative.")

        rhoC = self.rho0.copy()
        likelihood_time = []

        for tau in tau_list:
            U = self._propagator(float(tau), method=method)
            rhoC = self.J @ U @ rhoC
            val = np.real((self.trace_row @ rhoC).item())
            likelihood_time.append(float(val))

        if tfin != -1:
            tau_fin = float(tfin) - float(np.sum(tau_list))
            if tau_fin < 0:
                raise ValueError("tfin is smaller than the sum of waiting times.")
            Ufin = self._propagator(tau_fin, method=method)
            rhoC = Ufin @ rhoC
            val = np.real((self.trace_row @ rhoC).item())
            likelihood_time.append(float(val))

        return np.asarray(likelihood_time, dtype=float)


# =============================================================================
# TLS: creation, MC generation, analytic waiting-time tools (gamma fixed to 1)
# =============================================================================

def create_TLS_model(delta, omega):
    psi0 = qt.basis(2, 0)
    a = qt.destroy(2)
    c = a
    H = float(delta) * a.dag() * a + float(omega) * (a + a.dag())
    return QuantumModel(
        H=H,
        c_ops=[c],
        psi0=psi0,
        detected_indices=(0,),
        with_eigvals=True,
        channel_names=["emission"],
    )


def generate_clicks_TLS(params, njumpsMC=48, seed=None):
    delta = float(params[0])
    omega = float(params[1])
    model = create_TLS_model(delta=delta, omega=omega)
    return model.simulateTrajectoryFixedJumps(njumpsMC=njumpsMC, seed=seed)


def generate_clicks_TLS_fixed_time(params, tfin: float, seed=None):
    delta = float(params[0])
    omega = float(params[1])
    model = create_TLS_model(delta=delta, omega=omega)
    return model.simulateTrajectoryFixedTime(tfin=tfin, seed=seed)


def compute_waiting_time_list(params, taulist):
    Omega, Delta = params
    return np.asarray(compute_waiting_time(taulist, Omega, Delta), dtype=float)


def compute_waiting_time(tau: float, Omega: float, Delta: float):
    """
    Closed-form TLS waiting-time density for gamma=1.
    """
    tau_arr = np.asarray(tau, dtype=float)
    Omega_arr = np.asarray(Omega, dtype=float)
    Delta_arr = np.asarray(Delta, dtype=float)

    eps = 1e-12

    a_sq = -64.0 * Omega_arr**2 + (1.0 + 4.0 * Delta_arr**2 + 16.0 * Omega_arr**2) ** 2
    invalid_domain = a_sq < -eps
    A = np.sqrt(np.clip(a_sq, a_min=0.0, a_max=None))
    A = np.clip(A, a_min=1e-300, a_max=None)

    b_sq = -1.0 + 4.0 * Delta_arr**2 + 16.0 * Omega_arr**2 + A
    c_sq = 1.0 - 4.0 * Delta_arr**2 - 16.0 * Omega_arr**2 + A
    invalid_domain = invalid_domain | (b_sq < -eps) | (c_sq < -eps)

    b = np.sqrt(np.clip(b_sq, a_min=0.0, a_max=None)) / (2.0 * np.sqrt(2.0))
    c = np.sqrt(np.clip(c_sq, a_min=0.0, a_max=None)) / (2.0 * np.sqrt(2.0))

    pref = 8.0 * Omega_arr**2 / A
    term_cos = np.exp(-0.5 * tau_arr) * np.cos(b * tau_arr)
    term_cosh = 0.5 * (np.exp(-(0.5 - c) * tau_arr) + np.exp(-(0.5 + c) * tau_arr))
    w_tau = pref * (-term_cos + term_cosh)

    invalid_val = (~np.isfinite(w_tau)) | (w_tau <= 0.0) | invalid_domain
    w_tau = np.where(invalid_val, 1e-300, w_tau)

    if np.ndim(w_tau) == 0:
        return float(w_tau)
    return np.asarray(w_tau, dtype=float)


def tls_waiting_time_real(tau, omega, delta, eps=1e-300):
    vals = np.asarray(compute_waiting_time(tau, Omega=omega, Delta=delta), dtype=float)
    vals = np.maximum(vals, eps)
    return vals.item() if np.ndim(tau) == 0 else vals


def compute_log_likelihood_analytical(Delta: float, data: np.ndarray, Omega: float = 1.0):
    tau_list = np.asarray(data, dtype=float)
    Delta_arr = np.asarray(Delta, dtype=float)

    if Delta_arr.ndim == 0:
        wlist = compute_waiting_time(tau_list, Omega, float(Delta_arr))
        return float(np.sum(np.log(np.clip(wlist, 1e-300, None))))

    wgrid = compute_waiting_time(tau_list[None, :], Omega, Delta_arr[:, None])
    return np.sum(np.log(np.clip(wgrid, 1e-300, None)), axis=1)


def compute_neg_log_likelihood_analytical(delta: float, data: np.ndarray, omega: float = 1.0):
    return -compute_log_likelihood_analytical(delta, data, Omega=omega)


def compute_likelihood_analytical(Delta: float, data: np.ndarray, Omega: float = 1.0):
    return np.exp(compute_log_likelihood_analytical(Delta, data, Omega=Omega))


def compute_likelihood_analytical_Classical(Delta: float, data: np.ndarray, Omega: float = 1.0):
    N = len(data)
    tau = np.mean(data)
    sigma = np.sqrt(((1.0 + 4 * Delta**2)**2 - 8 * (1.0 - 12 * Delta**2) * Omega**2 + 64 * Omega**4) / (N * Omega**4)) / 4
    mu = (1.0 + 4 * Delta**2 + 8 * Omega**2) / (4 * Omega**2)
    return np.exp(-0.5 * ((tau - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def get_estimates_Bayesian(data: np.ndarray, DeltaMin: float = 0.0, DeltaMax: float = 5.0, nDeltaGrid: int = 500, output_probabilities=False, only_mean=False):
    DeltaBayesListFine = np.linspace(DeltaMin, DeltaMax, nDeltaGrid)
    likelihood_grid = np.asarray(compute_likelihood_analytical(DeltaBayesListFine, data), dtype=float)

    prob_grid = likelihood_grid / trapz_integral(likelihood_grid, DeltaBayesListFine)
    cdf_grid = cumulative_trapezoid(prob_grid, DeltaBayesListFine, initial=0.0)

    deltaMean = trapz_integral(prob_grid * DeltaBayesListFine, DeltaBayesListFine)
    if only_mean:
        return deltaMean

    deltaMedian = DeltaBayesListFine[np.argmin(np.abs(cdf_grid - 0.5))]
    deltaMax = DeltaBayesListFine[np.argmax(prob_grid)]

    if not output_probabilities:
        return deltaMean, deltaMedian, deltaMax
    return deltaMean, deltaMedian, deltaMax, prob_grid


def compute_population_ss(params: list):
    Omega, Delta = params
    return 4 * Omega**2 / (1.0 + 4 * Delta**2 + 8 * Omega**2)


def compute_likelihood(Delta: float, data: np.ndarray, r: float = 5.0, omega: float = 1.0):
    model = create_TLS_model(delta=Delta, omega=omega)
    out = model.computeLikelihood(np.asarray(data, dtype=float), r=r, tfin=-1, method="spectral")
    return out[-1] if out.size > 0 else 1.0


def compute_negative_log_likelihood(Delta: float, data: np.ndarray, r: float = 5.0, omega: float = 1.0):
    return -np.log(compute_likelihood(Delta, data, r=r, omega=omega) + 1e-300)


# =============================================================================
# Three-level models: physically corrected channel construction
# =============================================================================

def _normalize_scheme_key(scheme):
    key = str(scheme).strip().lower()
    if key == "lamda":
        key = "lambda"
    if key not in {"shelving", "lambda", "v", "ladder"}:
        raise ValueError("scheme must be one of {'shelving', 'lambda', 'v', 'ladder'}.")
    return key


def _build_three_level_channels(
    scheme,
    delta,
    omega,
    gamma_det=1.0,
    gamma_hidden=0.20,
    gamma_aux=0.0,
    delta_aux=0.0,
    omega_aux=0.0,
):
    """
    Returns:
        scheme_key, H, c_ops, reset_states, channel_names

    Conventions:
      |0> : ground / lower bright state
      |1> : primary excited state
      |2> : auxiliary dark / metastable / second lower-or-upper state depending on scheme

    Notes:
    - shelving is the standard bright-dark-repump picture.
    - lambda, v, ladder are written canonically, with gamma_aux used only as an OPTIONAL
      extra leakage / relaxation channel. Set gamma_aux=0 for the textbook versions.
    """
    scheme_key = _normalize_scheme_key(scheme)

    ket_0 = qt.basis(3, 0)
    ket_1 = qt.basis(3, 1)
    ket_2 = qt.basis(3, 2)

    proj_1 = ket_1 * ket_1.dag()
    proj_2 = ket_2 * ket_2.dag()

    sigma_01 = ket_0 * ket_1.dag()
    sigma_10 = sigma_01.dag()

    sigma_02 = ket_0 * ket_2.dag()
    sigma_20 = sigma_02.dag()

    sigma_12 = ket_1 * ket_2.dag()
    sigma_21 = sigma_12.dag()

    c_ops = []
    reset_states = []
    channel_names = []

    def add_channel(name, rate, op, reset_state):
        rate = float(rate)
        if rate > 0.0:
            c_ops.append(np.sqrt(rate) * op)
            reset_states.append(reset_state)
            channel_names.append(name)

    if scheme_key == "shelving":
        H = float(delta) * proj_1 + float(delta_aux) * proj_2 + float(omega) * (sigma_01 + sigma_10)

        # bright fluorescence: 1 -> 0
        add_channel("bright", gamma_det, sigma_01, ket_0)

        # shelving jump: 1 -> 2
        add_channel("shelve", gamma_hidden, sigma_21, ket_2)

        # repump / deshelving: 2 -> 0
        add_channel("repump", gamma_aux, sigma_02, ket_0)

    elif scheme_key == "lambda":
        H = (
            float(delta) * proj_1
            + float(delta_aux) * proj_2
            + float(omega) * (sigma_01 + sigma_10)
            + float(omega_aux) * (sigma_21 + sigma_12)
        )

        # excited -> lower state |0>
        add_channel("decay_10", gamma_det, sigma_01, ket_0)

        # excited -> lower state |2>
        add_channel("decay_12", gamma_hidden, sigma_21, ket_2)

        # optional extra leakage 2 -> 0 (not part of the canonical textbook lambda unless wanted)
        add_channel("leak_20", gamma_aux, sigma_02, ket_0)

    elif scheme_key == "v":
        H = (
            float(delta) * proj_1
            + float(delta_aux) * proj_2
            + float(omega) * (sigma_01 + sigma_10)
            + float(omega_aux) * (sigma_02 + sigma_20)
        )

        # |1> -> |0>
        add_channel("decay_10", gamma_det, sigma_01, ket_0)

        # |2> -> |0>
        add_channel("decay_20", gamma_hidden, sigma_02, ket_0)

        # optional relaxation |2> -> |1>
        add_channel("relax_21", gamma_aux, sigma_12, ket_1)

    elif scheme_key == "ladder":
        H = (
            float(delta) * proj_1
            + float(delta + delta_aux) * proj_2
            + float(omega) * (sigma_01 + sigma_10)
            + float(omega_aux) * (sigma_12 + sigma_21)
        )

        # |1> -> |0>
        add_channel("decay_10", gamma_det, sigma_01, ket_0)

        # |2> -> |1>
        add_channel("decay_21", gamma_hidden, sigma_12, ket_1)

        # optional direct leakage |2> -> |0>
        add_channel("leak_20", gamma_aux, sigma_02, ket_0)

    else:
        raise AssertionError("unreachable")

    if len(c_ops) == 0:
        raise ValueError("no collapse channels were created.")

    return scheme_key, H, c_ops, reset_states, channel_names


def create_three_level_model(
    scheme,
    delta,
    omega,
    gamma_det=1.0,
    gamma_hidden=0.20,
    gamma_aux=0.0,
    delta_aux=0.0,
    omega_aux=0.0,
    detected_channels=None,
):
    scheme_key, H, c_ops, reset_states, channel_names = _build_three_level_channels(
        scheme=scheme,
        delta=delta,
        omega=omega,
        gamma_det=gamma_det,
        gamma_hidden=gamma_hidden,
        gamma_aux=gamma_aux,
        delta_aux=delta_aux,
        omega_aux=omega_aux,
    )
    del reset_states  # not needed in the MC model itself

    detected_indices = _resolve_channel_selection(channel_names, detected_channels)
    model = QuantumModel(
        H=H,
        c_ops=c_ops,
        psi0=qt.basis(3, 0),
        detected_indices=detected_indices,
        with_eigvals=True,
        channel_names=channel_names,
    )
    model.scheme = scheme_key
    return model


# =============================================================================
# Spectral kernel model for genuinely non-iid observed jump-channel sequences
# =============================================================================

def _state_density_vector(state):
    if not isinstance(state, qt.Qobj):
        state = qt.Qobj(state)
    rho = qt.ket2dm(state.unit()) if state.isket else state
    if not rho.isoper:
        raise TypeError("reset states must be kets or density operators.")
    tr = complex(rho.tr())
    if abs(tr) <= 0.0:
        raise ValueError("reset-state density operator has zero trace.")
    rho = rho / tr
    return qt.operator_to_vector(rho).full().reshape(-1)


def build_jump_kernel_spectral_model(H, observed_ops, reset_states, hidden_ops=None, channel_names=None, params=None):
    """
    Build the channel-to-channel waiting-time kernel matrix
        K_{ij}(t) = Tr[ J_i exp(L0 t) rho_j^reset ],
    where
        L0 = L - sum_i J_i
    and hidden_ops remain inside L.

    Here j = previous observed channel, i = next observed channel.
    """
    observed_ops = list(observed_ops)
    reset_states = list(reset_states)
    hidden_ops = [] if hidden_ops is None else list(hidden_ops)

    if len(observed_ops) == 0:
        raise ValueError("observed_ops must be non-empty.")
    if len(observed_ops) != len(reset_states):
        raise ValueError("observed_ops and reset_states must have the same length.")

    if channel_names is None:
        channel_names = [f"obs{k}" for k in range(len(observed_ops))]
    if len(channel_names) != len(observed_ops):
        raise ValueError("channel_names must match observed_ops.")

    dim = H.shape[0]
    L = qt.liouvillian(H, observed_ops + hidden_ops)

    J_list = [qt.sprepost(op, op.dag()) for op in observed_ops]
    J_total = J_list[0]
    for Jk in J_list[1:]:
        J_total = J_total + Jk

    L0 = (L - J_total).full()
    evals, E = np.linalg.eig(L0)
    Einv = np.linalg.inv(E)

    trace_row = _trace_row(dim)
    reset_vecs = [_state_density_vector(state) for state in reset_states]

    coeff = np.zeros((len(observed_ops), len(observed_ops), len(evals)), dtype=np.complex128)

    left_rows = []
    for Jk in J_list:
        left_rows.append((trace_row @ Jk.full() @ E).reshape(-1))

    for i in range(len(observed_ops)):
        left_i = left_rows[i]
        for j in range(len(observed_ops)):
            right_j = (Einv @ reset_vecs[j]).reshape(-1)
            coeff[i, j, :] = left_i * right_j

    return {
        "evals": np.asarray(evals, dtype=np.complex128),
        "coeff": coeff,
        "channel_names": list(channel_names),
        "params": {} if params is None else dict(params),
    }


def evaluate_jump_kernel_matrix(z, spectral_model):
    z_arr = np.asarray(z, dtype=np.complex128)
    flat = z_arr.reshape(-1)
    phase = np.exp(np.outer(spectral_model["evals"], flat))
    vals = np.tensordot(spectral_model["coeff"], phase, axes=([2], [0]))  # (nout, ninit, nz)

    if z_arr.ndim == 0:
        return vals[:, :, 0]
    return np.moveaxis(vals, -1, 0)  # (nz, nout, ninit)


def evaluate_detected_kernel_matrix(z, spectral_model):
    return evaluate_jump_kernel_matrix(z, spectral_model)


def kernel_determinant_identity(z, spectral_model):
    K = evaluate_jump_kernel_matrix(z, spectral_model)
    if np.ndim(z) == 0:
        return np.linalg.det(np.eye(K.shape[0], dtype=np.complex128) - K)
    return np.asarray(
        [np.linalg.det(np.eye(Ki.shape[0], dtype=np.complex128) - Ki) for Ki in K],
        dtype=np.complex128,
    )


def kernel_transition_matrix(spectral_model, t_max=80.0, n_t=6000, tail_tol=1e-4, t_max_limit=1024.0):
    t_stop = float(t_max)
    while True:
        tau = np.linspace(0.0, t_stop, int(n_t), dtype=float)
        K = np.real_if_close(evaluate_jump_kernel_matrix(tau, spectral_model))
        K = np.maximum(np.asarray(K, dtype=float), 0.0)  # (nt, i, j)
        transition = trapz_integral(K, tau, axis=0)  # (i, j)
        col_sums = np.sum(transition, axis=0)
        if tail_tol is None or np.min(col_sums) >= 1.0 - float(tail_tol) or t_stop >= float(t_max_limit):
            return transition
        t_stop *= 2.0


def kernel_conditional_transition_matrix(transition_matrix):
    P = np.asarray(transition_matrix, dtype=float).copy()
    col_sums = np.sum(P, axis=0, keepdims=True)
    for j in range(P.shape[1]):
        s = float(col_sums[0, j])
        if s > 0.0 and np.isfinite(s):
            P[:, j] /= s
        else:
            P[:, j] = 1.0 / float(P.shape[0])
    return P


def kernel_stationary_distribution(transition_matrix):
    P = np.asarray(transition_matrix, dtype=float)
    vals, vecs = np.linalg.eig(P)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    vec = np.real_if_close(vecs[:, idx]).astype(float)
    if np.sum(vec) < 0.0:
        vec = -vec
    vec = np.maximum(vec, 0.0)
    if np.sum(vec) <= 0.0:
        vec = np.ones(P.shape[0], dtype=float)
    return vec / np.sum(vec)


def kernel_aggregated_waiting_time_real(tau, spectral_model, eps=1e-300):
    K = np.real_if_close(evaluate_jump_kernel_matrix(tau, spectral_model))
    P = kernel_conditional_transition_matrix(kernel_transition_matrix(spectral_model))
    pi = kernel_stationary_distribution(P)

    if np.ndim(tau) == 0:
        val = float(np.sum(np.maximum(K, 0.0) * pi[None, :]))
        return max(val, eps)

    K = np.maximum(np.asarray(K, dtype=float), 0.0)  # (nt, i, j)
    out = np.einsum("j,tj->t", pi, np.sum(K, axis=1))
    out = np.maximum(out, eps)
    return out


def build_noniid_three_level_kernel_model(
    delta,
    omega,
    gamma_det=1.0,
    gamma_hidden=0.20,
    gamma_aux=0.0,
    delta_aux=0.0,
    omega_aux=0.0,
    scheme="shelving",
    observed_channels=None,
    hidden_channels=None,
):
    """
    Build the first-order Markov kernel on OBSERVED jump channels.

    Default:
      - if observed_channels is None: all physical channels are observed
      - if hidden_channels is None: all non-observed channels are treated as hidden
    """
    scheme_key, H, c_ops, reset_states, channel_names = _build_three_level_channels(
        scheme=scheme,
        delta=delta,
        omega=omega,
        gamma_det=gamma_det,
        gamma_hidden=gamma_hidden,
        gamma_aux=gamma_aux,
        delta_aux=delta_aux,
        omega_aux=omega_aux,
    )

    observed_idx = _resolve_channel_selection(channel_names, observed_channels)
    if hidden_channels is None:
        hidden_idx = tuple(k for k in range(len(c_ops)) if k not in observed_idx)
    else:
        hidden_idx = _resolve_channel_selection(channel_names, hidden_channels)

    if set(observed_idx) & set(hidden_idx):
        raise ValueError("observed_channels and hidden_channels must be disjoint.")

    obs_ops = [c_ops[k] for k in observed_idx]
    obs_resets = [reset_states[k] for k in observed_idx]
    obs_names = [channel_names[k] for k in observed_idx]
    hid_ops = [c_ops[k] for k in hidden_idx]

    return build_jump_kernel_spectral_model(
        H=H,
        observed_ops=obs_ops,
        reset_states=obs_resets,
        hidden_ops=hid_ops,
        channel_names=obs_names,
        params={
            "scheme": scheme_key,
            "delta": float(delta),
            "omega": float(omega),
            "gamma_det": float(gamma_det),
            "gamma_hidden": float(gamma_hidden),
            "gamma_aux": float(gamma_aux),
            "delta_aux": float(delta_aux),
            "omega_aux": float(omega_aux),
            "observed_channels": tuple(obs_names),
            "hidden_channels": tuple(channel_names[k] for k in hidden_idx),
        },
    )


def noniid_three_level_kernel_matrix_complex(
    z,
    delta,
    omega,
    gamma_det=1.0,
    gamma_hidden=0.20,
    gamma_aux=0.0,
    delta_aux=0.0,
    omega_aux=0.0,
    scheme="shelving",
    observed_channels=None,
    hidden_channels=None,
):
    model = build_noniid_three_level_kernel_model(
        delta=delta,
        omega=omega,
        gamma_det=gamma_det,
        gamma_hidden=gamma_hidden,
        gamma_aux=gamma_aux,
        delta_aux=delta_aux,
        omega_aux=omega_aux,
        scheme=scheme,
        observed_channels=observed_channels,
        hidden_channels=hidden_channels,
    )
    return evaluate_jump_kernel_matrix(z, model)


def noniid_three_level_determinant_complex(
    z,
    delta,
    omega,
    gamma_det=1.0,
    gamma_hidden=0.20,
    gamma_aux=0.0,
    delta_aux=0.0,
    omega_aux=0.0,
    scheme="shelving",
    observed_channels=None,
    hidden_channels=None,
):
    model = build_noniid_three_level_kernel_model(
        delta=delta,
        omega=omega,
        gamma_det=gamma_det,
        gamma_hidden=gamma_hidden,
        gamma_aux=gamma_aux,
        delta_aux=delta_aux,
        omega_aux=omega_aux,
        scheme=scheme,
        observed_channels=observed_channels,
        hidden_channels=hidden_channels,
    )
    return kernel_determinant_identity(z, model)


def noniid_three_level_aggregated_waiting_time_real(
    tau,
    delta,
    omega,
    gamma_det=1.0,
    gamma_hidden=0.20,
    gamma_aux=0.0,
    delta_aux=0.0,
    omega_aux=0.0,
    scheme="shelving",
    observed_channels=None,
    hidden_channels=None,
    eps=1e-300,
):
    model = build_noniid_three_level_kernel_model(
        delta=delta,
        omega=omega,
        gamma_det=gamma_det,
        gamma_hidden=gamma_hidden,
        gamma_aux=gamma_aux,
        delta_aux=delta_aux,
        omega_aux=omega_aux,
        scheme=scheme,
        observed_channels=observed_channels,
        hidden_channels=hidden_channels,
    )
    return kernel_aggregated_waiting_time_real(tau, model, eps=eps)


# =============================================================================
# convenient wrappers by scheme
# =============================================================================

def build_shelving_kernel_model(delta, omega, gamma_bg=1.0, gamma_es=0.20, gamma_sg=0.05, delta_s=0.0, observed_channels=None, hidden_channels=None):
    return build_noniid_three_level_kernel_model(
        delta=delta,
        omega=omega,
        gamma_det=gamma_bg,
        gamma_hidden=gamma_es,
        gamma_aux=gamma_sg,
        delta_aux=delta_s,
        omega_aux=0.0,
        scheme="shelving",
        observed_channels=observed_channels,
        hidden_channels=hidden_channels,
    )


def build_lambda_kernel_model(delta, omega, gamma_det=1.0, gamma_hidden=0.20, gamma_aux=0.0, delta_aux=0.0, omega_aux=0.35, observed_channels=None, hidden_channels=None):
    return build_noniid_three_level_kernel_model(
        delta=delta,
        omega=omega,
        gamma_det=gamma_det,
        gamma_hidden=gamma_hidden,
        gamma_aux=gamma_aux,
        delta_aux=delta_aux,
        omega_aux=omega_aux,
        scheme="lambda",
        observed_channels=observed_channels,
        hidden_channels=hidden_channels,
    )


def build_v_kernel_model(delta, omega, gamma_det=1.0, gamma_hidden=0.20, gamma_aux=0.0, delta_aux=0.8, omega_aux=0.55, observed_channels=None, hidden_channels=None):
    return build_noniid_three_level_kernel_model(
        delta=delta,
        omega=omega,
        gamma_det=gamma_det,
        gamma_hidden=gamma_hidden,
        gamma_aux=gamma_aux,
        delta_aux=delta_aux,
        omega_aux=omega_aux,
        scheme="v",
        observed_channels=observed_channels,
        hidden_channels=hidden_channels,
    )


def build_ladder_kernel_model(delta, omega, gamma_det=1.0, gamma_hidden=0.20, gamma_aux=0.0, delta_aux=0.8, omega_aux=0.55, observed_channels=None, hidden_channels=None):
    return build_noniid_three_level_kernel_model(
        delta=delta,
        omega=omega,
        gamma_det=gamma_det,
        gamma_hidden=gamma_hidden,
        gamma_aux=gamma_aux,
        delta_aux=delta_aux,
        omega_aux=omega_aux,
        scheme="ladder",
        observed_channels=observed_channels,
        hidden_channels=hidden_channels,
    )


# =============================================================================
# TLS-only Torch theory for renewal benchmarks
# =============================================================================

def _as_real_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x if device is None else x.to(device=device)
    return torch.as_tensor(x, device=device)


def _as_complex_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x if device is None else x.to(device=device)
    return torch.as_tensor(x, device=device)


def _tls_waiting_real_branch(tau, delta, omega, eps=1e-300):
    tau = _as_real_tensor(tau)
    delta = _as_real_tensor(delta, device=tau.device).to(dtype=tau.dtype)
    omega = _as_real_tensor(omega, device=tau.device).to(dtype=tau.dtype)
    eps_tensor = torch.as_tensor(eps, device=tau.device, dtype=tau.dtype)
    tol = torch.as_tensor(1e-12, device=tau.device, dtype=tau.dtype)

    a_sq = -64.0 * omega * omega + torch.square(1.0 + 4.0 * delta * delta + 16.0 * omega * omega)
    a_safe = torch.sqrt(torch.clamp(a_sq, min=0.0))
    a_safe = torch.clamp(a_safe, min=eps_tensor)

    b_sq = -1.0 + 4.0 * delta * delta + 16.0 * omega * omega + a_safe
    c_sq = 1.0 - 4.0 * delta * delta - 16.0 * omega * omega + a_safe
    b = torch.sqrt(torch.clamp(b_sq, min=0.0)) / (2.0 * math.sqrt(2.0))
    c = torch.sqrt(torch.clamp(c_sq, min=0.0)) / (2.0 * math.sqrt(2.0))

    pref = 8.0 * omega * omega / a_safe
    term_cos = torch.exp(-0.5 * tau) * torch.cos(b * tau)
    term_cosh = 0.5 * (torch.exp(-(0.5 - c) * tau) + torch.exp(-(0.5 + c) * tau))
    w = pref * (-term_cos + term_cosh)

    invalid_domain = (a_sq < -tol) | (b_sq < -tol) | (c_sq < -tol)
    invalid_val = (~torch.isfinite(w)) | (w <= eps_tensor)
    return torch.where(invalid_domain | invalid_val, torch.full_like(w, eps_tensor), w)


def _tls_laplace_branch(delta, omega, s, eps=1e-300):
    delta = _as_real_tensor(delta)
    omega = _as_real_tensor(omega, device=delta.device).to(dtype=delta.dtype)
    s = _as_complex_tensor(s, device=delta.device)

    real_dtype = delta.dtype if delta.is_floating_point() else torch.float32
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    s = s.to(dtype=complex_dtype)

    eps_tensor = torch.as_tensor(eps, device=delta.device, dtype=real_dtype)
    a_sq = -64.0 * omega * omega + torch.square(1.0 + 4.0 * delta * delta + 16.0 * omega * omega)
    a_safe = torch.sqrt(torch.clamp(a_sq, min=0.0))
    a_safe = torch.clamp(a_safe, min=eps_tensor)

    b_sq = -1.0 + 4.0 * delta * delta + 16.0 * omega * omega + a_safe
    c_sq = 1.0 - 4.0 * delta * delta - 16.0 * omega * omega + a_safe
    b = torch.sqrt(torch.clamp(b_sq, min=0.0)) / (2.0 * math.sqrt(2.0))
    c = torch.sqrt(torch.clamp(c_sq, min=0.0)) / (2.0 * math.sqrt(2.0))

    pref = (8.0 * omega * omega / a_safe).to(dtype=complex_dtype)
    s_shift = s + torch.as_tensor(0.5, device=s.device, dtype=complex_dtype)
    b_sq_c = (b * b).to(dtype=complex_dtype)
    c_c = c.to(dtype=complex_dtype)
    return pref * (-(s_shift) / (s_shift * s_shift + b_sq_c) + 0.5 / (s_shift - c_c) + 0.5 / (s_shift + c_c))


def build_model_theory_torch(model_name):
    key = str(model_name).strip().lower()
    if key != "tls":
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Only TLS has a renewal Torch theory; "
            "three-level models must use nonrenewal kernel-matrix theory."
        )

    def waiting_time_real_torch(tau, delta, omega, eps=1e-300):
        return _tls_waiting_real_branch(tau=tau, delta=delta, omega=omega, eps=eps)

    def laplace_transform_torch(theta, s, eps=1e-300):
        theta = _as_real_tensor(theta)
        if theta.ndim == 2 and theta.shape[-1] == 2:
            outs = [_tls_laplace_branch(delta=th[0], omega=th[1], s=s, eps=eps) for th in theta]
            return torch.stack(outs, dim=0)
        if theta.ndim != 1 or theta.shape[0] != 2:
            raise ValueError(f"theta must have shape (2,) or (N,2), got {tuple(theta.shape)}")
        return _tls_laplace_branch(delta=theta[0], omega=theta[1], s=s, eps=eps)

    return waiting_time_real_torch, laplace_transform_torch
