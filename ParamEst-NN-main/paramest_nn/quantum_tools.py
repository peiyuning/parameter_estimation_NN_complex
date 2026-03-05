import os

import numpy as np
from numpy import exp, cos, cosh, sqrt

# SciPy
from scipy.integrate import cumulative_trapezoid
cumtrapz = cumulative_trapezoid


# QuTiP 5
import qutip as qt

class QuantumModel:
    def __init__(self,H,c_ops,J,psi0,with_eigvals=True):             
        self.H = H
        self.c_ops = c_ops
        self.J = J.full()
        self.L = qt.liouvillian(H, c_ops)  # type: ignore
        self.Lhat = self.L - J
        if not isinstance(psi0, qt.Qobj):
            psi0 = qt.Qobj(psi0, dims=H.dims[0])

        if psi0.isoper:
            raise TypeError(
                "psi0 must be a pure-state ket (Qobj), not a density matrix/operator."
            )

        if not psi0.isket:
            raise TypeError(f"psi0 must be a ket with shape (N,1). Got shape={psi0.shape}.")

        psi0 = psi0.unit()

        self.psi0 = psi0
        self.rho0 = qt.operator_to_vector(qt.ket2dm(self.psi0)).full()
        
        if with_eigvals==True:
            eigen_hat=self.Lhat.eigenstates()
            h2 = len(eigen_hat[1])
            E = np.transpose(np.reshape(
                np.array([eigen_hat[1][i].full() for i in range(h2)]), 
                (h2,h2)))
            Einv = np.linalg.inv(E)
            self.Ehat = E
            self.Einvhat = Einv
            self.eigs_hat = eigen_hat[0]
        else:
            self.Ehat = None
            self.Einvhat = None
            self.eigs_hat = None

    # 如果给进来的是算符/密度矩阵，就直接报错（因为 mcsolve 需要 ket）
        if psi0.isoper:
            raise TypeError(
            "psi0 looks like an operator/density matrix. "
            "mcsolve (QuTiP 5) needs a ket pure state. "
            "Please pass a state vector / ket."
        )
        if not psi0.isket:
            raise TypeError(f"psi0 must be a ket (shape (N,1)). Got shape={psi0.shape}.")    

    def simulateTrajectories(self, tfin: float, ntraj:int, tlistexp = None, seed = None):
        
        H = self.H
        c_ops = self.c_ops
        psi0 = self.psi0
        
        if seed is not None:
            np.random.seed(seed)
            # QuTiP 5: can pass a single int or a list/array of seeds per trajectory.
            seeds = seed

        if tlistexp is None:
            tlist = [0.0, tfin]
            
            sol = qt.mcsolve(
                H, psi0, tlist,
                c_ops,
                e_ops=[],
                ntraj=ntraj,
            )

            taus = []
            for times in sol.col_times:
                if len(times)>0:
                    taus.append(to_time_delay(times))
                else:
                    taus.append(np.array([]))
            return taus
        
        else:
            tlist = tlistexp
            sigma = c_ops[0]

            sol = qt.mcsolve(
                H, psi0, tlist,
                c_ops,
                e_ops=[],
                ntraj=ntraj,
            )

        taus = []
        for times in sol.col_times:
            if len(times) > 0:
                taus.append(to_time_delay(times))
            else:
                taus.append(np.array([]))

        statestraj = sol.states
        if not statestraj:
            raise RuntimeError("mcsolve returned empty states.")

        # Case A: statestraj is a single trajectory over time: [state_t0, state_t1, ...]
        # (common when ntraj=1)
        if isinstance(statestraj[0], qt.Qobj):
            traj_states = [statestraj]  
        # Case B: statestraj is already list of trajectories: [[...],[...],...]
        else:
             traj_states = statestraj

        op = sigma.dag() * sigma
        pop_array = np.asarray([[qt.expect(op, st) for st in traj] for traj in traj_states], dtype=float)
        return taus, pop_array


    def simulateTrajectoryFixedJumps(self,njumpsMC=48):    
        H = self.H
        c_ops = self.c_ops
        psi0 = self.psi0
        t0 = 0
        factor = 1.2
        jumpOp = c_ops[0]
        rho_ss = qt.steadystate(H, c_ops)
        n_expect = qt.expect(jumpOp.dag()*jumpOp, rho_ss)
        tf = factor*njumpsMC/n_expect
        tlist = [t0, tf]
        taus_list = []
        tshift = t0
        while len(taus_list) < njumpsMC:
            sol = qt.mcsolve(
                H, psi0, tlist,
                c_ops,
                e_ops=[],
                ntraj=1,
                options={"progress_bar": False},
            )
            states = getattr(sol, "states", None)
            if states is not None and len(states) > 0:
                psi0_next = states[0][-1] if isinstance(states[0], (list, tuple)) else states[-1]
                if not isinstance(psi0_next, qt.Qobj):
                    psi0_next = qt.Qobj(psi0_next, dims=H.dims[0])
                if psi0_next.isoper:
                    evals, evecs = psi0_next.eigenstates()
                    evals = np.real(np.asarray(evals))
                    idx = int(np.argmax(evals))
                    if np.isclose(evals[idx], 1.0, atol=1e-8):
                        psi0_next = evecs[idx]
                    else:
                        raise TypeError("mcsolve returned a mixed density matrix state")
                if not psi0_next.isket:
                    raise TypeError(f"expected ket after conversion, got shape={psi0_next.shape}")
                psi0 = psi0_next.unit()
            else:
                raise RuntimeError("sol.states is None")
            times = np.asarray(sol.col_times[0], dtype=float)
            if times.size == 0:
                tf *= 1.5
                tlist = [0.0, tf]
                continue
            times = tshift + times
            tshift = float(times[-1])
            taus_list += list(to_time_delay(times))
        taus = np.asarray(taus_list, dtype=float)
        return taus[:njumpsMC]

    def computeLikelihood(self, data: np.ndarray, r: float = 5.0, tfin: float = -1, method: str = "spectral") -> np.ndarray:
        tau_list = data
        t_total = np.sum(tau_list)
        njumps = len(tau_list)
    
        J = self.J
        psi0=self.psi0
        Lhat = self.Lhat
    
        hilbert = psi0.shape[0]
        rho0 = self.rho0
        rhoC = rho0
        rhoC_t = []
        
        if method=='spectral':
            # Spectral decomposition of (L-J)
            E=self.Ehat 
            Einv=self.Einvhat
            EinvTr = np.transpose(Einv)  # type: ignore

            eigvals = self.eigs_hat

                
            Uops = [  E@(np.transpose((exp(tau*eigvals))*EinvTr))       for tau in tau_list]
        
        
            # Now we need to use numpy arrays    
                
            for jump_idx  in range(njumps):
                renorm = tau_list[jump_idx]*r
                rhoC=J@Uops[jump_idx]@rhoC
                rhoC = renorm*rhoC
                rhoC_t.append(rhoC)

            # If the simulation set a final time (not a fixed # of clicks):
            if tfin!= -1:
                tau_fin = tfin-t_total
                renorm = tau_fin*r
                Ufin = E@(np.transpose((exp(tau_fin*eigvals))*EinvTr)) 
                rhoC=Ufin@rhoC
                rhoC = renorm*rhoC
                rhoC_t.append(rhoC)
            
            likelihoodTime = np.array([np.reshape(rhoCsel,(hilbert,hilbert)).trace() for rhoCsel in rhoC_t])   

        if method=='direct':
            Uops = [(tau*Lhat).expm() for tau in tau_list]

            for jump_idx  in range(njumps):
                renorm = tau_list[jump_idx]*r
                rhoC=J*Uops[jump_idx]*rhoC
                rhoC = renorm*rhoC
                rhoC_t.append(rhoC)

            # If the simulation set a final time (not a fixed # of clicks):
            if tfin!= -1:
                tau_fin = tfin-t_total
                renorm = tau_fin*r
                Ufin = (tau_fin*Lhat).expm()
                rhoC=Ufin*rhoC
                rhoC = renorm*rhoC
                rhoC_t.append(rhoC)

            likelihoodTime = np.array([qt.vector_to_operator(rhoCsel).tr() for rhoCsel in rhoC_t])   

        return likelihoodTime

def create_TLS_model(delta,omega,gamma=1):
    
    psi0=qt.basis(2,0); a=qt.destroy(2); c_ops=[]; c_ops.append(np.sqrt(gamma)*a)
    H=delta*a.dag()*a+omega*a+omega*a.dag()
    J = gamma*qt.sprepost(a,a.dag())

    model_TLS = QuantumModel(H,c_ops,J,psi0)
    return model_TLS

def generate_clicks_TLS(params,njumpsMC=48,gamma=1.):
   
    delta = params[0]
    omega = params[1]

    model = create_TLS_model(delta,omega,gamma) # type: ignore
    taus = model.simulateTrajectoryFixedJumps(njumpsMC)
    return taus

def to_time_delay(array: np.ndarray):
    return np.concatenate((np.asarray([array[0]]), np.diff(array)))

def to_time_delay_matrix(matrix: np.ndarray):
    first_column = np.reshape(matrix[:, 0], (matrix.shape[0], 1))
    time_delay_matrix = np.concatenate((first_column, np.diff(matrix)), axis=1)
    return time_delay_matrix

def generate_prob_from_model_list(data: np.ndarray, model_list : list, r: float = 5.0, tfin: float = -1, method: str = "spectral"):
    # Compute the likelihood L(D|model) for each model in 'model_list' and store in an array
    likelihood_time_delta = np.asarray([ model.computeLikelihood(data, r, tfin=tfin,method= method) for model in model_list])

    # Normalize the likelihood to obtain a probability distribution P(D|model)
    prob_delta = likelihood_time_delta/np.sum(likelihood_time_delta,axis=0)

    return prob_delta

def compute_log_likelihood_analytical(Delta: float, data: np.ndarray,gamma: float = 1., Omega: float = 1.) -> np.ndarray:
    tau_list = data
    wlist = np.array([compute_waiting_time(tau,Omega,Delta) for tau in tau_list])

    # Jumps are independent, therefore, the likelihood is the product of the probabilities, and the log-likelihood, the sum of the log-probabilities.
    log_likelihood = np.sum(np.log(wlist+1e-300))

    return log_likelihood

def compute_neg_log_likelihood_analytical(delta: float, data: np.ndarray,gamma: float = 1., omega: float = 1.) -> np.ndarray:
        
    return -compute_log_likelihood_analytical(delta, data,gamma, omega)

def compute_likelihood_analytical(Delta: float, data: np.ndarray, gamma: float = 1., Omega: float = 1.) -> np.ndarray:
    log_likelihood =  compute_log_likelihood_analytical(Delta, data, gamma, Omega)

    return np.exp(log_likelihood)

def compute_likelihood_analytical_Classical(Delta: float, data: np.ndarray, gamma: float = 1., Omega: float = 1.) -> np.ndarray :
    N = len(data)
    tau = np.mean(data)
    sigma = np.sqrt(((gamma**2 + 4*Delta**2)**2 - 8*(gamma**2 - 12*Delta**2)*Omega**2 + 64*Omega**4)/(N*gamma**2 * Omega**4))/4
    mu = (gamma**2 + 4*Delta**2 + 8*Omega**2)/(4*gamma*Omega**2)
    return np.exp(-0.5*((tau-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))


def get_estimates_Bayesian(data: np.ndarray, DeltaMin: float = 0., DeltaMax: float = 5., nDeltaGrid: int = 500, output_probabilities=False, only_mean = False):
    # Generate a fine grid of delta values
    DeltaBayesListFine = np.linspace(DeltaMin, DeltaMax, nDeltaGrid)

    # Calculate the likelihood grid using analytical computation
    likelihood_grid = np.array([compute_likelihood_analytical(Delta, data) for Delta in DeltaBayesListFine])

    # Compute the probability distribution by normalizing the likelihood
    prob_grid = likelihood_grid / np.trapz(likelihood_grid, DeltaBayesListFine)
    cdf_grid = cumtrapz(prob_grid, DeltaBayesListFine)

    # Calculate mean, median, and maximum estimates for delta
    deltaMean = np.trapz(prob_grid*DeltaBayesListFine, DeltaBayesListFine)

    if only_mean == True:
        return deltaMean

    deltaMedian = DeltaBayesListFine[np.argmin(np.abs(cdf_grid - 0.5))]
    deltaMax = DeltaBayesListFine[np.argmax(prob_grid)]

    # Check if output_probabilities is requested
    if output_probabilities == False:
        return deltaMean, deltaMedian, deltaMax
    else:
        return deltaMean, deltaMedian, deltaMax, prob_grid

    
def prob_no_click(Delta: float,tau:float,Omega:float =1.):
    return np.real(
        ((-1/8*1j)*
  (exp(((2 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))*
       tau)/4)*((-8*1j)*np.emath.sqrt(2)*Delta**3 + (2*1j)*np.emath.sqrt(2)*Delta*
      (-1 - 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
         8*Delta**2*(1 + 16*Omega**2))) + np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
       np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
     4*Delta**2*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
        np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
     16*Omega**2*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
        np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
     np.emath.sqrt((16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))*
       (1 - 4*Delta**2 - 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
          8*Delta**2*(1 + 16*Omega**2))))) + 
   exp(((2 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
        (2*1j)*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))*
       tau)/4)*((-8*1j)*np.emath.sqrt(2)*Delta**3 + (2*1j)*np.emath.sqrt(2)*Delta*
      (-1 - 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
         8*Delta**2*(1 + 16*Omega**2))) - np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
       np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
     4*Delta**2*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
        np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
     16*Omega**2*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
        np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
     np.emath.sqrt((16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))*
       (1 - 4*Delta**2 - 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
          8*Delta**2*(1 + 16*Omega**2))))) + 
   exp(((2 + 1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))*
       tau)/4)*((8*1j)*np.emath.sqrt(2)*Delta**3 + (2*1j)*np.emath.sqrt(2)*Delta*
      (1 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
         8*Delta**2*(1 + 16*Omega**2))) - (4*1j)*Delta**2*
      np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
          8*Delta**2*(1 + 16*Omega**2))) - 
     1j*(np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + 
           (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
       16*Omega**2*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
          np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
       np.emath.sqrt((16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))*
         (-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
            8*Delta**2*(1 + 16*Omega**2)))))) + 
   exp(((2 + 2*np.emath.sqrt(2)*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
        1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))*
       tau)/4)*((8*1j)*np.emath.sqrt(2)*Delta**3 + (2*1j)*np.emath.sqrt(2)*Delta*
      (1 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
         8*Delta**2*(1 + 16*Omega**2))) + (4*1j)*Delta**2*
      np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
          8*Delta**2*(1 + 16*Omega**2))) + 
     1j*(np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + 
           (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
       16*Omega**2*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
          np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
       np.emath.sqrt((16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))*
         (-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
            8*Delta**2*(1 + 16*Omega**2))))))))/
 (np.emath.sqrt(2)*Delta*
  exp(((4 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
         np.emath.sqrt(-64*Omega**2 + (1 + 4*Delta**2 + 16*Omega**2)**2)) + 
      1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
         np.emath.sqrt(-64*Omega**2 + (1 + 4*Delta**2 + 16*Omega**2)**2)))*tau)/4)*
  np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))

def compute_waiting_time_list(params, taulist):
    Omega, Delta = params
    wtau = np.asarray([compute_waiting_time(tau,Omega,Delta) for tau in taulist])
    return wtau


def compute_waiting_time(tau: float, Omega: float, Delta: float):
    tau = float(tau)
    Omega = float(Omega)
    Delta = float(Delta)

    # Numerical robustness: tiny negative radicands can appear from floating-point
    # cancellation near the model boundary. Clip within tolerance; reject clearly
    # invalid points to avoid propagating NaNs into Barankin/HCRB calculations.
    eps = 1e-12

    a_sq = -64.0 * Omega**2 + (1.0 + 4.0 * Delta**2 + 16.0 * Omega**2) ** 2
    if a_sq < -eps:
        return 1e-300
    A = np.sqrt(max(a_sq, 0.0))

    # Avoid division-by-zero at singular points.
    A = max(float(A), 1e-300)

    b_sq = -1.0 + 4.0 * Delta**2 + 16.0 * Omega**2 + A
    c_sq = 1.0 - 4.0 * Delta**2 - 16.0 * Omega**2 + A

    if b_sq < -eps or c_sq < -eps:
        return 1e-300

    b = np.sqrt(max(b_sq, 0.0)) / (2.0 * np.sqrt(2.0))
    c = np.sqrt(max(c_sq, 0.0)) / (2.0 * np.sqrt(2.0))

    pref = 8.0 * Omega**2 / A

    # Stable terms for exp(-tau/2)*cos and exp(-tau/2)*cosh.
    term_cos = np.exp(-0.5 * tau) * np.cos(b * tau)
    term_cosh = 0.5 * (np.exp(-(0.5 - c) * tau) + np.exp(-(0.5 + c) * tau))

    w_tau = pref * (-term_cos + term_cosh)

    # waiting-time pdf should be positive; clamp tiny/invalid values.
    if not np.isfinite(w_tau) or w_tau <= 0.0:
        return 1e-300
    return float(w_tau)

def compute_population_ss(params: list):
    Omega, Delta = params
    nsigma = 4*Omega**2/(1**2 + 4*Delta**2 + 8 * Omega**2)
    return nsigma

def compute_likelihood(Delta: float, data: np.ndarray, r: float = 5.0,gamma: float = 1., omega: float = 1.) -> np.ndarray:
    tau_list = data

    njumps = len(tau_list)
    psi0=qt.basis(2,0)
    a=qt.destroy(2)
    c_ops=[]
    c_ops.append(np.sqrt(gamma)*a)

    # Define system Hamiltonian
    H=Delta*a.dag()*a+omega*a+omega*a.dag() # type: ignore

    L = qt.liouvillian(H,c_ops) # type: ignore
    J = gamma*qt.sprepost(a,a.dag())
    Lhat = L-J
    Uops = [(tau*Lhat).expm() for tau in tau_list]
    rho0 = qt.operator_to_vector(qt.ket2dm(psi0))
    rhoC = rho0

    rhoC_t = []
    for jump_idx  in range(njumps):
        renorm = tau_list[jump_idx]*r
        rhoC=J*Uops[jump_idx]*rhoC
        rhoC = renorm*rhoC
        rhoC_t.append(rhoC)

    likelihoodTime = np.array([qt.vector_to_operator(rhoCsel).tr() for rhoCsel in rhoC_t])   


    return likelihoodTime[-1]


def compute_negative_log_likelihood(Delta: float, data: np.ndarray, r: float = 5.0,gamma: float = 1., omega: float = 1.):

    return -np.log(compute_likelihood(Delta, data, r, gamma, omega)+1e-300)

def create_directory(filepath:str):
    
    if not os.path.exists(filepath):
        print(f"Directory {filepath} not found: creating...")
        os.makedirs(filepath)
    else:
        print("Folder already exists.")
