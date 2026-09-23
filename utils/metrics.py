import numpy as np
from math import comb
from scipy.sparse.linalg import svds
import networkx as nx

"""
Metrics
"""

def calculate_diameters(A):
    """Calculate 3 diameters of the graph formed from the adjacency matrix A
    
    Returns:
        giant_diam (int): diameter of the giant component (or largest component if the giant doesn't exist)
        
        average_diam (int): average over all diameters

        giant_size (int): number of nodes in the giant component
    """
    G = nx.from_scipy_sparse_array(A)

    giant_diam = 0
    giant_size = 0
    sum_diam = 0

    connected_components = list(nx.connected_components(G))

    for c in connected_components:
        G_sub = G.subgraph(c)

        # Single-node component
        if len(G_sub) <= 1:
            curr_diam = 0
        else:
            curr_diam = nx.diameter(G_sub)

        m = len(G_sub)

        # Giant component diameter
        if m > giant_size or (m == giant_size and curr_diam > giant_diam):
            giant_diam = curr_diam
            giant_size = m

        sum_diam += curr_diam

    average_diam = sum_diam / len(connected_components)

    return giant_diam, average_diam, giant_size


def calculate_diameters_weakly_connected(A):
    """
    Calculate diameters of weakly connected components
    in a directed graph given by adjacency matrix A.

    Returns:
        giant_diam (int): diameter of the largest weakly connected component
        average_diam (float): average diameter over all components
        giant_size (int): size of the largest component
    """
    # Create directed graph
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)

    giant_diam = 0
    giant_size = 0
    sum_diam = 0

    components = list(nx.weakly_connected_components(G))

    for c in components:
        G_sub = G.subgraph(c)

        # Diameter is defined for undirected graphs
        G_undirected = G_sub.to_undirected()

        # Single-node component
        if len(G_undirected) <= 1:
            curr_diam = 0
        else:
            curr_diam = nx.diameter(G_undirected)

        m = len(G_sub)

        # Giant component diameter
        if m > giant_size or (m == giant_size and curr_diam > giant_diam):
            giant_diam = curr_diam
            giant_size = m

        sum_diam += curr_diam

    average_diam = sum_diam / len(components)

    return giant_diam, average_diam, giant_size


def nrmse(true, pred):
    """ Normalized root mean square error. (A metric for measuring difference in orbits)
    Parameters:
        Two mxn arrays. Axis zero is assumed to be the time axis (i.e. there are m time steps)
    Returns:
        err (ndarray): Error at each time value. 1D array with m entries
    """
    sig = np.std(true, axis=0)
    err = np.linalg.norm((true-pred) / sig, axis=1, ord=2)
    # err = np.linalg.norm((true-pred), axis=1, ord=2) # Just regular 2-norm
    return err


def valid_prediction_index(err, tol):
    """First index i where err[i] > tol. err is assumed to be 1D and tol is a float. If err is never greater than tol, then len(err) is returned."""
    mask = np.logical_or(err > tol, ~np.isfinite(err))
    if np.any(mask):
        return np.argmax(mask)
    return len(err)


def vpt_time(ts, Uts, pre, vpt_tol=5.):
    """
    Valid prediction time for a specific instance.
    """
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, vpt_tol)
    if idx == 0:
        vptime = 0.
    else:
        vptime = ts[idx-1] - ts[0]
    return vptime


def div_spatial(states):
    """
    Compute diversity scores of internal reservoir states,
    matching the element-wise absolute difference method from div_metric_tests.

    Parameters:
        states (ndarray): Reservoir states, shape (T, n)

    Returns:
        div_pos (float): Position-based diversity scores
    """
    T, n = states.shape
    abs_states = np.abs(states)

    # Compute pairwise differences with vectors
    diffs_pos = abs_states[:, :, None] - abs_states[:, None, :]

    # Take upper triangle without diagonal to avoid double counting and self pairs
    triu_indices = np.triu_indices(n, k=1)

    # Sum over time, average over number of pairs
    div_pos = np.sum(np.abs(diffs_pos[:, triu_indices[0], triu_indices[1]])) / (T * comb(n, 2))

    return div_pos


def div_rate(states):
    """
    Compute diversity scores of internal reservoir states' time derivatives,
    matching the element-wise absolute difference method from div_metric_tests.

    Parameters:
        states (ndarray): Reservoir states, shape (T, n)

    Returns:
        div_der (float): Derivative-based diversity scores
    """
    T, n = states.shape
    res_deriv = np.gradient(states, axis=0)

    # Compute abs values
    abs_deriv = np.abs(res_deriv)

    # Compute pairwise differences with vectors
    diffs_der = abs_deriv[:, :, None] - abs_deriv[:, None, :]

    # Take upper triangle without diagonal to avoid double counting and self pairs
    triu_indices = np.triu_indices(n, k=1)

    # Sum over time, average over number of pairs
    div_der = np.sum(np.abs(diffs_der[:, triu_indices[0], triu_indices[1]])) / (T * comb(n, 2))

    return div_der


def div_spectral(states):
    """
    Compute diversity scores of internal reservoir states' spectrum,
    matching the element-wise absolute difference method from div_metric_tests.

    Parameters:
        states (ndarray): Reservoir states, shape (T, n)

    Returns:
        div_spect (float): Spectrum based diversity score
    """
    cond_number = np.linalg.cond(states)

    # Inverted to avoid ill-conditioning numerical problems
    div_spect = 1./np.log(cond_number)

    return div_spect


def div_norm(states):
    """
    Compute diversity scores of internal reservoir states' rank,
    matching the element-wise absolute difference method from div_metric_tests.

    Parameters:
        states (ndarray): Reservoir states, shape (T, n)

    Returns:
        div_rank (float): Rank based diversity score
    """
    frobenius = np.linalg.norm(states, 'fro')**2
    two_norm = np.linalg.norm(states, 2)**2

    div_rank = frobenius / two_norm - 1.0

    return div_rank


def div_var_node(states):
    """
    Cross-Node Variation (D_CN).

        D_CN = (1/N) sum_{i=1}^{N} Var_t [r_i(t)]

    Temporal variance of each node's signed response, then averaged over nodes.
    Var is the population variance (1/T, numpy ddof=0).

    Parameters:
        states (ndarray): Reservoir states, shape (T, N)
    """
    return float(np.mean(np.var(states, axis=0, ddof=0)))


def div_var_state(states):
    """
    Cross-State Variation (D_CS).

        D_CS = (1/T) sum_t Var_i [|r_i(t)|]

    Snapshot variance of node response magnitudes |r_i(t)|, then averaged over
    time. Var is the population variance (1/N, numpy ddof=0).

    Parameters:
        states (ndarray): Reservoir states, shape (T, N)
    """
    return float(np.mean(np.var(np.abs(states), axis=1, ddof=0)))


def div_task(states, targets):
    """
    Task rank diversity (D_task).

        D_task = ||P_Y R||_F^2 / ||R||_F^2
        P_Y = Y Y^+

    R is the reservoir response (T, n) and Y is the target signal (T, m).
    Y^+ is the Moore-Penrose pseudoinverse. The T x T projector is never
    formed: P_Y R = Y (Y^+ R).

    Parameters:
        states (ndarray): Reservoir states R, shape (T, n)
        targets (ndarray): Target signal Y, shape (T, m)
    """
    R = np.asarray(states, dtype=float)
    Y = np.asarray(targets, dtype=float)
    if R.ndim != 2 or Y.ndim != 2:
        raise ValueError("states and targets must be 2-D arrays")
    if R.shape[0] != Y.shape[0]:
        T = min(R.shape[0], Y.shape[0])
        R = R[:T]
        Y = Y[:T]

    denom = np.linalg.norm(R, "fro") ** 2
    if denom == 0.0:
        return 0.0

    projected = Y @ (np.linalg.pinv(Y) @ R)
    return float(np.linalg.norm(projected, "fro") ** 2 / denom)


def div_metric_tests(states, targets=None, task_states=None):
    """
    Compute diversity scores of internal reservoir states.

    Parameters:
        states (ndarray): Reservoir states, shape (T, n)
        targets (ndarray, optional): Target signal Y, shape (T, m). If given,
            include D_task.
        task_states (ndarray, optional): Response matrix R to use for D_task
            when it must share a time axis with Y. Defaults to states.

    Returns:
        dict of float diversity scores
    """
    scores = {
        "div_pos": div_spatial(states),
        "div_der": div_rate(states),
        "div_spect": div_spectral(states),
        "div_rank": div_norm(states),
        "div_var_node": div_var_node(states),
        "div_var_state": div_var_state(states),
    }
    if targets is not None:
        scores["div_task"] = div_task(
            states if task_states is None else task_states,
            targets,
        )
    return scores


def consistency_analysis_sphering(x, y, max_cutoff=8000, alpha=1e-9):
    """ Based on the Appendix: Consistency Analysis walkthrough sent by Dr. Lymburn
        Parameters:
        ----------
        x: ndarray(N,L)
            States using initial state r0. L = time, N = number of nodes

        y: ndarray(N,L)
            States using different initial state r0 prime. L = time, N = number of nodes

        Returns:
        --------
        cap: float
            The consistency capacity of the system
        S: ndarray(N)
            Each node's consistency
    """

    def _transient_cutoff(u,v,eps=1e-3):
        """Calculate first potential cutoff for transients"""
        # Look at FFT High-Frequency Transients and Variance Analysis
        err = np.linalg.norm(u-v, axis=0)
        cutoffs = np.where(err < eps)[0]
        transient_cutoff = max_cutoff
        if len(cutoffs) > 0:
            transient_cutoff = min(cutoffs[0], max_cutoff)
        return transient_cutoff
    
    transient_cutoff = _transient_cutoff(x,y)
    
    # Cutoff transient states
    x = x[:,transient_cutoff:]
    y = y[:,transient_cutoff:]

    N, L = np.shape(x)

    # The next lines are the Consistency PDF attempt to get unit variance

    # Calculate Covariance Matrices
    # Transpose here to ensure N x N Covariance matrix
    Cxx = x @ x.T / L 
    Cyy = y @ y.T / L 

    # Save space by averaging the two
    C = (Cxx + Cyy) / 2.

    # Add regularization term 
    C = C + alpha * np.eye(N)

    # Compute SVD
    U, S, _ = np.linalg.svd(C)
    Qxx = U
    S_inv = np.diag(1. / np.sqrt(S))

    # Apply spherical transformation T_o
    # Multiplying by S_inv in the spherical transformation is equivalent 
    # to dividing by standard deviation (conjecture)
    T_o = Qxx @ S_inv @ Qxx.T
    x = T_o @ x
    y = T_o @ y

    # Calculate Cross-variance matrix
    Cxy = x @ y.T / L
    Css = (Cxy + Cxy.T) / 2.

    # Calculuate Consistency Capacity
    cap = np.abs(np.trace(Css)) / N

    return cap


def consistency_analysis_pearson(x, y, transient_cutoff=3000, alpha=1e-9):
    """ Based on the Echo State Paper - Pearson Correlation Coefficient Definition
        Parameters:
        ----------
        x: ndarray(N,L)
            States using initial state r0. L = time, N = number of nodes

        y: ndarray(N,L)
            States using different initial state r0 prime. L = time, N = number of nodes

        Returns:
        --------
        cap: float
            The consistency capacity of the system
        S: ndarray(N)
            Each node's consistency
    """

    def _rescale_states(v):
        """Rescale states to have 0 mean and unit variance"""
        mean = np.mean(v, axis=1)
        std = np.std(v, axis=1)
        v_mod = v.T - mean
        valid = std > alpha
        v_mod[:,valid] = v_mod[:,valid] / std[valid]
        return v_mod.T
    
    # Cutoff transient states
    x = x[:,transient_cutoff:]
    y = y[:,transient_cutoff:]

    N, L = np.shape(x)

    # Rescale x and y
    x = _rescale_states(x)
    y = _rescale_states(y)

    cap = np.sum(x*y) / (N*L)

    return cap