"""
CUDA-Q Quantum kernels for Counterdiabatic LABS optimization.

Implements the digitized counterdiabatic quantum algorithm using CUDA-Q
as described in:
"Scaling advantage with quantum-enhanced memetic tabu search for LABS"
"""

import cudaq
import numpy as np
from math import floor
from typing import List, Tuple, Dict


# ----------------------------------------------------------------------------
# Two-Qubit Kernels (R_YZ and R_ZY)
# ----------------------------------------------------------------------------
@cudaq.kernel
def r_yz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """
    R_YZ(theta) = exp(-i * theta/2 * Y_0 * Z_1)
    
    Strategy:
    1. Rotate q0 from Y basis to Z basis using Rx(pi/2)
    2. Apply controlled-Z rotation (via CNOT-RZ-CNOT)
    3. Rotate q0 back with Rx(-pi/2)
    """
    # Basis change: Y -> Z requires Rx(pi/2)
    rx(1.5707963267948966, q0)
    # Controlled-Z rotation
    x.ctrl(q0, q1)
    rz(theta, q1)
    x.ctrl(q0, q1)
    # Inverse basis change
    rx(-1.5707963267948966, q0)


@cudaq.kernel
def r_zy(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """
    R_ZY(theta) = exp(-i * theta/2 * Z_0 * Y_1)
    """
    # Basis change for q1
    rx(1.5707963267948966, q1)
    x.ctrl(q0, q1)
    rz(theta, q1)
    x.ctrl(q0, q1)
    rx(-1.5707963267948966, q1)


# ----------------------------------------------------------------------------
# Four-Qubit Kernels (R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY)
# ----------------------------------------------------------------------------
@cudaq.kernel
def r_yzzz(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
    """R_YZZZ term: Y on q0; Z on q1, q2, q3"""
    # Basis change Y->Z on q0
    rx(1.5707963267948966, q0)
    # Compute parity chain
    x.ctrl(q0, q1)
    x.ctrl(q1, q2)
    x.ctrl(q2, q3)
    # Apply rotation
    rz(theta, q3)
    # Uncompute parity
    x.ctrl(q2, q3)
    x.ctrl(q1, q2)
    x.ctrl(q0, q1)
    # Inverse basis change
    rx(-1.5707963267948966, q0)


@cudaq.kernel
def r_zyzz(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
    """R_ZYZZ term: Y on q1"""
    rx(1.5707963267948966, q1)
    x.ctrl(q0, q1)
    x.ctrl(q1, q2)
    x.ctrl(q2, q3)
    rz(theta, q3)
    x.ctrl(q2, q3)
    x.ctrl(q1, q2)
    x.ctrl(q0, q1)
    rx(-1.5707963267948966, q1)


@cudaq.kernel
def r_zzyz(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
    """R_ZZYZ term: Y on q2"""
    rx(1.5707963267948966, q2)
    x.ctrl(q0, q1)
    x.ctrl(q1, q2)
    x.ctrl(q2, q3)
    rz(theta, q3)
    x.ctrl(q2, q3)
    x.ctrl(q1, q2)
    x.ctrl(q0, q1)
    rx(-1.5707963267948966, q2)


@cudaq.kernel
def r_zzzy(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
    """R_ZZZY term: Y on q3"""
    rx(1.5707963267948966, q3)
    x.ctrl(q0, q1)
    x.ctrl(q1, q2)
    x.ctrl(q2, q3)
    rz(theta, q3)
    x.ctrl(q2, q3)
    x.ctrl(q1, q2)
    x.ctrl(q0, q1)
    rx(-1.5707963267948966, q3)


# ----------------------------------------------------------------------------
# Interaction Index Generation
# ----------------------------------------------------------------------------
def get_interactions(N: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists of ints.
    
    Args:
        N: Sequence length.
        
    Returns:
        G2: List of lists containing two body term indices
        G4: List of lists containing four body term indices
    """
    G2 = []
    G4 = []
    
    # --- Two-body terms (G2) ---
    # Equation outer loop: i from 1 to N-2
    # Python outer loop:   i from 0 to N-3 (range stops before N-2)
    for i in range(N - 2):
        # The equation limit for k is floor((N - i_1based) / 2)
        # Since our loop 'i' is 0-based, i_1based is (i + 1)
        k_limit = floor((N - (i + 1)) / 2)
        
        # k goes from 1 to k_limit (inclusive)
        for k in range(1, k_limit + 1):
            # Indices are i and i+k
            G2.append([i, i + k])

    # --- Four-body terms (G4) ---
    # Equation outer loop: i from 1 to N-3
    # Python outer loop:   i from 0 to N-4
    for i in range(N - 3):
        # Limit for t is floor((N - i_1based - 1) / 2)
        t_limit = floor((N - (i + 1) - 1) / 2)
        
        # t goes from 1 to t_limit (inclusive)
        for t in range(1, t_limit + 1):
            # Limit for k is N - i_1based - t
            k_limit = N - (i + 1) - t
            
            # k goes from t+1 to k_limit (inclusive)
            for k in range(t + 1, k_limit + 1):
                # Indices are i, i+t, i+k, i+k+t
                G4.append([i, i + t, i + k, i + k + t])
                
    return G2, G4


# ----------------------------------------------------------------------------
# Theta Calculation (requires labs_utils module)
# ----------------------------------------------------------------------------
def compute_theta_simple(t: float, dt: float, T: float, N: int) -> float:
    """
    Simplified theta computation for testing without labs_utils.
    Uses a simple sinusoidal schedule.
    
    Args:
        t: Current time
        dt: Time step
        T: Total evolution time
        N: Problem size
    
    Returns:
        Theta value for this time step
    """
    # Simple sinusoidal annealing schedule
    lambda_t = np.sin(np.pi * t / (2 * T)) ** 2
    d_lambda = np.pi / T * np.sin(np.pi * t / T) / 2
    
    # Simplified alpha coefficient
    alpha = 0.5 * d_lambda / (1 + lambda_t + 1e-10)
    
    return alpha * dt


# ----------------------------------------------------------------------------
# Trotterized Circuit
# ----------------------------------------------------------------------------
@cudaq.kernel
def trotterized_circuit(N: int, G2: list[list[int]], G4: list[list[int]], 
                        steps: int, dt: float, T: float, thetas: list[float]):
    """
    Full Trotterized circuit implementing Equation B3 from the paper.
    Applies counteradiabatic optimization for LABS problem.
    
    Args:
        N: Number of qubits (sequence length)
        G2: Two-body interaction indices
        G4: Four-body interaction indices
        steps: Number of Trotter steps
        dt: Time step size
        T: Total evolution time
        thetas: Pre-computed theta values for each step
    """
    reg = cudaq.qvector(N)
    
    # Initialize in |+> state (ground state of H_i = sum of X)
    h(reg)
    
    # Apply Trotter steps
    for step in range(steps):
        theta = thetas[step]
        
        # Apply 2-body terms (G2)
        for i in range(len(G2)):
            idx0 = G2[i][0]
            idx1 = G2[i][1]
            # R_YZ and R_ZY from Equation B3
            r_yz(reg[idx0], reg[idx1], 4.0 * theta)
            r_zy(reg[idx0], reg[idx1], 4.0 * theta)
        
        # Apply 4-body terms (G4)
        for i in range(len(G4)):
            idx0 = G4[i][0]
            idx1 = G4[i][1]
            idx2 = G4[i][2]
            idx3 = G4[i][3]
            # All 4 permutations from Equation B3
            r_yzzz(reg[idx0], reg[idx1], reg[idx2], reg[idx3], 8.0 * theta)
            r_zyzz(reg[idx0], reg[idx1], reg[idx2], reg[idx3], 8.0 * theta)
            r_zzyz(reg[idx0], reg[idx1], reg[idx2], reg[idx3], 8.0 * theta)
            r_zzzy(reg[idx0], reg[idx1], reg[idx2], reg[idx3], 8.0 * theta)


# ----------------------------------------------------------------------------
# High-level Quantum Sampling Function
# ----------------------------------------------------------------------------
def run_quantum_sampling(N: int, n_steps: int = 30, T: float = 1.0, 
                         shots: int = 1000, use_simple_theta: bool = True,
                         labs_utils=None) -> Dict[str, int]:
    """
    Run the quantum counterdiabatic circuit and return measurement counts.
    
    Args:
        N: Problem size (number of qubits)
        n_steps: Number of Trotter steps
        T: Total evolution time
        shots: Number of shots to sample
        use_simple_theta: Use simplified theta (True) or labs_utils (False)
        labs_utils: Optional labs_utils module for proper theta calculation
    
    Returns:
        Dictionary mapping bitstrings to counts
    """
    dt = T / n_steps
    G2, G4 = get_interactions(N)
    
    # Compute thetas for each step
    thetas = []
    for step in range(1, n_steps + 1):
        t = step * dt
        if use_simple_theta or labs_utils is None:
            theta_val = compute_theta_simple(t, dt, T, N)
        else:
            theta_val = labs_utils.compute_theta(t, dt, T, N, G2, G4)
        thetas.append(theta_val)
    
    # Sample the circuit
    counts = cudaq.sample(trotterized_circuit, N, G2, G4, n_steps, dt, T, thetas, 
                          shots_count=shots)
    
    return dict(counts.items())


def quantum_population_from_samples(counts: Dict[str, int], 
                                   population_size: int = 20) -> List[np.ndarray]:
    """
    Convert quantum sampling results to a population of sequences.
    
    Args:
        counts: Dictionary of bitstring -> count from quantum sampling
        population_size: Number of sequences to return
    
    Returns:
        List of numpy arrays representing sequences
    """
    from .energy import bitstring_to_sequence
    
    # Sort by count (most frequent first)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    population = []
    for bitstring, _ in sorted_counts:
        seq = bitstring_to_sequence(bitstring)
        population.append(seq)
        if len(population) >= population_size:
            break
    
    return population
