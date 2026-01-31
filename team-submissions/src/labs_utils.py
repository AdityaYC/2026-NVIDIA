import numpy as np
from math import floor, sin, cos, pi

def compute_labs_energy(sequence: np.ndarray) -> float:
    """
    Compute the LABS objective function for a given binary sequence.
    
    E(s) = sum_{k=1}^{N-1} C_k^2
    where C_k = sum_{i=1}^{N-k} s_i * s_{i+k}
    
    Args:
        sequence: Binary sequence of +1/-1 values
    
    Returns:
        Energy value (lower is better)
    """
    N = len(sequence)
    energy = 0
    
    for k in range(1, N):
        C_k = 0
        for i in range(N - k):
            C_k += sequence[i] * sequence[i + k]
        energy += C_k ** 2
    
    return float(energy)

def bitstring_to_sequence(bitstring: str) -> np.ndarray:
    """Convert binary bitstring to +1/-1 sequence."""
    return np.array([1 if bit == '0' else -1 for bit in bitstring])

def sequence_to_bitstring(sequence: np.ndarray) -> str:
    """Convert +1/-1 sequence to binary bitstring."""
    return ''.join(['0' if s == 1 else '1' for s in sequence])

def get_interactions(N: int):
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists of ints.
    
    Args:
        N (int): Sequence length.
        
    Returns:
        G2: List of lists containing two body term indices
        G4: List of lists containing four body term indices
    """
    G2 = []
    G4 = []
    
    # --- Two-body terms (G2) ---
    for i in range(N - 2):
        k_limit = floor((N - (i + 1)) / 2)
        for k in range(1, k_limit + 1):
            G2.append([i, i + k])

    # --- Four-body terms (G4) ---
    for i in range(N - 3):
        t_limit = floor((N - (i + 1) - 1) / 2)
        for t in range(1, t_limit + 1):
            k_limit = N - (i + 1) - t
            for k in range(t + 1, k_limit + 1):
                G4.append([i, i + t, i + k, i + k + t])
                
    return G2, G4

def compute_topology_overlaps(G2, G4):
    """
    Computes the topological invariants I_22, I_24, I_44 based on set overlaps.
    I_alpha_beta counts how many sets share IDENTICAL elements.
    """
    def count_matches(list_a, list_b):
        matches = 0
        set_b = set(tuple(sorted(x)) for x in list_b)
        for item in list_a:
            if tuple(sorted(item)) in set_b:
                matches += 1
        return matches

    I_22 = count_matches(G2, G2)
    I_44 = count_matches(G4, G4)
    I_24 = 0 
    
    return {'22': I_22, '44': I_44, '24': I_24}

def compute_theta(t, dt, total_time, N, G2, G4):
    """
    Computes theta(t) using the analytical solutions for Gamma1 and Gamma2.
    """
    if total_time == 0:
        return 0.0

    arg = (pi * t) / (2.0 * total_time)
    
    lam = sin(arg)**2
    # Derivative: (pi/2T) * sin(2 * arg) -> sin(pi * t / T) (Fixed from original comment which was simplified)
    # Original: lam_dot = (pi / (2.0 * total_time)) * sin((pi * t) / total_time)
    # Check derivative of sin^2(pi*t/2T) = 2*sin(..)*cos(..)*(pi/2T) = sin(pi*t/T)*(pi/2T)
    lam_dot = (pi / (2.0 * total_time)) * sin((pi * t) / total_time)
    
    # Gamma 1
    term_g1_2 = 16 * len(G2) * 2
    term_g1_4 = 64 * len(G4) * 4
    Gamma1 = term_g1_2 + term_g1_4
    
    # Gamma 2
    sum_G2 = len(G2) * (lam**2 * 2)
    sum_G4 = 4 * len(G4) * (16 * (lam**2) + 8 * ((1 - lam)**2))
    
    I_vals = compute_topology_overlaps(G2, G4)
    term_topology = 4 * (lam**2) * (4 * I_vals['24'] + I_vals['22']) + 64 * (lam**2) * I_vals['44']
    
    Gamma2 = -256 * (term_topology + sum_G2 + sum_G4)

    if abs(Gamma2) < 1e-12:
        alpha = 0.0
    else:
        alpha = - Gamma1 / Gamma2
        
    return dt * alpha * lam_dot
