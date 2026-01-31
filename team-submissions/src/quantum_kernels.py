import cudaq
import numpy as np

# ----------------------------------------------------------------------------
# Two-Qubit Kernels (R_YZ and R_ZY)
# ----------------------------------------------------------------------------
@cudaq.kernel
def r_yz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """
    R_YZ(theta) = exp(-i * theta/2 * Y_0 * Z_1)
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
# Four-Qubit Kernels (R_YZZZ, R_ZYZZ, etc.)
# ----------------------------------------------------------------------------
@cudaq.kernel
def r_yzzz(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
    """ R_YZZZ term: Y on q0; Z on q1, q2, q3 """
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
    """ R_ZYZZ term: Y on q1 """
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
    """ R_ZZYZ term: Y on q2 """
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
    """ R_ZZZY term: Y on q3 """
    rx(1.5707963267948966, q3)
    x.ctrl(q0, q1)
    x.ctrl(q1, q2)
    x.ctrl(q2, q3)
    rz(theta, q3)
    x.ctrl(q2, q3)
    x.ctrl(q1, q2)
    x.ctrl(q0, q1)
    rx(-1.5707963267948966, q3)


@cudaq.kernel
def trotterized_circuit(N: int, G2: list[list[int]], G4: list[list[int]], steps: int, dt: float, T: float, thetas: list[float]):
    """
    Full Trotterized circuit implementing Equation B3 from the paper.
    Applies counteradiabatic optimization for LABS problem.
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
