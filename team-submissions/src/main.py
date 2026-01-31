import cudaq
import argparse
import sys
import numpy as np
from src.labs_utils import get_interactions, compute_theta, bitstring_to_sequence, compute_labs_energy
from src.quantum_kernels import trotterized_circuit
from src.mts import MemeticTabuSearch

def main():
    parser = argparse.ArgumentParser(description="Quantum Enhanced LABS Solver")
    parser.add_argument("--N", type=int, default=20, help="Sequence length")
    parser.add_argument("--steps", type=int, default=30, help="Number of Trotter steps")
    parser.add_argument("--shots", type=int, default=1000, help="Number of quantum shots")
    parser.add_argument("--pop-size", type=int, default=10, help="MTS population size")
    parser.add_argument("--mts-iter", type=int, default=50, help="MTS iterations")
    parser.add_argument("--gpu-backend", action="store_true", help="Use NVIDIA GPU backend")
    args = parser.parse_args()

    N = args.N
    n_steps = args.steps
    T = 1.0
    dt = T / n_steps
    
    if args.gpu_backend:
        try:
            cudaq.set_target("nvidia")
            print("Using NVIDIA GPU backend.")
        except Exception as e:
            print(f"Warning: Could not set NVIDIA target: {e}. Falling back to default.")

    print(f"Initializing for N={N}...")
    
    # 1. Pre-compute interactions and scheduler
    G2, G4 = get_interactions(N)
    
    thetas = []
    for step in range(1, n_steps + 1):
        t = step * dt
        theta_val = compute_theta(t, dt, T, N, G2, G4)
        thetas.append(theta_val)
        
    # 2. Run Quantum Circuit
    print(f"Running Quantum Circuit (Steps={n_steps}, Shots={args.shots})...")
    counts = cudaq.sample(trotterized_circuit, N, G2, G4, n_steps, dt, T, thetas, shots_count=args.shots)
    
    # 3. Process Quantum Results
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    quantum_population = []
    
    for bitstring, count in sorted_counts:
        seq = bitstring_to_sequence(bitstring)
        # Verify length
        if len(seq) == N:
             quantum_population.append(seq)
        if len(quantum_population) >= args.pop_size:
            break
            
    print(f"Generated {len(quantum_population)} sequences from Quantum Circuit.")
    if quantum_population:
        best_q = min([compute_labs_energy(s) for s in quantum_population])
        print(f"Best Quantum Candidate Energy (Before MTS): {best_q}")
    else:
        print("Warning: No valid sequences generated from quantum circuit.")
        
    # 4. Run Hybrid MTS
    print("\\nStarting Quantum-Enhanced MTS...")
    qe_mts = MemeticTabuSearch(N, population_size=args.pop_size)
    best_seq, best_energy, _ = qe_mts.run(
        max_iterations=args.mts_iter,
        initial_sequences=quantum_population
    )
    
    print(f"QE-MTS Best Energy: {best_energy}")
    print(f"Best Sequence: {best_seq}")
    
if __name__ == "__main__":
    main()
