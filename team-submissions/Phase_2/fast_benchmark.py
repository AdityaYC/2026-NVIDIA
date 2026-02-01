"""
Fast GPU Benchmark for Large N

Optimizations:
1. Reduced Trotter steps (5 instead of 30)
2. Reduced shots (500 instead of 1000)
3. Uses tensornet backend for tensor network simulation (faster for larger N)
"""

import sys
import os
import time
import json
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    print("CUDA-Q not available")
    sys.exit(1)

from labs_solver.energy import compute_labs_energy, bitstring_to_sequence
from labs_solver.mts import MemeticTabuSearch
from labs_solver.quantum import get_interactions, trotterized_circuit, compute_theta_simple


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_fast_quantum(N: int, n_steps: int = 5, shots: int = 500):
    """Run quantum circuit with reduced steps for speed."""
    T = 1.0
    dt = T / n_steps
    G2, G4 = get_interactions(N)
    
    # Compute thetas
    thetas = []
    for step in range(1, n_steps + 1):
        t = step * dt
        theta_val = compute_theta_simple(t, dt, T, N)
        thetas.append(theta_val)
    
    print(f"  G2 terms: {len(G2)}, G4 terms: {len(G4)}")
    print(f"  Trotter steps: {n_steps}")
    
    start = time.time()
    counts = cudaq.sample(trotterized_circuit, N, G2, G4, n_steps, dt, T, thetas, shots_count=shots)
    quantum_time = time.time() - start
    
    # Analyze
    counts_dict = dict(counts.items())
    energies = []
    for bitstring, _ in sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)[:20]:
        seq = bitstring_to_sequence(bitstring)
        energies.append(compute_labs_energy(seq))
    
    return {
        "quantum_time_seconds": quantum_time,
        "n_steps": n_steps,
        "shots": shots,
        "unique_samples": len(counts_dict),
        "min_energy": min(energies) if energies else -1,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--steps", type=int, default=5, help="Trotter steps (fewer = faster)")
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--backend", type=str, default="nvidia", choices=["nvidia", "tensornet", "nvidia-mgpu"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"FAST BENCHMARK: N={args.n}, Steps={args.steps}, Backend={args.backend}")
    print(f"{'='*60}\n")
    
    # Set target
    try:
        cudaq.set_target(args.backend)
        print(f"Target: {args.backend}")
    except Exception as e:
        print(f"Warning: {args.backend} not available, using default")
        cudaq.reset_target()
    
    results = {
        "metadata": {
            "N": args.n,
            "backend": args.backend,
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    # Quantum benchmark
    print("\n[1/2] Running Quantum Circuit...")
    quantum_results = run_fast_quantum(N=args.n, n_steps=args.steps, shots=args.shots)
    results["quantum"] = quantum_results
    print(f"  Time: {quantum_results['quantum_time_seconds']:.2f}s")
    print(f"  Min Energy: {quantum_results['min_energy']}")
    
    # MTS benchmark
    print("\n[2/2] Running MTS...")
    np.random.seed(42)
    mts = MemeticTabuSearch(N=args.n, population_size=20)
    start = time.time()
    _, best_energy, _ = mts.run(max_iterations=30)
    mts_time = time.time() - start
    results["mts"] = {"time": mts_time, "best_energy": best_energy}
    print(f"  Time: {mts_time:.2f}s")
    print(f"  Best Energy: {best_energy}")
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nSaved to: {args.output}")
    else:
        print(json.dumps(results, indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
