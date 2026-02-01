"""
GPU Benchmark Script for LABS Quantum-Enhanced Optimization

This script benchmarks the CUDA-Q quantum circuit and GPU-accelerated MTS
on NVIDIA GPUs via Brev platform.

Usage:
    # CPU mode (for validation)
    python run_gpu_benchmark.py --mode cpu --n 20
    
    # GPU mode (requires NVIDIA GPU)
    python run_gpu_benchmark.py --mode gpu --n 40
    
    # Dry run (no actual computation)
    python run_gpu_benchmark.py --dry-run
"""

import argparse
import time
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Conditionally import cudaq (may not be available on all systems)
try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    print("Warning: CUDA-Q not available. Running in limited mode.")

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labs_solver.energy import compute_labs_energy, bitstring_to_sequence
from labs_solver.mts import MemeticTabuSearch


def set_target(mode: str) -> str:
    """
    Set the CUDA-Q target based on mode.
    
    Args:
        mode: 'cpu' or 'gpu'
    
    Returns:
        Target name that was set
    """
    if not CUDAQ_AVAILABLE:
        return "none"
    
    if mode == "gpu":
        # Try nvidia target first, fall back to nvidia-mqpu
        try:
            cudaq.set_target("nvidia")
            return "nvidia"
        except Exception:
            try:
                cudaq.set_target("nvidia-mgpu")
                return "nvidia-mgpu"
            except Exception:
                print("Warning: GPU target not available, using default")
                return "default"
    else:
        # CPU simulation
        cudaq.reset_target()
        return "default"


def run_quantum_benchmark(N: int, n_steps: int = 30, T: float = 1.0, 
                          shots: int = 1000) -> Dict:
    """
    Run the quantum circuit benchmark.
    
    Args:
        N: Problem size
        n_steps: Trotter steps
        T: Total evolution time
        shots: Number of measurement shots
    
    Returns:
        Dictionary with benchmark results
    """
    from labs_solver.quantum import get_interactions, run_quantum_sampling
    
    # Time the circuit execution
    start_time = time.time()
    
    counts = run_quantum_sampling(
        N=N, 
        n_steps=n_steps, 
        T=T, 
        shots=shots,
        use_simple_theta=True
    )
    
    quantum_time = time.time() - start_time
    
    # Analyze results
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate energies of top samples
    energies = []
    for bitstring, count in sorted_counts[:20]:
        seq = bitstring_to_sequence(bitstring)
        energy = compute_labs_energy(seq)
        energies.append(energy)
    
    return {
        "N": N,
        "n_steps": n_steps,
        "shots": shots,
        "quantum_time_seconds": quantum_time,
        "unique_samples": len(counts),
        "min_energy": min(energies) if energies else -1,
        "mean_energy_top20": np.mean(energies) if energies else -1,
        "best_bitstring": sorted_counts[0][0] if sorted_counts else "",
    }


def run_mts_benchmark(N: int, population_size: int = 20, 
                      iterations: int = 50,
                      initial_sequences: List[np.ndarray] = None) -> Dict:
    """
    Run the MTS benchmark.
    
    Args:
        N: Problem size
        population_size: MTS population size
        iterations: Number of MTS iterations
        initial_sequences: Optional quantum seed population
    
    Returns:
        Dictionary with benchmark results
    """
    start_time = time.time()
    
    mts = MemeticTabuSearch(N=N, population_size=population_size)
    best_seq, best_energy, final_pop = mts.run(
        max_iterations=iterations,
        initial_sequences=initial_sequences
    )
    
    mts_time = time.time() - start_time
    
    # Calculate population statistics
    pop_energies = [compute_labs_energy(s) for s in final_pop]
    
    return {
        "N": N,
        "population_size": population_size,
        "iterations": iterations,
        "seeded": initial_sequences is not None,
        "mts_time_seconds": mts_time,
        "best_energy": best_energy,
        "mean_population_energy": np.mean(pop_energies),
        "std_population_energy": np.std(pop_energies),
    }


def run_full_benchmark(N: int, mode: str = "cpu") -> Dict:
    """
    Run complete quantum-enhanced MTS benchmark.
    
    Args:
        N: Problem size
        mode: 'cpu' or 'gpu'
    
    Returns:
        Complete benchmark results
    """
    print(f"\n{'='*60}")
    print(f"LABS Benchmark: N={N}, Mode={mode.upper()}")
    print(f"{'='*60}\n")
    
    target = set_target(mode)
    print(f"CUDA-Q Target: {target}")
    
    results = {
        "metadata": {
            "N": N,
            "mode": mode,
            "target": target,
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    # Run quantum circuit
    if CUDAQ_AVAILABLE:
        print("\n[1/3] Running Quantum Circuit...")
        quantum_results = run_quantum_benchmark(N=N)
        results["quantum"] = quantum_results
        print(f"  Time: {quantum_results['quantum_time_seconds']:.2f}s")
        print(f"  Min Energy: {quantum_results['min_energy']}")
        print(f"  Unique Samples: {quantum_results['unique_samples']}")
        
        # Extract population for seeding MTS
        from labs_solver.quantum import quantum_population_from_samples, run_quantum_sampling
        counts = run_quantum_sampling(N=N, n_steps=30, shots=1000, use_simple_theta=True)
        quantum_pop = quantum_population_from_samples(counts, population_size=20)
    else:
        print("\n[1/3] Skipping Quantum Circuit (CUDA-Q not available)")
        results["quantum"] = {"error": "CUDA-Q not available"}
        quantum_pop = None
    
    # Run MTS with random initialization
    print("\n[2/3] Running Classical MTS (Random Init)...")
    np.random.seed(42)
    classical_results = run_mts_benchmark(N=N, initial_sequences=None)
    results["classical_mts"] = classical_results
    print(f"  Time: {classical_results['mts_time_seconds']:.2f}s")
    print(f"  Best Energy: {classical_results['best_energy']}")
    
    # Run MTS with quantum seeding
    if quantum_pop is not None:
        print("\n[3/3] Running Quantum-Enhanced MTS...")
        np.random.seed(42)
        qe_results = run_mts_benchmark(N=N, initial_sequences=quantum_pop)
        results["quantum_enhanced_mts"] = qe_results
        print(f"  Time: {qe_results['mts_time_seconds']:.2f}s")
        print(f"  Best Energy: {qe_results['best_energy']}")
        
        # Calculate improvement
        improvement = classical_results['best_energy'] - qe_results['best_energy']
        results["improvement"] = improvement
        print(f"\n  Improvement: {improvement}")
    else:
        print("\n[3/3] Skipping QE-MTS (no quantum population)")
        results["quantum_enhanced_mts"] = {"error": "No quantum population"}
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="LABS GPU Benchmark Script"
    )
    parser.add_argument(
        "--mode", 
        choices=["cpu", "gpu"], 
        default="cpu",
        help="Execution mode (cpu or gpu)"
    )
    parser.add_argument(
        "--n", 
        type=int, 
        default=20,
        help="Problem size N"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run configuration:")
        print(f"  Mode: {args.mode}")
        print(f"  N: {args.n}")
        print(f"  Output: {args.output or 'stdout'}")
        print(f"  CUDA-Q Available: {CUDAQ_AVAILABLE}")
        return
    
    # Run benchmark
    results = run_full_benchmark(N=args.n, mode=args.mode)
    
    # Output results
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nResults:")
        print(json.dumps(results, indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
