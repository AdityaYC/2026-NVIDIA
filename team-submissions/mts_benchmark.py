"""
MTS-Only Benchmark for Large N (No CUDA-Q Required)
Runs Memetic Tabu Search for N=30, 40, 50, etc.
"""

import sys
import os
import time
import json
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labs_solver.energy import compute_labs_energy, get_known_optimum
from labs_solver.mts import MemeticTabuSearch


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_mts_benchmark(N: int, iterations: int = 50, population_size: int = 20):
    """Run MTS benchmark for given N."""
    print(f"\n{'='*60}")
    print(f"MTS Benchmark: N={N}")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    mts = MemeticTabuSearch(N=N, population_size=population_size)
    
    start_time = time.time()
    best_seq, best_energy, final_pop = mts.run(max_iterations=iterations, verbose=True)
    elapsed = time.time() - start_time
    
    # Get known optimum for comparison
    known_opt = get_known_optimum(N)
    
    # Calculate population stats
    pop_energies = [compute_labs_energy(s) for s in final_pop]
    
    results = {
        "N": N,
        "time_seconds": elapsed,
        "best_energy": int(best_energy),
        "known_optimum": known_opt,
        "gap": int(best_energy) - known_opt if known_opt else None,
        "mean_pop_energy": float(np.mean(pop_energies)),
        "std_pop_energy": float(np.std(pop_energies)),
        "iterations": iterations,
        "population_size": population_size,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n  Time: {elapsed:.2f}s")
    print(f"  Best Energy: {best_energy}")
    if known_opt:
        print(f"  Known Optimum: {known_opt}")
        print(f"  Gap: {int(best_energy) - known_opt}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MTS-Only Benchmark (No CUDA-Q)")
    parser.add_argument("--n", type=int, nargs="+", default=[30, 40], help="Problem size(s)")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    all_results = {}
    
    for N in args.n:
        results = run_mts_benchmark(N, iterations=args.iterations)
        all_results[f"N={N}"] = results
    
    print(f"\n{'='*60}")
    print("ALL BENCHMARKS COMPLETE")
    print(f"{'='*60}\n")
    
    # Summary table
    print(f"{'N':>5} | {'Time':>10} | {'Best E':>8} | {'Optimum':>8} | {'Gap':>6}")
    print("-" * 50)
    for key, res in all_results.items():
        opt = res['known_optimum'] or 'N/A'
        gap = res['gap'] if res['gap'] is not None else 'N/A'
        print(f"{res['N']:>5} | {res['time_seconds']:>8.2f}s | {res['best_energy']:>8} | {opt:>8} | {gap:>6}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
