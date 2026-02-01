"""
GPU-Accelerated Memetic Tabu Search using CuPy.

This module provides GPU-accelerated versions of the MTS algorithm
components for running on NVIDIA GPUs.

Requirements:
    pip install cupy-cuda12x  # for CUDA 12.x
    # or
    pip install cupy-cuda11x  # for CUDA 11.x
"""

import numpy as np
from typing import List, Tuple, Optional
import random

# Try to import CuPy, fall back to NumPy if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np  # Fallback to numpy
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. Using NumPy fallback (no GPU acceleration).")


def compute_labs_energy_gpu(sequence: cp.ndarray) -> float:
    """
    GPU-accelerated LABS energy calculation using CuPy.
    
    Uses vectorized operations for each correlation C_k.
    
    Args:
        sequence: Binary sequence of +1/-1 values on GPU
    
    Returns:
        Energy value (lower is better)
    """
    N = len(sequence)
    energy = 0.0
    
    for k in range(1, N):
        # Vectorized correlation calculation
        C_k = cp.sum(sequence[:N-k] * sequence[k:])
        energy += float(C_k ** 2)
    
    return energy


def compute_labs_energy_batch(sequences: cp.ndarray) -> cp.ndarray:
    """
    Compute LABS energy for a batch of sequences in parallel on GPU.
    
    Args:
        sequences: 2D array of shape (batch_size, N) with ±1 values
    
    Returns:
        1D array of energies for each sequence
    """
    batch_size, N = sequences.shape
    energies = cp.zeros(batch_size, dtype=cp.float32)
    
    for k in range(1, N):
        # Vectorized across entire batch
        C_k = cp.sum(sequences[:, :N-k] * sequences[:, k:], axis=1)
        energies += C_k ** 2
    
    return energies


def batch_neighbor_evaluation(sequence: cp.ndarray) -> Tuple[cp.ndarray, int]:
    """
    Evaluate all single-flip neighbors in parallel on GPU.
    
    Instead of evaluating one flip at a time, this creates all N
    neighbors and evaluates them simultaneously.
    
    Args:
        sequence: Current sequence on GPU
    
    Returns:
        Tuple of (neighbor energies array, best flip position)
    """
    N = len(sequence)
    
    # Create all neighbors by tiling and flipping
    # Shape: (N, N) where row i has bit i flipped
    neighbors = cp.tile(sequence, (N, 1))
    flip_indices = cp.arange(N)
    neighbors[flip_indices, flip_indices] *= -1
    
    # Compute all energies in parallel
    energies = compute_labs_energy_batch(neighbors)
    
    # Find best neighbor
    best_idx = int(cp.argmin(energies))
    
    return energies, best_idx


class TabuSearchGPU:
    """
    GPU-accelerated Tabu Search using CuPy.
    
    Uses batch neighbor evaluation for parallel energy computation.
    """
    
    def __init__(self, tabu_tenure: int = 10):
        """
        Initialize TabuSearchGPU.
        
        Args:
            tabu_tenure: Number of iterations a move stays in the tabu list
        """
        self.tabu_tenure = tabu_tenure
    
    def search(self, sequence: np.ndarray, max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Perform GPU-accelerated tabu search.
        
        Args:
            sequence: Initial binary sequence (numpy array)
            max_iterations: Maximum number of iterations
        
        Returns:
            Tuple of (best sequence found as numpy, its energy)
        """
        N = len(sequence)
        
        # Transfer to GPU
        current = cp.asarray(sequence.copy())
        current_energy = compute_labs_energy_gpu(current)
        
        best = current.copy()
        best_energy = current_energy
        
        tabu_list = []
        
        for iteration in range(max_iterations):
            # Batch evaluate all neighbors
            neighbor_energies, _ = batch_neighbor_evaluation(current)
            
            # Apply tabu mask (set tabu positions to high energy)
            tabu_mask = cp.zeros(N, dtype=bool)
            for pos in tabu_list:
                tabu_mask[pos] = True
            neighbor_energies[tabu_mask] = float('inf')
            
            # Find best non-tabu neighbor
            best_flip_pos = int(cp.argmin(neighbor_energies))
            best_neighbor_energy = float(neighbor_energies[best_flip_pos])
            
            if best_neighbor_energy == float('inf'):
                break  # All moves are tabu
            
            # Make the move
            current[best_flip_pos] *= -1
            current_energy = best_neighbor_energy
            
            # Update tabu list
            tabu_list.append(best_flip_pos)
            if len(tabu_list) > self.tabu_tenure:
                tabu_list.pop(0)
            
            # Update global best
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        
        # Transfer back to CPU
        return cp.asnumpy(best), best_energy


class MemeticTabuSearchGPU:
    """
    GPU-accelerated Memetic Tabu Search.
    
    Uses CuPy for GPU acceleration of:
    - Population energy evaluation
    - Tabu search local optimization
    """
    
    def __init__(self, N: int, population_size: int = 20, p_mutate: float = 0.1):
        """
        Initialize MemeticTabuSearchGPU.
        
        Args:
            N: Sequence length
            population_size: Size of the population
            p_mutate: Mutation probability
        """
        self.N = N
        self.population_size = population_size
        self.p_mutate = p_mutate
        self.tabu_search = TabuSearchGPU(tabu_tenure=10)
    
    def initialize_population(self, initial_sequences: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Initialize population randomly or with provided sequences.
        """
        if initial_sequences is not None and len(initial_sequences) >= self.population_size:
            return initial_sequences[:self.population_size]
        
        population = []
        if initial_sequences:
            population.extend(initial_sequences)
        
        while len(population) < self.population_size:
            seq = np.random.choice([-1, 1], size=self.N).astype(np.int32)
            population.append(seq)
        
        return population
    
    def evaluate_population_gpu(self, population: List[np.ndarray]) -> cp.ndarray:
        """
        Evaluate entire population energy on GPU in parallel.
        """
        # Stack population into 2D array
        pop_array = cp.asarray(np.stack(population))
        return compute_labs_energy_batch(pop_array)
    
    def run(self, max_iterations: int = 100, 
            initial_sequences: Optional[List[np.ndarray]] = None,
            verbose: bool = False) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """
        Run GPU-accelerated Memetic Tabu Search.
        
        Args:
            max_iterations: Number of MTS iterations
            initial_sequences: Optional seed population
            verbose: Print progress
        
        Returns:
            Tuple of (best sequence, its energy, final population)
        """
        # Initialize population
        population = self.initialize_population(initial_sequences)
        
        # Evaluate initial population on GPU
        pop_energies = self.evaluate_population_gpu(population)
        best_idx = int(cp.argmin(pop_energies))
        best_sequence = population[best_idx].copy()
        best_energy = float(pop_energies[best_idx])
        
        # Main MTS loop
        for iteration in range(max_iterations):
            # Select parents
            parent1 = population[random.randint(0, self.population_size - 1)]
            parent2 = population[random.randint(0, self.population_size - 1)]
            
            # Create child
            if random.random() < 0.5:
                child = self._combine(parent1, parent2)
            else:
                child = population[random.randint(0, self.population_size - 1)].copy()
            
            # Mutate child
            if random.random() < self.p_mutate:
                child = self._mutate(child)
            
            # Apply GPU tabu search
            child, child_energy = self.tabu_search.search(child, max_iterations=50)
            
            # Update if improved
            if child_energy < best_energy:
                best_sequence = child.copy()
                best_energy = child_energy
                
                replace_idx = random.randint(0, self.population_size - 1)
                population[replace_idx] = child
                
                if verbose:
                    print(f"Iteration {iteration}: New best energy = {best_energy}")
        
        return best_sequence, best_energy, population
    
    def _combine(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Uniform crossover combining two parents."""
        mask = np.random.random(self.N) < 0.5
        child = np.where(mask, p1, p2).astype(np.int32)
        return child
    
    def _mutate(self, sequence: np.ndarray) -> np.ndarray:
        """Mutate sequence by flipping random bits."""
        mutated = sequence.copy()
        flip_mask = np.random.random(self.N) < 0.1
        mutated[flip_mask] *= -1
        return mutated


def benchmark_gpu_vs_cpu(N: int = 20, iterations: int = 30):
    """
    Benchmark GPU vs CPU MTS performance.
    """
    import time
    from labs_solver.mts import MemeticTabuSearch
    
    print(f"\n{'='*60}")
    print(f"GPU vs CPU Benchmark: N={N}, Iterations={iterations}")
    print(f"CuPy Available: {CUPY_AVAILABLE}")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    # CPU benchmark
    print("\n[CPU] Running MTS...")
    start = time.time()
    mts_cpu = MemeticTabuSearch(N=N, population_size=20)
    _, energy_cpu, _ = mts_cpu.run(max_iterations=iterations)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time:.3f}s")
    print(f"  Best Energy: {energy_cpu}")
    
    # GPU benchmark
    if CUPY_AVAILABLE:
        print("\n[GPU] Running MTS...")
        start = time.time()
        mts_gpu = MemeticTabuSearchGPU(N=N, population_size=20)
        _, energy_gpu, _ = mts_gpu.run(max_iterations=iterations)
        gpu_time = time.time() - start
        print(f"  Time: {gpu_time:.3f}s")
        print(f"  Best Energy: {energy_gpu}")
        
        print(f"\nSpeedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("\n[GPU] Skipped (CuPy not available)")


if __name__ == "__main__":
    benchmark_gpu_vs_cpu(N=20, iterations=30)
