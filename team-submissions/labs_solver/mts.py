"""
Memetic Tabu Search (MTS) algorithm for LABS optimization.

This module implements the classical optimization component of the
Quantum-Enhanced MTS workflow as described in:
"Scaling advantage with quantum-enhanced memetic tabu search for LABS"
"""

import numpy as np
import random
from typing import List, Tuple, Optional

from .energy import compute_labs_energy


class TabuSearch:
    """
    Tabu search local optimization for LABS.
    
    Uses a tabu list to avoid cycling and escape local minima.
    """
    
    def __init__(self, tabu_tenure: int = 10):
        """
        Initialize TabuSearch.
        
        Args:
            tabu_tenure: Number of iterations a move stays in the tabu list
        """
        self.tabu_tenure = tabu_tenure
    
    def search(self, sequence: np.ndarray, max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Perform tabu search starting from given sequence.
        
        Args:
            sequence: Initial binary sequence
            max_iterations: Maximum number of iterations
        
        Returns:
            Tuple of (best sequence found, its energy)
        """
        N = len(sequence)
        current = sequence.copy()
        current_energy = compute_labs_energy(current)
        
        best = current.copy()
        best_energy = current_energy
        
        tabu_list = []
        
        for iteration in range(max_iterations):
            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_energy = float('inf')
            best_flip_pos = -1
            
            for pos in range(N):
                if pos not in tabu_list:
                    # Flip bit at position pos
                    neighbor = current.copy()
                    neighbor[pos] *= -1
                    neighbor_energy = compute_labs_energy(neighbor)
                    
                    if neighbor_energy < best_neighbor_energy:
                        best_neighbor = neighbor
                        best_neighbor_energy = neighbor_energy
                        best_flip_pos = pos
            
            if best_neighbor is None:
                break
            
            # Move to best neighbor
            current = best_neighbor
            current_energy = best_neighbor_energy
            
            # Update tabu list
            tabu_list.append(best_flip_pos)
            if len(tabu_list) > self.tabu_tenure:
                tabu_list.pop(0)
            
            # Update global best
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        
        return best, best_energy


def combine(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """
    Combine two parent sequences using uniform crossover.
    Each position is randomly selected from either parent.
    
    Args:
        parent1: First parent sequence
        parent2: Second parent sequence
    
    Returns:
        Child sequence
    """
    N = len(parent1)
    child = np.zeros(N, dtype=int)
    
    for i in range(N):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    
    return child


def mutate(sequence: np.ndarray, p_mutate: float = 0.1) -> np.ndarray:
    """
    Mutate a sequence by flipping bits with probability p_mutate.
    
    Args:
        sequence: Input sequence
        p_mutate: Probability of flipping each bit
    
    Returns:
        Mutated sequence
    """
    N = len(sequence)
    mutated = sequence.copy()
    
    for i in range(N):
        if random.random() < p_mutate:
            mutated[i] *= -1
    
    return mutated


class MemeticTabuSearch:
    """
    Memetic Tabu Search for LABS problem.
    
    Combines genetic algorithm population management with tabu search
    local optimization.
    """
    
    def __init__(self, N: int, population_size: int = 20, p_mutate: float = 0.1):
        """
        Initialize MemeticTabuSearch.
        
        Args:
            N: Sequence length
            population_size: Size of the population
            p_mutate: Mutation probability
        """
        self.N = N
        self.population_size = population_size
        self.p_mutate = p_mutate
        self.tabu_search = TabuSearch(tabu_tenure=10)
    
    def initialize_population(self, initial_sequences: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Initialize population randomly or with provided sequences.
        
        Args:
            initial_sequences: Optional list of seed sequences (e.g., from quantum sampling)
        
        Returns:
            List of population sequences
        """
        if initial_sequences is not None and len(initial_sequences) >= self.population_size:
            return initial_sequences[:self.population_size]
        
        population = []
        
        # Add provided sequences first
        if initial_sequences:
            population.extend(initial_sequences)
        
        # Fill remaining with random sequences
        while len(population) < self.population_size:
            sequence = np.random.choice([-1, 1], size=self.N)
            population.append(sequence)
        
        return population
    
    def run(self, max_iterations: int = 100, initial_sequences: Optional[List[np.ndarray]] = None, 
            verbose: bool = False) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """
        Run Memetic Tabu Search algorithm.
        
        Args:
            max_iterations: Number of MTS iterations
            initial_sequences: Optional initial population from quantum preprocessing
            verbose: Print progress information
        
        Returns:
            Tuple of (best sequence, its energy, final population)
        """
        # Initialize population
        population = self.initialize_population(initial_sequences)
        
        # Find initial best
        best_sequence = None
        best_energy = float('inf')
        
        for seq in population:
            energy = compute_labs_energy(seq)
            if energy < best_energy:
                best_sequence = seq.copy()
                best_energy = energy
        
        # Main MTS loop
        for iteration in range(max_iterations):
            # Select parents
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            # Create child
            if random.random() < 0.5:
                child = combine(parent1, parent2)
            else:
                child = random.choice(population).copy()
            
            # Mutate child
            if random.random() < self.p_mutate:
                child = mutate(child, p_mutate=0.1)
            
            # Apply tabu search
            child, child_energy = self.tabu_search.search(child, max_iterations=50)
            
            # Update population if child is better
            if child_energy < best_energy:
                best_sequence = child.copy()
                best_energy = child_energy
                
                # Replace random member of population
                replace_idx = random.randint(0, self.population_size - 1)
                population[replace_idx] = child
                
                if verbose:
                    print(f"Iteration {iteration}: New best energy = {best_energy}")
        
        return best_sequence, best_energy, population
