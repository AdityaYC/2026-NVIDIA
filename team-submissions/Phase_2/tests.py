"""
LABS Solver Test Suite

Rigorous test suite for validating the LABS solver components.
This file is a required deliverable for Phase 2 (Milestone 3).

Run with: pytest tests.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add the labs_solver package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labs_solver.energy import (
    compute_labs_energy, 
    bitstring_to_sequence, 
    sequence_to_bitstring,
    get_known_optimum
)
from labs_solver.mts import TabuSearch, MemeticTabuSearch, combine, mutate

# Conditionally import quantum module (requires CUDA-Q)
try:
    from labs_solver.quantum import get_interactions
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    # Provide a local implementation for testing without CUDA-Q
    from math import floor
    def get_interactions(N):
        """Local implementation for testing without CUDA-Q."""
        G2 = []
        G4 = []
        for i in range(N - 2):
            k_limit = floor((N - (i + 1)) / 2)
            for k in range(1, k_limit + 1):
                G2.append([i, i + k])
        for i in range(N - 3):
            t_limit = floor((N - (i + 1) - 1) / 2)
            for t in range(1, t_limit + 1):
                k_limit = N - (i + 1) - t
                for k in range(t + 1, k_limit + 1):
                    G4.append([i, i + t, i + k, i + k + t])
        return G2, G4


# ============================================================================
# TEST 1: Energy Function - Manual Calculation Check
# ============================================================================
class TestEnergyFunction:
    """Test the LABS energy function with hand-calculated values."""
    
    def test_energy_n3_all_ones(self):
        """
        E([1,1,1]):
          C_1 = s_0*s_1 + s_1*s_2 = 1*1 + 1*1 = 2
          C_2 = s_0*s_2 = 1*1 = 1
          E = C_1^2 + C_2^2 = 4 + 1 = 5
        """
        assert compute_labs_energy(np.array([1, 1, 1])) == 5
    
    def test_energy_n3_mixed(self):
        """
        E([1,1,-1]):
          C_1 = s_0*s_1 + s_1*s_2 = 1*1 + 1*(-1) = 0
          C_2 = s_0*s_2 = 1*(-1) = -1
          E = C_1^2 + C_2^2 = 0 + 1 = 1
        """
        assert compute_labs_energy(np.array([1, 1, -1])) == 1
    
    def test_energy_n3_alternating(self):
        """
        E([1,-1,1]):
          C_1 = s_0*s_1 + s_1*s_2 = 1*(-1) + (-1)*1 = -2
          C_2 = s_0*s_2 = 1*1 = 1
          E = C_1^2 + C_2^2 = 4 + 1 = 5
        """
        assert compute_labs_energy(np.array([1, -1, 1])) == 5
    
    def test_energy_n3_all_negative_ones(self):
        """E([-1,-1,-1]) should equal E([1,1,1]) by symmetry = 5"""
        assert compute_labs_energy(np.array([-1, -1, -1])) == 5
    
    def test_energy_returns_nonnegative(self):
        """Energy should always be >= 0 (it's a sum of squares)."""
        np.random.seed(42)
        for _ in range(20):
            s = np.random.choice([-1, 1], size=np.random.randint(3, 20))
            assert compute_labs_energy(s) >= 0


# ============================================================================
# TEST 2: Sign-Flip Symmetry: E(s) = E(-s)
# ============================================================================
class TestSignFlipSymmetry:
    """Test that LABS energy is invariant under sign flip."""
    
    def test_sign_flip_symmetry_fixed(self):
        """Test specific sequences."""
        sequences = [
            np.array([1, 1, 1]),
            np.array([1, -1, 1, -1]),
            np.array([1, 1, -1, -1, 1]),
        ]
        for s in sequences:
            assert compute_labs_energy(s) == compute_labs_energy(-s)
    
    def test_sign_flip_symmetry_random(self):
        """Test random sequences of various sizes."""
        np.random.seed(42)
        for _ in range(10):
            N = np.random.randint(5, 25)
            s = np.random.choice([-1, 1], size=N)
            e_s = compute_labs_energy(s)
            e_neg_s = compute_labs_energy(-s)
            assert e_s == e_neg_s, f"Symmetry failed for N={N}: E(s)={e_s}, E(-s)={e_neg_s}"


# ============================================================================
# TEST 3: Reversal Symmetry: E(s) = E(s[::-1])
# ============================================================================
class TestReversalSymmetry:
    """Test that LABS energy is invariant under sequence reversal."""
    
    def test_reversal_symmetry_fixed(self):
        """Test specific sequences."""
        sequences = [
            np.array([1, 1, 1]),
            np.array([1, -1, 1]),
            np.array([1, 1, -1, -1]),
        ]
        for s in sequences:
            assert compute_labs_energy(s) == compute_labs_energy(s[::-1])
    
    def test_reversal_symmetry_random(self):
        """Test random sequences."""
        np.random.seed(123)
        for _ in range(10):
            N = np.random.randint(5, 25)
            s = np.random.choice([-1, 1], size=N)
            e_s = compute_labs_energy(s)
            e_rev = compute_labs_energy(s[::-1])
            assert e_s == e_rev, f"Reversal symmetry failed for N={N}"


# ============================================================================
# TEST 4: Interaction Index Validation (G2, G4)
# ============================================================================
class TestInteractionIndices:
    """Test the G2 and G4 index generation functions."""
    
    def test_g2_count_n6(self):
        """For N=6, there should be 6 two-body terms."""
        G2, G4 = get_interactions(6)
        assert len(G2) == 6
    
    def test_g2_first_index_n8(self):
        """For N=8, the first G2 index should be [0, 1]."""
        G2, G4 = get_interactions(8)
        assert G2[0] == [0, 1]
    
    def test_g2_indices_are_valid(self):
        """All G2 indices should be within bounds."""
        for N in range(4, 15):
            G2, G4 = get_interactions(N)
            for pair in G2:
                assert len(pair) == 2
                assert 0 <= pair[0] < N
                assert 0 <= pair[1] < N
                assert pair[0] < pair[1]  # i < i+k
    
    def test_g4_indices_are_valid(self):
        """All G4 indices should be within bounds."""
        for N in range(5, 12):
            G2, G4 = get_interactions(N)
            for quad in G4:
                assert len(quad) == 4
                for idx in quad:
                    assert 0 <= idx < N
    
    def test_g4_empty_for_small_n(self):
        """G4 should be empty for N < 4."""
        G2, G4 = get_interactions(3)
        assert len(G4) == 0


# ============================================================================
# TEST 5: MTS Algorithm Convergence
# ============================================================================
class TestMTSConvergence:
    """Test that MTS converges to good solutions."""
    
    def test_mts_improves_energy(self):
        """MTS should find energy < 40 for N=12 within 30 iterations."""
        np.random.seed(42)
        mts = MemeticTabuSearch(N=12, population_size=15)
        _, best_energy, _ = mts.run(max_iterations=30)
        assert best_energy < 40, f"MTS did not converge: best_energy={best_energy}"
    
    def test_mts_with_seed_population(self):
        """MTS should work with a provided seed population."""
        np.random.seed(42)
        N = 10
        seed_pop = [np.random.choice([-1, 1], size=N) for _ in range(5)]
        mts = MemeticTabuSearch(N=N, population_size=10)
        best_seq, best_energy, _ = mts.run(max_iterations=20, initial_sequences=seed_pop)
        assert len(best_seq) == N
        assert best_energy >= 0
    
    def test_tabu_search_improves_random(self):
        """Tabu search should improve on a random sequence."""
        np.random.seed(42)
        N = 12
        initial = np.random.choice([-1, 1], size=N)
        initial_energy = compute_labs_energy(initial)
        
        ts = TabuSearch(tabu_tenure=10)
        final, final_energy = ts.search(initial, max_iterations=50)
        
        # Final should be no worse than initial
        assert final_energy <= initial_energy


# ============================================================================
# TEST 6: Bitstring Conversion Functions
# ============================================================================
class TestBitstringConversion:
    """Test bitstring to sequence and back conversion."""
    
    def test_bitstring_to_sequence(self):
        """Test conversion from bitstring to ±1 array."""
        assert np.array_equal(bitstring_to_sequence("000"), np.array([1, 1, 1]))
        assert np.array_equal(bitstring_to_sequence("111"), np.array([-1, -1, -1]))
        assert np.array_equal(bitstring_to_sequence("010"), np.array([1, -1, 1]))
    
    def test_sequence_to_bitstring(self):
        """Test conversion from ±1 array to bitstring."""
        assert sequence_to_bitstring(np.array([1, 1, 1])) == "000"
        assert sequence_to_bitstring(np.array([-1, -1, -1])) == "111"
        assert sequence_to_bitstring(np.array([1, -1, 1])) == "010"
    
    def test_roundtrip_conversion(self):
        """Converting back and forth should preserve the sequence."""
        np.random.seed(42)
        for _ in range(10):
            N = np.random.randint(5, 20)
            s = np.random.choice([-1, 1], size=N)
            bitstring = sequence_to_bitstring(s)
            recovered = bitstring_to_sequence(bitstring)
            assert np.array_equal(s, recovered)


# ============================================================================
# TEST 7: Combine and Mutate Functions
# ============================================================================
class TestGeneticOperators:
    """Test genetic algorithm operators."""
    
    def test_combine_output_size(self):
        """Combine should return a sequence of the same size."""
        p1 = np.array([1, 1, 1, 1])
        p2 = np.array([-1, -1, -1, -1])
        child = combine(p1, p2)
        assert len(child) == len(p1)
    
    def test_combine_values_from_parents(self):
        """Child values should come from one of the parents at each position."""
        np.random.seed(42)
        p1 = np.array([1, 1, 1, 1, 1])
        p2 = np.array([-1, -1, -1, -1, -1])
        child = combine(p1, p2)
        for i, val in enumerate(child):
            assert val in [p1[i], p2[i]]
    
    def test_mutate_preserves_values(self):
        """Mutated sequence should only contain ±1 values."""
        s = np.array([1, 1, 1, 1, 1])
        mutated = mutate(s, p_mutate=0.5)
        assert all(v in [-1, 1] for v in mutated)
    
    def test_mutate_changes_some_values(self):
        """With p_mutate=1.0, all values should flip."""
        s = np.array([1, 1, 1, 1, 1])
        mutated = mutate(s, p_mutate=1.0)
        assert np.array_equal(mutated, -s)


# ============================================================================
# QUANTUM TESTS (Optional - require CUDA-Q environment)
# ============================================================================
class TestQuantumComponents:
    """
    Tests for quantum components.
    These may be skipped if CUDA-Q is not available.
    """
    
    @pytest.fixture
    def check_cudaq(self):
        """Skip tests if CUDA-Q is not available."""
        try:
            import cudaq
            return True
        except ImportError:
            pytest.skip("CUDA-Q not available")
    
    def test_quantum_sampling_output_length(self, check_cudaq):
        """Quantum samples should have correct bitstring length."""
        from labs_solver.quantum import run_quantum_sampling
        
        N = 6
        counts = run_quantum_sampling(N=N, n_steps=5, shots=100, use_simple_theta=True)
        
        if len(counts) > 0:
            first_bitstring = list(counts.keys())[0]
            assert len(first_bitstring) == N
    
    def test_quantum_population_extraction(self, check_cudaq):
        """Should be able to extract population from quantum samples."""
        from labs_solver.quantum import run_quantum_sampling, quantum_population_from_samples
        
        N = 6
        counts = run_quantum_sampling(N=N, n_steps=5, shots=100, use_simple_theta=True)
        population = quantum_population_from_samples(counts, population_size=5)
        
        assert len(population) <= 5
        if len(population) > 0:
            assert len(population[0]) == N


# ============================================================================
# Entry point for running tests
# ============================================================================
if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
