import pytest
import numpy as np
import sys
import os

# Add parent dir to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.labs_utils import compute_labs_energy, get_interactions, compute_theta
from src.mts import MemeticTabuSearch, TabuSearch

def test_energy_function_manual_cases():
    """Validate energy function against manual N=3 calculations."""
    test_cases = {
        (1, 1, 1): 5,      # C1=2, C2=1 -> 5
        (1, 1, -1): 1,     # C1=0, C2=-1 -> 1
        (1, -1, 1): 5,     # C1=-2, C2=1 -> 5
    }
    for seq, expected in test_cases.items():
        assert compute_labs_energy(np.array(seq)) == expected

def test_energy_symmetry_sign_flip():
    """Test E(s) = E(-s) symmetry."""
    np.random.seed(42)
    for _ in range(5):
        s = np.random.choice([-1, 1], size=10)
        assert compute_labs_energy(s) == compute_labs_energy(-s)

def test_energy_symmetry_reversal():
    """Test E(s) = E(s[::-1]) symmetry."""
    np.random.seed(42)
    for _ in range(5):
        s = np.random.choice([-1, 1], size=10)
        assert compute_labs_energy(s) == compute_labs_energy(s[::-1])

def test_interactions_logic():
    """Test G2 and G4 generation logic."""
    G2, G4 = get_interactions(6)
    # For N=6, G2 should have size...
    # i=0: k=1,2 (2)
    # i=1: k=1,2 (2)
    # i=2: k=1 (1)
    # i=3: k=1 (1)
    # Total = 6
    assert len(G2) == 6
    
    G2_8, _ = get_interactions(8)
    assert G2_8[0] == [0, 1]

def test_mts_convergence_small():
    """Test that MTS finds optimal/near-optimal for small N."""
    # N=5 ground truth is 2 (e.g. [1, 1, -1, 1, 1] -> C1=0, C2=1, C3=0, C4=1 -> 2)
    mts = MemeticTabuSearch(N=5, population_size=10)
    best_seq, best_energy, _ = mts.run(max_iterations=10)
    # Known min energy for N=5 is 2. Allow it to find it.
    assert best_energy <= 3 # Being lenient for randomized test

def test_compute_theta_zero_time():
    """Test compute_theta handles edge cases."""
    val = compute_theta(0.0, 0.1, 1.0, 10, [], [])
    assert val == 0.0
