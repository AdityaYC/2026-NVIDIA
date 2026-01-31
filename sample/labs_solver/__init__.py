# LABS Solver Package
# Quantum-Enhanced Memetic Tabu Search for LABS Optimization

__version__ = "1.0.0"

from .energy import compute_labs_energy, bitstring_to_sequence, sequence_to_bitstring
from .mts import TabuSearch, MemeticTabuSearch, combine, mutate

__all__ = [
    "compute_labs_energy",
    "bitstring_to_sequence", 
    "sequence_to_bitstring",
    "TabuSearch",
    "MemeticTabuSearch",
    "combine",
    "mutate",
]
