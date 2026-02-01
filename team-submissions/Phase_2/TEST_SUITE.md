# Test Suite Documentation

**Team:** QuantumSpark  
**Project:** QRadarX - Quantum Enhanced LABS Optimization  
**Test File:** [tests.py](tests.py)

---

## Verification Strategy

We implemented a rigorous, multi-layered test suite to validate all components of the LABS solver. Our strategy was:

1. **Physics-First Testing** — Verify mathematical correctness against hand-calculated values and known symmetries
2. **Boundary Condition Checks** — Ensure loop indices and array bounds match the paper's equations
3. **Algorithm Convergence** — Confirm optimization algorithms find valid solutions
4. **AI Hallucination Detection** — Tests specifically designed to catch common AI coding errors

---

## Test Categories (26+ Assertions)

| Category | Tests | Purpose | Coverage Rationale |
|----------|-------|---------|-------------------|
| **Energy Function** | 5 | `E([1,1,1]) = 5`, non-negative outputs | Core physics correctness |
| **Sign-Flip Symmetry** | 2 | `E(s) = E(-s)` for fixed + random sequences | LABS Hamiltonian invariant |
| **Reversal Symmetry** | 2 | `E(s) = E(s[::-1])` | LABS Hamiltonian invariant |
| **G2/G4 Indices** | 5 | Correct counts, valid bounds, edge cases | Eq. 15 loop bounds from paper |
| **MTS Convergence** | 3 | Algorithm improves energy, handles seeds | Optimization correctness |
| **Bitstring Conversion** | 3 | Roundtrip preservation | Data integrity |
| **Genetic Operators** | 4 | Combine/mutate output validity | Memetic algorithm operators |
| **Quantum Output** | 2 | Correct bitstring length | Circuit-to-classical interface |

---

## Coverage Reasoning

### Why These Tests?

1. **Hand-Calculated Energy (N=3)**  
   Small enough to verify by hand, catches off-by-one errors in the energy summation.
   ```python
   # E([1,1,1]) = C_1² + C_2² = 2² + 1² = 5
   assert compute_labs_energy(np.array([1, 1, 1])) == 5
   ```

2. **Symmetry Properties**  
   The LABS energy function has two fundamental symmetries: sign-flip and reversal. Testing these catches bugs in array indexing.
   ```python
   assert compute_labs_energy(s) == compute_labs_energy(-s)      # Sign flip
   assert compute_labs_energy(s) == compute_labs_energy(s[::-1]) # Reversal
   ```

3. **Interaction Indices (G2/G4)**  
   The Trotterized quantum circuit depends on correct loop bounds from Eq. 15. Getting these wrong produces invalid circuits.
   ```python
   # N=6 should have exactly 6 two-body terms
   assert len(G2) == 6
   ```

4. **MTS Convergence**  
   Validates that the optimization actually improves energy. A threshold test catches broken implementations.
   ```python
   assert best_energy < 40  # N=12 should achieve this
   ```

5. **Quantum Output Length**  
   Catches mismatches between qubit count and expected bitstring length—a common AI error.
   ```python
   assert len(bitstring) == N
   ```

---

## AI Bug Detection

Our tests specifically caught this AI hallucination:

```python
# AI wrote (WRONG):
ry(theta/2, q0)  # Confused rotation angle with basis change

# We fixed to (CORRECT):
rx(1.5707963267948966, q0)  # π/2 for Y→Z basis change
```

**Test 7** (Quantum Sampling Quality) detected this because quantum samples performed no better than random—indicating the circuit wasn't implementing the intended unitary.

---

## Running the Tests

```bash
# From team-submissions directory
pytest tests.py -v

# Expected output:
# 26 passed (or 24 passed, 2 skipped if CUDA-Q unavailable)
```

---

## Test File Structure

```
tests.py
├── TestEnergyFunction        (5 tests)
├── TestSignFlipSymmetry      (2 tests)
├── TestReversalSymmetry      (2 tests)
├── TestInteractionIndices    (5 tests)
├── TestMTSConvergence        (3 tests)
├── TestBitstringConversion   (3 tests)
├── TestGeneticOperators      (4 tests)
└── TestQuantumComponents     (2 tests, optional)
```

---

*Team QuantumSpark — NVIDIA iQuHACK 2026*
