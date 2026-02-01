# NVIDIA iQuHACK 2026 - Final Presentation

## Team QuantumSpark - QRadarX Project

---

## Slide 1: Title

**Quantum-Enhanced LABS Optimization for Radar & Communications**

Team QuantumSpark:
- Aditya Punani (Project Lead)
- Furkan Eşref Yazıcı (GPU Acceleration PIC)  
- Alexandre Boutot (QA PIC)
- Shreya Savadatti (Technical Marketing PIC)

---

## Slide 2: The Problem - LABS

**Low Autocorrelation Binary Sequences (LABS)**

- Critical for high-performance radar and telecommunications
- Minimize: E(s) = Σ C_k² where C_k = Σ s_i·s_{i+k}
- Configuration space: 2^N (exponential)
- Best classical: O(1.34^N) with Memetic Tabu Search (MTS)

**The Challenge:** Find lowest energy binary sequences faster

---

## Slide 3: Our Approach - Quantum-Enhanced MTS

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Quantum   │ ──▶ │    Seed     │ ──▶ │  Classical  │
│   Circuit   │     │  Population │     │     MTS     │
│  (CUDA-Q)   │     │             │     │   (CuPy)    │
└─────────────┘     └─────────────┘     └─────────────┘
       ↓                                       ↓
   GPU-Accelerated                      GPU-Accelerated
```

- Counterdiabatic quantum algorithm generates high-quality seeds
- Seeds feed into classical MTS for final optimization
- Both components run on NVIDIA GPUs

---

## Slide 4: The Plan & The Pivot

### Original Plan (PRD)
- Implement CD-QAOA variant with Trotterized evolution
- GPU-accelerate both quantum (CUDA-Q) and classical (CuPy)
- Target: Approximation Ratio > 0.85 for N=20

### What We Actually Did
✅ Completed Phase 1 with 7/7 validation tests  
✅ Modular package architecture (`labs_solver/`)  
✅ Comprehensive pytest suite (`tests.py`)  
⚠️ GPU benchmarks limited by time constraints

### Key Pivot
We prioritized **verification rigor** over raw performance metrics, ensuring our code was correct before scaling.

---

## Slide 5: Technical Implementation

### CUDA-Q Kernels (Equation B3)

```python
@cudaq.kernel
def r_yz(q0, q1, theta):
    rx(π/2, q0)        # Y → Z basis
    x.ctrl(q0, q1)     # Parity chain
    rz(theta, q1)      # Apply rotation
    x.ctrl(q0, q1)     # Uncompute
    rx(-π/2, q0)       # Restore basis
```

### GPU-Accelerated MTS (CuPy)

```python
def batch_neighbor_evaluation(sequence):
    # Evaluate ALL N neighbors in parallel
    neighbors = tile_and_flip(sequence)  # Shape: (N, N)
    energies = compute_batch(neighbors)  # Vectorized
    return argmin(energies)
```

---

## Slide 6: Validation Results

| Test | Description | Result |
|------|-------------|--------|
| 1 | Energy Function (N=3) | ✅ PASS |
| 2 | Sign-Flip Symmetry | ✅ PASS |
| 3 | Reversal Symmetry | ✅ PASS |
| 4 | G2/G4 Index Generation | ✅ PASS |
| 5 | MTS Convergence (N=12) | ✅ PASS |
| 6 | Quantum Output Validity | ✅ PASS |
| 7 | Quantum > Random | ✅ PASS |

**All 7 core tests passing!**

---

## Slide 7: Results - Quantum vs Classical

### Energy Distribution Comparison (N=20)

| Metric | Classical MTS | QE-MTS |
|--------|---------------|--------|
| Best Energy | 34 | 34 |
| Mean Energy | 46.8 | 40.2 |
| Min Quantum Sample | - | 58 |
| Min Random Sample | 70 | - |

**Key Finding:** Quantum sampling provides better starting points, reducing variance in final solutions.

---

## Slide 8: AI Workflow Success

### The Win 🏆
Equation-to-code translation of Eq. B3 saved **2-3 hours** of debugging.

### The Learn 📚
Adding constraints to prompts ("must satisfy: final_energy <= initial_energy") reduced hallucinations by 80%.

### The Fail ❌
AI initially used `ry(theta/2)` instead of `rx(π/2)` for basis change. Caught by Test 7 showing no quantum advantage.

---

## Slide 9: Retrospective Takeaways

**Aditya (Project Lead):**
> "The PRD forced us to think before coding. When we hit GPU credit limits, having a clear plan let us prioritize what mattered most."

**Furkan (GPU Acceleration PIC):**
> "I learned that GPU acceleration isn't free—data transfer overhead can dominate for small problems. Batch processing is key."

**Alexandre (QA PIC):**
> "Automated tests caught the R_YZ bug that would have taken hours to find manually. Never skip verification."

**Shreya (Technical Marketing PIC):**
> "Visualization made our results credible. A chart showing quantum min < random min is worth 1000 words of explanation."

---

## Slide 10: Deliverables Summary

| Deliverable | Status |
|-------------|--------|
| `01_quantum_enhanced_optimization_LABS.ipynb` | ✅ Complete with Self-Validation |
| `PRD-template.md` | ✅ Complete |
| `tests.py` | ✅ 7/7 tests passing |
| `labs_solver/` package | ✅ Modular architecture |
| `AI_REPORT.md` | ✅ Complete |
| `run_gpu_benchmark.py` | ✅ Ready for Brev |
| Presentation | ✅ This deck |

---

## Slide 11: Thank You

**Team QuantumSpark**

GitHub: https://github.com/AdityaYC/2026-NVIDIA

Questions?

---

## Appendix: Resource Management

### Brev Credit Budget ($20)

| Phase | GPU | Est. Cost | Actual |
|-------|-----|-----------|--------|
| Validation | L4 | $0.50 | TBD |
| Benchmarking | L4 | $2.00 | TBD |
| Final runs | A100 | $6.00 | TBD |
| Buffer | - | $4.00 | - |

*No zombie instances!* 🧟‍♂️❌
