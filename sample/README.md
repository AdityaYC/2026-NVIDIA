# NVIDIA iQuHACK 2026 - Phase 2 Submission

## Team QuantumSpark - QRadarX Project

**Members:** Aditya Punjani (Lead), Furkan Eşref Yazıcı, Alexandre Boutot, Shreya Savadatti

---

## Quick Start for Judges

```bash
# Run tests (24/24 pass locally, 26/26 on qBraid with CUDA-Q)
pip install pytest numpy
pytest tests.py -v

# Run GPU benchmark (requires NVIDIA GPU)
python3 run_gpu_benchmark.py --mode gpu --n 20 --output results.json
```

---

## File Structure

```
sample/
├── labs_solver/               # Modular Python package
│   ├── __init__.py
│   ├── energy.py              # LABS energy calculation
│   ├── mts.py                 # Memetic Tabu Search (CPU)
│   ├── mts_cupy.py            # GPU-accelerated MTS (CuPy)
│   └── quantum.py             # CUDA-Q kernels
├── tests.py                   # Pytest test suite (7 test classes)
├── run_gpu_benchmark.py       # Main GPU benchmark script
├── fast_benchmark.py          # Optimized benchmark for larger N
├── results_n20.json           # GPU benchmark results
├── AI_REPORT.md               # AI Post-Mortem Report
├── PRESENTATION.md            # Presentation outline
└── README.md                  # This file
```

---

## Benchmark Results

### GPU: NVIDIA L4 (Brev)

| N | Quantum Circuit Time | Target |
|---|---------------------|--------|
| 20 | **7.77 seconds** | nvidia |

### MTS Comparison (N=20)

| Implementation | Time | Best Energy |
|---------------|------|-------------|
| CPU NumPy | 1.08s | 26 |
| GPU CuPy | 15.4s | 26 |

> Note: GPU overhead dominated for small N=20. Batch evaluation benefits would appear at larger population sizes.

---

## Test Coverage

| Test Class | Description | Status |
|------------|-------------|--------|
| TestEnergyFunction | Manual energy calculations | ✅ 5/5 |
| TestSignFlipSymmetry | E(s) = E(-s) | ✅ 2/2 |
| TestReversalSymmetry | E(s) = E(s[::-1]) | ✅ 2/2 |
| TestInteractionIndices | G2/G4 generation | ✅ 5/5 |
| TestMTSConvergence | Algorithm finds good solutions | ✅ 3/3 |
| TestBitstringConversion | Roundtrip preservation | ✅ 3/3 |
| TestGeneticOperators | Combine/mutate functions | ✅ 4/4 |
| TestQuantumComponents | CUDA-Q output | ✅ 2/2 (on qBraid) |

**Total: 26/26 tests passing on qBraid**

---

## Key Accomplishments

1. **GPU Migration**: Successfully migrated to NVIDIA L4 on Brev
2. **Quantum Acceleration**: Ran counterdiabatic circuit on GPU (7.77s for N=20)
3. **Classical GPU Acceleration**: Implemented CuPy-based MTS in `mts_cupy.py`
4. **Rigorous Testing**: 26 tests covering energy function, symmetries, and convergence
5. **Documentation**: Full AI report, presentation, and README

---

## Hardware Used

- **CPU Testing**: qBraid (CPU-only CUDA-Q sandbox)
- **GPU Testing**: Brev L4 GPU (CUDA 12.8, 23GB VRAM)

---

## Files to Review

1. **[tests.py](tests.py)** - Test suite demonstrating verification approach
2. **[AI_REPORT.md](AI_REPORT.md)** - AI workflow, verification strategy, Win/Learn/Fail log
3. **[results_n20.json](results_n20.json)** - Raw GPU benchmark data
4. **[PRESENTATION.md](PRESENTATION.md)** - Presentation outline

---

*Submitted by Team QuantumSpark for NVIDIA iQuHACK 2026*
