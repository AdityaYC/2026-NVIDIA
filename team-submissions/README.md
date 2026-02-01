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
.
├── labs_solver/               # Modular Python package
│   ├── __init__.py
│   ├── energy.py              # LABS energy calculation
│   ├── mts.py                 # Memetic Tabu Search (CPU)
│   ├── mts_cupy.py            # GPU-accelerated MTS (CuPy)
│   └── quantum.py             # CUDA-Q kernels
├── tests.py                   # Pytest test suite (7 test classes)
├── run_gpu_benchmark.py       # Main GPU benchmark script
├── fast_benchmark.py          # Optimized benchmark for larger N
├── mts_benchmark.py           # Standalone MTS benchmark
├── results_n25.json           # Quantum GPU benchmark (N=25)
├── results_n30_quantum.json   # Quantum CPU benchmark (N=30)
├── mts_results.json           # MTS benchmark results
├── mts_results_n35.json       # MTS benchmark (N=35)
├── generate_charts.py         # Script to plot benchmark results
├── figures/chart_*.png        # Generated benchmark visualizations
├── AI_REPORT.md               # AI Post-Mortem Report
├── PRESENTATION_SLIDES.md     # Presentation outline
├── Quantum_NVIDIA_IQUHACKS_Presentation.pdf # Final Presentation PDF
├── TEST_SUITE.md              # Detailed verification docs
├── PRDfinal.md                # Phase 1 Product Requirements
├── 01_quantum_enhanced_optimization_LABS_final.ipynb # Phase 1 Notebook
└── README.md                  # This file
```

---

## Benchmark Results

### Quantum Circuit Benchmarks

| N | Platform | Quantum Time | Trotter Steps | Min Energy | Status |
|---|----------|--------------|---------------|------------|--------|
| 20 | Brev L4 GPU | **7.77s** | 30 | 90 | ✅ |
| 25 | Brev L4 GPU | **13.50s** | 5 | 156 | ✅ |
| 30 | qBraid CPU | **61.68s** | 3 | 187 | ✅ |
| 35 | — | — | — | — | ❌ Memory limit (~550GB required) |
| 40 | — | — | — | — | ❌ Memory limit (~16TB required) |

### MTS (Classical) Benchmarks

| N | Time | Best Energy | Known Optimum |
|---|------|-------------|---------------|
| 20 | 1.08s | 26 | 37 |
| 25 | 2.40s | 48 | — |
| 30 | 6.39s | 83 | 59 |
| 35 | 10.56s | 97 | — |
| 40 | 16.52s | 128 | 108 |

### Key Findings

- **Exponential scaling**: Quantum simulation hits memory limits at N=35 (2^35 = 34B states = ~550GB)
- **GPU acceleration**: 7.77s on GPU vs. 61.68s on CPU for comparable N
- **MTS scales linearly**: Classical algorithm handles N=40 in 16.5s

### Visualizations

![Quantum Circuit Time vs N](figures/chart_quantum_time.png)
*Quantum circuit execution time showing GPU vs CPU performance*

![MTS Algorithm Scaling](figures/chart_mts_time.png)
*Classical MTS algorithm shows linear scaling with problem size*

![Combined Performance](figures/chart_combined.png)
*Side-by-side comparison of quantum and classical performance*

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
### Phase 1

1. **[PRDfinal.md](PRDfinal.md)** - Product Requirements Document (Phase 1)
2. **[01_quantum_enhanced_optimization_LABS_final.ipynb](01_quantum_enhanced_optimization_LABS_final.ipynb)** - Phase 1 Tutorial Notebook (with Self-Validation)

### Phase 2

1. **[tests.py](tests.py)** - Test suite demonstrating verification approach
2. **[AI_REPORT.md](AI_REPORT.md)** - AI workflow, verification strategy, Win/Learn/Fail log
3. **[results_n25.json](results_n25.json)** - Raw GPU benchmark data
4. **[PRESENTATION_SLIDES.md](PRESENTATION_SLIDES.md)** - Presentation outline
5. **[TEST_SUITE.md](TEST_SUITE.md)** - Detailed test suite documentation
6. **[Quantum_NVIDIA_IQUHACKS_Presentation.pdf](Quantum_NVIDIA_IQUHACKS_Presentation.pdf)** - Final Presentation Slides

---

*Submitted by Team QuantumSpark for NVIDIA iQuHACK 2026*
