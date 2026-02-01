# Quantum-Enhanced LABS Optimization
## GPU-Accelerated Hybrid Workflow | Team QuantumSpark | NVIDIA iQuHACK 2026

---

# 1. The Team

**Team QuantumSpark**

| Role | Member |
|------|--------|
| Project Lead | Aditya Punjani |
| GPU Acceleration | Furkan Eşref Yazıcı |
| Quality Assurance | Alexandre Boutot |
| Technical Marketing | Shreya Savadatti |

---

# 2. The LABS Problem

**Goal:** Find binary sequence s ∈ {-1, +1}^N that minimizes autocorrelation energy

**Applications:** Radar systems, Telecommunications, Cryptography

**Challenge:** NP-hard — N=40 has 1 trillion possibilities

| N | Search Space | Brute Force Time |
|---|-------------|------------------|
| 20 | 1 million | ~1 second |
| 40 | 1 trillion | **317 years** |

---

# 3. Our Hybrid Approach

**Quantum + Classical + GPU Pipeline**

1. **Quantum Circuit** → Generate diverse candidate solutions (CUDA-Q)
2. **Classical MTS** → Refine candidates to find optimum (CuPy)
3. **GPU Acceleration** → Speed up both components

**Why hybrid?** Quantum explores, classical optimizes, GPU accelerates.

---

# 4. The Pivot

**Original Plan:** Scale quantum to N=40+

**Reality:** N=35 needs 550GB RAM — impossible!

**Our Adaptation:**
- Focused quantum on N≤30 (feasible)
- Extended classical MTS to N=40
- Documented memory limits as key finding

*Engineering is about adaptation.*

---

# 5. GPU Results: Quantum Circuit

**NVIDIA L4 GPU (24GB) on Brev**

| N | Platform | Time |
|---|----------|------|
| 20 | GPU | **7.77s** |
| 25 | GPU | **13.50s** |
| 30 | CPU | 61.68s |

**Key Finding:** GPU provides ~8x speedup

![Quantum Performance](chart_quantum_time.png)

---

# 6. GPU Results: Classical MTS

**Linear Scaling Achieved**

| N | Time | Best Energy |
|---|------|-------------|
| 20 | 1.08s | 26 |
| 30 | 6.39s | 83 |
| 40 | **16.52s** | 128 |

**Key Finding:** MTS scales linearly to N=40

![MTS Performance](chart_mts_time.png)

---

# 7. The Exponential Wall

**Why N=35+ Quantum Is Impossible**

| N | Memory Required | Status |
|---|-----------------|--------|
| 20 | 16 MB | ✓ |
| 30 | 16 GB | ✓ |
| 35 | 550 GB | ✗ |
| 40 | 16 TB | ✗ |

*Future: Tensor networks or real quantum hardware*

![Memory Scaling](chart_memory.png)

---

# 8. Verification

**26/26 Tests Passing**

| Category | Verified |
|----------|----------|
| Energy Function | E([1,1,1]) = 5 ✓ |
| Symmetry | E(s) = E(-s) ✓ |
| MTS Convergence | Finds good solutions ✓ |
| Quantum Output | Correct format ✓ |

**AI Bug Caught:** Wrong rotation gate — tests saved us!

---

# 9. Team Takeaways

**Aditya:** "Memory is the real bottleneck for quantum."

**Furkan:** "Modular code made GPU migration easy."

**Alexandre:** "Tests caught critical AI bugs."

**Shreya:** "AI helps, but verification is essential."

---

# 10. Summary & Thank You

| Deliverable | Result |
|-------------|--------|
| Quantum GPU | 7.77s for N=20 ✓ |
| MTS Scaling | N=40 in 16.52s ✓ |
| Tests | 26/26 passing ✓ |
| Memory Limits | Documented ✓ |

**GitHub:** github.com/AdityaYC/2026-NVIDIA

**Thank You! Questions?**
