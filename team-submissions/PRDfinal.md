# Product Requirements Document (PRD)

**Project Name:** [QRadarX]
**Team Name:** [QuantumSpark]
**GitHub Repository:** [https://github.com/AdityaYC/2026-NVIDIA.git]

---

> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have a solid plan, not just to fill in boxes.

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Aditya Punani | AdityaYC | aditya_52163 |
| **GPU Acceleration PIC** (Builder) | Furkan Eşref Yazıcı | Nonsensicalinsane | furkaneyazici |
| **Quality Assurance PIC** (Verifier) | Alexandre Boutot | TDC28 | tdc05 |
| **Technical Marketing PIC** (Storyteller) | Shreya Savadatti | shreyasavadatti | shreyasavadatti_95898 |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm

**Primary Algorithm:** Counterdiabatic Quantum Optimization (CD-QAOA variant)
* Based on Trotterized adiabatic evolution with counterdiabatic driving terms
* Implements Equation B3: 2-body terms (R_YZ, R_ZY) and 4-body terms (R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY)

**Motivation:**
* Counterdiabatic approach suppresses diabatic transitions → faster annealing than standard QAOA
* LABS Hamiltonian structure (2-body and 4-body Pauli terms) maps naturally to this ansatz
* Paper demonstrates scaling: QE-MTS O(1.24^N) vs classical MTS O(1.34^N)

---

### Phase 1 Status: ✅ COMPLETED

| Component | Status | Details |
|-----------|--------|---------|
| Energy Function | ✅ | Verified with 7 automated tests |
| MTS Algorithm | ✅ | Converges for N=12 in 30 iterations |
| CUDA-Q Kernels | ✅ | Fixed R_YZ basis (rx→rz), np.pi→literal |
| Trotterized Circuit | ✅ | 30 steps, T=1.0, N=20 validated |
| Validation Suite | ✅ | 7/7 tests pass, quantum min(58) < random min(70) |


---

### Phase 2 Strategy: Build & Accelerate

#### Step A: CPU Validation (Milestone 3A)
* **Status:** Ready to proceed
* **Target:** Validate custom algorithm for N=3 to N=10 with rigorous `tests.py`
* **Approach:** Port validation tests from notebook to standalone pytest suite

#### Step B: GPU Migration (Milestone 3B)
* **Hardware:** Brev L4 instance (initial), scale to A100 for large N
* **Target Backend:** `nvidia` single-GPU, then `nvidia-mgpu` for N>30
* **Metrics:** Time-to-solution comparison vs CPU baseline

#### Step C: Classical GPU Acceleration (Milestone 3C)
* **Strategy:** Replace NumPy → CuPy for MTS energy calculations
* **Optimization:** Batch neighbor evaluation (1000 flips simultaneously)
* **Goal:** Fully GPU-accelerated hybrid workflow

---

### Literature Review

| Reference | Relevance |
|-----------|-----------|
| Chandarana et al. "Scaling quantum optimization" | Primary source: CD protocol, R_YZ/R_YZZZ gate equations |
| LABS problem literature (Böhm & Goertz) | Ground truth energies for validation |
| CUDA-Q Documentation | Kernel syntax, backend selection, multi-GPU |

---


## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Current Implementation:** 
    * CUDA-Q kernels for all interaction operators (r_yz, r_zy, r_yzzz, r_zyzz, r_zzyz, r_zzzy)
    * Trotterized circuit with parameterized theta values from `utils.compute_theta`
* **Phase 2 Strategy:**
    * Target `nvidia` backend for single-GPU acceleration
    * For N > 30, explore `nvidia-mgpu` for multi-GPU distribution
    * Use cuQuantum's tensor network simulator for memory-efficient large N simulation

### Classical Acceleration (MTS)
* **Current Implementation:** Pure Python/NumPy MTS with Tabu Search
* **Phase 2 Strategy:**
    * Replace NumPy arrays with CuPy for GPU-accelerated energy calculations
    * Batch neighbor evaluation: evaluate 1000 bit-flip neighbors simultaneously
    * Parallelize population evaluation across GPU threads

### Hardware Targets
* **Dev Environment:** qBraid (CPU) - all validation done here first
* **Initial GPU Testing:** Brev L4 instance (cheapest GPU option)
* **Production Benchmarks:** Brev A100-80GB for N=40+ experiments

---


## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** In-notebook assertions with `assert` statements and visual validation plots
* **AI Hallucination Guardrails:** 
    * All AI-generated kernels must pass **7 automated tests** before integration
    * Manual energy calculations verified by hand for N=3 test cases
    * Cross-validation between quantum and classical results
    * Visual plots generated to confirm mathematical properties

### Core Correctness Checks
* **Check 1 (Energy Function):** Manual calculation for N=3 sequences
    * `E([1,1,1]) == 5`, `E([1,1,-1]) == 1`, `E([1,-1,1]) == 5`
* **Check 2 (Sign-Flip Symmetry):** `assert energy(S) == energy(-S)` for 5 random sequences
* **Check 3 (Reversal Symmetry):** `assert energy(S) == energy(S[::-1])` for 3 random sequences
* **Check 4 (Index Generation):** For N=6, `len(G2) == 6`; For N=8, `G2[0] == [0, 1]`
* **Check 5 (MTS Convergence):** For N=12, best energy < 40 after 30 iterations
* **Check 6 (Quantum Output):** Sampled bitstring length == N
* **Check 7 (Quantum Quality):** `quantum_min_energy <= random_min_energy`

### Critical Bug Fixes Documented
* **R_YZ Basis Transformation:** Changed `ry(theta/2)` → `rx(π/2)` (correct Y→Z basis change)
* **CUDA-Q Literal Values:** Replaced `np.pi` with `1.5707963267948966` (np.pi undefined in compiled kernels)
* **Empty Kernel Fix:** Completed `trotterized_circuit` which was only doing random sampling

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **IDE:** VS Code with Gemini Code Assist (Antigravity)
* **Process:** 
    1. Agent generates code based on paper equations
    2. Run notebook in qBraid environment
    3. Execute 7-test validation suite
    4. If tests fail → paste error back to Agent → refactor
    5. Repeat until all tests pass
* **Documentation:** Created `walkthrough.md` documenting all changes and physics rationale

### Success Metrics
| Metric | Target | Achieved |
|--------|--------|----------|
| Core Tests | 5/5 PASSED | ✅ 5/5 PASSED |
| Quantum Output Valid | Bitstring length = N | ✅ 20 = 20 |
| Quantum Advantage | min(quantum) ≤ min(random) | ✅ 58 ≤ 70 |
| Simulation Scale | N = 20, 30 Trotter steps | ✅ Completed |

### Visualization Plan
* **Plot 1:** Self-Validation Results (3-panel visualization)
    * Core Test Results (bar chart)
    * Energy Function Verification (expected vs computed)
    * Symmetry Scatter Plot (E(s) vs E(-s))
* **Plot 2:** Classical vs Quantum-Enhanced Population Energy Distribution
    * Side-by-side histograms showing population energy distributions
    * Best energy markers for both methods

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

### Current Phase (Phase 1 - Prototyping)
* **Environment:** qBraid CPU-only environment
* **GPU Credits Used:** 0 (all development on CPU simulator)
* **Strategy:** Complete all validation tests on CPU before any GPU usage

### Phase 2 Strategy (Acceleration)
* **Step 1:** Port validated code to Brev L4 instance (cheapest GPU option)
* **Step 2:** Run initial GPU benchmarks, fix any GPU-specific issues
* **Step 3:** Scale to A100-80GB only for final N=40+ benchmarks
* **Credit Protection:** 
    * GPU Acceleration PIC responsible for instance shutdown during breaks
    * Set calendar reminders for instance checks every 2 hours
    * Use qBraid for all debugging, Brev only for validated runs

