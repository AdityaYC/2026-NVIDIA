# Video Presentation Script
**Note:** Read the text in quotes (`"..."`). The speaker notes are in *italics*.

---

## Slide 1: Title
**"Hello judges. We are Team QuantumSpark, and this is QRadarX: Quantum-Enhanced LABS Optimization."**

*"We are presenting our solution for the NVIDIA Challenge, featuring a GPU-accelerated hybrid workflow."*

---

## Slide 2: The Problem (LABS)
**"The LABS problem requires finding a binary sequence with minimal off-peak autocorrelation. This is critical for radar, telecom, and cryptography."**

*"As you can see, the search space explodes exponentially. For N=40, a brute force search would take 317 years. We need a better way."*

---

## Slide 3: Our Approach (Hybrid Workflow)
**"We built a hybrid pipeline. We use a quantum circuit running on NVIDIA cuQuantum to explore the energy landscape and generate diverse seeds."**

*"These seeds are then fed into our Classical Memetic Tabu Search (MTS) running on GPU, which refines them to the optimal solution. It combines quantum exploration with classical exploitation."*

---

## Slide 4: The Pivot
**"We initially planned to scale quantum circuits to N=40. But we hit reality: N=35 requires 550GB of RAM."**

*"So we pivoted. We focused on benchmarking quantum up to N=30 and using our classical MTS solver to reach N=40, proving the limits of current state-vector simulation."*

---

## Slide 5: GPU Acceleration
**"We migrated our workload to an NVIDIA L4 GPU on Brev.dev. The migration was seamless—just one line of code to switch the CUDA-Q target."**

*"This simple switch unlocked massive performance gains compared to our CPU baselines."*

---

## Slide 6: Quantum Results
**"Here is the data. N=30 on a CPU took over a minute. On the L4 GPU, N=20 ran in just 7.77 seconds."**

*"The chart clearly shows the speedup. The GPU backend is roughly 8x faster than the CPU backend for these state vector simulations."*

---

## Slide 7: Classical MTS Results
**"For our classical solver, we used CuPy to accelerate matrix operations. We achieved linear scaling."**

*"While the quantum part scales exponentially, our GPU-accelerated classical solver can handle N=40 in just 16 seconds. This confirms the power of the hybrid approach."*

---

## Slide 8: The Exponential Wall
**"This chart explains why we pivoted. The memory usage doubles with every N. At N=35, we hit the 'Memory Wall'—needing 550GB."**

*"Even an A100 cannot handle N=35 via state vector simulation. This highlights the need for tensor networks or real quantum hardware for larger problems."*

---

## Slide 9: Verification
**"We implemented a rigorous test suite with 26 tests, all passing. This was critical."**

*"In fact, our tests caught an AI hallucination where it used the wrong rotation gate. Without these tests, our results would have been invalid."*

---

## Slide 10: Retrospective (Team Takeaways)

*(Each member reads their line)*

**Aditya:**
"I learned that GPU acceleration is powerful, but memory is the real bottleneck. Quantum simulation needs hardware, not just GPUs."

**Furkan:**
"For me, I saw how modular code made GPU migration trivial. It was just a one-line change from CPU to GPU."

**Alexandre:**
"I discovered that test-driven development is essential. It caught bugs the AI introduced that would have otherwise ruined our results."

**Shreya:**
"My takeaway is that AI accelerates coding, but human verification is essential. We must trust but verify."

---

## Conclusion
**"In summary, we delivered a verified, GPU-accelerated solver. Thank you from Team QuantumSpark."**
