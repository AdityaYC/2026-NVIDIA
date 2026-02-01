# VIDEO RECORDING SCRIPT
## Team QuantumSpark - NVIDIA iQuHACK 2026
### Total Duration: 7-8 minutes

---

## BEFORE RECORDING

**Setup:**
- Open your 10-slide presentation (Gamma or PowerPoint)
- Have charts ready: `chart_quantum_time.png`, `chart_mts_time.png`, `chart_memory.png`
- Quiet environment, good microphone
- Record screen + voice

---

## SLIDE 1: TITLE (30 seconds)

**[Show title slide]**

> "Hello everyone! We are Team QuantumSpark, and today we're presenting our solution for the NVIDIA iQuHACK 2026 challenge.
>
> Our project is called QRadarX — a quantum-enhanced optimization system for solving the Low Autocorrelation Binary Sequences problem, accelerated by NVIDIA GPUs.
>
> I'm Aditya Punjani, the project lead. My teammates are Furkan, who handled GPU acceleration, Alexandre on quality assurance, and Shreya on technical marketing."

---

## SLIDE 2: THE LABS PROBLEM (45 seconds)

**[Show LABS problem slide]**

> "So what is the LABS problem? It's about finding the optimal binary sequence — a string of plus ones and minus ones — that minimizes something called autocorrelation energy.
>
> This has real applications in radar systems, where low autocorrelation means better target detection, and in telecommunications for reducing signal interference.
>
> The challenge? It's NP-hard. For a sequence of length 20, there are 1 million possibilities. For length 40, there are over 1 TRILLION. Brute force would take 317 years. We need smarter algorithms."

---

## SLIDE 3: OUR HYBRID APPROACH (45 seconds)

**[Show hybrid approach slide]**

> "Our solution is a hybrid quantum-classical workflow with GPU acceleration.
>
> Here's how it works:
>
> First, we use a quantum circuit built with NVIDIA's CUDA-Q framework to generate diverse candidate solutions. Quantum provides intelligent exploration of the search space.
>
> Second, these quantum samples seed a classical algorithm called Memetic Tabu Search, or MTS. This refines the candidates to find the true optimum.
>
> Both components are GPU-accelerated — quantum using CUDA-Q's nvidia backend, and classical using CuPy for parallel computation.
>
> The key insight: quantum explores, classical optimizes, GPU makes it fast."

---

## SLIDE 4: THE PIVOT (45 seconds)

**[Show pivot slide]**

> "Now, every engineering project has challenges. Let me tell you about our pivot.
>
> Our original plan was to scale the quantum simulation to N equals 40 or beyond, then do a fair comparison with classical.
>
> But reality hit. We discovered that N equals 35 would require 550 gigabytes of RAM — impossible on any GPU!
>
> So we adapted. We focused our quantum experiments on N up to 30, which is the maximum feasible size. We extended our classical MTS algorithm to N equals 40. And most importantly, we documented these memory limits as a key scientific finding.
>
> Engineering is about adaptation, not perfection."

---

## SLIDE 5: QUANTUM GPU RESULTS (60 seconds)

**[Show quantum results slide with chart]**

> "Let's look at our quantum results.
>
> We ran our quantum circuits on an NVIDIA L4 GPU using the Brev platform. This chart shows execution time versus problem size.
>
> [Point to chart]
>
> For N equals 20, the quantum circuit runs in just 7.77 seconds on GPU. For N equals 25, it's 13.5 seconds. When we ran N equals 30 on CPU for comparison, it took over 61 seconds.
>
> That's roughly an 8x speedup from GPU acceleration!
>
> The CUDA-Q framework made this straightforward — we just set the target to 'nvidia' and the GPU handled the rest."

---

## SLIDE 6: CLASSICAL MTS RESULTS (45 seconds)

**[Show MTS results slide with chart]**

> "Now let's look at our classical MTS algorithm.
>
> This chart shows time versus problem size for the Memetic Tabu Search.
>
> Notice something important: it scales linearly! N equals 20 takes about 1 second. N equals 30 takes 6 seconds. N equals 40 — which was impossible for quantum — takes just 16.5 seconds.
>
> This is why hybrid approaches work. Quantum provides good starting points, but classical can handle the scale."

---

## SLIDE 7: THE EXPONENTIAL WALL (60 seconds)

**[Show memory scaling slide with chart]**

> "This slide explains why we couldn't go beyond N equals 30 for quantum.
>
> Look at this memory chart. Every time N increases by 1, the required memory DOUBLES. That's exponential scaling.
>
> For N equals 20, we need just 16 megabytes — easy.
> For N equals 30, we need 16 gigabytes — fits on an L4 GPU.
> For N equals 35, we would need 550 gigabytes — more than any GPU has.
> For N equals 40, we're talking 16 TERABYTES — that's datacenter scale.
>
> This is the fundamental challenge of quantum simulation. To go beyond N equals 32 or so, we need either tensor network methods or actual quantum hardware. This finding is itself valuable for the field."

---

## SLIDE 8: VERIFICATION (45 seconds)

**[Show verification slide]**

> "Rigorous engineering requires rigorous testing. Our test suite has 26 tests, all passing.
>
> We verified the energy function against hand calculations. We tested symmetry properties — for example, flipping all signs shouldn't change the energy. We verified that MTS converges to good solutions. And we checked that quantum outputs are in the correct format.
>
> One important catch: during development, we used AI tools to help write code. Our tests caught a bug where the AI suggested using the wrong rotation gate in the quantum kernel. Without tests, this would have silently corrupted our results."

---

## SLIDE 9: TEAM TAKEAWAYS (45 seconds)

**[Show takeaways slide]**

> "Let me share what each team member learned.
>
> As project lead, I learned that memory is the real bottleneck for quantum simulation — not computation time.
>
> Furkan found that our modular code design made GPU migration surprisingly easy — just one line changed.
>
> Alexandre proved that test-driven development catches critical bugs before they become problems.
>
> And Shreya discovered that while AI accelerates coding, human verification remains essential. Trust, but verify."

---

## SLIDE 10: SUMMARY & THANK YOU (45 seconds)

**[Show summary slide]**

> "Let me summarize what we accomplished:
>
> Quantum GPU acceleration — we achieved 7.77 seconds for N equals 20 on the L4 GPU.
>
> Classical MTS scaling — we successfully ran up to N equals 40 in under 17 seconds.
>
> Verification — 26 out of 26 tests passing, ensuring correctness.
>
> And we documented the memory limits that prevent quantum simulation beyond N equals 30.
>
> Our code is available on GitHub at github.com/AdityaYC/2026-NVIDIA.
>
> Thank you for watching! We're Team QuantumSpark, and this has been our solution for the NVIDIA iQuHACK 2026 challenge."

---

## AFTER RECORDING

**Checklist:**
- [ ] Video is 5-10 minutes long
- [ ] Audio is clear
- [ ] All slides are visible
- [ ] Charts are readable
- [ ] Export as MP4
- [ ] Upload to team-submissions folder

---

## TOTAL TIME BREAKDOWN

| Slide | Topic | Duration |
|-------|-------|----------|
| 1 | Title & Team | 30s |
| 2 | LABS Problem | 45s |
| 3 | Hybrid Approach | 45s |
| 4 | The Pivot | 45s |
| 5 | Quantum Results | 60s |
| 6 | MTS Results | 45s |
| 7 | Memory Wall | 60s |
| 8 | Verification | 45s |
| 9 | Takeaways | 45s |
| 10 | Summary | 45s |
| **TOTAL** | | **~7.5 min** |

---

*Good luck with your recording!*
