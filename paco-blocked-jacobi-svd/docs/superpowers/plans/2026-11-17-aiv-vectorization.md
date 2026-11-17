# AIV batched Jacobi optimization

The existing authorization covers implementation and server measurements. Preserve
one mixed launch and Cube-only full-panel `R X` updates. Keep numerical tolerances.

The local Gram solver uses a circle schedule conjugated into a `[P rows, Q rows]`
layout. Each of d-1 rounds handles d/2 disjoint pivots. Static Gather maps advance
the schedule; after a sweep all row labels return to their original order. Apply
both sides of `J G J^T`, including interactions between different pivot pairs.
Use compact Gram storage, vector angle calculation and batched row normalization.
There are no per-pivot scalar reads or scalar/vector handshakes.

- [x] Preserve pre-change executable and inspect measured AIV bottlenecks.
- [x] Implement static schedule, vector parameters, batched Gram/R updates within
      192 KiB UB; retain at least 128 KiB shared pre/postprocessing scratch.
- [x] Build on server; run independent numerical tests, control/overlap regression.
- [x] Compare old/new 4096x4096 timings on identical seeded input, four repeats,
      discard first, validate outside timing; tune grouping using measurements.
- [x] Profile one mixed launch and actual AIV/AIC utilization. Report limits.
- [x] Update design/results, synchronize both locations, verify hashes/log layout.

Review follow-ups: protect reused Gram storage with V_MTE2 between consecutive
jobs (the partial-grid regression exposed a real DMA/vector race); use
WholeReduceMax to avoid ReduceMax's internal scalar copy and extra handshakes;
use distinct source/destination matrices to remove one row-copy instruction
from each of three transforms per circle round without adding UB.
