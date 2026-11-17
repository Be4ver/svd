# Block64 AIV implementation plan

Goal: support paired 64-row blocks (128x128 Gram) while preserving one mixed
kernel launch and full-panel AIC rotation. No baseline directory is modified.

Keep G and R resident; use bounded tiles for in-place row updates and vector
repeat strides for column updates. Replace full transpose Gather maps and R
permutation maps with in-place moves. Retain small pivot and column permutation
indices where necessary. GEMM already computes Gram with transposed B; its
transpose flag cannot replace the local AIV bilateral Jacobi update.

- [x] Add block64 numerical cases and observe current CLI rejection.
- [x] Implement bounded UB layout, tiled updates and block64 CLI support.
- [x] Define FP32 stopping tolerance and distinguish it from output accuracy;
      do not claim measured equivalence to GPU XGESVDJ without GPU evidence.
- [x] Compile and validate block16/32/64 and single-launch control serially.
- [x] Benchmark and independently verify 4096x4096, compare at equal tolerance,
      and document useful-core limits. All logs go in logs/.
- [x] Complete independent static review; no critical implementation defect found.
- [x] Complete final local/server hash verification.

Evidence and limitations: [block64-results.md](block64-results.md).
