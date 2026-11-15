# QDWH/SDC + Single-Sided Jacobi SVD Fusion Design

Date: 2026-11-12
Location: `/home/zhushengrui/project/svd/qdwh_sdc_fused_jacobi`

## Goal

Fuse the existing device-side QDWH/SDC operator with the 64x64-oriented single-sided Jacobi backend so one host launch transforms an input matrix into PSD symmetric SDC leaves and directly returns the singular values of the original matrix.

The primary workload is small batch latency (batch 1--40) and leaf sizes no larger than 64. The first executable path remains the existing single-matrix QDWH/SDC interface; a large matrix may produce multiple SDC leaves.

## Existing constraints

- The current QDWH/SDC implementation is one mixed AIC/AIV kernel launch.
- QDWH writes the root H into the workspace arena; SDC writes compact leaf matrices into `L.leaf` and metadata into `L.leafN`, `L.leafLd`, and `L.leafOff`.
- The current launch uses 20 AIC plus 40 AIV logical cores (`BLOCKDIM=20`).
- SDC leaves have dimensions at most `threshold` and the target setting is `threshold=64`.
- The 64x64 Jacobi research implementation keeps one matrix resident in UB and runs 63 tournament rounds per sweep.

## Proposed architecture

The kernel remains `qdwh_sdc_fused_custom` and is launched exactly once.

1. Phase 1 runs the existing QDWH path.
2. The existing phase barrier releases the SDC phase.
3. SDC recursively produces leaves exactly as it does now.
4. A second device barrier is added after the SDC queue is completely drained and leaf metadata is committed.
5. AIV cores partition the finalized leaf index range. Each leaf owner loads the compact leaf into UB, symmetrizes it with `H = 0.5*(H + H^T)`, runs resident tournament Jacobi for `n` rows, using at most 63 rounds per sweep, and writes `n` singular values to a new `L.sigma` region plus per-leaf sigma metadata.
6. A final cross-core barrier guarantees all sigma writes are visible before kernel exit.

The PSD property means the eigenvalues of each symmetrized leaf are the singular values contributed by that leaf. The Jacobi path therefore operates on the leaf itself; no U/V vectors are required.

## Output contract

The workspace gains:

- `sigma`: contiguous float32 output with capacity equal to `leafCap` scalar slots. The first `sum(leafN[i])` values are valid.
- `sigmaOff`: per-leaf scalar offset into `sigma`.
- `sigmaCount`: per-leaf count (equal to `leafN[i]`).
- `sigmaLeafCount`: total leaf count copied from the SDC counter.

Each leaf sigma segment is sorted in descending order. Cross-leaf global sorting is intentionally outside the fused kernel in this first version: it adds synchronization and work that is not needed to expose all singular values, while metadata gives the host a deterministic merge path. Output leaf order follows SDC leaf order.

The existing leaf matrix output and metadata remain available for verification and debugging.

## Synchronization and ownership

The SDC AIC side owns the final leaf count and leaf metadata. The post-SDC barrier is entered only after AIC0 writes `LeafCount` and all AIC cores clean metadata cache lines. Every AIV invalidates the metadata cache line before reading it.

For postprocessing, leaf index `i` is owned by AIV logical core `i % (2*BLOCKDIM)`. Each owner writes disjoint sigma and metadata entries, so no atomic operations are needed. A final barrier is used before kernel return.

The existing SDC barrier flags are not reused until the SDC loop has exited. New flags are allocated in the unused debug/fusion range and paired by the existing mode=2 cross-core protocol.

## UB and performance strategy

- The 64x64 fast path uses the resident tournament implementation from `/tmp/jacobi64-study-20261112-iOuF3K/src-tournament`.
- For `n<64`, rows are loaded with the existing padded row-copy helper and the inactive tail is ignored by the Jacobi loop.
- For a leaf larger than 64, the kernel treats it as an error/overflow condition; normal SDC operation must guarantee `n<=threshold<=64`.
- Symmetrization is fused into the UB load/store phase and does not create a separate kernel or GM round trip.
- The postprocessing loop is skipped when the leaf count is zero or the SDC overflow flag is set.
- Only one AIV owner processes a leaf, avoiding duplicate GM traffic. Multiple leaves overlap across AIVs; one 64x64 leaf uses one AIV.

## Host changes

The host allocates additional sigma and sigma metadata ranges through `FusedLayout::Build`, initializes counters, launches the existing kernel once, and downloads only sigma/meta for the fused SVD result. Optional debug mode still downloads leaves and H/U.

For timing, the host adds ACL event pairs around the fused launch, an independent QDWH/SDC launch, and an independent Jacobi launch on the produced leaves. Each measurement uses the same input, one warm-up, and repeated steady-state samples. Device time is ACL event elapsed time; host file I/O and result copies are excluded.

## Correctness and acceptance

For each leaf, compare fused sigma against CPU reference eigenvalue/SVD after sorting descending. The full output is correct when maximum relative/absolute sigma error meets the existing Jacobi tolerance, sigma count equals the sum of SDC leaf sizes, metadata offsets are in range and non-overlapping, exactly one fused kernel launch is issued, and fused device time is no greater than independent QDWH/SDC plus Jacobi event times within measurement noise.

Test shapes include a single 64x64 input and a larger input producing multiple leaves. Standalone Jacobi microbenchmark batches 1, 8, and 40 are retained; the fused operator remains single-matrix until a batched QDWH input interface is added.

## Risks and fallback

- If the compiler rejects a cross-core flag or API used by the resident Jacobi path, use validated barrier primitives and scalar compare/swap fallback with the same ownership protocol.
- If a leaf is numerically nonsymmetric beyond tolerance, symmetrization is mandatory before Jacobi and asymmetry is recorded in debug counters.
- The first fused version does not globally merge leaf sigma segments on device. A host merge is required when a globally sorted spectrum is requested.