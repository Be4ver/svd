# QDWH/SDC + Jacobi Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Extend the existing one-launch QDWH/SDC kernel to symmetrize each finalized SDC leaf, compute its singular values with resident tournament Jacobi, and return sigma and metadata.

**Architecture:** Preserve QDWH and SDC through leaf finalization. Add a cross AIC/AIV completion barrier, then assign leaves to AIV owners for UB-resident symmetrization and Jacobi. Extend the workspace, host output, numerical verification, and ACL event measurements.

**Tech Stack:** Ascend C/C220, ACL runtime, CMake, Python and NumPy.

**Spec:** docs/superpowers/specs/2026-11-12-qdwh-sdc-jacobi-fusion-design.md

## Global Constraints

- Fused kernel launch count is exactly one.
- SDC leaves have dimensions at most 64 and threshold is 64.
- Output preserves SDC leaf order; each leaf sigma segment is sorted descending; cross-leaf merge stays on host.
- The original qdwh_sdc_fused directory is never modified.
- Fused event time must be compared with independent QDWH/SDC and Jacobi event times on the same input.

---

### Task 1: Extend shared workspace and host output

**Files:**
- Modify fused_layout.hpp
- Modify fused_main.cpp

**Interfaces:**
- Add sigma, sigmaOff, sigmaCount, and sigmaLeafCount workspace offsets.
- sigma uses leafCap floats; sigmaOff and sigmaCount use maxLeaves int32 entries; sigmaLeafCount is one int32.

- [ ] Add the four fields to FusedLayout and reserve the corresponding ranges in Build, keeping offsets in float units.
- [ ] Initialize the sigma metadata ranges to zero alongside the existing leaf arrays before launch.
- [ ] After kernel synchronization, download sigmaLeafCount and valid sigma metadata, then write sigma.bin and sigma_meta.txt.
- [ ] Build with source /usr/local/Ascend/ascend-toolkit/set_env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target fused_main -j16.

### Task 2: Implement leaf-resident Jacobi and synchronization

**Files:**
- Create jacobi_leaf.hpp
- Modify fused_qdwh_sdc.hpp

**Interfaces:**
- JacobiLeafWorker::Init(TPipe*) initializes UB buffers and hard-event IDs.
- JacobiLeafWorker::Run loads one leaf, symmetrizes it, applies resident tournament Jacobi, computes row norms as singular values, sorts descending, and stores sigma.

- [ ] Port only the resident 64x64 Jacobi worker from /tmp/jacobi64-study-20261112-iOuF3K/src-tournament/jacobi_svd_custom.cpp; omit the standalone kernel and host.
- [ ] Load the compact leaf into a 64x64 UB tile, zero inactive rows and columns, and compute each symmetric pair as 0.5*(H[r,c]+H[c,r]).
- [ ] Run 63 tournament rounds per sweep, preserving the matching schedule and existing convergence threshold.
- [ ] After Jacobi, reduce each row's squared elements, take sqrt, sort n values descending, and store them to sigma.
- [ ] Add one AIC-to-AIV completion handshake after SdcCube has written and cleaned final leaf metadata; AIV invalidates and reads leaf count and metadata before processing.
- [ ] Assign leaf i to AIV i modulo 2*BLOCKDIM; add a final AIV-to-AIC/AIC-to-AIV handshake before kernel return.
- [ ] Compile with source /usr/local/Ascend/ascend-toolkit/set_env.sh && cmake --build build --target fused_kernels -j16.

### Task 3: Verify fused sigma output

**Files:**
- Create verify_sigma.py

**Interface:**
- Command: python3 verify_sigma.py --output-dir output --threshold 64.
- Read sigma.bin, sigma_meta.txt, and leaf_h.bin; compare each segment with sorted absolute eigenvalues of the symmetrized CPU leaf.

- [ ] Implement file-size, leaf-count, offset, and count validation.
- [ ] Compare each sigma segment with numpy.linalg.eigvalsh(0.5*(H+H.T)), sorted descending and absolute-valued; report max absolute and relative errors.
- [ ] Return a failing exit status if an offset is invalid, count differs from leaf size, or max error exceeds the existing Jacobi tolerance.
- [ ] Run the verifier against existing output to confirm it reports missing sigma files clearly, then against fresh device output.

### Task 4: Measure ACL device time

**Files:**
- Modify fused_main.cpp
- Modify utils.h

**Interfaces:**
- Add an ACL-event measurement helper using aclrtCreateEvent, aclrtRecordEvent, aclrtSynchronizeEvent, and aclrtEventElapsedTime.
- Report device event milliseconds separately from host wall time.

- [ ] Add one warm-up and five measured fused launches, reinitializing mutable workspace metadata before each launch.
- [ ] Keep one launch in the production invocation; benchmark mode may repeat that one-launch operation five times.
- [ ] Print the exact kernel launch count and event timing without including host copies or file I/O.
- [ ] Establish independent QDWH/SDC and Jacobi event samples on identical input and report their sum and fused/sum ratio.

### Task 5: Build, numerical, and performance validation

**Files:**
- Modify run.sh only if needed for validation switches
- Create output/validation_notes.txt

- [ ] Build all targets and run a 64x64 threshold-64 input; run python3 verify.py --threshold 64 and python3 verify_sigma.py --threshold 64.
- [ ] Run a 1024x1024 threshold-64 input to exercise multiple SDC leaves; run both verifiers.
- [ ] Confirm launch count equals one for the fused invocation and sigma count equals the sum of leaf sizes.
- [ ] Record event samples, independent stage sum, ratio, device load, shapes, leaf counts, and numeric errors.
- [ ] If device utilization remains above 90 percent, report the measurements as noisy and repeat once the device is quieter; do not claim the timing target passed without comparable event data.