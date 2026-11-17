# AIV/AIC phase overlap implementation plan

**Goal:** Overlap local AIV Jacobi parameter solving with AIC Gram/apply GEMMs,
keeping one mixed-kernel launch and the existing output and convergence rules.

**Approved design:** Two disjoint groups within each round. Pack all panels,
then execute Gram(A); Gram(B) alongside Parameters(A); Apply(A) alongside
Parameters(B); Apply(B). Each semicolon is a full device barrier. Retain the
round-end barrier because subsequent pairings reuse the updated matrix rows.
Per-job GM panel/Gram/R storage is already separate, so no additional UB or GM
matrix buffers are required. No cross-round overlap or asynchronous polling.

**Execution:** Inline in the existing user-requested local/server project
directories. Keep the PACO baseline read-only and preserve all existing outputs.
Do not move this untracked project into a different worktree or commit unrelated
workspace contents. Logs must remain under `logs/` on both hosts.

**Measured refinement:** Small cases with jobs <= AIV count nearly doubled
parameter time after splitting. Auto mode therefore keeps these cases serial;
explicit positive splits can still force overlap. Target measurements found
40+24 faster than 32+32. Auto now balances parameter waves: W=ceil(jobs/AIV),
split=floor(W/2)*AIV. This keeps the original total wave count, fills A's AIV
waves, and selects 40+24 for 4096/block32.

## Task 1: Behavioral regression before implementation

- [x] Add `tests/run_overlap.py` exercising real ACL runs, independent NumPy
  validation and exact output/sweep equality against `--no-overlap`.
- [x] Cover default even split, explicit uneven split, odd task count, singleton,
  partial AIV grids, repeated calls and fixed-sweep termination.
- [x] Run one default overlap case against the current binary and observe that
  `phase_overlap=false` fails the new execution contract.

## Task 2: Kernel and CLI

- [x] Refactor Cube Gram/apply loops into range helpers in `fused_cube.hpp`.
  Local task indices map to global job indices using `begin + localJob`.
- [x] Add `FusedVector::Parameters(begin, end, localSweeps, tol)` with local
  grid-stride assignment so all available AIVs can process either group.
- [x] Add kernel `split` argument. Split zero retains the serial path. Positive
  split uses the four stages above; all cores join every barrier, including
  those with no assigned work. Accumulate active flags from both groups.
- [x] Add `--no-overlap` and `--pipeline-jobs N`. Default N=0 uses the measured
  wave-aligned split above; smaller workloads stay serial. Reject negative
  sizes and a nonzero split >= jobs.
  Record actual groups, phase overlap and per-phase useful-core bounds.
- [x] Extend `scripts/benchmark_large.py` to pass these options through.

## Task 3: Real-device validation and measurements

- [x] Preserve the pre-change server executable as an artifact before rebuilding.
- [x] Build with CANN 8.2.RC1 on server, logging to `logs/`.
- [x] Run overlap comparisons, existing NPU numerical/control regressions and
  local CPU tests. No tolerance changes.
- [x] Run 4096x4096/block32/local-sweeps1/tol1e-6 using the same seeded input,
  four repeats per configuration, discarding the first. Compare serial, auto
  32+32 and explicit 40+24. Validate outputs outside timing.
- [x] Collect profiler evidence of one mixed launch with 20 AIC/40 AIV and report
  measured utilization independently from non-profiled timings.
- [x] Update README/design/results with actual behavior, timing boundaries and
  limitations. Sync source and evidence to local/server and verify hashes.
