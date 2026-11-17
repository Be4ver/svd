# Blocked Jacobi implementation plan

Goal: deliver a buildable, verified 910B3 batch operator with Cube Gram and rotation.
Architecture: round-robin packed panels; AIC Gram; AIV small symmetric eigensolve;
AIC dense application. A single mixed kernel and device barriers establish dependencies.
Tech stack: Ascend C, ACL, CMake, Python/NumPy. Spec: design.md.

- [x] Add numerical tests for schedule coverage, accumulated rotation, singular
  values, orthogonality, zero/repeated/rank-deficient/rectangular cases; observe
  missing implementation failure, then implement a CPU reference.
- [x] Implement standalone ACL driver and Ascend kernels, reusing PACO GEMM with
  explicit split-K accumulation. Validate on server against independent NumPy SVD.
- [x] Run an automated NPU case suite and benchmark representative baseline/new
  shapes; capture device core counts, task/core upper bounds and profiler evidence.
- [x] Document build/run, numerical and occupancy limitations; sync both trees
  and verify source hashes and git diff cleanliness.

Profiling evidence and ACL event timings are recorded under `artifacts/` on the
server; see the deliverable README for the profiling invocation and limits.

The original phase-launch implementation is superseded by the single-launch
revision in [single-launch-plan.md](single-launch-plan.md).
