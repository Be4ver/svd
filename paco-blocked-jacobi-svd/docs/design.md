# One-side blocked Jacobi on Ascend 910B3

Keep the PACO batch CLI and FP32 row-major input/output. First release supports
1 <= rows <= 4096, 1 <= cols <= 4096, positive batch. The existing PACO row-oriented
one-sided convention is retained: B = Q A; normalized B rows form Vt.

Each sweep uses a complete round-robin tournament over even, padded row blocks.
Default block size is 16 (configurable 16/32); a panel contains two blocks.
Zero rows cover partial blocks. Every round covers every block exactly once.

1. All available AIVs pack disjoint block pairs into contiguous GM panels.
2. All available AICs form panel Gram matrices X X^T using FP32 Cube GEMM.
3. AIVs perform symmetric Jacobi on the small Gram matrices and accumulate dense
   row rotations R, using batched vector angles and contiguous vector updates.
4. All available AICs apply R X and scatter the two result blocks back to B.
   Input panels are separate from B, preventing in-place GEMM aliasing.

One KERNEL_TYPE_MIX_AIC_1_2 kernel contains all operations and sweep control.
The host uploads the raw unpadded matrix, calls blocked_jacobi_fused once, then
downloads final B_rot/sigma/Vt. Each independent --repeat run calls it once.
Normalization, padding, pair generation, convergence, norms, sorting and output
normalization execute on AIV. Gram and the full panel rotation execute on AIC.

Every resident core participates in each DeviceBarrier: all AIVs join a mode 0
barrier, then send mode 2 to their AIC; all AICs join a mode 0 barrier and each
broadcasts ONE mode 2 flag to both AIVs. Sending twice leaves extra notifications
and corrupts repeated barriers. Counts are capped at resident physical cores.
GM control slots use DMA reads on AIV and explicit cache invalidation on AIC.
Grid-stride dispatch splits rotation GEMM by row block and column tile. The
number of useful active cores is bounded by independent work.

The local AIV solver stores G and R compactly, without eightfold replication.
For d=64, each of 63 circle rounds processes 32 disjoint pivots. A fixed
permutation advances the circle in a [P rows, Q rows] layout. Parameters for
all pivots are calculated with vector sqrt/div/compare/select. Each row-update
instruction covers 2048 useful FP32 elements (32 full vector repeats). G is
updated on both sides as J G J^T using transpose Gather maps, including the
cross-pair entries. R receives J R. The schedule permutation returns to identity
after d-1 rounds. Gather maps are initialized once per AIV and reused. Updates
alternate source/destination matrices without copying rows back: six vector
arithmetic instructions per row transform instead of seven, with the same UB.

Only a local-sweep-end WholeReduceMax is read by the scalar pipeline. There are no
per-pivot UB scalar reads, angle branches or S/V handshakes. All R row lengths
are reduced together with WholeReduceSum and corrected by a vector Newton step.
UB allocation is 180 KiB, including 128 KiB reused by preprocessing and output;
immutable maps sit outside this shared region. Block16 uses the same algorithm
with d=32, 16 simultaneous pivots and 512-element row updates.

Between consecutive parameter jobs, V_MTE2 orders the next Gram DMA after the
previous vector use of the reused g_ buffer. MTE3_V alone only delays later V
instructions, so it does not protect this MTE2 overwrite. This dependency is
required when one AIV processes several jobs without a device barrier. Within
each local circle round, parameter calculation and row transforms stay on the
vector pipeline without scalar reads or scalar/vector event handshakes.

The new pivot order changes individual output vectors and may change the number
of sweeps relative to the former lexicographic solver. Compare invariant
numerical properties against independent FP64 references. Serial and overlapped
schedules using the same new solver must still produce identical binary output.

The default schedule splits each round's independent block-pair jobs into A/B.
After all panels have been packed, it executes Gram(A), then Gram(B) alongside
Parameters(A), then Apply(A) alongside Parameters(B), then Apply(B). Every stage
ends in DeviceBarrier, including stages containing concurrent AIC/AIV work.
The Gram, R and panel arrays are already disjoint per job, so this requires no
additional matrix storage or UB. A group's parameter solve only reads its own
Gram and writes its own R; applying another group's R writes disjoint matrix
rows. No next-round reads begin before every current-round write completes.

`--pipeline-jobs 0` balances parameter waves across two groups. With C resident
AIVs, W=ceil(jobs/C) and split=floor(W/2)*C. This gives a whole number of full
AIV waves to group A without increasing the total number of parameter waves.
W=1 gives split=0 (serial); 64 jobs/40 AIV gives 40+24. Explicit positive split
still permits arbitrary experiments, including 32+32 on the target workload.
`--pipeline-jobs N` sets a positive explicit split smaller than jobs.
`--no-overlap` retains the serial schedule; jobs=1 automatically uses it.
All cores execute every barrier even for an empty local task assignment.
Activity flags accumulate across both parameter groups, so sweep termination
is unchanged. Each group's grid-stride indices start at zero before adding
its global job offset. This avoids dropping AIV/AIC utilization on group B.

This first schedule has a global stage boundary rather than per-job readiness;
it adds one barrier per round and may reduce AIV utilization when splitting a
small job count. Measure actual decomposition latency against `--no-overlap`.
Overlapping phases alone does not guarantee a speedup or full Cube utilization.

Stop only after a complete sweep whose every initial panel Gram is converged,
or maxSweeps. Local eigensolve has its own bounded sweep count. Singular values
come from actual final B row norms, never from sqrt(Gram eigenvalues).
Per-matrix max-absolute-value scaling and powers-of-two staging on AIV protect
extreme input scales. Final restoration and binary layout conversion also run
on AIV; the host only checks completed outputs for representability/finite data.

The whole matrix and packed panels stay in GM. AIV UB holds only the current
small Gram/R or a row cache plus O(rows) norm/sort arrays, with the Gram scratch
reused between these phases. Postprocessing uses a descending heap permutation.
Before exporting R, a Newton row-length correction suppresses accumulated FP32
Givens norm drift over thousands of block rounds. All full-panel updates remain
on Cube. Current AIV measurements and numerical evidence are in
[aiv-vectorization-results.md](aiv-vectorization-results.md). Earlier overlap
measurements are in overlap-results.md; 4096-results.md preserves the earlier
serial measurements.

FP32 Gram can lose small singular subspaces; test random, rectangular, zero,
identity, repeated spectrum, rank deficient and controlled-condition inputs.
Measure kernel/device and end-to-end latency separately. Report limitations.

All source changes stay in this directory; baseline PACO is read-only.
Server mirror: /home/zhushengrui/project/paco-blocked-jacobi-svd.
