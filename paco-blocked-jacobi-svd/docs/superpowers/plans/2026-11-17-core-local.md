# Core-local Jacobi pipeline implementation plan

**Goal:** Replace round phase barriers with two independent AIV producer slots per AIC and real GM block dependencies, then measure full 4096 decomposition.

**Architecture:** Each AIV owns an ascending queue of (round, job) tasks and waits for both block versions. Its AIC polls two request mailboxes, executes Gram or full-panel Apply, and publishes acknowledgements after DMA completion. Each 64-byte signal line has one writer at a time; block versions transfer ownership only after the previous Apply. Initialization, convergence and output retain DeviceBarrier.

**Constraints:** One mixed launch; block16/32/64; FP32; local-sweeps=1; exact existing pivot schedule; unchanged tolerance and CLI; preserve existing edits; no commits/pushes. Existing --no-overlap and explicit --pipeline-jobs paths remain available for comparison. Run all device jobs serially. All new logs live under logs/core-local (test logs may retain a case subtree).

- [x] Read current sources and compare local/server content after newline normalization; archive both initial trees and baseline executable.
- [x] Inspect installed CANN C220 core numbering, KFC scalar mailbox and GroupBarrier implementations.
- [x] Extend static dependency testing to real worker queues, padding, multiple batches, partial and idle groups, and two-slot scheduler progress under randomized delays.
- [x] Add task-specific Pack/Gram/Parameters/Apply entry points, 64B request/ack/version lines, explicit MTE3_S/FIX_S releases and scalar cache invalidation before acquire.
- [x] Device-construct deterministic queues; add default async branch while retaining serial/split controls. Reset all signals on every launch and use monotonic versions through sweep boundaries.
- [x] Build on server; run independent FP64 single-sweep comparison before full convergence testing. Inspect build and profiler evidence for AIC work and ordering; readable AICore disassembly was unavailable, so document this limit.
- [x] Serially run numerical, block64, control, split/serial equivalence and repeated-call regressions; exercise scheduler handoffs beyond one wave.
- [x] Time same SHA256 4096 input repeatedly; discard first timing; independently validate FP64 errors; compare 966.515 ms and preserved executable.
- [x] Audit unused functions; remove Process2 and the constant cubeidx/dead notification branch, rebuild, repeat all suites and 4096 timings, and profile the cleaned executable. Document every maintained file in docs/file-guide.md.
- [x] Record exact protocol, evidence, limitations and performance decision; synchronize source/docs/tests/logs and compare hashes. Leave changes uncommitted.

The user has explicitly authorized implementation and validation using the most suitable schedule; routine implementation decisions are resolved within that scope. This plan is executed inline.

