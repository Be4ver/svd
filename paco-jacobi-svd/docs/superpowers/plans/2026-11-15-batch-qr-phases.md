# Batch QR phases implementation plan

Goal: all matrices in a batch finish W projection before any matrix enters C update; split QR runs once for all internal nodes in a group.

Architecture: retain ordered K accumulation, store disjoint near/tail/Q projections, publish them with one batch barrier, and distribute output tiles independently. Reuse finished QDWH buffers for split P±, Q, UΩ, and H±.

Constraints: remote production is authoritative; local root sources are stale. NPU runs are serial. Strict full-sigma atol=rtol=1e-5; target condition <=1000 uses four iterations, larger conditions use six initially.

- [x] Snapshot current production into perf_20261115_batch_phases/baseline and candidate.
- [x] Refactor qr_custom.cpp to projection, publication, update, completion phases across the whole batch. Use the same final-panel path for all shapes. Preserve stable near K reduction and tail/Q K order.
- [x] Batch split stages in sdc_node.hpp and fused_qdwh_sdc.hpp. Give each node independent trace, Q, P±, UΩ, and H± slices. Add Q offset to AIV identity initialization.
- [x] Build and validate direct QR on square, tall, wide, mixed batches with 16/20 AICs; validate fused sigma across sizes, structured matrices, and conditions.
- [x] Obtain independent review, address findings, compare timings against the current production snapshot with alternating order.
- [ ] Improve large-matrix performance and resolve 16-core regression before production deployment. Candidate, patch, metrics and report are exported; production remains unchanged.

Buffer lifetimes: let F=sum(nPad_i^2). Split Q=qbuf[0:F], UΩ=qbuf[F:3F]; after QR, TmpAll reuses qbuf[F:2F]. P± use pt/xt[0:F], and after their consumers finish H± reuse those slices. Traces use a0[128*i:128*(i+1)] after QDWH completion, accommodating the host limit of 64 AICs. CopyRowsPad consumes the matching pt/xt node slice before the next group reuses these buffers.
