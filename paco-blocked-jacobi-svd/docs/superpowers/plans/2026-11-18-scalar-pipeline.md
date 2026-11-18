# Scalar, angle and AIV pipeline experiments

Continue the authorized optimization without changing the CLI, FP32-epsilon
criterion, exact-once pair schedule, single mixed launch, or the cross-group
release/acquire protocol. Preserve all pre-existing edits and do not commit.

1. Save the current source/executable hashes and a recoverable source snapshot.
2. Inspect C220 vector and synchronization implementations. Measure candidates
   separately: angle arithmetic; inactive-slot polling/static mailbox payload;
   UB/DMA lifetime changes. Do not infer speed from instruction counts.
3. For arithmetic changes run the independent FP64 single-sweep comparison and
   full numerical cases before accepting a 4096 full-SVD result. Keep failures.
4. Time the fixed 4096 input at unchanged tolerance, compare warm medians to the
   saved current executable and 966.515 ms historical result. All NPU work runs
   serially. Keep newly created logs under logs/scalar-pipeline/.
5. Retain only verified improvements. Run schedule, numerical, control, repeated
   invocation and alternate-scheduling regression; profile the final executable.
6. Document arithmetic, communication, actual overlap and remaining limits;
   synchronize maintained files and compact evidence, checking their hashes.

The previous profiler reports AIC scalar 90.8% (including busy waits), AIV vector
57.4%, scalar 19.4%, MTE2 2.6% and MTE3 2.4%. These counters overlap and do not
establish per-function runtime or prove double-buffer overlap.
