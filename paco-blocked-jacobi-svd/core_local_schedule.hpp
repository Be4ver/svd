#pragma once
#include <cstdint>

namespace blocked_jacobi {
__aicore__ inline int32_t PairBlock(int32_t pair, int32_t half, int32_t round, int32_t nb)
{
    const int32_t pos = half == 0 ? pair : nb - 1 - pair;
    return pos == 0 ? 0 : 1 + (pos - 1 + nb - 1 - round) % (nb - 1);
}

// Adjacent AIV IDs share an AIC. Distribute the first wave across all AICs
// before assigning their second AIV; queues remain ascending in (round, job).
__aicore__ inline int32_t OwnerWorker(int32_t job, int32_t groups)
{ return 2 * (job % groups) + (job / groups) % 2; }

__aicore__ inline int32_t WorkerJob(int32_t worker, int32_t ordinal, int32_t groups)
{ return worker / 2 + (worker % 2) * groups + ordinal * 2 * groups; }

// AIC derives the same queue as its AIV; advance only after Apply is complete.
__aicore__ inline void AdvanceWorkerJob(int32_t& job, int32_t& round,
    int32_t worker, int32_t groups, int32_t jobs)
{
    job += 2 * groups;
    if (job >= jobs) { job = WorkerJob(worker, 0, groups); ++round; }
}
}
