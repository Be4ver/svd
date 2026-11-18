#pragma once
#include "core_local_sync.hpp"

namespace blocked_jacobi {
#ifdef __DAV_C220_VEC__
__aicore__ inline bool VectorSweep(FusedVector& vec, CoreSignals& signals,
    int nb, int jobs, int sweep, int localSweeps, float tol, int64_t& sequence)
{
    const int worker = GetBlockIdx(), groups = GetBlockNum(), np = nb / 2;
    bool active = false;
    for (int round = 0; round < nb - 1; ++round) {
        for (int ordinal = 0; WorkerJob(worker, ordinal, groups) < jobs; ++ordinal) {
            const int job = WorkerJob(worker, ordinal, groups);
            const int m = job / np, pair = job % np;
            const int a = m * nb + PairBlock(pair, 0, round, nb);
            const int b = m * nb + PairBlock(pair, 1, round, nb);
            const int epoch = sweep * (nb - 1) + round;
            if (!signals.Wait(signals.Version(a), epoch)) return active;
            if (!signals.Wait(signals.Version(b), epoch)) return active;
            vec.PackJob(job, round);
            signals.Publish(signals.Request(worker), ++sequence);
            if (!signals.Wait(signals.Ack(worker), sequence)) return active;
            const bool changed = vec.ParametersJob(job, localSweeps, tol, round != 0);
            if (changed) active = true;
            ++sequence;
            // A negative even request denotes an exactly identity rotation.
            // Ack remains positive; no extra GM field or cache line is needed.
            signals.Publish(signals.Request(worker), changed ? sequence : -sequence);
            if (!signals.Wait(signals.Ack(worker), sequence)) return active;
            // X/R scratch can now be reused; both sides derive the next job.
        }
    }
    return active;
}
#elif defined(__DAV_C220_CUBE__)
__aicore__ inline void CubeSweep(FusedCube& cube, CoreSignals& signals,
    int nb, int jobs, int sweep, int64_t (&served)[2])
{
    const int group = GetBlockIdx(), groups = GetBlockNum(), np = nb / 2;
    const int count = group < jobs ? (jobs - 1 - group) / groups + 1 : 0;
    const int lanes = count < 2 ? count : 2;
    int32_t cursor[2] = {group, group + groups}, rounds[2] = {0, 0};
    int64_t completed = 0;
    int polls = 0;
    int64_t lastProgress = GetSystemCycle();
    // Poll every populated slot. Blocking on a particular lane could prevent a ready
    // Apply on the other lane from releasing a dependency of the first one.
    while (completed < static_cast<int64_t>(count) * (nb - 1)) {
        if ((++polls & 1023) == 0) {
            if (signals.Aborted()) return;
            if (GetSystemCycle() - lastProgress > 2000000000LL) { signals.Abort(); return; }
        }
        for (int lane = 0; lane < lanes; ++lane) {
            const int worker = group * 2 + lane;
            const int64_t request = signals.Read(signals.Request(worker));
            const int64_t token = request < 0 ? -request : request;
            if (token != served[lane] + 1) continue;
            const int job = cursor[lane], round = rounds[lane];
            if (token % 2) {
                cube.GramJob(job);
            } else {
                if (request > 0) cube.ApplyJob(job, round);
                const int m = job / np, pair = job % np;
                const int epoch = sweep * (nb - 1) + round + 1;
                // A changed panel has completed FIX_S for BOTH blocks. An
                // identity leaves the already-acquired B untouched. In either
                // case both inputs are valid before publishing either version.
                signals.Publish(signals.Version(m * nb + PairBlock(pair, 0, round, nb)), epoch);
                signals.Publish(signals.Version(m * nb + PairBlock(pair, 1, round, nb)), epoch);
                AdvanceWorkerJob(cursor[lane], rounds[lane], worker, groups, jobs);
                ++completed;
            }
            signals.Publish(signals.Ack(worker), token);
            served[lane] = token;
            lastProgress = GetSystemCycle();
        }
    }
}
#endif
}
