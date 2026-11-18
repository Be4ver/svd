#include "fused_common.hpp"
#include "fused_vector.hpp"
#include "fused_cube.hpp"
#include "core_local_pipeline.hpp"
using namespace AscendC;
using namespace blocked_jacobi;

extern "C" __global__ __aicore__ void blocked_jacobi_fused(
    GM_ADDR input, GM_ADDR matrix, GM_ADDR panels, GM_ADDR gram, GM_ADDR rotations,
    GM_ADDR flags, GM_ADDR scales, GM_ADDR status, GM_ADDR bout, GM_ADDR sigma, GM_ADDR vt, GM_ADDR sync,
    int32_t rows, int32_t cols, int32_t batch, int32_t block, int32_t maxSweeps,
    int32_t localSweeps, float tol, int32_t fixed, int32_t split)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    int nb = ((rows + block - 1) / block + 1) / 2 * 2;
    if (nb < 2) nb = 2;
    const int cp = (cols + 31) / 32 * 32;
    const int jobs = batch * (nb / 2);
    CoreSignals signals;
    signals.Init(sync, GetBlockNum() * 2, batch * nb);
#ifdef __DAV_C220_VEC__
    FusedVector vec;
    vec.Init(&pipe, input, matrix, panels, gram, rotations, flags, scales, status,
             bout, sigma, vt, rows, cols, batch, block);
    vec.Initialize();
    int64_t sequence = 0;
    if (split < 0) {
        for (int line = GetBlockIdx(); line < 4 * GetBlockNum() + batch * nb + 1; line += 2 * GetBlockNum())
            signals.Publish(line, 0);
    }
#else
    FusedCube cube;
    cube.Init(&pipe, matrix, panels, gram, rotations, block, nb, cp);
    GlobalTensor<int32_t> state;
    state.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status) + GetBlockNum() * 2 * 8);
    int64_t served[2] = {0, 0};
#endif
    DeviceBarrier();
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
#ifdef __DAV_C220_VEC__
        bool active = false;
#endif
        if (split < 0) {
#ifdef __DAV_C220_VEC__
            active = VectorSweep(vec, signals, nb, jobs, sweep, localSweeps, tol, sequence);
#else
            CubeSweep(cube, signals, nb, jobs, sweep, served);
#endif
        } else for (int round = 0; round < nb - 1; ++round) {
#ifdef __DAV_C220_VEC__
            vec.Pack(round);
#endif
            DeviceBarrier();
#ifdef __DAV_C220_CUBE__
            cube.Gram(0, split > 0 ? split : jobs);
#endif
            DeviceBarrier();
            // A and B are disjoint block-pair groups in this round. Both core
            // types execute useful work between the same two global barriers.
            if (split > 0) {
#ifdef __DAV_C220_CUBE__
                cube.Gram(split, jobs);
#else
                if (vec.Parameters(0, split, localSweeps, tol, round != 0)) active = true;
#endif
                DeviceBarrier();
#ifdef __DAV_C220_CUBE__
                cube.Apply(0, split, round);
#else
                if (vec.Parameters(split, jobs, localSweeps, tol, round != 0)) active = true;
#endif
                DeviceBarrier();
#ifdef __DAV_C220_CUBE__
                cube.Apply(split, jobs, round);
#endif
            } else {
#ifdef __DAV_C220_VEC__
                if (vec.Parameters(0, jobs, localSweeps, tol, round != 0)) active = true;
#endif
                DeviceBarrier();
#ifdef __DAV_C220_CUBE__
                cube.Apply(0, jobs, round);
#endif
            }
            // Every row is complete before the next round changes pairings.
            DeviceBarrier();
        }
#ifdef __DAV_C220_VEC__
        vec.PublishActive(active);
#endif
        DeviceBarrier();
#ifdef __DAV_C220_VEC__
        vec.ReduceConvergence(sweep + 1);
#endif
        DeviceBarrier();
#ifdef __DAV_C220_VEC__
        const bool converged = vec.Converged();
#else
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(state[0]);
        const bool converged = state.GetValue(1) != 0;
#endif
        if (split < 0 && signals.Aborted()) break;
        if (!fixed && converged) break;
    }
#ifdef __DAV_C220_VEC__
    if (split >= 0 || !signals.Aborted()) vec.Finalize();
#endif
    DeviceBarrier();
}
