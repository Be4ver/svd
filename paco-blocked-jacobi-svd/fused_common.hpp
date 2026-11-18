#pragma once
#include "kernel_operator.h"
#include "core_local_schedule.hpp"

namespace blocked_jacobi {
using namespace AscendC;
constexpr int32_t kDone = 0x424a5356;

// Every resident AIV and AIC participates, including cores with no work.
// Mode 0 orders all producers of one core type. Mode 2 joins both types.
// Flags are balanced on every iteration before reuse.
__aicore__ inline void DeviceBarrier()
{
#ifdef __DAV_C220_VEC__
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0);
    CrossCoreWaitFlag(0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(2);
    CrossCoreWaitFlag(3);
#else
    CrossCoreWaitFlag(2);
    CrossCoreSetFlag<0x0, PIPE_FIX>(1);
    CrossCoreWaitFlag(1);
    CrossCoreSetFlag<0x2, PIPE_FIX>(3);
#endif
}
}
