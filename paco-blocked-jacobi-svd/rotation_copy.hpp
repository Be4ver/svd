#pragma once
#include "kernel_operator.h"

namespace blocked_jacobi {
// Rotation workspaces use 32 FP32 elements or a multiple of 64, at most
// 8192 elements. Copy's mask is per repeat; it has no count-only overload.
// Every call has disjoint accessed source/destination elements. In-place
// row shifts still go through the existing tile scratch, never a memmove.
__aicore__ inline void CopyRotationBuffer(AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src, int32_t count)
{
    AscendC::Copy(dst, src, static_cast<uint64_t>(count < 64 ? count : 64),
        static_cast<uint8_t>((count + 63) / 64), {1, 1, 8, 8});
}
}
