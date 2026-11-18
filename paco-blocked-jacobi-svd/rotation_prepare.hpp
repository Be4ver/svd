#pragma once
#include "kernel_operator.h"
#include "rotation_copy.hpp"

namespace blocked_jacobi {
namespace rotation_prepare {
using namespace AscendC;

// All helpers run before the first Gram load. Reuse G/R as temporary grids;
// immutable maps/masks live beyond the shared preprocessing scratch region.
__aicore__ inline void Templates(LocalTensor<float> sequence, LocalTensor<float> next,
    LocalTensor<float> work, LocalTensor<uint8_t> mask, int32_t d, bool crossOnly)
{
    // C220 CreateVecIndex uses eight scalar seed stores internally, then Adds.
    // Generate once, including padding for the 32-element Compare case.
    const int32_t count = d < 64 ? 64 : d;
    CreateVecIndex(sequence, 0.0f, static_cast<uint32_t>(count));
    if (crossOnly) {
        // Keep P fixed and cycle Q left, pairing each P with every Q once.
        Adds(next, sequence, 1.0f, count);
        CompareScalar(mask, sequence, static_cast<float>(d / 2), CMPMODE::LT, count);
        Select(next, mask, sequence, next, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        CompareScalar(mask, sequence, static_cast<float>(d - 1), CMPMODE::NE, count);
        Select(next, mask, next, static_cast<float>(d / 2), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        return;
    }
    Adds(next, sequence, 1.0f, count);
    Adds(work, sequence, -1.0f, count);
    CompareScalar(mask, sequence, static_cast<float>(d / 2), CMPMODE::LT, count);
    Select(next, mask, work, next, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    CompareScalar(mask, sequence, 0.0f, CMPMODE::NE, count);
    Select(next, mask, next, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    // Tensor-scalar Select keeps the tensor where the predicate is true.
    CompareScalar(mask, sequence, 1.0f, CMPMODE::NE, count);
    Select(next, mask, next, static_cast<float>(d / 2), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    CompareScalar(mask, sequence, static_cast<float>(d - 1), CMPMODE::NE, count);
    Select(next, mask, next, static_cast<float>(d / 2 - 1), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
}

__aicore__ inline void Columns(LocalTensor<float> columns, LocalTensor<float> seed, int32_t d)
{
    CopyRotationBuffer(columns, seed, d);
    // Double the number of complete rows with each instruction sequence.
    for (int32_t count = d; count < d * d; count *= 2)
        CopyRotationBuffer(columns[count], columns, count);
}

__aicore__ inline void Grids(LocalTensor<float> rows, LocalTensor<float> columns,
    LocalTensor<float> seed, LocalTensor<float> work, int32_t d)
{
    Brcb(work, seed, d / 8, {1, 8});
    Brcb(columns, work, d, {1, 8});
    // Brcb broadcasts to 64 FP32 columns. Compact for d=32, or duplicate
    // both halves for d=128; each call covers every row in the matrix.
    const int32_t width = d < 64 ? d : 64;
    Copy(rows, columns, static_cast<uint64_t>(width), d,
         {1, 1, static_cast<uint8_t>(d / 8), 8});
    if (d == 128)
        Copy(rows[64], columns, uint64_t(64), d, {1, 1, 16, 8});
    Columns(columns, seed, d);
}

__aicore__ inline void Indices(LocalTensor<uint32_t> dst, LocalTensor<float> src, int32_t count)
{
    // All offsets are exact nonnegative FP32 integers, at most 65532.
    Cast(dst.ReinterpretCast<int32_t>(), src, RoundMode::CAST_RINT, count);
}

__aicore__ inline void Pivots(LocalTensor<uint32_t> a, LocalTensor<uint32_t> b,
    LocalTensor<uint32_t> c, LocalTensor<float> sequence, LocalTensor<float> work, int32_t d)
{
    const int32_t pairs = d / 2;
    Muls(work, sequence, static_cast<float>((d + 1) * 4), pairs);
    Indices(a, work, pairs);
    Adds(work, work, static_cast<float>(pairs * (d + 1) * 4), pairs);
    Indices(b, work, pairs);
    Adds(work, work, static_cast<float>(-pairs * d * 4), pairs);
    Indices(c, work, pairs);
}
}
}
