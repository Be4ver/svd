#pragma once
#include "kernel_operator.h"

namespace blocked_jacobi {
// One normal-mask repeat computes every pivot (16, 32 or 64 FP32 lanes).
// Count-mode Level-2 APIs switch/reset mask mode for EACH arithmetic operation;
// the Level-0 forms below keep the identical arithmetic with less scalar setup.
__aicore__ inline void RotationThreshold(AscendC::LocalTensor<float> params,
    AscendC::LocalTensor<uint8_t> activeMask,
    int32_t n, float tol)
{
    using namespace AscendC;
    const uint64_t mask = n;
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    auto a = params, b = params[64], c = params[128];
    auto ac = params[320], u = params[448];
    auto one = params[832], active = params[896], accum = params[960];
    Mul(u, a, b, mask, 1, binary); Abs(u, u, mask, 1, unary); Sqrt(u, u, mask, 1, unary);
    Muls(u, u, tol, mask, 1, unary); Adds(u, u, 1e-30f, mask, 1, unary);
    Abs(ac, c, mask, 1, unary);
    // Explicit full repeat also covers n<64; only the first n bits are used.
    Compare(activeMask, ac, u, CMPMODE::GT, uint64_t(64), 1, binary);
    Duplicate(one, 1.0f, mask, 1, 1, 8);
    Select(active, activeMask, one, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, 1, binary);
    Max(accum, accum, active, mask, 1, binary);
}

__aicore__ inline void RotationCoefficients(AscendC::LocalTensor<float> params,
    AscendC::LocalTensor<uint8_t> activeMask, AscendC::LocalTensor<uint8_t> signMask,
    int32_t n)
{
    using namespace AscendC;
    const uint64_t mask = n;
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    auto a = params, b = params[64], c = params[128];
    auto delta = params[192], ad = params[256], ac = params[320];
    auto scale = params[384], u = params[448], v = params[512];
    auto denom = params[576], t = params[640], cs = params[704], sn = params[768];
    auto one = params[832];
    Sub(delta, b, a, mask, 1, binary); Muls(delta, delta, 0.5f, mask, 1, unary);
    Abs(ad, delta, mask, 1, unary);
    Max(scale, ad, ac, mask, 1, binary); Maxs(scale, scale, 1e-30f, mask, 1, unary);
    // For an active pivot |c|>1e-30, scale is the true max. The bounded
    // ratio avoids overflow and uses one division for hypot instead of two.
    Min(u, ad, ac, mask, 1, binary); Div(u, u, scale, mask, 1, binary);
    Mul(u, u, u, mask, 1, binary); Adds(u, u, 1.0f, mask, 1, unary);
    Sqrt(u, u, mask, 1, unary); Mul(u, u, scale, mask, 1, binary); Add(denom, ad, u, mask, 1, binary);
    Maxs(denom, denom, 1e-30f, mask, 1, unary); Div(t, c, denom, mask, 1, binary);
    Muls(v, t, -1.0f, mask, 1, unary);
    CompareScalar(signMask, delta, 0.0f, CMPMODE::GE, uint64_t(64), 1, unary);
    Select(t, signMask, t, v, SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, 1, binary);
    Select(t, activeMask, t, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, 1, binary);
    Mul(u, t, t, mask, 1, binary); Adds(u, u, 1.0f, mask, 1, unary); Sqrt(u, u, mask, 1, unary);
    Div(cs, one, u, mask, 1, binary); Mul(sn, cs, t, mask, 1, binary);
}

__aicore__ inline void RotationAngles(AscendC::LocalTensor<float> params,
    AscendC::LocalTensor<uint8_t> activeMask, AscendC::LocalTensor<uint8_t> signMask,
    int32_t n, float tol)
{
    RotationThreshold(params, activeMask, n, tol);
    RotationCoefficients(params, activeMask, signMask, n);
}
}
