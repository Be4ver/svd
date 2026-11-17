#pragma once
#include "kernel_operator.h"

namespace blocked_jacobi {
using namespace AscendC;

// Two resident matrices; all other storage is bounded tile scratch.
// Full-panel rotation R X is performed exclusively by FusedCube.
class BlockRotation128 {
public:
    __aicore__ inline void Init(TPipe* pipe)
    {
        pipe->InitBuffer(buf_, 180 * 1024);
        auto m = buf_.Get<float>();
        g_ = m; r_ = m[16384];
        tmp_ = m[32768]; tmp2_ = m[34816];
        cos_ = m[36864]; sin_ = m[38912];
        broadcast_ = m[40960]; params_ = m[41984];
        auto idx = m.ReinterpretCast<uint32_t>();
        columnIdx_ = idx[43200];
        aIdx_ = idx[45248]; bIdx_ = idx[45312]; cIdx_ = idx[45376];
        identityMask_ = m[45440].ReinterpretCast<uint8_t>();
        activeMask_ = params_[1088].ReinterpretCast<uint8_t>();
        signMask_ = params_[1120].ReinterpretCast<uint8_t>();
        prepared_ = false;
        m2v_ = pipe->FetchEventID(HardEvent::MTE2_V);
        v2m2_ = pipe->FetchEventID(HardEvent::V_MTE2);
        v2s_ = pipe->FetchEventID(HardEvent::V_S);
        s2v_ = pipe->FetchEventID(HardEvent::S_V);
        v2m3_ = pipe->FetchEventID(HardEvent::V_MTE3);
        m3v_ = pipe->FetchEventID(HardEvent::MTE3_V);
    }

    __aicore__ inline LocalTensor<float> Scratch() { return g_; }

    __aicore__ inline bool Run(GlobalTensor<float> gram, GlobalTensor<float> rotations,
                               GlobalTensor<float> flags, int32_t job,
                               int32_t sweeps, float tol)
    {
        if (!prepared_) Prepare();
        SetFlag<HardEvent::V_MTE2>(v2m2_); WaitFlag<HardEvent::V_MTE2>(v2m2_);
        DataCopy(g_, gram[static_cast<int64_t>(job) * 16384], 16384);
        SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
        Duplicate(r_, 1.0f, 16384);
        Select(r_, identityMask_, r_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 16384);
        bool initialActive = false;
        for (int32_t sweep = 0; sweep < sweeps; ++sweep) {
            Duplicate(params_[960], 0.0f, 64);
            for (int32_t round = 0; round < 127; ++round) {
                Angles(tol);
                RotateBothRows();
                RotateColumns();
                // The same permutation conjugates G and permutes R's rows.
                PermuteRows(g_);
                PermuteColumns();
                PermuteRows(r_);
            }
            WholeReduceMax(params_[1024], params_[960], 64, 1, 1, 1, 8,
                           ReduceOrder::ORDER_ONLY_VALUE);
            SetFlag<HardEvent::V_S>(v2s_); WaitFlag<HardEvent::V_S>(v2s_);
            const bool active = params_.GetValue(1024) > 0.0f;
            if (sweep == 0) initialActive = active;
            if (!active) break;
        }
        // FP32 WholeReduceSum has a 64-element mask: sum both halves of
        // each 128-element row explicitly, with sixteen rows per instruction.
        for (int32_t row = 0; row < 128; row += 16) {
            Mul(tmp_, r_[row * 128], r_[row * 128], 2048);
            WholeReduceSum(params_, tmp_, 64, 16, 1, 1, 16);
            WholeReduceSum(params_[128], tmp_[64], 64, 16, 1, 1, 16);
            Add(params_, params_, params_[128], 16);
            Muls(params_, params_, -0.5f, 16);
            Adds(params_, params_, 1.5f, 16);
            BroadcastRows(cos_, params_);
            Mul(r_[row * 128], r_[row * 128], cos_, 2048);
        }
        Duplicate(params_[1024], initialActive ? 1.0f : 0.0f, 8);
        SetFlag<HardEvent::V_MTE3>(v2m3_); WaitFlag<HardEvent::V_MTE3>(v2m3_);
        DataCopy(rotations[static_cast<int64_t>(job) * 16384], r_, 16384);
        DataCopy(flags[static_cast<int64_t>(job) * 8], params_[1024], 8);
        SetFlag<HardEvent::MTE3_V>(m3v_); WaitFlag<HardEvent::MTE3_V>(m3v_);
        return initialActive;
    }

private:
    __aicore__ inline int32_t Next(int32_t i)
    {
        if (i == 0) return 0;
        if (i == 1) return 64;
        if (i < 64) return i - 1;
        if (i == 127) return 63;
        return i + 1;
    }
    __aicore__ inline void Prepare()
    {
        // All immutable indices are beyond the shared preprocessing scratch.
        for (int32_t row = 0; row < 16; ++row)
            for (int32_t col = 0; col < 128; ++col)
                columnIdx_.SetValue(row * 128 + col, (row * 128 + Next(col)) * 4);
        for (int32_t i = 0; i < 64; ++i) {
            aIdx_.SetValue(i, (i * 128 + i) * 4);
            bIdx_.SetValue(i, ((i + 64) * 128 + i + 64) * 4);
            cIdx_.SetValue(i, (i * 128 + i + 64) * 4);
        }
        for (int32_t i = 0; i < 2048; ++i) {
            const int32_t diagonal = (i * 8 + 128) / 129;
            const int32_t bit = diagonal * 129 - i * 8;
            identityMask_.SetValue(i, bit < 8 ? static_cast<uint8_t>(1U << bit) : 0);
        }
        SetFlag<HardEvent::S_V>(s2v_); WaitFlag<HardEvent::S_V>(s2v_);
        prepared_ = true;
    }
    __aicore__ inline void Angles(float tol)
    {
        auto a = params_, b = params_[64], c = params_[128];
        auto delta = params_[192], ad = params_[256], ac = params_[320];
        auto scale = params_[384], u = params_[448], v = params_[512];
        auto denom = params_[576], t = params_[640], cs = params_[704], sn = params_[768];
        auto one = params_[832], active = params_[896], accum = params_[960];
        Gather(a, g_, aIdx_, 0, 64);
        Gather(b, g_, bIdx_, 0, 64);
        Gather(c, g_, cIdx_, 0, 64);
        Mul(u, a, b, 64); Abs(u, u, 64); Sqrt(u, u, 64);
        Muls(u, u, tol, 64); Adds(u, u, 1e-30f, 64);
        Abs(ac, c, 64);
        Compare(activeMask_, ac, u, CMPMODE::GT, uint64_t(64), 1, {1, 1, 1, 8, 8, 8});
        Duplicate(one, 1.0f, 64);
        Select(active, activeMask_, one, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
        Max(accum, accum, active, 64);
        Sub(delta, b, a, 64); Muls(delta, delta, 0.5f, 64); Abs(ad, delta, 64);
        Max(scale, ad, ac, 64); Maxs(scale, scale, 1e-30f, 64);
        Div(u, delta, scale, 64); Div(v, c, scale, 64);
        Mul(u, u, u, 64); Mul(v, v, v, 64); Add(u, u, v, 64);
        Sqrt(u, u, 64); Mul(u, u, scale, 64); Add(denom, ad, u, 64);
        Maxs(denom, denom, 1e-30f, 64); Div(t, c, denom, 64);
        Muls(v, t, -1.0f, 64);
        CompareScalar(signMask_, delta, 0.0f, CMPMODE::GE, uint64_t(64), 1, {1, 1, 8, 8});
        Select(t, signMask_, t, v, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
        Select(t, activeMask_, t, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
        Mul(u, t, t, 64); Adds(u, u, 1.0f, 64); Sqrt(u, u, 64);
        Div(cs, one, u, 64); Mul(sn, cs, t, 64);
    }
    __aicore__ inline void BroadcastRows(LocalTensor<float> dst, LocalTensor<float> src)
    {
        Brcb(tmp_, src, 2, {1, 8});
        Brcb(broadcast_, tmp_, 16, {1, 8});
        // FP32 vector mask is at most 64, so broadcast both row halves.
        Adds(dst, broadcast_, 0.0f, uint64_t(64), 16, {1, 1, 16, 8});
        Adds(dst[64], broadcast_, 0.0f, uint64_t(64), 16, {1, 1, 16, 8});
    }
    __aicore__ inline void RotateTile(LocalTensor<float> matrix, int32_t row)
    {
            auto p = matrix[row * 128], q = matrix[(row + 64) * 128];
            Mul(tmp_, p, cos_, 2048);
            Mul(tmp2_, q, sin_, 2048);
            Sub(tmp_, tmp_, tmp2_, 2048);
            Mul(tmp2_, p, sin_, 2048);
            Mul(q, q, cos_, 2048);
            Add(q, q, tmp2_, 2048);
            Adds(p, tmp_, 0.0f, 2048);
    }
    __aicore__ inline void RotateBothRows()
    {
        for (int32_t row = 0; row < 64; row += 16) {
            BroadcastRows(cos_, params_[704 + row]);
            BroadcastRows(sin_, params_[768 + row]);
            RotateTile(g_, row);
            RotateTile(r_, row);
        }
    }
    __aicore__ inline void RotateColumns()
    {
        // Right multiplication by J^T, without transposing G. Each repeat
        // handles 64 paired columns; 16 repeats cover sixteen physical rows.
        for (int32_t row = 0; row < 128; row += 16) {
            auto p = g_[row * 128], q = g_[row * 128 + 64];
            Mul(tmp_, p, params_[704], uint64_t(64), 16, {1, 1, 1, 8, 16, 0});
            Mul(tmp2_, q, params_[768], uint64_t(64), 16, {1, 1, 1, 8, 16, 0});
            Sub(tmp_, tmp_, tmp2_, 1024);
            Mul(tmp2_, p, params_[768], uint64_t(64), 16, {1, 1, 1, 8, 16, 0});
            Mul(q, q, params_[704], uint64_t(64), 16, {1, 1, 1, 16, 16, 0});
            Add(q, q, tmp2_, uint64_t(64), 16, {1, 1, 1, 16, 16, 8});
            Adds(p, tmp_, 0.0f, uint64_t(64), 16, {1, 1, 16, 8});
        }
    }
    __aicore__ inline void PermuteRows(LocalTensor<float> matrix)
    {
        // next=[0,64,1,...,62,65,...,127,63]. Two monotonic tiled
        // shifts implement it without a full matrix-sized Gather map.
        Adds(cos_, matrix[64 * 128], 0.0f, 128);
        Adds(sin_, matrix[63 * 128], 0.0f, 128);
        for (int32_t end = 63; end > 1;) {
            const int32_t count = end - 1 < 16 ? end - 1 : 16;
            const int32_t start = end - count;
            Adds(tmp_, matrix[start * 128], 0.0f, count * 128);
            Adds(matrix[(start + 1) * 128], tmp_, 0.0f, count * 128);
            end = start;
        }
        for (int32_t start = 65; start < 128;) {
            const int32_t count = 128 - start < 16 ? 128 - start : 16;
            Adds(tmp_, matrix[start * 128], 0.0f, count * 128);
            Adds(matrix[(start - 1) * 128], tmp_, 0.0f, count * 128);
            start += count;
        }
        Adds(matrix[128], cos_, 0.0f, 128);
        Adds(matrix[127 * 128], sin_, 0.0f, 128);
    }
    __aicore__ inline void PermuteColumns()
    {
        for (int32_t row = 0; row < 128; row += 16) {
            Gather(tmp_, g_[row * 128], columnIdx_, 0, 2048);
            Adds(g_[row * 128], tmp_, 0.0f, 2048);
        }
    }
    TBuf<TPosition::VECCALC> buf_;
    LocalTensor<float> g_, r_, tmp_, tmp2_, cos_, sin_, broadcast_, params_;
    LocalTensor<uint32_t> columnIdx_, aIdx_, bIdx_, cIdx_;
    LocalTensor<uint8_t> identityMask_, activeMask_, signMask_;
    int32_t m2v_, v2m2_, v2s_, s2v_, v2m3_, m3v_;
    bool prepared_;
};
}
