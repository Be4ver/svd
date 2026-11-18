#pragma once
#include "kernel_operator.h"
#include "rotation_prepare.hpp"
#include "rotation_angles.hpp"

namespace blocked_jacobi {
using namespace AscendC;

// Two resident matrices; all other storage is bounded tile scratch.
// Full-panel rotation R X is performed exclusively by FusedCube.
class BlockRotation128 {
public:
    __aicore__ inline void Init(TPipe* pipe)
    {
        // C220 reserves the top 8 KiB of its 192 KiB UB for API scratch.
        // Keep user storage below the 184 KiB usable limit.
        pipe->InitBuffer(buf_, 180 * 1024);
        auto m = buf_.Get<float>();
        g_ = m; r_ = m[16384];
        tmp_ = m[32768]; tmp2_ = m[36864];
        // Two identical 8-float coefficient blocks per 128-element row.
        cos_ = m[40960]; sin_ = m[41472]; params_ = m[41984];
        auto idx = m.ReinterpretCast<uint32_t>();
        columnIdx_ = idx[43200];
        aIdx_ = idx[45248]; bIdx_ = idx[45312]; cIdx_ = idx[45376];
        identityMask_ = m[45440].ReinterpretCast<uint8_t>();
        activeMask_ = params_[1088].ReinterpretCast<uint8_t>();
        signMask_ = params_[1120].ReinterpretCast<uint8_t>();
        prepared_ = false; preparedCrossOnly_ = false;
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
                               int32_t sweeps, float tol, bool crossOnly, bool requireIdentity = true)
    {
        if (!prepared_ || preparedCrossOnly_ != crossOnly) Prepare(crossOnly);
        SetFlag<HardEvent::V_MTE2>(v2m2_); WaitFlag<HardEvent::V_MTE2>(v2m2_);
        DataCopy(g_, gram[static_cast<int64_t>(job) * 16384], 16384);
        SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
        const bool skip = crossOnly && !CrossActive(tol);
        // Core-local identity requests never read R or the per-job flag.
        // The global phase path still requires a materialized identity R.
        if (skip && !requireIdentity) return false;
        Duplicate(r_, 1.0f, 16384);
        Select(r_, identityMask_, r_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 16384);
        bool initialActive = false;
        for (int32_t sweep = 0; sweep < sweeps && !skip; ++sweep) {
            Duplicate(params_[960], 0.0f, 64);
            for (int32_t round = 0; round < (crossOnly ? 64 : 127); ++round) {
                if (Angles(tol, crossOnly)) {
                    RotateBothRows();
                    RotateColumns();
                }
                // The same permutation conjugates G and permutes R's rows.
                PermuteRows(g_, crossOnly);
                PermuteColumns();
                PermuteRows(r_, crossOnly);
            }
            WholeReduceMax(params_[1024], params_[960], 64, 1, 1, 1, 8,
                           ReduceOrder::ORDER_ONLY_VALUE);
            SetFlag<HardEvent::V_S>(v2s_); WaitFlag<HardEvent::V_S>(v2s_);
            const bool active = params_.GetValue(1024) > 0.0f;
            if (sweep == 0) initialActive = active;
            if (!active) break;
        }
        // FP32 WholeReduceSum has a 64-element mask: sum both halves of
        // each 128-element row explicitly, with 32 rows per instruction.
        for (int32_t row = 0; row < 128; row += 32) {
            Mul(tmp_, r_[row * 128], r_[row * 128], 4096);
            WholeReduceSum(params_, tmp_, 64, 32, 1, 1, 16);
            WholeReduceSum(params_[128], tmp_[64], 64, 32, 1, 1, 16);
            Add(params_, params_, params_[128], 32);
            Muls(params_, params_, -0.5f, 32);
            Adds(params_, params_, 1.5f, 32);
            BroadcastRows(cos_, params_);
            Mul(r_[row * 128], r_[row * 128], cos_, uint64_t(64), 64,
                {1, 1, 0, 8, 8, 1});
        }
        Duplicate(params_[1024], initialActive ? 1.0f : 0.0f, 8);
        SetFlag<HardEvent::V_MTE3>(v2m3_); WaitFlag<HardEvent::V_MTE3>(v2m3_);
        DataCopy(rotations[static_cast<int64_t>(job) * 16384], r_, 16384);
        DataCopy(flags[static_cast<int64_t>(job) * 8], params_[1024], 8);
        SetFlag<HardEvent::MTE3_V>(m3v_); WaitFlag<HardEvent::MTE3_V>(m3v_);
        return initialActive;
    }

private:
    __aicore__ inline bool CrossActive(float tol)
    {
        // With no active cross pair, all 64 rounds are identities. Check the
        // SAME per-pair threshold on the unmodified Gram, using R as temporary
        // storage before its identity initialization. No pair is omitted.
        Gather(params_, g_, aIdx_, 0, 64);
        Gather(params_[64], g_, bIdx_, 0, 64);
        for (int32_t row = 0; row < 64; row += 32) {
            BroadcastRows(cos_, params_[row]);
            Mul(tmp_, params_[64], cos_, uint64_t(64), 32, {1, 1, 0, 8, 0, 2});
            Abs(tmp_, tmp_, 2048); Sqrt(tmp_, tmp_, 2048);
            Muls(tmp_, tmp_, tol, 2048); Adds(tmp_, tmp_, 1e-30f, 2048);
            Abs(r_[row * 64], g_[row * 128 + 64], uint64_t(64), 32, {1, 1, 8, 16});
            Sub(r_[row * 64], r_[row * 64], tmp_, 2048);
        }
        ReduceMax(params_[128], r_, tmp2_, 4096, false);
        SetFlag<HardEvent::V_S>(v2s_); WaitFlag<HardEvent::V_S>(v2s_);
        return params_.GetValue(128) > 0.0f;
    }
    __aicore__ inline void Prepare(bool crossOnly)
    {
        // Prepare only once per participating AIV, before its first Gram load.
        SetFlag<HardEvent::S_V>(s2v_); WaitFlag<HardEvent::S_V>(s2v_);
        auto sequence = tmp_, next = tmp_[128];
        rotation_prepare::Templates(sequence, next, tmp2_, activeMask_, 128, crossOnly);
        rotation_prepare::Pivots(aIdx_, bIdx_, cIdx_, sequence, tmp2_, 128);
        Muls(g_, next, 4.0f, 128);
        for (int32_t count = 128; count < 2048; count *= 2)
            Adds(g_[count], g_, static_cast<float>(count * 4), count);
        rotation_prepare::Indices(columnIdx_, g_, 2048);
        rotation_prepare::Grids(r_, g_, sequence, tmp2_, 128);
        Compare(identityMask_, g_, r_, CMPMODE::EQ, 16384);
        prepared_ = true; preparedCrossOnly_ = crossOnly;
    }
    __aicore__ inline bool Angles(float tol, bool checkActive)
    {
        Gather(params_, g_, aIdx_, 0, 64);
        Gather(params_[64], g_, bIdx_, 0, 64);
        Gather(params_[128], g_, cIdx_, 0, 64);
        RotationThreshold(params_, activeMask_, 64, tol);
        if (checkActive) {
            WholeReduceMax(params_[1024], params_[896], 64, 1, 1, 1, 8,
                           ReduceOrder::ORDER_ONLY_VALUE);
            SetFlag<HardEvent::V_S>(v2s_); WaitFlag<HardEvent::V_S>(v2s_);
            if (params_.GetValue(1024) == 0.0f) return false;
        }
        RotationCoefficients(params_, activeMask_, signMask_, 64);
        return true;
    }
    __aicore__ inline void BroadcastRows(LocalTensor<float> dst, LocalTensor<float> src)
    {
        // Brcb expands 32 scalars into every other 32-byte block. Fill the
        // gaps once; the row arithmetic reuses each block across 64 lanes
        // with src1BlkStride=0. Only 512 floats instead of 4096 are needed.
        Brcb(dst, src, 4, {2, 16});
        // Read even blocks and write odd blocks: actual elements are disjoint.
        Copy(dst[8], dst, uint64_t(8), 32, {1, 1, 2, 2});
    }
    __aicore__ inline void RotateTile(LocalTensor<float> matrix, int32_t row)
    {
            // AIV: local G/R updates only; FusedCube::Apply performs R X.
            auto p = matrix[row * 128], q = matrix[(row + 64) * 128];
            Mul(tmp_, p, cos_, uint64_t(64), 64, {1, 1, 0, 8, 8, 1});
            Mul(tmp2_, q, sin_, uint64_t(64), 64, {1, 1, 0, 8, 8, 1});
            Sub(tmp_, tmp_, tmp2_, 4096);
            Mul(tmp2_, p, sin_, uint64_t(64), 64, {1, 1, 0, 8, 8, 1});
            Mul(q, q, cos_, uint64_t(64), 64, {1, 1, 0, 8, 8, 1});
            Add(q, q, tmp2_, 4096);
            CopyRotationBuffer(p, tmp_, 4096);
    }
    __aicore__ inline void RotateBothRows()
    {
        for (int32_t row = 0; row < 64; row += 32) {
            BroadcastRows(cos_, params_[704 + row]);
            BroadcastRows(sin_, params_[768 + row]);
            RotateTile(g_, row);
            RotateTile(r_, row);
        }
    }
    __aicore__ inline void RotateColumns()
    {
        // Right multiplication by J^T, without transposing G. Each repeat
        // handles 64 paired columns; 32 repeats cover 32 physical rows.
        for (int32_t row = 0; row < 128; row += 32) {
            auto p = g_[row * 128], q = g_[row * 128 + 64];
            Mul(tmp_, p, params_[704], uint64_t(64), 32, {1, 1, 1, 8, 16, 0});
            Mul(tmp2_, q, params_[768], uint64_t(64), 32, {1, 1, 1, 8, 16, 0});
            Sub(tmp_, tmp_, tmp2_, 2048);
            Mul(tmp2_, p, params_[768], uint64_t(64), 32, {1, 1, 1, 8, 16, 0});
            Mul(q, q, params_[704], uint64_t(64), 32, {1, 1, 1, 16, 16, 0});
            Add(q, q, tmp2_, uint64_t(64), 32, {1, 1, 1, 16, 16, 8});
            Copy(p, tmp_, uint64_t(64), 32, {1, 1, 16, 8});
        }
    }
    __aicore__ inline void PermuteRows(LocalTensor<float> matrix, bool crossOnly)
    {
        if (crossOnly) {
            CopyRotationBuffer(cos_, matrix[64 * 128], 128);
            for (int32_t start = 65; start < 128;) {
                const int32_t count = 128 - start < 32 ? 128 - start : 32;
                CopyRotationBuffer(tmp_, matrix[start * 128], count * 128);
                CopyRotationBuffer(matrix[(start - 1) * 128], tmp_, count * 128);
                start += count;
            }
            CopyRotationBuffer(matrix[127 * 128], cos_, 128);
            return;
        }
        // next=[0,64,1,...,62,65,...,127,63]. Two monotonic tiled
        // shifts implement it without a full matrix-sized Gather map.
        CopyRotationBuffer(cos_, matrix[64 * 128], 128);
        CopyRotationBuffer(sin_, matrix[63 * 128], 128);
        for (int32_t end = 63; end > 1;) {
            const int32_t count = end - 1 < 32 ? end - 1 : 32;
            const int32_t start = end - count;
            CopyRotationBuffer(tmp_, matrix[start * 128], count * 128);
            CopyRotationBuffer(matrix[(start + 1) * 128], tmp_, count * 128);
            end = start;
        }
        for (int32_t start = 65; start < 128;) {
            const int32_t count = 128 - start < 32 ? 128 - start : 32;
            CopyRotationBuffer(tmp_, matrix[start * 128], count * 128);
            CopyRotationBuffer(matrix[(start - 1) * 128], tmp_, count * 128);
            start += count;
        }
        CopyRotationBuffer(matrix[128], cos_, 128);
        CopyRotationBuffer(matrix[127 * 128], sin_, 128);
    }
    __aicore__ inline void PermuteColumns()
    {
        // Retain a 16-row map to fit the 184 KiB user UB budget.
        for (int32_t row = 0; row < 128; row += 16) {
            Gather(tmp_, g_[row * 128], columnIdx_, 0, 2048);
            CopyRotationBuffer(g_[row * 128], tmp_, 2048);
        }
    }
    TBuf<TPosition::VECCALC> buf_;
    LocalTensor<float> g_, r_, tmp_, tmp2_, cos_, sin_, params_;
    LocalTensor<uint32_t> columnIdx_, aIdx_, bIdx_, cIdx_;
    LocalTensor<uint8_t> identityMask_, activeMask_, signMask_;
    int32_t m2v_, v2m2_, v2s_, s2v_, v2m3_, m3v_;
    bool prepared_, preparedCrossOnly_;
};
}
