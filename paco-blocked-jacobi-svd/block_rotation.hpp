#pragma once
#include "kernel_operator.h"
#include "rotation_prepare.hpp"
#include "rotation_angles.hpp"
#include "block_rotation128.hpp"

namespace blocked_jacobi {
using namespace AscendC;

// Small symmetric eigensolve: R G R^T = diagonal. Only Cube applies R to X.
// A circle schedule batches d/2 disjoint pivots per round. The schedule is
// conjugated into [P rows, Q rows], so all row arithmetic is contiguous.
class BlockRotationSmall {
public:
    __aicore__ inline void Init(TPipe* pipe)
    {
        // Compact matrices + batched workspaces + immutable Gather maps.
        // First 128 KiB is also pre/postprocessing scratch; total UB = 180 KiB.
        pipe->InitBuffer(buf_, 180 * 1024);
        auto memory = buf_.Get<float>();
        g_ = memory; r_ = memory[4096]; trans_ = memory[8192]; rNext_ = memory[12288];
        tmp_ = memory[16384]; tmp2_ = memory[18432];
        cos_ = memory[20480]; sin_ = memory[22528]; broadcast_ = memory[24576];
        params_ = memory[28672];
        auto indices = memory.ReinterpretCast<uint32_t>();
        transposeIdx_ = indices[32768]; nextTransposeIdx_ = indices[36864];
        nextRowIdx_ = indices[40960];
        aIdx_ = indices[45056]; bIdx_ = indices[45120]; cIdx_ = indices[45184];
        identityMask_ = memory[45248].ReinterpretCast<uint8_t>();
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
                               GlobalTensor<float> flags, int32_t job, int32_t d,
                               int32_t sweeps, float tol, bool crossOnly)
    {
        if (!prepared_ || preparedCrossOnly_ != crossOnly) Prepare(d, crossOnly);
        // A prior job's vector row correction still reads g_. MTE3_V orders
        // later vector work, but cannot stop this independent MTE2 overwrite.
        SetFlag<HardEvent::V_MTE2>(v2m2_); WaitFlag<HardEvent::V_MTE2>(v2m2_);
        DataCopy(g_, gram[static_cast<int64_t>(job) * d * d], d * d);
        SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
        Duplicate(r_, 1.0f, d * d);
        Select(r_, identityMask_, r_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, d * d);
        bool initialActive = false;
        const int32_t pairs = d / 2;
        for (int32_t sweep = 0; sweep < sweeps; ++sweep) {
            Duplicate(params_[960], 0.0f, pairs);
            for (int32_t round = 0; round < (crossOnly ? d / 2 : d - 1); ++round) {
                Angles(d, tol);
                RotateRows(trans_, g_, d);
                RotateRows(rNext_, r_, d);
                // A second row transform on G^T supplies the right-hand J^T.
                // This includes all interactions between distinct pivot pairs.
                Gather(g_, trans_, transposeIdx_, 0, d * d);
                RotateRows(trans_, g_, d);
                Gather(g_, trans_, nextTransposeIdx_, 0, d * d);
                Gather(r_, rNext_, nextRowIdx_, 0, d * d);
            }
            // Only one V->S result per local sweep, rather than per pivot.
            WholeReduceMax(params_[1024], params_[960], pairs, 1, 1, 1, 8,
                           ReduceOrder::ORDER_ONLY_VALUE);
            VectorToScalar();
            const bool active = params_.GetValue(1024) > 0.0f;
            if (sweep == 0) initialActive = active;
            if (!active) break;
        }
        // Correct all R row lengths together: one reduction instruction with
        // d repeats, followed by a vector Newton step. No scalar row reads.
        Mul(trans_, r_, r_, d * d);
        WholeReduceSum(params_, trans_, d, d, 1, 1, d / 8);
        Muls(params_, params_, -0.5f, d);
        Adds(params_, params_, 1.5f, d);
        BroadcastRows(g_, params_, d, d);
        Mul(r_, r_, g_, d * d);
        Duplicate(params_[1024], initialActive ? 1.0f : 0.0f, 8);
        SetFlag<HardEvent::V_MTE3>(v2m3_); WaitFlag<HardEvent::V_MTE3>(v2m3_);
        DataCopy(rotations[static_cast<int64_t>(job) * d * d], r_, d * d);
        DataCopy(flags[static_cast<int64_t>(job) * 8], params_[1024], 8);
        SetFlag<HardEvent::MTE3_V>(m3v_); WaitFlag<HardEvent::MTE3_V>(m3v_);
        return initialActive;
    }

private:
    __aicore__ inline void Prepare(int32_t d, bool crossOnly)
    {
        // Order prior scalar accesses to shared scratch before vector reuse.
        SetFlag<HardEvent::S_V>(s2v_); WaitFlag<HardEvent::S_V>(s2v_);
        auto sequence = tmp_, next = tmp_[128];
        rotation_prepare::Templates(sequence, next, tmp2_, activeMask_, d, crossOnly);
        rotation_prepare::Pivots(aIdx_, bIdx_, cIdx_, sequence, tmp2_, d);
        rotation_prepare::Grids(r_, g_, sequence, tmp2_, d);
        Compare(identityMask_, g_, r_, CMPMODE::EQ, d * d);
        Muls(g_, g_, static_cast<float>(d * 4), d * d);
        Muls(r_, r_, 4.0f, d * d);
        Add(trans_, g_, r_, d * d);
        rotation_prepare::Indices(transposeIdx_, trans_, d * d);

        rotation_prepare::Grids(r_, g_, next, tmp2_, d);
        Muls(g_, g_, static_cast<float>(d * 4), d * d);
        Muls(r_, r_, 4.0f, d * d);
        Add(trans_, g_, r_, d * d);
        rotation_prepare::Indices(nextTransposeIdx_, trans_, d * d);
        Muls(r_, r_, static_cast<float>(d), d * d);
        rotation_prepare::Columns(g_, sequence, d);
        Muls(g_, g_, 4.0f, d * d);
        Add(trans_, r_, g_, d * d);
        rotation_prepare::Indices(nextRowIdx_, trans_, d * d);
        prepared_ = true; preparedCrossOnly_ = crossOnly;
    }
    __aicore__ inline void Angles(int32_t d, float tol)
    {
        const int32_t n = d / 2;
        Gather(params_, g_, aIdx_, 0, n);
        Gather(params_[64], g_, bIdx_, 0, n);
        Gather(params_[128], g_, cIdx_, 0, n);
        RotationAngles(params_, activeMask_, signMask_, n, tol);
        BroadcastRows(cos_, params_[704], n, d);
        BroadcastRows(sin_, params_[768], n, d);
    }
    __aicore__ inline void BroadcastRows(LocalTensor<float> dst, LocalTensor<float> src,
                                        int32_t rows, int32_t d)
    {
        Brcb(tmp_, src, rows / 8, {1, 8});
        Brcb(broadcast_, tmp_, rows, {1, 8});
        // Broadcast to 64 columns, then compact to 32 columns for block16.
        Copy(dst, broadcast_, static_cast<uint64_t>(d), rows,
             {1, 1, static_cast<uint8_t>(d / 8), 8});
    }
    __aicore__ inline void RotateRows(LocalTensor<float> dst, LocalTensor<float> src, int32_t d)
    {
        // AIV: update the small Gram or accumulated R, never the input panel X.
        // Distinct source/destination matrices remove the trailing row copy,
        // and keep both original halves live until their last multiplication.
        const int32_t n = d * d / 2;
        Mul(tmp_, src, cos_, n);
        Mul(tmp2_, src[n], sin_, n);
        Sub(dst, tmp_, tmp2_, n);
        Mul(tmp2_, src, sin_, n);
        Mul(dst[n], src[n], cos_, n);
        Add(dst[n], dst[n], tmp2_, n);
    }
    __aicore__ inline void VectorToScalar()
    { SetFlag<HardEvent::V_S>(v2s_); WaitFlag<HardEvent::V_S>(v2s_); }
    TBuf<TPosition::VECCALC> buf_;
    LocalTensor<float> g_, r_, trans_, rNext_, tmp_, tmp2_, cos_, sin_, broadcast_, params_;
    LocalTensor<uint32_t> transposeIdx_, nextTransposeIdx_, nextRowIdx_, aIdx_, bIdx_, cIdx_;
    LocalTensor<uint8_t> identityMask_, activeMask_, signMask_;
    int32_t m2v_, v2m2_, v2s_, s2v_, v2m3_, m3v_;
    bool prepared_, preparedCrossOnly_;
};

// Only one Init branch allocates UB in this kernel invocation.
class BlockRotation {
public:
    __aicore__ inline void Init(TPipe* pipe, int32_t d)
    {
        large_ = d == 128;
        if (large_) largeSolver_.Init(pipe);
        else smallSolver_.Init(pipe);
    }
    __aicore__ inline LocalTensor<float> Scratch()
    { return large_ ? largeSolver_.Scratch() : smallSolver_.Scratch(); }
    __aicore__ inline bool Run(GlobalTensor<float> gram, GlobalTensor<float> rotations,
                               GlobalTensor<float> flags, int32_t job, int32_t d,
                               int32_t sweeps, float tol, bool crossOnly, bool requireIdentity = true)
    {
        if (large_) return largeSolver_.Run(gram, rotations, flags, job, sweeps, tol, crossOnly, requireIdentity);
        return smallSolver_.Run(gram, rotations, flags, job, d, sweeps, tol, crossOnly);
    }
private:
    BlockRotationSmall smallSolver_;
    BlockRotation128 largeSolver_;
    bool large_;
};
}
