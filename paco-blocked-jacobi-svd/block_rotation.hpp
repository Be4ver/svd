#pragma once
#include "kernel_operator.h"
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
                               GlobalTensor<float> flags, int32_t job, int32_t d,
                               int32_t sweeps, float tol)
    {
        if (!prepared_) Prepare(d);
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
            for (int32_t round = 0; round < d - 1; ++round) {
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
    // Map a position in [P,Q] to the conventional circle order, and back.
    // This involution makes the initial row order the identity.
    __aicore__ inline int32_t Circle(int32_t i, int32_t d)
    { return i < d / 2 ? i : d + d / 2 - 1 - i; }
    __aicore__ inline int32_t Next(int32_t i, int32_t d)
    {
        const int32_t c = Circle(i, d);
        return Circle(c == 0 ? 0 : (c == 1 ? d - 1 : c - 1), d);
    }
    __aicore__ inline void Prepare(int32_t d)
    {
        // Initialized once on each AIV, then reused for every panel and round.
        // These maps lie outside the shared pre/postprocessing scratch region.
        for (int32_t i = 0; i < d; ++i) {
            for (int32_t j = 0; j < d; ++j) {
                transposeIdx_.SetValue(i * d + j, (j * d + i) * sizeof(float));
                nextTransposeIdx_.SetValue(i * d + j, (Next(j, d) * d + Next(i, d)) * sizeof(float));
                nextRowIdx_.SetValue(i * d + j, (Next(i, d) * d + j) * sizeof(float));
            }
        }
        for (int32_t i = 0; i < d / 2; ++i) {
            aIdx_.SetValue(i, (i * d + i) * sizeof(float));
            bIdx_.SetValue(i, ((i + d / 2) * d + i + d / 2) * sizeof(float));
            cIdx_.SetValue(i, (i * d + i + d / 2) * sizeof(float));
        }
        for (int32_t i = 0; i < d * d / 8; ++i) {
            const int32_t diagonal = (i * 8 + d) / (d + 1);
            const int32_t bit = diagonal * (d + 1) - i * 8;
            identityMask_.SetValue(i, bit < 8 ? static_cast<uint8_t>(1U << bit) : 0);
        }
        SetFlag<HardEvent::S_V>(s2v_); WaitFlag<HardEvent::S_V>(s2v_);
        prepared_ = true;
    }
    __aicore__ inline void Angles(int32_t d, float tol)
    {
        const int32_t n = d / 2;
        auto a = params_, b = params_[64], c = params_[128];
        auto delta = params_[192], ad = params_[256], ac = params_[320];
        auto scale = params_[384], u = params_[448], v = params_[512];
        auto denom = params_[576], t = params_[640], cs = params_[704], sn = params_[768];
        auto one = params_[832], active = params_[896], accum = params_[960];
        Gather(a, g_, aIdx_, 0, n);
        Gather(b, g_, bIdx_, 0, n);
        Gather(c, g_, cIdx_, 0, n);
        Mul(u, a, b, n); Abs(u, u, n); Sqrt(u, u, n);
        Muls(u, u, tol, n); Adds(u, u, 1e-30f, n);
        Abs(ac, c, n);
        // C220 count-mode Compare truncates count/64; n is only 16 or 32.
        // Issue one full repeat, then consume only the first n predicate bits.
        Compare(activeMask_, ac, u, CMPMODE::GT, uint64_t(64), 1, {1, 1, 1, 8, 8, 8});
        Duplicate(one, 1.0f, n);
        Select(active, activeMask_, one, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
        Max(accum, accum, active, n);
        Sub(delta, b, a, n); Muls(delta, delta, 0.5f, n); Abs(ad, delta, n);
        Max(scale, ad, ac, n); Maxs(scale, scale, 1e-30f, n);
        Div(u, delta, scale, n); Div(v, c, scale, n);
        Mul(u, u, u, n); Mul(v, v, v, n); Add(u, u, v, n);
        Sqrt(u, u, n); Mul(u, u, scale, n); Add(denom, ad, u, n);
        Maxs(denom, denom, 1e-30f, n);
        Div(t, c, denom, n);
        Muls(v, t, -1.0f, n);
        CompareScalar(signMask_, delta, 0.0f, CMPMODE::GE, uint64_t(64), 1, {1, 1, 8, 8});
        Select(t, signMask_, t, v, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        Select(t, activeMask_, t, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
        Mul(u, t, t, n); Adds(u, u, 1.0f, n); Sqrt(u, u, n);
        Div(cs, one, u, n); Mul(sn, cs, t, n);
        BroadcastRows(cos_, cs, n, d);
        BroadcastRows(sin_, sn, n, d);
    }
    __aicore__ inline void BroadcastRows(LocalTensor<float> dst, LocalTensor<float> src,
                                        int32_t rows, int32_t d)
    {
        Brcb(tmp_, src, rows / 8, {1, 8});
        Brcb(broadcast_, tmp_, rows, {1, 8});
        // Broadcast to 64 columns, then compact to 32 columns for block16.
        Adds(dst, broadcast_, 0.0f, static_cast<uint64_t>(d), rows,
             {1, 1, static_cast<uint8_t>(d / 8), 8});
    }
    __aicore__ inline void RotateRows(LocalTensor<float> dst, LocalTensor<float> src, int32_t d)
    {
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
    bool prepared_;
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
                               int32_t sweeps, float tol)
    {
        if (large_) return largeSolver_.Run(gram, rotations, flags, job, sweeps, tol);
        return smallSolver_.Run(gram, rotations, flags, job, d, sweeps, tol);
    }
private:
    BlockRotationSmall smallSolver_;
    BlockRotation128 largeSolver_;
    bool large_;
};
}
