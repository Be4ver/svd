#pragma once
#include "fused_common.hpp"
#include "paco_gemm.hpp"

namespace blocked_jacobi {
#ifdef __DAV_C220_CUBE__
class FusedCube {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR matrix, GM_ADDR panels,
        GM_ADDR gram, GM_ADDR rotations, int block, int nb, int cp)
    {
        block_ = block; nb_ = nb; cp_ = cp;
        d_ = 2 * block; np_ = nb / 2; rp_ = nb * block;
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrix));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(panels));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gram));
        r_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotations));
        mm_.Init(pipe);
        fixs_ = pipe->FetchEventID(HardEvent::FIX_S);
        sm2_ = pipe->FetchEventID(HardEvent::S_MTE2);
    }

    __aicore__ inline void AcquireInputs()
    { SetFlag<HardEvent::S_MTE2>(sm2_); WaitFlag<HardEvent::S_MTE2>(sm2_); }

    __aicore__ inline void ReleaseOutputs()
    { SetFlag<HardEvent::FIX_S>(fixs_); WaitFlag<HardEvent::FIX_S>(fixs_); }

    __aicore__ inline void GramJob(int job)
    {
        AcquireInputs();
        auto x = x_[static_cast<int64_t>(job) * d_ * cp_];
        auto g = g_[static_cast<int64_t>(job) * d_ * d_];
        mm_.SetMatrix(x, x, g, d_, cp_);
        for (int k = 0; k < cp_; k += 128) {
            const int nk = cp_ - k < 128 ? cp_ - k : 128;
            mm_.Process(k, k, 0, d_, d_, nk, true, k > 0);
        }
        ReleaseOutputs();
    }

    __aicore__ inline void ApplyJob(int job, int round)
    {
        AcquireInputs();
        const int m = job / np_, pair = job % np_;
        auto r = r_[static_cast<int64_t>(job) * d_ * d_];
        auto x = x_[static_cast<int64_t>(job) * d_ * cp_];
        auto b = b_[static_cast<int64_t>(m) * rp_ * cp_];
        mm_.SetMatrix(r, x, b, cp_, d_);
        for (int half = 0; half < 2; ++half) {
            const int rb = PairBlock(pair, half, round, nb_);
            for (int col = 0; col < cp_; col += 128) {
                const int nc = cp_ - col < 128 ? cp_ - col : 128;
                mm_.Process(half * block_ * d_, col, rb * block_ * cp_ + col,
                            block_, nc, d_, false, false);
            }
        }
        ReleaseOutputs();
    }

    __aicore__ inline void Gram(int begin, int end)
    {
        for (int local = GetBlockIdx(); local < end - begin; local += GetBlockNum()) {
            const int job = begin + local;
            auto x = x_[static_cast<int64_t>(job) * d_ * cp_];
            auto g = g_[static_cast<int64_t>(job) * d_ * d_];
            mm_.SetMatrix(x, x, g, d_, cp_);
            // One AIC owns this Gram. Queue events protect buffer reuse and
            // M -> FIX dependencies; the phase barrier publishes G to AIVs.
            // Preserve the 128-column partial sums (FP32 reduction order).
            // AtomicAdd performs their arithmetic sum; it is not a core lock.
            for (int k = 0; k < cp_; k += 128) {
                const int nk = cp_ - k < 128 ? cp_ - k : 128;
                mm_.Process(k, k, 0, d_, d_, nk, true, k > 0);
            }
        }
    }

    __aicore__ inline void Apply(int begin, int end, int round)
    {
        const int nt = (cp_ + 127) / 128;
        for (int task = GetBlockIdx(); task < (end - begin) * 2 * nt; task += GetBlockNum()) {
            const int job = begin + task / (2 * nt);
            const int half = (task / nt) % 2, col = (task % nt) * 128;
            const int m = job / np_, pair = job % np_;
            const int rb = PairBlock(pair, half, round, nb_);
            auto r = r_[static_cast<int64_t>(job) * d_ * d_];
            auto x = x_[static_cast<int64_t>(job) * d_ * cp_];
            auto b = b_[static_cast<int64_t>(m) * rp_ * cp_];
            mm_.SetMatrix(r, x, b, cp_, d_);
            const int nc = cp_ - col < 128 ? cp_ - col : 128;
            mm_.Process(half * block_ * d_, col, rb * block_ * cp_ + col,
                        block_, nc, d_, false, false);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    MatmulCustom<float, float, float> mm_;
    GlobalTensor<float> b_, x_, g_, r_;
    int block_, nb_, cp_, d_, np_, rp_;
    int fixs_, sm2_;
};
#endif
}
