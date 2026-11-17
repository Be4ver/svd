#pragma once
#include "fused_common.hpp"
#include "block_rotation.hpp"

namespace blocked_jacobi {
#ifdef __DAV_C220_VEC__
class FusedVector {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR input, GM_ADDR matrix, GM_ADDR panels,
        GM_ADDR gram, GM_ADDR rotations, GM_ADDR flags, GM_ADDR scales, GM_ADDR status,
        GM_ADDR bout, GM_ADDR sigma, GM_ADDR vt, int rows, int cols, int batch, int block)
    {
        rows_ = rows; cols_ = cols; batch_ = batch; block_ = block;
        nb_ = ((rows + block - 1) / block + 1) / 2 * 2;
        if (nb_ < 2) nb_ = 2;
        rp_ = nb_ * block; cp_ = (cols + 31) / 32 * 32;
        d_ = 2 * block; np_ = nb_ / 2;
        cid_ = GetBlockIdx(); cores_ = GetBlockNum() * 2;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrix));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(panels));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gram));
        r_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotations));
        f_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(flags));
        scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
        status_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
        bout_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bout));
        sigma_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sigma));
        vt_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vt));
        rotation_.Init(pipe, d_);
        // At most (3*4096 + 3*4096 + 16)*4 < 128 KiB. The Gram
        // eigensolver uses this storage only during Parameters(), between
        // preprocessing and postprocessing, so total UB stays below 192 KiB.
        const int ra = (rows_ + 7) / 8 * 8;
        row_ = rotation_.Scratch(); work_ = row_[cp_]; temp_ = row_[2 * cp_];
        norms_ = row_[3 * cp_]; sorted_ = norms_[ra];
        perm_ = row_.ReinterpretCast<int32_t>()[3 * cp_ + 2 * ra];
        ctrl_ = row_.ReinterpretCast<int32_t>()[3 * cp_ + 3 * ra];
        scaleBuf_ = row_[3 * cp_ + 3 * ra + 8];
        m2s_ = pipe->FetchEventID(HardEvent::MTE2_S);
        m2v_ = pipe->FetchEventID(HardEvent::MTE2_V);
        v2s_ = pipe->FetchEventID(HardEvent::V_S);
        s2v_ = pipe->FetchEventID(HardEvent::S_V);
        v2m3_ = pipe->FetchEventID(HardEvent::V_MTE3);
        s2m3_ = pipe->FetchEventID(HardEvent::S_MTE3);
        m3s_ = pipe->FetchEventID(HardEvent::MTE3_S);
        m3v_ = pipe->FetchEventID(HardEvent::MTE3_V);
        m3m2_ = pipe->FetchEventID(HardEvent::MTE3_MTE2);
        v2m2_ = pipe->FetchEventID(HardEvent::V_MTE2);
    }

    __aicore__ inline void Initialize()
    {
        for (int m = cid_; m < batch_; m += cores_) {
            float scale = 0.0f;
            for (int i = 0; i < rows_; ++i) {
                LoadRaw(m, i);
                Abs(work_, row_, cp_);
                ReduceMax(temp_, work_, row_, cp_, false);
                V2S();
                const float value = temp_.GetValue(0);
                if (value > scale) scale = value;
            }
            if (scale == 0.0f) scale = 1.0f;
            scaleBuf_.SetValue(0, scale);
            S2M3(); DataCopy(scales_[m * 8], scaleBuf_, 8); M3S();
            // Two powers-of-two stages keep the reciprocal representable.
            const float pre = scale < 1e-20f ? 18446744073709551616.0f :
                              (scale > 1e20f ? 5.421010862427522e-20f : 1.0f);
            const float inverse = 1.0f / (scale * pre);
            for (int i = 0; i < rp_; ++i) {
                if (i < rows_) LoadRaw(m, i);
                else { S2V(); Duplicate(row_, 0.0f, cp_); }
                Muls(row_, row_, pre, cp_);
                Muls(row_, row_, inverse, cp_);
                Store(b_, (static_cast<int64_t>(m) * rp_ + i) * cp_, cp_);
            }
        }
    }

    __aicore__ inline void Pack(int round)
    {
        for (int task = cid_; task < batch_ * rp_; task += cores_) {
            const int job = task / d_, i = task % d_, m = job / np_, pair = job % np_;
            const int rb = PairBlock(pair, i / block_, round, nb_);
            Load(b_, (static_cast<int64_t>(m) * rp_ + rb * block_ + i % block_) * cp_);
            Store(x_, static_cast<int64_t>(task) * cp_, cp_);
        }
    }

    __aicore__ inline bool Parameters(int begin, int end, int localSweeps, float tol)
    {
        M3M2();
        bool active = false;
        for (int job = begin + cid_; job < end; job += cores_)
            if (rotation_.Run(g_, r_, f_, job, d_, localSweeps, tol)) active = true;
        return active;
    }

    __aicore__ inline void PublishActive(bool active)
    {
        ctrl_.SetValue(0, active ? 1 : 0);
        S2M3(); DataCopy(status_[cid_ * 8], ctrl_, 8); M3S();
    }

    __aicore__ inline void ReduceConvergence(int sweeps)
    {
        if (cid_ != 0) return;
        bool active = false;
        for (int core = 0; core < cores_; ++core) {
            DataCopy(ctrl_, status_[core * 8], 8); M2S();
            if (ctrl_.GetValue(0) != 0) active = true;
        }
        ctrl_.SetValue(0, sweeps); ctrl_.SetValue(1, active ? 0 : 1);
        ctrl_.SetValue(2, 0);
        S2M3(); DataCopy(status_[cores_ * 8], ctrl_, 8); M3S();
    }

    __aicore__ inline bool Converged()
    {
        DataCopy(ctrl_, status_[cores_ * 8], 8); M2S();
        return ctrl_.GetValue(1) != 0;
    }

    __aicore__ inline void Finalize()
    {
        const int rank = rows_ < cols_ ? rows_ : cols_;
        for (int m = cid_; m < batch_; m += cores_) {
            DataCopy(scaleBuf_, scales_[m * 8], 8); M2S();
            const float scale = scaleBuf_.GetValue(0);
            for (int i = 0; i < rows_; ++i) {
                Load(b_, (static_cast<int64_t>(m) * rp_ + i) * cp_);
                Mul(work_, row_, row_, cp_);
                ReduceSum(temp_, work_, row_, cp_);
                V2S();
                const float v = temp_.GetValue(0);
                norms_.SetValue(i, sqrt(v > 0 ? v : 0.0f));
                perm_.SetValue(i, i);
            }
            // Min-heap sort gives descending indices in O(rows*log(rows)).
            // B_rot preserves the unsorted row order.
            for (int root = rows_ / 2 - 1; root >= 0; --root) SiftDown(root, rows_);
            for (int end = rows_ - 1; end > 0; --end) {
                const int saved = perm_.GetValue(0);
                perm_.SetValue(0, perm_.GetValue(end)); perm_.SetValue(end, saved);
                SiftDown(0, end);
            }
            const float largest = norms_.GetValue(perm_.GetValue(0));
            for (int k = 0; k < rank; ++k) {
                const int i = perm_.GetValue(k);
                const float norm = norms_.GetValue(i);
                sorted_.SetValue(k, norm * scale);
                Load(b_, (static_cast<int64_t>(m) * rp_ + i) * cp_);
                if (norm > 1e-6f * (largest > 1e-30f ? largest : 1e-30f))
                    Muls(row_, row_, 1.0f / norm, cp_);
                else Duplicate(row_, 0.0f, cp_);
                Store(vt_, (static_cast<int64_t>(m) * rank + k) * cols_, cols_);
            }
            S2M3();
            DataCopyExtParams out{1, static_cast<uint32_t>(rank * sizeof(float)), 0, 0, 0};
            DataCopyPad(sigma_[static_cast<int64_t>(m) * rank], sorted_, out); M3S();
            for (int i = 0; i < rows_; ++i) {
                Load(b_, (static_cast<int64_t>(m) * rp_ + i) * cp_);
                Muls(row_, row_, scale, cp_);
                Store(bout_, (static_cast<int64_t>(m) * rows_ + i) * cols_, cols_);
            }
        }
        if (cid_ == 0) {
            DataCopy(ctrl_, status_[cores_ * 8], 8); M2S();
            ctrl_.SetValue(2, kDone);
            S2M3(); DataCopy(status_[cores_ * 8], ctrl_, 8); M3S();
        }
    }

private:
    __aicore__ inline void SiftDown(int root, int count)
    {
        const int saved = perm_.GetValue(root);
        const float value = norms_.GetValue(saved);
        while (2 * root + 1 < count) {
            int child = 2 * root + 1;
            if (child + 1 < count && norms_.GetValue(perm_.GetValue(child + 1)) < norms_.GetValue(perm_.GetValue(child))) ++child;
            const int next = perm_.GetValue(child);
            if (norms_.GetValue(next) >= value) break;
            perm_.SetValue(root, next); root = child;
        }
        perm_.SetValue(root, saved);
    }
    __aicore__ inline void LoadRaw(int m, int i)
    {
        M3M2();
        S2V(); Duplicate(row_, 0.0f, cp_);
        SetFlag<HardEvent::V_MTE2>(v2m2_); WaitFlag<HardEvent::V_MTE2>(v2m2_);
        // The full destination row was zeroed, including padding beyond 32B.
        DataCopyExtParams in{1, static_cast<uint32_t>(cols_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>((8 - cols_ % 8) % 8), 0};
        DataCopyPad(row_, a_[(static_cast<int64_t>(m) * rows_ + i) * cols_], in, pad);
        M2V();
    }
    __aicore__ inline void Load(GlobalTensor<float> src, int64_t offset)
    { M3M2(); DataCopy(row_, src[offset], cp_); M2V(); }
    __aicore__ inline void Store(GlobalTensor<float> dst, int64_t offset, int count)
    {
        V2M3();
        DataCopyExtParams out{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(dst[offset], row_, out); M3V();
    }
    __aicore__ inline void M2S() { SetFlag<HardEvent::MTE2_S>(m2s_); WaitFlag<HardEvent::MTE2_S>(m2s_); }
    __aicore__ inline void M2V() { SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_); }
    __aicore__ inline void V2S() { SetFlag<HardEvent::V_S>(v2s_); WaitFlag<HardEvent::V_S>(v2s_); }
    __aicore__ inline void S2V() { SetFlag<HardEvent::S_V>(s2v_); WaitFlag<HardEvent::S_V>(s2v_); }
    __aicore__ inline void V2M3() { SetFlag<HardEvent::V_MTE3>(v2m3_); WaitFlag<HardEvent::V_MTE3>(v2m3_); }
    __aicore__ inline void S2M3() { SetFlag<HardEvent::S_MTE3>(s2m3_); WaitFlag<HardEvent::S_MTE3>(s2m3_); }
    __aicore__ inline void M3S() { SetFlag<HardEvent::MTE3_S>(m3s_); WaitFlag<HardEvent::MTE3_S>(m3s_); }
    __aicore__ inline void M3V() { SetFlag<HardEvent::MTE3_V>(m3v_); WaitFlag<HardEvent::MTE3_V>(m3v_); }
    __aicore__ inline void M3M2() { SetFlag<HardEvent::MTE3_MTE2>(m3m2_); WaitFlag<HardEvent::MTE3_MTE2>(m3m2_); }
    GlobalTensor<float> a_, b_, x_, g_, r_, f_, scales_, bout_, sigma_, vt_;
    GlobalTensor<int32_t> status_;
    LocalTensor<float> row_, work_, temp_, norms_, sorted_, scaleBuf_;
    LocalTensor<int32_t> perm_, ctrl_;
    BlockRotation rotation_;
    int rows_, cols_, batch_, block_, nb_, rp_, cp_, d_, np_, cid_, cores_;
    int m2s_, m2v_, v2s_, s2v_, v2m3_, s2m3_, m3s_, m3v_, m3m2_, v2m2_;
};
#endif
}
