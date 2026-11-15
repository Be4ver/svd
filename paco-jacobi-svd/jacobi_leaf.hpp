#pragma once
#include "kernel_operator.h"

namespace fused_jacobi {
using namespace AscendC;

class JacobiLeafWorker {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(tileBuf_, 8192 * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(tempBuf_, 2048 * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(redBuf_, 256 * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(prodBuf_, 4096 * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(angBuf_, 16 * 64 * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(maskBuf_, 256);
        tile_ = tileBuf_.Get<float>();
        rowA_ = tempBuf_.Get<float>();
        red_ = redBuf_.Get<float>();
        prod_ = prodBuf_.Get<float>();
        LocalTensor<float> av = angBuf_.Get<float>();
        scls_ = av[0]; thr_ = av[64]; abs_ = av[128]; diff_ = av[192];
        tau_ = av[256]; den_ = av[320]; sv_ = av[384]; tmp_ = av[448];
        t_ = av[512]; cs_ = av[576]; sn_ = av[640]; ones_ = av[704];
        zeros_ = av[768]; active_ = av[896]; ls_ = av[960];
        Duplicate(ones_, 1.0f, 64);
        Duplicate(zeros_, 0.0f, 64);
        mask_ = maskBuf_.Get<uint8_t>();
        mask2_ = maskBuf_.Get<uint8_t>()[128];
        evM2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        evV2M3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        evM3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        evM3M2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Run(__gm__ float *leaf, int32_t n, int32_t ld, int32_t leafOff,
                               __gm__ float *sigma, int32_t sigmaOff, int32_t maxSweeps)
    {
        if (n <= 0 || n > 64 || ld < n) return;
        LoadSymmetric(leaf, n, ld, leafOff);
        LocalTensor<float> cur = tile_;
        LocalTensor<float> next = tile_[4096];
        const BinaryRepeatParams br{1, 1, 0, 8, 8, 1};
        for (int32_t sweep = 0; sweep < maxSweeps; ++sweep) {
            Duplicate(ls_, 0.0f, 64);
            for (int32_t round = 0; round < 63; ++round) {
                LocalTensor<float> u = cur;
                LocalTensor<float> v = cur[2048];
                Mul(prod_, u, u, 2048);
                WholeReduceSum(red_[0], prod_, 64, 32, 1, 1, 8);
                Mul(prod_, v, v, 2048);
                WholeReduceSum(red_[64], prod_, 64, 32, 1, 1, 8);
                Mul(prod_, u, v, 2048);
                WholeReduceSum(red_[128], prod_, 64, 32, 1, 1, 8);
                Mul(scls_, red_[0], red_[64], 64);
                Sqrt(scls_, scls_, 64);
                Muls(thr_, scls_, 1e-6f, 64);
                Abs(abs_, red_[128], 64);
                Compare(mask_, abs_, thr_, CMPMODE::GT, 64);
                Select(active_, mask_, ones_, zeros_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Max(ls_, ls_, active_, 64);
                Sub(diff_, red_[64], red_[0], 64);
                Select(diff_, mask_, diff_, zeros_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Muls(den_, red_[128], 2.0f, 64);
                Select(den_, mask_, den_, ones_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Div(tau_, diff_, den_, 64);
                Abs(diff_, tau_, 64);
                Mul(sv_, tau_, tau_, 64);
                Adds(sv_, sv_, 1.0f, 64);
                Sqrt(sv_, sv_, 64);
                Add(den_, diff_, sv_, 64);
                Div(t_, ones_, den_, 64);
                Compare(mask2_, tau_, zeros_, CMPMODE::GE, 64);
                Muls(tmp_, t_, -1.0f, 64);
                Select(t_, mask2_, t_, tmp_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Select(t_, mask_, t_, zeros_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Mul(tmp_, t_, t_, 64);
                Adds(tmp_, tmp_, 1.0f, 64);
                Sqrt(tmp_, tmp_, 64);
                Div(cs_, ones_, tmp_, 64);
                Mul(sn_, cs_, t_, 64);
                Muls(sn_, sn_, -1.0f, 64);
                Brcb(rowA_[0], cs_, 4, {1, 8});
                Brcb(rowA_[256], sn_, 4, {1, 8});
                Brcb(rowA_[512], tmp_, 4, {1, 8});
                Brcb(rowA_[768], t_, 4, {1, 8});
                Mul(u, u, rowA_[0], 64, 32, br);
                MulAddDst(u, v, rowA_[256], 64, 32, br);
                Mul(v, v, rowA_[512], 64, 32, br);
                MulAddDst(v, u, rowA_[768], 64, 32, br);
                Adds(next[0], cur[0], 0.0f, 64);
                Adds(next[64], cur[2048], 0.0f, 64);
                Adds(next[128], cur[64], 0.0f, 30 * 64);
                Adds(next[2048], cur[2112], 0.0f, 31 * 64);
                Adds(next[4032], cur[1984], 0.0f, 64);
                LocalTensor<float> sw = cur; cur = next; next = sw;
            }
            WholeReduceMax(red_[200], ls_, 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            if (red_.GetValue(200) == 0.0f) break;
        }
        Mul(prod_, cur, cur, 4096);
        WholeReduceSum(red_[0], prod_, 64, 64, 1, 1, 8);
        for (int32_t i = 0; i < 64; ++i) {
            float x = red_.GetValue(i);
            red_.SetValue(128 + i, sqrt(x > 0.0f ? x : 0.0f));
        }
        // For n<64, Jacobi rotations can spread valid-row energy across all rows.
        // Sort every norm before keeping the largest n singular values.
        for (int32_t i = 0; i < 64; ++i) {
            int32_t best = i;
            float bv = red_.GetValue(128 + i);
            for (int32_t j = i + 1; j < 64; ++j) {
                float v = red_.GetValue(128 + j);
                if (v > bv) { bv = v; best = j; }
            }
            if (best != i) {
                float q = red_.GetValue(128 + i);
                red_.SetValue(128 + i, red_.GetValue(128 + best));
                red_.SetValue(128 + best, q);
            }
        }
        GlobalTensor<float> out;
        out.SetGlobalBuffer(sigma);
        for (int32_t i = 0; i < n; ++i) out.SetValue(static_cast<uint64_t>(sigmaOff + i), red_.GetValue(128 + i));
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(out[sigmaOff]);
    }

private:
    __aicore__ inline void LoadSymmetric(__gm__ float *leaf, int32_t n, int32_t ld, int32_t off)
    {
        Duplicate(tile_, 0.0f, 8192);
        GlobalTensor<float> src;
        src.SetGlobalBuffer(leaf + off);
        for (int32_t r = 0; r < n; ++r) {
            DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pp{true, 0, static_cast<uint8_t>(64 - n), 0};
            SetFlag<HardEvent::MTE3_MTE2>(evM3M2_);
            WaitFlag<HardEvent::MTE3_MTE2>(evM3M2_);
            DataCopyPad(tile_[r * 64], src[static_cast<int64_t>(r) * ld], cp, pp);
            SetFlag<HardEvent::MTE2_V>(evM2V_);
            WaitFlag<HardEvent::MTE2_V>(evM2V_);
        }
        for (int32_t r = 0; r < 64; ++r) {
            for (int32_t c = r + 1; c < 64; ++c) {
                float x = 0.5f * (tile_.GetValue(r * 64 + c) + tile_.GetValue(c * 64 + r));
                tile_.SetValue(r * 64 + c, x);
                tile_.SetValue(c * 64 + r, x);
            }
        }
    }

    TBuf<TPosition::VECCALC> tileBuf_, tempBuf_, redBuf_, prodBuf_, angBuf_, maskBuf_;
    LocalTensor<float> tile_, rowA_, red_, prod_;
    LocalTensor<float> scls_, thr_, abs_, diff_, tau_, den_, sv_, tmp_, t_, cs_, sn_, ones_, zeros_, active_, ls_;
    LocalTensor<uint8_t> mask_, mask2_;
    int32_t evM2V_{0}, evV2M3_{0}, evM3V_{0}, evM3M2_{0};
};
} // namespace fused_jacobi


