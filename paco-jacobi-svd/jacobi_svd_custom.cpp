#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include <cstdint>

#include "kernel_operator.h"
#include "gemm_trans.hpp"

using namespace AscendC;

/**
 * 批量单边（块状）雅可比 SVD —— 小 shape + 大 batch
 *
 * 调度（多层划分）
 * ----------------
 * 1) pad（device 侧）：A_raw(rows x cols) -> A_pad(rowsPad x colsPad)，补 0；
 *    行 64 对齐（AIC 分块），列 32 对齐（cube fractal 是 16，AIV 只需 8）。
 * 2) 每个 AIV 平均分到 perCore = ceil(batch/coreNum) 个矩阵：
 *    - perCore == 1：一矩阵一核；
 *    - perCore >  1：再按 UB 容量切块，一次装 K 个矩阵（按行交错 tile[(i*K+k)*colsW]），
 *      同一对 (i,j) 的 Mul 合并成 1 条、归约合并成 1 条 WholeReduceSum（出 3K 个结果）。
 * 3) AIC：等全部 AIV 完成后分摊矩阵，用 MatmulCustom(isTransposeB) 算 G = B·Bᵀ，
 *    主机取对角线开根得到奇异值。
 *
 * sweep 停止：一个 sweep 内没有任何旋转（|c| <= 1e-6*sqrt(a*b)）即收敛；maxSweeps 兜底。
 */

#define FLAG_VEC_BAR 0x0
#define FLAG_VEC_DONE 0x2
#define FLAG_AIC_BAR 0x3
#define FLAG_GRAM_DONE 0x4   // AIC -> AIV：Gram 已算好

namespace jacobi_svd {

constexpr int32_t kB = 64;
constexpr int32_t kMaxK = 64;               // 一次最多 64 个矩阵（WholeReduceSum 的 3K<=255）
constexpr int32_t kMultiLen = 2048;         // K*colsW 上限（UB 预算内）
constexpr int32_t kCopyChunk = 16384;       // 单条 DataCopy 上限 64KB

__aicore__ inline int32_t MinI(int32_t a, int32_t b) { return a < b ? a : b; }

class UnitWorker {
public:
    __aicore__ inline UnitWorker() {}

    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(tileBuf_, 32768 * static_cast<int32_t>(sizeof(float)));   // 128KB
        pipe->InitBuffer(tempBuf_, 4 * kMultiLen * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(redBuf_, 256 * static_cast<int32_t>(sizeof(float)));             // 3*r8 <= 192
        pipe->InitBuffer(prodBuf_, kMultiLen * static_cast<int32_t>(sizeof(float)));      // 只需 K*colsW
        pipe->InitBuffer(angBuf_, 16 * 64 * static_cast<int32_t>(sizeof(float)));   // 每种 K<=64 个
        pipe->InitBuffer(maskBuf_, 256);

        tile_ = tileBuf_.Get<float>();
        LocalTensor<float> t = tempBuf_.Get<float>();
        rowA_ = t[0];
        rowB_ = t[kMultiLen];
        rowC_ = t[2 * kMultiLen];
        rowW_ = t[3 * kMultiLen];
        red_ = redBuf_.Get<float>();
        prod3_ = prodBuf_.Get<float>();
        mask_ = maskBuf_.Get<uint8_t>();
        {
            LocalTensor<float> av = angBuf_.Get<float>();
            scls_ = av[0];   thrV_ = av[64];   absV_ = av[128];  diffV_ = av[192];
            tauV_ = av[256]; denV_ = av[320];  svV_ = av[384];   tmpV_ = av[448];
            tV_ = av[512];   csV_ = av[576];   snV_ = av[640];   onesV_ = av[704];
            zerosV_ = av[768];
            sgnV_ = av[832];
            actV_ = av[896];
            ls_ = av[960];
            Duplicate(onesV_, 1.0f, 64);
            Duplicate(zerosV_, 0.0f, 64);
            mask2_ = maskBuf_.Get<uint8_t>()[128];
        }

        evVtoS_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        evStoV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        evS2M3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        evM2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        evV2M3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        evM3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        evM3M2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    // ------------------------------------------------------------ 搬运
    __aicore__ inline void LoadRows(const LocalTensor<float> &dst, const GlobalTensor<float> &src,
                                    int64_t srcOff, int32_t count)
    {
        for (int32_t done = 0; done < count; done += kCopyChunk) {
            const int32_t n = MinI(kCopyChunk, count - done);
            SetFlag<HardEvent::MTE3_MTE2>(evM3M2_);
            WaitFlag<HardEvent::MTE3_MTE2>(evM3M2_);
            DataCopy(dst[done], src[srcOff + done], n);
            SetFlag<HardEvent::MTE2_V>(evM2V_);
            WaitFlag<HardEvent::MTE2_V>(evM2V_);
        }
    }

    __aicore__ inline void StoreRows(const GlobalTensor<float> &dst, int64_t dstOff,
                                     const LocalTensor<float> &src, int32_t count)
    {
        for (int32_t done = 0; done < count; done += kCopyChunk) {
            const int32_t n = MinI(kCopyChunk, count - done);
            SetFlag<HardEvent::V_MTE3>(evV2M3_);
            WaitFlag<HardEvent::V_MTE3>(evV2M3_);
            DataCopy(dst[dstOff + done], src[done], n);
            SetFlag<HardEvent::MTE3_V>(evM3V_);
            WaitFlag<HardEvent::MTE3_V>(evM3V_);
        }
    }

    /** pad：raw 第 r 行 -> pad 第 r 行（任意列数，右侧补 0 到 8 对齐） */
    __aicore__ inline void PadRow(GlobalTensor<float> rawG, GlobalTensor<float> padG, int32_t r,
                                  int32_t cols, int32_t colsPad)
    {
        const int32_t colsW = ((cols + 7) / 8) * 8;
        DataCopyExtParams cp{1, static_cast<uint32_t>(cols * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pp{true, 0, static_cast<uint8_t>(colsW - cols), 0};
        SetFlag<HardEvent::MTE3_MTE2>(evM3M2_);
        WaitFlag<HardEvent::MTE3_MTE2>(evM3M2_);
        DataCopyPad(tile_, rawG[static_cast<int64_t>(r) * cols], cp, pp);
        SetFlag<HardEvent::MTE2_V>(evM2V_);
        WaitFlag<HardEvent::MTE2_V>(evM2V_);
        StoreRows(padG, static_cast<int64_t>(r) * colsPad, tile_, colsW);
    }

    // ------------------------------------------------------------ 单矩阵
    /** 一次旋转：u' = cs*u - sn*v, v' = sn*u + cs*v（只留一次 Adds 拷回） */
    __aicore__ inline bool RotateRows(const LocalTensor<float> &u, const LocalTensor<float> &v,
                                      int32_t cols)
    {
        if (cols <= 64) {
            Mul(prod3_[0], u, u, cols);
            Mul(prod3_[cols], v, v, cols);
            Mul(prod3_[2 * cols], u, v, cols);
            WholeReduceSum(red_, prod3_, cols, 3, 1, 1, cols / 8);
        } else {
            Mul(rowA_, u, u, cols);
            Mul(rowB_, v, v, cols);
            Mul(rowC_, u, v, cols);
            ReduceSum(red_[0], rowA_, rowW_, cols);
            ReduceSum(red_[8], rowB_, rowW_, cols);
            ReduceSum(red_[16], rowC_, rowW_, cols);
        }

        const int32_t st = (cols <= 64) ? 1 : 8;
        const float a = red_.GetValue(0);
        const float b = red_.GetValue(st);
        const float c = red_.GetValue(2 * st);
        const float sc = sqrt(a * b);
        if (c <= 1e-6f * sc && c >= -1e-6f * sc) return false;

        const float tau = (b - a) / (2.0f * c);
        const float t = (tau >= 0.0f) ? (1.0f / (tau + sqrt(1.0f + tau * tau)))
                                      : (1.0f / (tau - sqrt(1.0f + tau * tau)));
        const float invCs = sqrt(1.0f + t * t);
        const float cs = 1.0f / invCs;
        const float sn = cs * t;

        // u' = cs*u - sn*v（就地）；v' = (1/cs)*v + t*u'
        // 标量系数走寄存器、不读 UB，所以不需要 S->V 同步
        Muls(u, u, cs, cols);
        Axpy(u, v, -sn, cols);
        Muls(v, v, invCs, cols);
        Axpy(v, u, t, cols);
        return true;
    }

    __aicore__ inline bool RunBlock(GlobalTensor<float> aG, int32_t row0, int32_t colsPad,
                                    int32_t colsWork, int32_t nRows)
    {
        bool rotated = false;
        const int32_t elems = nRows * colsPad;
        LoadRows(tile_, aG, static_cast<int64_t>(row0) * colsPad, elems);
        for (int32_t i = 0; i < nRows; ++i) {
            for (int32_t j = i + 1; j < nRows; ++j) {
                if (RotateRows(tile_[i * colsPad], tile_[j * colsPad], colsWork)) rotated = true;
            }
        }
        StoreRows(aG, static_cast<int64_t>(row0) * colsPad, tile_, elems);
        return rotated;
    }

    __aicore__ inline bool RunPair(GlobalTensor<float> aG, int32_t rowP, int32_t rowQ,
                                   int32_t colsPad, int32_t colsWork, int32_t nP, int32_t nQ)
    {
        bool rotated = false;
        const int32_t elemsP = nP * colsPad;
        const int32_t elemsQ = nQ * colsPad;
        LoadRows(tile_, aG, static_cast<int64_t>(rowP) * colsPad, elemsP);
        LoadRows(tile_[kB * colsPad], aG, static_cast<int64_t>(rowQ) * colsPad, elemsQ);
        for (int32_t i = 0; i < nP; ++i) {
            for (int32_t j = 0; j < nQ; ++j) {
                if (RotateRows(tile_[i * colsPad], tile_[(kB + j) * colsPad], colsWork)) rotated = true;
            }
        }
        StoreRows(aG, static_cast<int64_t>(rowP) * colsPad, tile_, elemsP);
        StoreRows(aG, static_cast<int64_t>(rowQ) * colsPad, tile_[kB * colsPad], elemsQ);
        return rotated;
    }

    __aicore__ inline void CopyWhole(GlobalTensor<float> aG, GlobalTensor<float> bG, int32_t elems)
    {
        constexpr int32_t kTileElems = 32768;
        for (int32_t off = 0; off < elems; off += kTileElems) {
            const int32_t n = MinI(kTileElems, elems - off);
            LoadRows(tile_, aG, off, n);
            StoreRows(bG, off, tile_, n);
        }
    }

    // ------------------------------------------------------------ K 个矩阵合并
    /** 载入 K 个矩阵，按行交错：tile[(i*K+k)*colsW] */
    __aicore__ inline void LoadMulti(__gm__ float *base, int32_t m0, int32_t nRows, int32_t colsW,
                                     int32_t colsPad, int32_t K, int32_t batchSize, int64_t matElems)
    {
        for (int32_t k = 0; k < K; ++k) {
            if (m0 + k >= batchSize) break;
            GlobalTensor<float> g;
            g.SetGlobalBuffer(base + static_cast<int64_t>(m0 + k) * matElems);
            for (int32_t i = 0; i < nRows; ++i) {
                SetFlag<HardEvent::MTE3_MTE2>(evM3M2_);
                WaitFlag<HardEvent::MTE3_MTE2>(evM3M2_);
                DataCopy(tile_[(i * K + k) * colsW], g[static_cast<int64_t>(i) * colsPad], colsW);
                SetFlag<HardEvent::MTE2_V>(evM2V_);
                WaitFlag<HardEvent::MTE2_V>(evM2V_);
            }
        }
    }

    __aicore__ inline void StoreMulti(__gm__ float *base, int32_t m0, int32_t nRows, int32_t colsW,
                                      int32_t colsPad, int32_t K, int32_t batchSize, int64_t matElems)
    {
        for (int32_t k = 0; k < K; ++k) {
            if (m0 + k >= batchSize) break;
            GlobalTensor<float> g;
            g.SetGlobalBuffer(base + static_cast<int64_t>(m0 + k) * matElems);
            for (int32_t i = 0; i < nRows; ++i) {
                SetFlag<HardEvent::V_MTE3>(evV2M3_);
                WaitFlag<HardEvent::V_MTE3>(evV2M3_);
                DataCopy(g[static_cast<int64_t>(i) * colsPad], tile_[(i * K + k) * colsW], colsW);
                SetFlag<HardEvent::MTE3_V>(evM3V_);
                WaitFlag<HardEvent::MTE3_V>(evM3V_);
            }
        }
    }

    /** K 个矩阵一起做一个 sweep：Mul 合并 + 归约合并 */
    __aicore__ inline bool RunMulti(int32_t nRows, int32_t colsW, int32_t K)
    {
        bool rotated = false;
        const int32_t n = K * colsW;
        // r8 = K 向上取 8 的倍数（32B 对齐）；只占 3*r8 个 float，prod 只需 n 个
        const int32_t r8 = ((K + 7) / 8) * 8;
        for (int32_t i = 0; i < nRows; ++i) {
            LocalTensor<float> u = tile_[(i * K) * colsW];
            for (int32_t j = i + 1; j < nRows; ++j) {
                LocalTensor<float> v = tile_[(j * K) * colsW];
                // 三次 WholeReduceSum，结果分别落在 red_[0] / red_[r8] / red_[2*r8]
                Mul(prod3_, u, u, n);
                WholeReduceSum(red_[0], prod3_, colsW, K, 1, 1, colsW / 8);
                Mul(prod3_, v, v, n);
                WholeReduceSum(red_[r8], prod3_, colsW, K, 1, 1, colsW / 8);
                Mul(prod3_, u, v, n);
                WholeReduceSum(red_[2 * r8], prod3_, colsW, K, 1, 1, colsW / 8);
                //SetFlag<HardEvent::V_S>(evVtoS_);
                //WaitFlag<HardEvent::V_S>(evVtoS_);
                // sc 用矢量算（count = r8，8 的倍数/32B 对齐）；GetValue 自带同步，不再手动 V->S
                Mul(scls_, red_, red_[r8], r8);
                Sqrt(scls_, scls_, r8);
                Sub(tauV_, red_[r8], red_, r8);      // b - a
                Muls(denV_, red_[2 * r8], 2.0f, r8); // 2c
                Div(tauV_, tauV_, denV_, r8);        // tau = (b-a)/(2c)
                Mul(svV_, tauV_, tauV_, r8);
                Adds(svV_, svV_, 1.0f, r8);
                Sqrt(svV_, svV_, r8);                // s = sqrt(1+tau^2)
                for (int32_t k = 0; k < K; ++k) {
                    const float c = red_.GetValue(2 * r8 + k);
                    const float sc = scls_.GetValue(k);
                    if (c <= 1e-6f * sc && c >= -1e-6f * sc) continue;
                    const float tau = tauV_.GetValue(k);
                    const float svk = svV_.GetValue(k);
                    const float t = (tau >= 0.0f) ? (1.0f / (tau + svk)) : (1.0f / (tau - svk));
                    const float invCs = sqrt(1.0f + t * t);
                    const float cs = 1.0f / invCs;
                    const float sn = cs * t;
                    rotated = true;
                    LocalTensor<float> uk = u[k * colsW];
                    LocalTensor<float> vk = v[k * colsW];
                    // u' = cs*u - sn*v（就地）；v' = (1/cs)*v + t*u'，全程无 Adds
                    Muls(uk, uk, cs, colsW);
                    Axpy(uk, vk, -sn, colsW);
                    
                    Muls(vk, vk, invCs, colsW);
                    Axpy(vk, uk, t, colsW);
                }
            }
        }
        return rotated;
    }

    /** 从 G 的对角线取 sqrt 写 sigma（AIC 算完 Gram 之后由 AIV 调用） */
    __aicore__ inline void WriteSigmaFromG(__gm__ float *gBase, __gm__ float *sigmaBase, int32_t mStart,
                                           int32_t mEnd, int32_t rows, int32_t rowsPad)
    {
        for (int32_t m = mStart; m < mEnd; ++m) {
            GlobalTensor<float> gT, sT;
            gT.SetGlobalBuffer(gBase + static_cast<int64_t>(m) * rowsPad * rowsPad);
            sT.SetGlobalBuffer(sigmaBase + static_cast<int64_t>(m) * rowsPad);   // 每矩阵 rowsPad 槽位，最后一段可写满 8
            LoadRows(tile_, gT, 0, rowsPad * rowsPad);
            for (int32_t i = 0; i < rows; i += 8) {
                const int32_t cnt = MinI(8, rows - i);
                for (int32_t t = 0; t < cnt; ++t) {
                    const int32_t d = i + t;
                    red_.SetValue(200 + t, sqrt(tile_.GetValue(d * rowsPad + d)));
                }
                SetFlag<HardEvent::S_MTE3>(evS2M3_);
                WaitFlag<HardEvent::S_MTE3>(evS2M3_);
                DataCopy(sT[static_cast<int64_t>(i)], red_[200], 8);
                SetFlag<HardEvent::MTE3_V>(evM3V_);
                WaitFlag<HardEvent::MTE3_V>(evM3V_);
            }
        }
    }

    __aicore__ inline LocalTensor<float> Tile() { return tile_; }
    __aicore__ inline void SetDbg(__gm__ float *p) { dbg_ = p; }
    __aicore__ inline int32_t EvV2M3() const { return evV2M3_; }
    __aicore__ inline int32_t EvM3V() const { return evM3V_; }

private:
    TBuf<TPosition::VECCALC> tileBuf_;
    TBuf<TPosition::VECCALC> tempBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    TBuf<TPosition::VECCALC> prodBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> angBuf_;
    LocalTensor<float> scls_, thrV_, absV_, diffV_, tauV_, denV_, svV_, tmpV_, tV_, csV_, snV_, onesV_, zerosV_;
    LocalTensor<float> sgnV_, actV_, ls_;
    __gm__ float *dbg_{nullptr};
    int32_t dbgCnt_{0};
    LocalTensor<uint8_t> mask2_;

    LocalTensor<float> tile_;
    LocalTensor<float> rowA_, rowB_, rowC_, rowW_;
    LocalTensor<float> red_, prod3_;
    LocalTensor<uint8_t> mask_;

    int32_t evVtoS_{0}, evStoV_{0}, evS2M3_{0}, evM2V_{0}, evV2M3_{0}, evM3V_{0}, evM3M2_{0};
};

}  // namespace jacobi_svd

extern "C" __global__ __aicore__ void jacobi_svd_batched(
    int32_t rows, int32_t cols, int32_t batchSize, int32_t maxSweeps, int32_t coreNum, int32_t aicNum,
    GM_ADDR A, GM_ADDR APad, GM_ADDR B, GM_ADDR G, GM_ADDR sigma, GM_ADDR flags, int32_t stages)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr int32_t kB = jacobi_svd::kB;
    if (rows <= 0 || cols <= 0 || batchSize <= 0 || coreNum <= 0) return;

    const int32_t rowsPad = ((rows + kB - 1) / kB) * kB;
    const int32_t colsPad = ((cols + 31) / 32) * 32;
    const int32_t colsWork = ((cols + 7) / 8) * 8;
    const int32_t nb = (rows + kB - 1) / kB;
    const int64_t rawElems = static_cast<int64_t>(rows) * cols;
    const int64_t matElems = static_cast<int64_t>(rowsPad) * colsPad;

#ifdef __DAV_C220_VEC__
    const int32_t cid = static_cast<int32_t>(GetBlockIdx());

    TPipe pipe;
    jacobi_svd::UnitWorker op;
    op.Init(&pipe);
    op.SetDbg(reinterpret_cast<__gm__ float *>(flags));

    GlobalTensor<float> aG, bG;
    GlobalTensor<float> flagsA;
    flagsA.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(flags));

    // ---- 阶段 0：device 侧 pad ----
    // ---- 阶段 0：device 侧 pad ----
    if (cols == colsPad) {
        // 列已对齐：一个矩阵连续，一条 DataCopy 搬完
        for (int32_t m = cid; m < batchSize; m += coreNum) {
            GlobalTensor<float> rawG, padG;
            rawG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A) + static_cast<int64_t>(m) * rawElems);
            padG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(APad) + static_cast<int64_t>(m) * matElems);
            op.CopyWhole(rawG, padG, static_cast<int32_t>(rawElems));
        }
    } else {
        // 列不对齐：cols 是 8 的倍数时用带 stride 的多行搬运；否则退回逐行 DataCopyPad
        if ((cols % 8) == 0) {
            const int32_t blockLen32 = (cols * 4) / 32;
            const int32_t dstStride32 = (colsPad - cols) / 8;
            const int32_t rowsPer = jacobi_svd::MinI(rows, 2048 / cols);
            for (int32_t m = cid; m < batchSize; m += coreNum) {
                GlobalTensor<float> rawG, padG;
                rawG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A) + static_cast<int64_t>(m) * rawElems);
                padG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(APad) + static_cast<int64_t>(m) * matElems);
                for (int32_t r0 = 0; r0 < rows; r0 += rowsPer) {
                    const int32_t cnt = jacobi_svd::MinI(rowsPer, rows - r0);
                    op.LoadRows(op.Tile(), rawG, static_cast<int64_t>(r0) * cols, cnt * cols);
                    DataCopyParams cp{static_cast<uint16_t>(cnt), static_cast<uint16_t>(blockLen32), 0,
                                      static_cast<uint16_t>(dstStride32)};
                    SetFlag<HardEvent::V_MTE3>(op.EvV2M3());
                    WaitFlag<HardEvent::V_MTE3>(op.EvV2M3());
                    DataCopy(padG[static_cast<int64_t>(r0) * colsPad], op.Tile(), cp);
                    SetFlag<HardEvent::MTE3_V>(op.EvM3V());
                    WaitFlag<HardEvent::MTE3_V>(op.EvM3V());
                }
            }
        } else {
            // cols 不是 8 的倍数：逐行 DataCopyPad（右侧补 0）
            int32_t task = 0;
            for (int32_t m = 0; m < batchSize; ++m) {
                GlobalTensor<float> rawG, padG;
                rawG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A) + static_cast<int64_t>(m) * rawElems);
                padG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(APad) + static_cast<int64_t>(m) * matElems);
                for (int32_t r = 0; r < rows; ++r) {
                    if ((task % coreNum) == cid) op.PadRow(rawG, padG, r, cols, colsPad);
                    ++task;
                }
            }
        }
    }
    CrossCoreSetFlag<0x0, PIPE_MTE3>(FLAG_VEC_BAR);
    CrossCoreWaitFlag(FLAG_VEC_BAR);
    if (stages < 1) return;   // 只测 pad

    // ---- 阶段 1：旋转 ----
    const int32_t perCore = (batchSize + coreNum - 1) / coreNum;
    const int32_t mStart = cid * perCore;
    const int32_t mEnd = jacobi_svd::MinI(batchSize, mStart + perCore);

    if (rows <= kB && colsWork <= 64 && perCore > 1) {
        constexpr int32_t kUbBudget = 120 * 1024;   // 实测 K 再大收益很小，保守取 120KB
        const int32_t perMatBytes = (rows + 4) * colsWork * static_cast<int32_t>(sizeof(float));
        int32_t K = (perMatBytes > 0) ? (kUbBudget / perMatBytes) : 1;
        if (K < 1) K = 1;
        if (K > jacobi_svd::kMaxK) K = jacobi_svd::kMaxK;
        for (int32_t m0 = mStart; m0 < mEnd; m0 += K) {
            const int32_t kk = jacobi_svd::MinI(K, mEnd - m0);
            op.LoadMulti(reinterpret_cast<__gm__ float *>(APad), m0, rows, colsWork, colsPad, kk,
                         batchSize, matElems);
            for (int32_t sweep = 0; sweep < maxSweeps; ++sweep) {
                if (!op.RunMulti(rows, colsWork, kk)) break;
            }
            op.StoreMulti(reinterpret_cast<__gm__ float *>(B), m0, rows, colsWork, colsPad, kk,
                          batchSize, matElems);
        }
    } else {
        for (int32_t m = mStart; m < mEnd; ++m) {
            aG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(APad) + static_cast<int64_t>(m) * matElems);
            bG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B) + static_cast<int64_t>(m) * matElems);
            for (int32_t sweep = 0; sweep < maxSweeps; ++sweep) {
                bool rotated = false;
                for (int32_t p = 0; p < nb; ++p) {
                    const int32_t nRows = jacobi_svd::MinI(kB, rows - p * kB);
                    if (nRows > 1 && op.RunBlock(aG, p * kB, colsPad, colsWork, nRows)) rotated = true;
                }
                for (int32_t rnd = 0; rnd < nb; ++rnd) {
                    for (int32_t p = 0; p < nb; ++p) {
                        const int32_t q = (rnd - p + nb) % nb;
                        if (p >= q) continue;
                        const int32_t nP = jacobi_svd::MinI(kB, rows - p * kB);
                        const int32_t nQ = jacobi_svd::MinI(kB, rows - q * kB);
                        if (nP > 0 && nQ > 0 &&
                            op.RunPair(aG, p * kB, q * kB, colsPad, colsWork, nP, nQ)) {
                            rotated = true;
                        }
                    }
                }
                if (!rotated) break;
            }
            op.CopyWhole(aG, bG, static_cast<int32_t>(matElems));
        }
    }
    CrossCoreSetFlag<0x0, PIPE_MTE3>(FLAG_VEC_BAR);
    CrossCoreWaitFlag(FLAG_VEC_BAR);
    if (stages < 2) return;   // 只测 pad+旋转
    CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_VEC_DONE);

    // ---- 等本簇 AIC 算好 G，AIV 读对角线开根写 sigma ----
    // 记录本 AIV 看到的最大 sweep 数（flags[cid]）
    {
        GlobalTensor<float> fg;
        fg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(flags));
        DataCopy(fg[static_cast<int64_t>(cid) * 8], op.Tile(), 8), void();
    }
    if (stages < 3) return;   // 只测 pad+旋转+Gram
    CrossCoreWaitFlag(FLAG_GRAM_DONE);
    op.WriteSigmaFromG(reinterpret_cast<__gm__ float *>(G), reinterpret_cast<__gm__ float *>(sigma),
                       mStart, mEnd, rows, rowsPad);
#endif

#ifdef __DAV_C220_CUBE__
    const int32_t cid = static_cast<int32_t>(GetBlockIdx());

    if (stages < 2) return;
    CrossCoreWaitFlag(FLAG_VEC_DONE);
    CrossCoreSetFlag<0x0, PIPE_FIX>(FLAG_AIC_BAR);
    CrossCoreWaitFlag(FLAG_AIC_BAR);

    TPipe pipe;
    MatmulCustom<float, float, float> mm;
    mm.Init(&pipe);
    constexpr int32_t kTile = 128;
    // 与本簇两个 AIV 的矩阵范围一致（AIV 2*cid 与 2*cid+1）
    const int32_t perCore = (batchSize + (2 * aicNum) - 1) / (2 * aicNum);
    const int32_t aStart = 2 * cid * perCore;
    const int32_t aEnd = jacobi_svd::MinI(batchSize, aStart + 2 * perCore);
    for (int32_t m = aStart; m < aEnd; ++m) {
        GlobalTensor<float> bT, gT;
        bT.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B) + static_cast<int64_t>(m) * matElems);
        gT.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(G) + static_cast<int64_t>(m) * rowsPad * rowsPad);
        mm.SetMatrix(bT, bT, gT, rowsPad, colsPad);
        for (int32_t i0 = 0; i0 < rowsPad; i0 += kTile) {
            const int32_t mLen = (rowsPad - i0 < kTile) ? (rowsPad - i0) : kTile;
            for (int32_t j0 = 0; j0 < rowsPad; j0 += kTile) {
                const int32_t nLen = (rowsPad - j0 < kTile) ? (rowsPad - j0) : kTile;
                for (int32_t k0 = 0; k0 < colsPad; k0 += kTile) {
                    const int32_t kLen = (colsPad - k0 < kTile) ? (colsPad - k0) : kTile;
                    mm.Process(static_cast<int32_t>(static_cast<int64_t>(i0) * colsPad + k0),
                               static_cast<int32_t>(static_cast<int64_t>(j0) * colsPad + k0),
                               static_cast<int32_t>(static_cast<int64_t>(i0) * rowsPad + j0),
                               mLen, nLen, kLen, -1, true, k0 > 0);
                }
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_GRAM_DONE);
    CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_GRAM_DONE);   // 两个 AIV 各等一次
#endif
}
