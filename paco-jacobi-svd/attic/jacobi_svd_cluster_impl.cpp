#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include <cstdint>

#include "kernel_operator.h"
#include "gemm.hpp"

using namespace AscendC;

/**
 * 批量单边（块状）雅可比 SVD —— AIC/AIV 分工 + 簇内流水（昇腾 910B3）
 *
 * 硬件模型
 * --------
 * 910B3 的一个"核" = 1 个 AIC(Cube) + 2 个 AIV(Vector)。MIX_AIC_1_2 模式下，
 * 核内 AIV 与 AIC 用 CrossCoreSetFlag/CrossCoreWaitFlag（mode 0x2）同步，同步
 * 只在核内有效。因此把矩阵按核分配，每个核独立处理自己的矩阵：
 *     AIV（2 个）: 算 Gram 矩阵 / 行范数 / 组装旋转矩阵 R
 *     AIC（1 个）: 用 Cube 执行旋转  W <- R * W
 * 两者用两个 flag 流水：AIV 写好 R 后 set(READY)；AIC 等到后做矩阵乘，完成后
 * set(DONE)。AIV 在 set 之后、wait 之前先把下一轮的 R 组装进另一个缓冲，
 * 与 AIC 的矩阵乘重叠执行。
 *
 * 算法
 * ----
 * 把矩阵的每一行看作一个列向量，做单边雅可比：A <- G A，使各行两两正交，
 * 行范数即奇异值。矩阵按 128 行分块，一轮 sweep 为：
 *   1) 块内旋转：每个块内所有行对各旋转一次（两两恰好一次）；
 *   2) 块间旋转：所有块对 (P,Q) 各旋转一次（每个跨块行对恰好一次），
 *      块对按轮转顺序 (P+Q) mod nb 处理，同一轮内块对互不重叠。
 * 每个单元（一块 / 一个块对）内：先算 Gram 与行范数，再按 round-robin 匹配把
 * b/2 个 Givens 旋转组装成 n x n 正交矩阵 R（n = b 或 2b），由 AIC 一次矩阵乘
 * 完成本单元该轮的全部旋转。
 */

#define FLAG_R_READY 0x2   // AIV -> AIC: R 已就绪
#define FLAG_R_DONE  0x3   // AIC -> AIV: 本轮旋转已应用

namespace jacobi_svd {

constexpr int32_t kBlockRows = 128;      // 块行数 b
constexpr int32_t kMaxVecLen = 512;      // 列向量长度上限
constexpr int32_t kPTile = 16;           // Gram 计算 P 方向分块
constexpr int32_t kQTile = 8;            // 搬运 / Gram 的 Q 方向分块
constexpr int32_t kRRowChunk = 8;        // 组装 R 时每次写入的行数
constexpr int32_t kCopyChunk = 2048;     // 单次 DataCopy 元素数

__aicore__ inline int32_t MinI(int32_t a, int32_t b) { return a < b ? a : b; }

/** 由 a=|u|^2, b=|v|^2, c=<u,v> 求 Givens 旋转的 cos/sin（使二者正交） */
__aicore__ inline void Givens(float a, float b, float c, float &cs, float &sn)
{
    const float tol = 1e-6f;
    const float scale = sqrt(a * b);
    if (c <= tol * scale && c >= -tol * scale) {
        cs = 1.0f;
        sn = 0.0f;
        return;
    }
    const float tau = (b - a) / (2.0f * c);
    const float t = (tau >= 0.0f) ? (1.0f / (tau + sqrt(1.0f + tau * tau)))
                                  : (1.0f / (tau - sqrt(1.0f + tau * tau)));
    cs = 1.0f / sqrt(1.0f + t * t);
    sn = cs * t;
}

// ================================================================= AIV 侧
class VecWorker {
public:
    __aicore__ inline VecWorker() {}

    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(pTileBuf_, kPTile * kMaxVecLen * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(qTileBuf_, kQTile * kMaxVecLen * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(tmpBuf_, kMaxVecLen * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(diagBuf_, 2 * kBlockRows * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(gramBuf_, kBlockRows * kBlockRows * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(redBuf_, 64 * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(rRowBuf_, kRRowChunk * 2 * kBlockRows * static_cast<int32_t>(sizeof(float)));

        pTile_ = pTileBuf_.Get<float>();
        qTile_ = qTileBuf_.Get<float>();
        tmpRow_ = tmpBuf_.Get<float>();
        diag_ = diagBuf_.Get<float>();
        gram_ = gramBuf_.Get<float>();
        red_ = redBuf_.Get<float>();
        rRows_ = rRowBuf_.Get<float>();

        evVtoS_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        evS2M3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        evM2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        evV2M3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        evM3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        evM3M2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void LoadRows(const LocalTensor<float> &dst, const GlobalTensor<float> &src,
                                    int64_t srcOff, int32_t count)
    {
        int32_t done = 0;
        while (done < count) {
            const int32_t n = MinI(kCopyChunk, count - done);
            SetFlag<HardEvent::MTE3_MTE2>(evM3M2_);
            WaitFlag<HardEvent::MTE3_MTE2>(evM3M2_);
            DataCopy(dst[done], src[srcOff + done], n);
            SetFlag<HardEvent::MTE2_V>(evM2V_);
            WaitFlag<HardEvent::MTE2_V>(evM2V_);
            done += n;
        }
    }

    __aicore__ inline void StoreRows(const GlobalTensor<float> &dst, int64_t dstOff,
                                     const LocalTensor<float> &src, int32_t count)
    {
        int32_t done = 0;
        while (done < count) {
            const int32_t n = MinI(kCopyChunk, count - done);
            SetFlag<HardEvent::V_MTE3>(evV2M3_);
            WaitFlag<HardEvent::V_MTE3>(evV2M3_);
            DataCopy(dst[dstOff + done], src[done], n);
            SetFlag<HardEvent::MTE3_V>(evM3V_);
            WaitFlag<HardEvent::MTE3_V>(evM3V_);
            done += n;
        }
    }

    __aicore__ inline void CopyGmGm(const GlobalTensor<float> &dstG, int64_t dstOff,
                                    const GlobalTensor<float> &srcG, int64_t srcOff, int32_t count)
    {
        int32_t done = 0;
        while (done < count) {
            const int32_t n = MinI(kCopyChunk, count - done);
            LoadRows(pTile_, srcG, srcOff + done, n);
            StoreRows(dstG, dstOff + done, pTile_, n);
            done += n;
        }
    }

    /** 行范数 -> sigma（本 AIV 负责 [lo, hi) 行） */
    __aicore__ inline void ComputeRowNorms(GlobalTensor<float> aG, GlobalTensor<float> sigmaG,
                                           int32_t cols, int32_t lo, int32_t hi)
    {
        for (int32_t r = lo; r < hi; ++r) {
            LoadRows(tmpRow_, aG, static_cast<int64_t>(r) * cols, cols);
            Mul(tmpRow_, tmpRow_, tmpRow_, cols);
            ReduceSum(red_[0], tmpRow_, tmpRow_, cols);
            SetFlag<HardEvent::V_S>(evVtoS_);
            WaitFlag<HardEvent::V_S>(evVtoS_);
            sigmaG.SetValue(r, sqrt(red_.GetValue(0)));
        }
    }

    /** 一个单元的全部 AIV 工作 */
    __aicore__ inline void RunUnit(GlobalTensor<float> aG, GlobalTensor<float> wG,
                                   GlobalTensor<float> rG, int32_t blockRow, int32_t vecLen,
                                   int32_t pairRow, int32_t lane)
    {
        const bool isPair = (pairRow >= 0);
        const int32_t space = kBlockRows;
        const int32_t n = isPair ? 2 * space : space;

        CopyGmGm(wG, 0, aG, static_cast<int64_t>(blockRow) * vecLen, space * vecLen);
        if (isPair) {
            CopyGmGm(wG, static_cast<int64_t>(space) * vecLen, aG,
                     static_cast<int64_t>(pairRow) * vecLen, space * vecLen);
        }

        ComputeDiag(wG, n, vecLen);

        const int32_t ownLo = isPair ? 0 : lane * space / 2;
        const int32_t ownHi = isPair ? space : (lane + 1) * space / 2;
        ComputeGram(wG, vecLen, space, ownLo, ownHi, isPair ? space : 0);

        // 逐轮组装 R 并交由 AIC 旋转（双缓冲，组装与矩阵乘重叠）
        AssembleR(rG, 0, 0, space, n, isPair, lane);
        for (int32_t r = 0; r < space; ++r) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_R_READY);
            if (r + 1 < space) {
                AssembleR(rG, (r + 1) & 1, r + 1, space, n, isPair, lane);
            }
            CrossCoreWaitFlag(FLAG_R_DONE);
        }

        // 写回（轮数为偶数，结果在 work[0]）
        CopyGmGm(aG, static_cast<int64_t>(blockRow) * vecLen, wG, 0, space * vecLen);
        if (isPair) {
            CopyGmGm(aG, static_cast<int64_t>(pairRow) * vecLen, wG,
                     static_cast<int64_t>(space) * vecLen, space * vecLen);
        }
    }

private:
    __aicore__ inline void ComputeDiag(const GlobalTensor<float> &wG, int32_t n, int32_t vecLen)
    {
        for (int32_t c = 0; c < n; c += kQTile) {
            const int32_t rows = MinI(kQTile, n - c);
            LoadRows(qTile_, wG, static_cast<int64_t>(c) * vecLen, rows * vecLen);
            for (int32_t t = 0; t < rows; ++t) {
                Mul(tmpRow_, qTile_[t * vecLen], qTile_[t * vecLen], vecLen);
                ReduceSum(red_[0], tmpRow_, tmpRow_, vecLen);
                SetFlag<HardEvent::V_S>(evVtoS_);
                WaitFlag<HardEvent::V_S>(evVtoS_);
                diag_.SetValue(c + t, red_.GetValue(0));
            }
        }
    }

    __aicore__ inline void ComputeGram(const GlobalTensor<float> &wG, int32_t vecLen, int32_t space,
                                       int32_t ownLo, int32_t ownHi, int32_t qBase)
    {
        for (int32_t p0 = ownLo; p0 < ownHi; p0 += kPTile) {
            const int32_t pRows = MinI(kPTile, ownHi - p0);
            LoadRows(pTile_, wG, static_cast<int64_t>(p0) * vecLen, pRows * vecLen);
            for (int32_t q0 = 0; q0 < space; q0 += kQTile) {
                const int32_t qRows = MinI(kQTile, space - q0);
                LoadRows(qTile_, wG, static_cast<int64_t>(qBase + q0) * vecLen, qRows * vecLen);
                for (int32_t i = 0; i < pRows; ++i) {
                    for (int32_t j = 0; j < qRows; ++j) {
                        Mul(tmpRow_, pTile_[i * vecLen], qTile_[j * vecLen], vecLen);
                        ReduceSum(red_[(i * qRows + j) % 8], tmpRow_, tmpRow_, vecLen);
                        if (((i * qRows + j) % 8 == 7) || (i + 1 == pRows && j + 1 == qRows)) {
                            SetFlag<HardEvent::V_S>(evVtoS_);
                            WaitFlag<HardEvent::V_S>(evVtoS_);
                            const int32_t cnt = (i * qRows + j) % 8 + 1;
                            for (int32_t k = 0; k < cnt; ++k) {
                                const int32_t idx = i * qRows + j - (cnt - 1) + k;
                                gram_.SetValue((p0 + idx / qRows) * space + q0 + idx % qRows,
                                               red_.GetValue(k));
                            }
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void AssembleR(GlobalTensor<float> rG, int32_t buf, int32_t round,
                                     int32_t space, int32_t n, bool isPair, int32_t lane)
    {
        const int32_t lo = isPair ? lane * space : lane * space / 2;
        const int32_t hi = isPair ? (lane + 1) * space : (lane + 1) * space / 2;
        for (int32_t r0 = lo; r0 < hi; r0 += kRRowChunk) {
            const int32_t cnt = MinI(kRRowChunk, hi - r0);
            Duplicate(rRows_, 0.0f, cnt * n);
            SetFlag<HardEvent::V_S>(evVtoS_);
            WaitFlag<HardEvent::V_S>(evVtoS_);
            for (int32_t t = 0; t < cnt; ++t) {
                const int32_t row = r0 + t;
                if (!isPair) {
                    const int32_t j = ((round - row) % space + space) % space;
                    const int32_t mn = (row < j) ? row : j;
                    const int32_t mx = (row < j) ? j : row;
                    float cs = 1.0f;
                    float sn = 0.0f;
                    Givens(diag_.GetValue(mn), diag_.GetValue(mx), gram_.GetValue(row * space + j),
                           cs, sn);
                    if (row == j) {
                        rRows_.SetValue(t * n + row, 1.0f);
                    } else if (row < j) {
                        rRows_.SetValue(t * n + row, cs);
                        rRows_.SetValue(t * n + j, -sn);
                    } else {
                        rRows_.SetValue(t * n + j, sn);
                        rRows_.SetValue(t * n + row, cs);
                    }
                } else if (row < space) {
                    const int32_t j = ((round - row) % space + space) % space;
                    float cs = 1.0f;
                    float sn = 0.0f;
                    Givens(diag_.GetValue(row), diag_.GetValue(space + j),
                           gram_.GetValue(row * space + j), cs, sn);
                    rRows_.SetValue(t * n + row, cs);
                    rRows_.SetValue(t * n + space + j, -sn);
                } else {
                    const int32_t q = row - space;
                    const int32_t i = ((round - q) % space + space) % space;
                    float cs = 1.0f;
                    float sn = 0.0f;
                    Givens(diag_.GetValue(i), diag_.GetValue(space + q),
                           gram_.GetValue(i * space + q), cs, sn);
                    rRows_.SetValue(t * n + i, sn);
                    rRows_.SetValue(t * n + row, cs);
                }
            }
            SetFlag<HardEvent::S_MTE3>(evS2M3_);
            WaitFlag<HardEvent::S_MTE3>(evS2M3_);
            StoreRows(rG, static_cast<int64_t>(buf) * n * n + static_cast<int64_t>(r0) * n,
                      rRows_, cnt * n);
        }
    }

    TBuf<TPosition::VECCALC> pTileBuf_;
    TBuf<TPosition::VECCALC> qTileBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> diagBuf_;
    TBuf<TPosition::VECCALC> gramBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    TBuf<TPosition::VECCALC> rRowBuf_;

    LocalTensor<float> pTile_;
    LocalTensor<float> qTile_;
    LocalTensor<float> tmpRow_;
    LocalTensor<float> diag_;
    LocalTensor<float> gram_;
    LocalTensor<float> red_;
    LocalTensor<float> rRows_;

    int32_t evVtoS_{0}, evS2M3_{0}, evM2V_{0}, evV2M3_{0}, evM3V_{0}, evM3M2_{0};
};

// ================================================================= AIC 侧
__aicore__ inline void RunUnitCube(MatmulCustom_01<float> &mm, __gm__ float *wPtr, __gm__ float *rPtr,
                                   int32_t vecLen, int32_t pairRow)
{
    const bool isPair = (pairRow >= 0);
    const int32_t space = kBlockRows;
    const int32_t n = isPair ? 2 * space : space;
    for (int32_t r = 0; r < space; ++r) {
        CrossCoreWaitFlag(FLAG_R_READY);
        const int32_t srcBuf = r & 1;
        const int32_t dstBuf = 1 - srcBuf;
        GlobalTensor<float> aT;
        GlobalTensor<float> bT;
        GlobalTensor<float> cT;
        aT.SetGlobalBuffer(rPtr + static_cast<int64_t>(srcBuf) * n * n);
        bT.SetGlobalBuffer(wPtr + static_cast<int64_t>(srcBuf) * n * vecLen);
        cT.SetGlobalBuffer(wPtr + static_cast<int64_t>(dstBuf) * n * vecLen);
        mm.SetMatrix(aT, bT, cT, vecLen, n);
        mm.Process(0, 0, 0, n, vecLen, n);
        CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_R_DONE);
    }
}

}  // namespace jacobi_svd

/**
 * 批量单边（块状）雅可比 SVD 内核入口（AIC/AIV 分工，按核分配矩阵）
 *
 * @param rows       每个矩阵的行数（须为 128 的倍数）
 * @param cols       行向量长度（8 的倍数，<= 512）
 * @param batchSize  批大小
 * @param maxSweeps  最大扫描次数
 * @param clusterNum 核（簇）数
 * @param A          输入/输出矩阵（batchSize * rows * cols），收敛后原地存放旋转结果
 * @param sigma      输出（batchSize * rows 行范数，未排序）
 * @param workGm     每核 2 * (2*128) * cols 的乒乓工作区
 * @param rGm        每核 2 * (2*128)^2 的旋转矩阵缓冲
 */
extern "C" __global__ __aicore__ void jacobi_svd_batched(
    int32_t rows, int32_t cols, int32_t batchSize, int32_t maxSweeps, int32_t clusterNum,
    GM_ADDR A, GM_ADDR sigma, GM_ADDR workGm, GM_ADDR rGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr int32_t b = jacobi_svd::kBlockRows;
    if (rows <= 0 || cols <= 0 || batchSize <= 0 || (rows % b) != 0) return;

    const int32_t nb = rows / b;
    const int64_t matElems = static_cast<int64_t>(rows) * cols;
    const int64_t wPerCluster = static_cast<int64_t>(2) * (2 * b) * cols;
    const int64_t rPerCluster = static_cast<int64_t>(2) * (2 * b) * (2 * b);

#ifdef __DAV_C220_VEC__
    const int32_t cid = GetBlockIdx() / 2;
    const int32_t lane = GetBlockIdx() % 2;
    if (cid >= clusterNum) return;

    TPipe pipe;
    jacobi_svd::VecWorker op;
    op.Init(&pipe);

    GlobalTensor<float> aG, sigmaG, wG, rG;
    for (int32_t m = cid; m < batchSize; m += clusterNum) {
        aG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A) + m * matElems);
        sigmaG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sigma) + static_cast<int64_t>(m) * rows);
        wG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workGm) + static_cast<int64_t>(cid) * wPerCluster);
        rG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rGm) + static_cast<int64_t>(cid) * rPerCluster);

        for (int32_t sweep = 0; sweep < maxSweeps; ++sweep) {
            for (int32_t p = 0; p < nb; ++p) {
                op.RunUnit(aG, wG, rG, p * b, cols, -1, lane);
            }
            for (int32_t rnd = 0; rnd < nb; ++rnd) {
                for (int32_t p = 0; p < nb; ++p) {
                    const int32_t q = (rnd - p + nb) % nb;
                    if (p < q) {
                        op.RunUnit(aG, wG, rG, p * b, cols, q * b, lane);
                    }
                }
            }
        }

        op.ComputeRowNorms(aG, sigmaG, cols, lane * rows / 2, (lane + 1) * rows / 2);
    }
#endif

#ifdef __DAV_C220_CUBE__
    const int32_t cid = GetBlockIdx();
    if (cid >= clusterNum) return;

    TPipe pipe;
    MatmulCustom_01<float> mm;
    mm.Init(&pipe);

    __gm__ float *wPtr = reinterpret_cast<__gm__ float *>(workGm) + static_cast<int64_t>(cid) * wPerCluster;
    __gm__ float *rPtr = reinterpret_cast<__gm__ float *>(rGm) + static_cast<int64_t>(cid) * rPerCluster;

    for (int32_t m = cid; m < batchSize; m += clusterNum) {
        for (int32_t sweep = 0; sweep < maxSweeps; ++sweep) {
            for (int32_t p = 0; p < nb; ++p) {
                jacobi_svd::RunUnitCube(mm, wPtr, rPtr, cols, -1);
            }
            for (int32_t rnd = 0; rnd < nb; ++rnd) {
                for (int32_t p = 0; p < nb; ++p) {
                    const int32_t q = (rnd - p + nb) % nb;
                    if (p < q) {
                        jacobi_svd::RunUnitCube(mm, wPtr, rPtr, cols, q * b);
                    }
                }
            }
        }
    }
#endif
}
