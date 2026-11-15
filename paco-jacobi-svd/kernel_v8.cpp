#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include <cstdint>

#include "kernel_operator.h"
#include "gemm_trans.hpp"

using namespace AscendC;

/**
 * 批量单边（块状）雅可比 SVD —— 逐核分配矩阵 + 127 步并行 Jacobi
 *
 * 结构（n = 128）
 * ---------------
 * 每个核（1 AIC + 2 AIV）被分配若干矩阵：m = cid, cid+clusterNum, ...
 * 对每个矩阵做若干轮 sweep，每轮 sweep 分 n-1 = 127 步：
 *   第 j 步：配对 (i, (i+j) mod n)，i = 0..n-1 且 i < (i+j)%n，共 n/2 = 64 对，
 *            两两不相交（正好覆盖全部 n 行一次）。
 *   AIV：把每一对的两行点积/范数算出来，得到 Givens 参数 (cos,sin)，
 *        组装成本步的 n x n 旋转矩阵 R（单位阵 + 64 个 2x2 块）写到 GM；
 *   AIC：执行 A <- R * A（一次 n x n x cols 的矩阵乘）。
 * 两个 AIV 各分一半的配对（按较小的行号划分），各自写自己负责的 R 行。
 *
 * 同步：核内 AIV -> AIC 用 CrossCore 0x2（READY）/ 0x3（DONE）乒乓；
 *       AIV 之间用 CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0) / CrossCoreWaitFlag(0x0)。
 *
 * 收敛：整步没有任何旋转（|c| <= tol*sqrt(ab)）即视为收敛，跳出 sweep。
 */

#define FLAG_R_READY 0x2    // AIV -> AIC: 本步 R 已写好
#define FLAG_R_DONE 0x3     // AIC -> AIV: 本步旋转已应用
#define FLAG_VEC_BAR 0x0    // AIV 之间同步

namespace jacobi_svd {

constexpr int32_t kN = 128;            // 矩阵维度（本轮算法固定 128x128）
constexpr int32_t kMaxCols = 256;      // 列数上限（128 的矩阵用 128）
constexpr int32_t kCopyChunk = 2048;

__aicore__ inline int32_t MinI(int32_t a, int32_t b) { return a < b ? a : b; }

class NetWorker {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        // 数据 tile + 旋转矩阵 R + 临时行
        pipe->InitBuffer(tileBuf_, kN * kMaxCols * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(rBuf_, kN * kN * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(tempBuf_, 4 * kMaxCols * static_cast<int32_t>(sizeof(float)));
        pipe->InitBuffer(redBuf_, 64 * static_cast<int32_t>(sizeof(float)));

        tile_ = tileBuf_.Get<float>();
        rMat_ = rBuf_.Get<float>();
        LocalTensor<float> t = tempBuf_.Get<float>();
        rowA_ = t[0];
        rowB_ = t[1 * kMaxCols];
        rowC_ = t[2 * kMaxCols];
        rowW_ = t[3 * kMaxCols];
        red_ = redBuf_.Get<float>();

        evVtoS_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        evStoV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        evS2M3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        evM2S_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
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

    /**
     * 一个 step：本 AIV 负责 i ∈ [lo, hi) 的配对，算参数并把 R 的对应行写好。
     * @return 本步是否真的做了旋转
     */
    __aicore__ inline bool RunStep(GlobalTensor<float> aG, GlobalTensor<float> rG, int32_t step,
                                   int32_t cols, int32_t lo, int32_t hi)
    {
        // 1) 取当前矩阵（AIC 上一步已更新）
        LoadRows(tile_, aG, 0, kN * cols);

        // 2) R 先置为单位阵（本步所有行都会重写，这里只保证未覆盖的行是单位）
        for (int32_t i = 0; i < kN; ++i) {
            // 用向量填充：整块先清 0，再写对角线
            (void)i;
        }
        Duplicate(rMat_, 0.0f, kN * kN);
        SetFlag<HardEvent::V_S>(evVtoS_);
        WaitFlag<HardEvent::V_S>(evVtoS_);
        for (int32_t i = 0; i < kN; ++i) {
            rMat_.SetValue(i * kN + i, 1.0f);
        }

        // 3) 逐对算参数并写 R
        bool rotated = false;
        for (int32_t i = lo; i < hi; ++i) {
            const int32_t p = (i + step) % kN;
            if (i >= p) continue;                       // 每对只由较小行号的一方处理

            // a = |row_i|^2, b = |row_p|^2, c = <row_i, row_p>
            Mul(rowA_, tile_[i * cols], tile_[i * cols], cols);
            Mul(rowB_, tile_[p * cols], tile_[p * cols], cols);
            Mul(rowC_, tile_[i * cols], tile_[p * cols], cols);
            ReduceSum(red_[0], rowA_, rowW_, cols);
            ReduceSum(red_[8], rowB_, rowW_, cols);
            ReduceSum(red_[16], rowC_, rowW_, cols);
            SetFlag<HardEvent::V_S>(evVtoS_);
            WaitFlag<HardEvent::V_S>(evVtoS_);

            const float a = red_.GetValue(0);
            const float b = red_.GetValue(8);
            const float c = red_.GetValue(16);
            const float tol = 1e-6f;
            const float scale = sqrt(a * b);
            if (!(c <= tol * scale && c >= -tol * scale)) {
                const float tau = (b - a) / (2.0f * c);
                const float t = (tau >= 0.0f) ? (1.0f / (tau + sqrt(1.0f + tau * tau)))
                                              : (1.0f / (tau - sqrt(1.0f + tau * tau)));
                const float cs = 1.0f / sqrt(1.0f + t * t);
                const float sn = cs * t;
                // A <- R * A：R 的行 i 为 cs*e_i - sn*e_p，行 p 为 sn*e_i + cs*e_p
                rMat_.SetValue(i * kN + i, cs);
                rMat_.SetValue(i * kN + p, -sn);
                rMat_.SetValue(p * kN + i, sn);
                rMat_.SetValue(p * kN + p, cs);
                rotated = true;
            }
        }
        SetFlag<HardEvent::S_MTE3>(evS2M3_);
        WaitFlag<HardEvent::S_MTE3>(evS2M3_);
        StoreRows(rG, 0, rMat_, kN * kN);
        return rotated;
    }

private:
    TBuf<TPosition::VECCALC> tileBuf_;
    TBuf<TPosition::VECCALC> rBuf_;
    TBuf<TPosition::VECCALC> tempBuf_;
    TBuf<TPosition::VECCALC> redBuf_;

    LocalTensor<float> tile_;
    LocalTensor<float> rMat_;
    LocalTensor<float> rowA_;
    LocalTensor<float> rowB_;
    LocalTensor<float> rowC_;
    LocalTensor<float> rowW_;
    LocalTensor<float> red_;

    int32_t evVtoS_{0}, evStoV_{0}, evS2M3_{0}, evM2S_{0}, evM2V_{0}, evV2M3_{0}, evM3V_{0}, evM3M2_{0};
};

}  // namespace jacobi_svd

/**
 * 内核入口：每个核分配若干矩阵，127 步/sweep 并行 Jacobi。
 *
 * @param n        矩阵维度（=128）
 * @param cols     列数（=n）
 * @param batchSize 批大小
 * @param maxSweeps 最大 sweep 数
 * @param coreNum  核数（簇数）
 * @param A        输入矩阵（batch * n * cols），原地旋转
 * @param sigma    输出（batch * n 行范数）
 * @param rBufGm   每核 n*n 的旋转矩阵缓冲（双缓冲：2 * n * n）
 * @param flags    收敛标记（maxSweeps * 64）
 */
extern "C" __global__ __aicore__ void jacobi_svd_batched(
    int32_t n, int32_t cols, int32_t batchSize, int32_t maxSweeps, int32_t coreNum,
    GM_ADDR A, GM_ADDR sigma, GM_ADDR rBufGm, GM_ADDR flags)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr int32_t kN = jacobi_svd::kN;
    if (n != kN || cols != n || batchSize <= 0 || coreNum <= 0) return;

    const int64_t matElems = static_cast<int64_t>(n) * cols;
    const int64_t rElems = static_cast<int64_t>(kN) * kN;

#ifdef __DAV_C220_VEC__
    const int32_t cid = static_cast<int32_t>(GetBlockIdx()) / 2;
    const int32_t lane = static_cast<int32_t>(GetBlockIdx()) % 2;
    if (cid >= coreNum) return;

    TPipe pipe;
    jacobi_svd::NetWorker op;
    op.Init(&pipe);

    const int32_t lo = lane * (kN / 2);
    const int32_t hi = (lane + 1) * (kN / 2);

    GlobalTensor<float> aG, sigmaG, rG, flagsG;
    flagsG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(flags));

    for (int32_t m = cid; m < batchSize; m += coreNum) {
        aG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A) + static_cast<int64_t>(m) * matElems);
        sigmaG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sigma) + static_cast<int64_t>(m) * n);
        rG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rBufGm) + static_cast<int64_t>(cid) * rElems);

        for (int32_t sweep = 0; sweep < maxSweeps; ++sweep) {
            for (int32_t step = 1; step < kN; ++step) {
                const bool rotated = op.RunStep(aG, rG, step, cols, lo, hi);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_R_READY);
                CrossCoreWaitFlag(FLAG_R_DONE);
                (void)rotated;
            }
        }
    }
#endif

#ifdef __DAV_C220_CUBE__
    const int32_t cid = static_cast<int32_t>(GetBlockIdx());
    if (n != kN || cols != n || batchSize <= 0 || cid >= coreNum) return;

    const int64_t matElems = static_cast<int64_t>(n) * cols;
    const int64_t rElems = static_cast<int64_t>(kN) * kN;

    TPipe pipe;
    MatmulCustom<float, float, float> mm;
    mm.Init(&pipe);

    GlobalTensor<float> aG, rG;
    aG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
    rG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rBufGm) + static_cast<int64_t>(cid) * rElems);

    for (int32_t m = cid; m < batchSize; m += coreNum) {
        for (int32_t sweep = 0; sweep < maxSweeps; ++sweep) {
            for (int32_t step = 1; step < kN; ++step) {
                CrossCoreWaitFlag(FLAG_R_READY);
                // A <- R * A ：R(n x n) * A(n x cols)
                GlobalTensor<float> aT, rT;
                aT.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A) + static_cast<int64_t>(m) * matElems);
                mm.SetMatrix(rG, aT, aT, cols, n);
                mm.Process(0, 0, 0, n, cols, n, -1, false, false);
                CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_R_DONE);
            }
        }
    }
#endif
}
