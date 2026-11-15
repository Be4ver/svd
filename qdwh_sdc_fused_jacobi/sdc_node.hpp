#pragma once
#define QDWH_BUILD
#include "qr_custom.cpp"
#include "fused_batch_cfg.hpp"

// ============================================================================
// Phase 2：同层 batch 的 SDC split
//
//   原来的流程是「一个节点 = 一次 sdc_split_node」：λ / A0 / sign-QDWH 4 轮 /
//   k / P,P_ / UΩ / QR / H± 。其中 **sign-QDWH 的 4 轮迭代环**（每轮
//   Prepare + ComputeAll(QR) + qr_update + QdwhPmat）每轮都要付
//   一次「逐行 DataCopyPad + 跨核 flag + panel 循环」的固定成本。
//
//   现在把**同一个 BFS 层内的 2~4 个节点**打包成一个「组」：
//     1) 每个内部节点各自算 λ，并把 A0 = pad(H-λI) 写到 a0 的第 b 个切片；
//     2) 这 B 个矩阵跑**一次 batch QDWH**（m[b]=2*nPad[b]、n[b]=nPad[b]，
//        qdwh_vec / qdwh_cube / QR 本来就是 batch 实现），固定成本除以 B；
//     3) 之后仍是逐节点的 tail（k / P,P_ / UΩ / QR / H±），这部分不变。
//
//   缓冲按「批内切片」分配（见 fused_layout.hpp 与 sdc_build_plan 里的偏移），
//   AIV 与 AIC 用同一个 plan 算出同样的偏移。
// ============================================================================

// 组内计划：哪些节点要 split、批内下标、nPad、以及在 a0/xsdc 里的切片偏移
struct SdcBatchPlan {
    int cntI;                          // 内部节点个数（>= threshold 的节点）
    int idx[kSdcBatchMax];             // 节点 b 在批内的下标（-1 = 叶块）
    int nPad[kSdcBatchMax];            // 批内第 i 个矩阵的 nPad
    int aOff[kSdcBatchMax];            // 批内第 i 个矩阵在 a0/xsdc/pt/xt 中的偏移
};

__aicore__ inline void sdc_build_plan(const int *gN, int cnt, int threshold, SdcBatchPlan &plan)
{
    plan.cntI = 0;
    int accA = 0;
    for (int b = 0; b < cnt; ++b) {
        if (gN[b] <= threshold) {
            plan.idx[b] = -1;
            continue;
        }
        const int nPad = ((gN[b] + 127) / 128) * 128;
        const int i = plan.cntI;
        plan.idx[b] = i;
        plan.nPad[i] = nPad;
        plan.aOff[i] = accA;
        accA += nPad * nPad;
        plan.cntI++;
    }
}

// λ = trace(H)/n（每个 AIV 独立算，值相同）；shift 恒为 0
__aicore__ inline float sdc_node_lambda(GM_ADDR h, int ldH, int n, float shift)
{
    GlobalTensor<float> H;
    H.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(h));
    float lambda = 0.0f;
    for (int i = 0; i < n; ++i) {
        lambda += H.GetValue(i * ldH + i);
    }
    lambda /= (float)n;
    return lambda + shift;
}

// ---------------------------------------------------------------------------
// 组内 batch QDWH（AIV 侧）
//   InitX -> I x [wait Xt -> Prepare -> QR -> wait Q -> scale Q1]
//   -> final Xt scale -> wait reconstructed X
// x/a0/qbuf/hwork/tau/T/W/Y 都是「按批切片」的缓冲，偏移由 SetDimsBatch 内部
// 用 m[]/n[] 递推（与 qdwh_cube 完全一致）。
// ---------------------------------------------------------------------------
__aicore__ inline void sdc_qdwh_batch_aiv(
    qdwh_vec &op,
    GM_ADDR hwork, GM_ADDR tau, GM_ADDR T, GM_ADDR W, GM_ADDR Y, GM_ADDR pQdwh,
    GM_ADDR x, GM_ADDR a0, GM_ADDR qbuf, GM_ADDR hout, GM_ADDR coef, GM_ADDR pt, GM_ADDR xt,
    bool skipInitX,
    const SdcBatchPlan &plan, int maxIter, float alpha, float l0)
{
#ifdef __DAV_C220_VEC__
    int mArr[kSdcBatchMax], nArr[kSdcBatchMax];
    for (int i = 0; i < plan.cntI; ++i) {
        mArr[i] = 2 * plan.nPad[i];
        nArr[i] = plan.nPad[i];
    }
    const int blockNum = GetBlockNum() << 1;
    const int blockIdx = GetBlockIdx();

    op.SetMatrix(hwork, tau, T, W, Y, pQdwh, plan.cntI);
    op.SetDimsBatch(mArr, nArr, plan.cntI);
    op.SetQdwhBuffers(x, a0, qbuf, hout, coef, alpha, pt, xt);
    op.SetL0(l0);

    // 【优化】alpha == 1 时 QdwhInitX（X = A0/alpha）是纯拷贝。split 侧 A0 只作为
    // QDWH 输入（phase 1 才需要留给 QdwhFinalGemm），所以调用方让 BuildA0FromH 直接
    // 写进 X 缓冲（xsdc）并传 skipInitX=true —— 省掉每节点 nPad² 读 + nPad² 写。
    if (!skipInitX) {
        op.QdwhInitX();
    }
    op.AivBarrier();
    // Publish initial X. Flag 0x5 has I+2 matched sets/waits per group:
    // initial X, I scaled Q1 matrices, final Xt scale.
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);

    for (int it = 0; it < maxIter; ++it) {
        CrossCoreWaitFlag(0x4);
        op.QdwhPrepare();
        op.AivBarrier();
        op.ComputeAll(blockNum, blockIdx);
        CrossCoreWaitFlag(0x6);
        op.AivBarrier();
        op.QdwhScaleQ1(blockIdx, blockNum);
        op.AivBarrier();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
    }
    CrossCoreWaitFlag(0x4);
    op.QdwhFinalizeScale();
    op.AivBarrier();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
    CrossCoreWaitFlag(0x6); // X was reconstructed before releasing the tails

#endif
}

// ---------------------------------------------------------------------------
// 组内 batch QDWH（CUBE 侧）
//   Each iteration: QR -> publish Q -> wait scaled Q1 -> atomic Xt -> 0x4.
//   The final scale handshake precedes reconstruction of X for the tails.
// ---------------------------------------------------------------------------
__aicore__ inline void sdc_qdwh_batch_cube(
    qdwh_cube &op,
    GM_ADDR hwork, GM_ADDR tau, GM_ADDR T, GM_ADDR W, GM_ADDR Y, GM_ADDR qbuf,
    GM_ADDR pQdwh, GM_ADDR eye, GM_ADDR x, GM_ADDR a0, GM_ADDR coef,
    GM_ADDR pt, GM_ADDR xt, GM_ADDR eyeN, int ldEye,
    const SdcBatchPlan &plan, int maxIter)
{
#ifdef __DAV_C220_CUBE__
    int mArr[kSdcBatchMax], nArr[kSdcBatchMax];
    for (int i = 0; i < plan.cntI; ++i) {
        mArr[i] = 2 * plan.nPad[i];
        nArr[i] = plan.nPad[i];
    }

    op.Init(hwork, tau, T, W, Y, qbuf, pQdwh, eye, plan.cntI);
    op.SetDimsBatch(mArr, nArr, plan.cntI);   // 覆盖 ReadShapes 读到的 P
    // 【Xt】把 pt/xt 也交给 cube 侧：QdwhPmat 只有在 hasPt 时才顺带算 Pᵀ
    // （漏了这两个参数 → Pt 里是上一次的残留 → AIV 维护的 Xᵀ 全错）。
    op.SetQdwhBuffers(x, a0, coef, pt, xt);

    // Xᵀ = I·Xᵀ：AIV 写完初值 X 后的一次 GEMM（替代 AIV 的 nPad² 次 scalar 转置）
    CrossCoreWaitFlag(0x5);
    op.QdwhInitXt(xt, eye);
    op.AicBarrier();                          // 所有 AIC 的 Xt 都写完
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);     // 放行第一轮 AIV prepare

    for (int it = 0; it < maxIter; ++it) {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x2);
        op.qr_update();
        op.AicBarrier();
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);
        CrossCoreWaitFlag(0x5);
        op.AicBarrier();
        op.QdwhPmat(true, true);
        op.AicBarrier();
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
    }
    CrossCoreWaitFlag(0x5); // final Xt scale done
    op.QdwhRestoreX(eye);
    op.AicBarrier();
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);

#endif
}

// ---------------------------------------------------------------------------
// 单节点 tail（AIV）：P = U+I / P_ = U-I -> 0xD -> split 的 QR（AIV 面板）
//   uOff = 本节点 U 在 x（xsdc）缓冲里的 float 偏移（批内切片）
// ---------------------------------------------------------------------------
__aicore__ inline void sdc_tail_aiv(
    qdwh_vec &op, long uOff, int n, int nPad,
    GM_ADDR tmp, GM_ADDR qt, GM_ADDR hwork, GM_ADDR tau, GM_ADDR T, GM_ADDR W, GM_ADDR Y,
    GM_ADDR pQr, GM_ADDR trBuf, int blockIdx, int blockNum)
{
#ifdef __DAV_C220_VEC__
    // ---- Hwork 的 padding 先清 0（向量版；AIC 此刻在做 UΩ，可重叠）----
    // 必须在 0xD 之前完成：AIC 的转置 GEMM 在 0xD 之后才写 [0:n, 0:n]。
    op.clear_pad(hwork, n, nPad, blockIdx, blockNum);
    // ---- P = U+I / P_ = U-I 由 AIV 用 DataCopyPad 构造 ----
    op.BuildPmats(tmp, qt, n, nPad, nPad, blockIdx, blockNum, uOff, trBuf);
    // Q 的初值 = 单位阵（AIV 写，避免占 AIC 关键路径上的一次 nPad^3 GEMM）
    op.BuildIdentityQ(n, nPad, blockIdx, blockNum);
    // Mode 2 below only joins one AIC and its two AIVs. All partials and
    // all P/Q rows must be visible before any AIC consumes them.
    op.AivBarrier();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0xD);   // 0xD = AIV->AIC：P/P_（和 Q=I）已落 GM

    // ---- split QR：一次 QR(A)，和 cube 侧严格配对 ----
    // 使用 0x9/0xA，避开 QDWH 最后一轮遗留的 0x1/0x2。
    op.SetMatrix(hwork, tau, T, W, Y, pQr, 1);
    op.SetDims(nPad, nPad);
    op.ComputeAll(blockNum, blockIdx, 0x9);
#endif
}

// ---------------------------------------------------------------------------
// 单节点 tail（CUBE）：
//   k=(trace(U)+nPad)/2 -> 等 0xD -> UΩ1 / P_Ω2 -> 构造 Aᵀ -> Q=I ->
//   qr_update(0x9) -> Q₊ᵀ/Q₋ᵀ -> H+ / H- （全部按 128 对齐的 ld 落盘）
// 返回 k。
// ---------------------------------------------------------------------------
__aicore__ inline int sdc_tail_cube(
    qdwh_cube &op,
    GM_ADDR h, int ldH, int n, int nPad, int ldOmega, long uOff,
    GM_ADDR x,
    GM_ADDR tmp, GM_ADDR qt, GM_ADDR hwork, GM_ADDR qbuf, GM_ADDR uomega,
    GM_ADDR hplus, GM_ADDR hminus,
    GM_ADDR T, GM_ADDR omega1, GM_ADDR omega2, GM_ADDR eyeN, GM_ADDR dbg, GM_ADDR trBuf,
    int blockIdx, int blockNum)
{
#ifdef __DAV_C220_CUBE__
    GlobalTensor<float> H, Hplus, Hminus, Hwork, QdwhQ, UOmega, QT, Tmp, X;
    GlobalTensor<float> Omega1, Omega2;
    GlobalTensor<int32_t> Dbg;
    H.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(h));
    X.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
    Hplus.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hplus));
    Hminus.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hminus));
    Hwork.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hwork));
    QdwhQ.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qbuf));
    UOmega.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(uomega));
    QT.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qt));
    Tmp.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tmp));
    Omega1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(omega1));
    Omega2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(omega2));
    Dbg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dbg));

    // P+/- construction also publishes blockNum*2 partial traces.
    // Reuse lambdaTmp only after the batch's NodeLambda consumers have finished.
    CrossCoreWaitFlag(0xD);
    GlobalTensor<float> Tr;
    Tr.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(trBuf));
    const int aivCount = blockNum * 2;
    for (int i = 0; i < aivCount; i += 16) {
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(Tr[i]);
    }
    float tr = 0.0f;
    for (int i = 0; i < aivCount; ++i) tr += Tr.GetValue(i);
    int k = (int)((tr + (float)n) * 0.5f + 0.5f);
    if (k < 0) k = 0;
    if (k > n) k = n;

    // Own complete 128x128 output tiles, with both independent products
    // in one task space. K stays ordered on one core; no inter-core reduction.
    MatmulCustom<float, float, float> &mm = op.Mm();
    const int rc = (n + 127) / 128;
    const int omegaTiles = rc * rc;
    for (int t = blockIdx; t < 2 * omegaTiles; t += blockNum) {
        const bool minus = t >= omegaTiles;
        const int q = minus ? t - omegaTiles : t;
        const int row = (q / rc) * 128;
        const int col = (q % rc) * 128;
        const int rows = min(128, n - row);
        const int cols = min(128, n - col);
        if (minus) mm.SetMatrixLd(QT, Omega2, UOmega[(long)nPad * nPad], nPad, n, nPad, ldOmega);
        else mm.SetMatrixLd(Tmp, Omega1, UOmega, nPad, n, nPad, ldOmega);
        for (int kc = 0; kc < n; kc += 128) {
            const int kk = min(128, n - kc);
            mm.Process(row * nPad + kc, kc * ldOmega + col, row * nPad + col,
                       rows, cols, kk, -1, false, kc != 0);
        }
    }
    op.AicBarrier();

    // ---- 构造一个“满秩”的 n x n 矩阵 A 的转置，送给 QR：
    //   A[:, 0:k]    = PΩ1[:, 0:k]          （张成 P）
    //   A[:, k:n]    = P_-Ω2[:, 0:n-k]      （张成 I-P）
    // A 满秩，QR(A) 的 Q 前 k 列张成 P，后 n-k 列张成 I-P。
    //
    // 【P3】Aᵀ 不再逐元素 scalar 转置（原来 nPad² 次 GetValue/SetValue + 每元素一次
    // DCCI，是 split 里 AIC scalar 的最大一块），改成两次 GEMM 的隐式转置：
    //   Hwork[0:k, 0:n] = I·(UOmega 的前 k 列)ᵀ    —— isTransposeB（B 物理 = N×K）
    //   Hwork[k:n, 0:n] = I·(UOmega2 的前 m2 列)ᵀ
    // C 的 dstStride 恒等于该次 Process 的 N 参数（这里给 nPad，正好是 Hwork 的行宽），
    // 输出列数由 blockN=n 限制，因此只写 [0:n, 0:n]；padding（行 ≥ n 的整行、列 ≥ n）
    // 先清 0 即可（清 0 在 GEMM 之前，写的是 GEMM 不会碰的列，不存在 Fixpipe 与
    // scalar 写同一行 cache line 的竞争）。
    const int m2 = n - k;
    {
        // padding（行 ≥ n 的整行、列 ≥ n）已由 AIV 在 0xD 之前清 0
        //（qdwh_vec::clear_pad，向量写），这里不再做逐元素 scalar 清 0 + DCCI。
        GlobalTensor<float> EyeN;
        EyeN.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eyeN));
        const int plusRows = (k + 127) / 128;
        const int minusRows = (m2 + 127) / 128;
        const int plusTiles = plusRows * rc;
        for (int t = blockIdx; t < (plusRows + minusRows) * rc; t += blockNum) {
            const bool minus = t >= plusTiles;
            const int q = minus ? t - plusTiles : t;
            const int size = minus ? m2 : k;
            const int row = (q / rc) * 128;
            const int col = (q % rc) * 128;
            const int rows = min(128, size - row);
            const int cols = min(128, n - col);
            if (minus) mm.SetMatrixLd(EyeN, UOmega[(long)nPad * nPad], Hwork[(long)k * nPad],
                                      nPad, m2, ldOmega, nPad);
            else mm.SetMatrixLd(EyeN, UOmega, Hwork, nPad, k, ldOmega, nPad);
            for (int kc = 0; kc < size; kc += 128) {
                const int kk = min(128, size - kc);
                mm.Process(row * ldOmega + kc, col * nPad + kc, row * nPad + col,
                           rows, cols, kk, -1, true, kc != 0);
            }
        }
        op.AicBarrier();
    }

    // ---- 一次 QR(A) ----
    // Q 的初值（单位阵）由 AIV 在 0xD 之前写好（qdwh_vec::BuildIdentityQ），
    // 这里只等 0xA 的时序即可，不再做 GEMM 与整块 DCCI。
    op.ResetMatrixDims(hwork, qbuf, nPad, nPad, 1);
    // Hwork 现在由 Fixpipe（GEMM 隐式转置）写，AIV 的 ComputeAll 要 MTE2 读它，
    // 所以 0xA 必须挂在 PIPE_FIX 上（原来 Hwork 是 scalar 写，PIPE_MTE3 也"碰巧"对）。
    CrossCoreSetFlag<0x2, PIPE_FIX>(0xA);
    Dbg.SetValue(blockIdx, 0x9);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(Dbg[blockIdx]);
    op.qr_update(0x9);
    op.AicBarrier();
    // 【P3】Q 的 consumer 现在全是 MTE（下面两处 Q±ᵀ 都改成 GEMM 的 MTE2 读），
    // 不再有 scalar 读 → 原来整块 nPad²/32 行的 DCCI 可以删掉。
    // （Fixpipe 写 → MTE2 读本身是 GM 一致的，不需要 invalidate。）

    // ---- 【P4】一次 GEMM 直接得到 TmpAll = Qᵀ·H（nPad x nPad，复用原 tmp 缓冲）----
    // 原来是 4 组：QT=Q+ᵀ(kPad x n) → Tmp=QT·H → QT=Q-ᵀ(m2Pad x n) → Tmp=QT·H。
    // 现在 A = Q 本身：K x M = nPad x nPad 的**紧凑**行主序（行步长就是 M = nPad，
    // 即 Q 的 ld，零拷贝、不物化），isTransposeA 让 cube 按“转置形态”取 A 的块、
    // 再在 L1->L0 做 enTranspose，于是 C = Aᵀ·H = Qᵀ·H：
    //     TmpAll[0:k, :] = Q+ᵀ H        TmpAll[k:n, :] = Q-ᵀ H
    // 这条 A 转置通路 phase 1 的 QdwhFinalGemm（Q = UᵀA0）一直在用。
    // 收益：少 2 次 GEMM 启动 + 2 次 AicBarrier，不再写 QT（省掉 (kPad+m2Pad)*n
    // 个 float 的 Fixpipe 写回），Q+ᵀ/Q-ᵀ 的写后读假依赖也消失。
    //
    // K/M/N 三维都跑满 nPad（不是 n）：Q 的 padding 行/列由 AIV 的 BuildIdentityQ
    // 写成 0，Householder 只从右边乘、不改行的支撑，所以整个 QR 之后 Q 的 padding
    // 恒为 0 —— 跑满 nPad 与只跑 n 的结果一样，但每个 tile 都是满 128
    //（A 转置通路只在整块下可用；末尾 <128 的残块会把 kernel 卡死在这组 GEMM 里）。
    // 多读的那几行 H 乘的是 0；arena 已在 phase 1 开头清 0，所以这些“多读”
    // 一定是有限值，不会把 NaN/Inf 带进结果。
    const int kPad = ((k + 127) / 128) * 128;
    const int m2Pad = ((m2 + 127) / 128) * 128;

    mm.SetMatrixLdM(QdwhQ, H, Tmp, /*M*/ nPad, /*N(= C ld)*/ nPad, /*K*/ nPad,
                    /*ldA*/ nPad, /*ldB*/ ldH);
    const int pc = nPad / 128;
    for (int t = blockIdx; t < pc * pc; t += blockNum) {
        const int row = (t / pc) * 128;
        const int col = (t % pc) * 128;
        for (int kc = 0; kc < nPad; kc += 128) {
            mm.Process(row + kc * nPad, kc * ldH + col, row * nPad + col,
                       128, 128, 128, -1, false, kc != 0, true);
        }
    }
    op.AicBarrier();

    // ---- H+ = TmpAll[0:k, :] · Q+ ----
    // B = Q 的前 kPad 列（非转置、自定义 ldB = nPad）；tile 按 128 补齐
    //（M = kPad、N = kPad），与旧实现 H+ = Tmp·QTᵀ 的几何完全一致：
    // 只有「B 从转置形态的 QT 换成不转置的 Q」这一点变了。
    mm.SetMatrixLd(Tmp, QdwhQ, Hplus, /*N(= C ld)*/ kPad, /*K*/ n, nPad, nPad);
    for (int t = blockIdx; t < (kPad + 127) / 128; t += blockNum) {
        const int rowStart = t * 128;
        const int rows = min(128, kPad - rowStart);
        for (int kc = 0; kc < n; kc += 128) {
            const int kk = min(128, n - kc);
            mm.Process(rowStart * nPad + kc, kc * nPad, rowStart * kPad,
                       rows, kPad, kk, -1, false, kc != 0);
        }
    }
    op.AicBarrier();

    // ---- device 侧对称化 H+：数学上已对称，kSymmetrize=false 时跳过 ----

    // ---- H- = TmpAll[k:n, :] · Q- ----
    // A 的行基址多走 k 行、B 的列基址多走 k 列（Q- = Q[:, k:n]）；几何同上按 128 补齐。
    mm.SetMatrixLd(Tmp, QdwhQ, Hminus, /*N(= C ld)*/ m2Pad, /*K*/ n, nPad, nPad);
    for (int t = blockIdx; t < (m2Pad + 127) / 128; t += blockNum) {
        const int rowStart = t * 128;
        const int rows = min(128, m2Pad - rowStart);
        for (int kc = 0; kc < n; kc += 128) {
            const int kk = min(128, n - kc);
            mm.Process((k + rowStart) * nPad + kc, k + kc * nPad, rowStart * m2Pad,
                       rows, m2Pad, kk, -1, false, kc != 0);
        }
    }
    op.AicBarrier();

    // ---- device 侧对称化 H-：kSymmetrize=false 时跳过 ----
    // vec.hpp 里 ComputeAll 已跳过最后一 panel 的 SetFlag，QR 的 flag 握手
    // 严格配平，这里不再需要消费残留 flag。

    return k;
#else
    return n;   // AIV 分支不需要 k（AIC 的返回值只在本核使用）
#endif
}
