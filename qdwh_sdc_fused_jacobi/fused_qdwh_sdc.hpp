#pragma once
// ============================================================================
// QdwhSdcFused：把 QDWH 极分解（Phase 1）与 device 内谱分治 SDC（Phase 2）
// 融合进**一个 kernel**。
//
//   fused_main:  上传 A 一次 -> launch qdwh_sdc_fused_custom 一次 -> 下载 leaves
//
// 计算期间 host 不再与 device 有任何数据传输（不落 h.bin/u.bin，不读节点
// 元数据）；H/U 只是可选下载用于验证。
//
// 复用的既有实现（不重写算法）：
//   * qdwh_vec / qdwh_cube（qr_custom.cpp -> vec.hpp / gemm.cpp）：QDWH 的
//     全部迭代步骤（QdwhInitX / QdwhPrepare / ComputeAll / QdwhInitXt /
//     QdwhCopyH 与 qr_update / QdwhPmat / QdwhRestoreX / QdwhFinalGemm）；
//   * sdc_build_plan / sdc_tail_aiv / sdc_tail_cube（sdc_node.hpp）：单节点
//     split（λ / sign / PΩ / QR / H±），Phase 2 只是把节点循环搬进类方法。
//
// flag 分配（与提示词一致，逐位不重叠）：
//   Phase 1 QDWH : 0x1/0x2/0x3（QR 内部）、0x4/0x5/0x6（QDWH 阶段握手）、
//                  0x7（AIC 内部 barrier）、0x8（AIV 内部 barrier）
//   Phase 1->2   : 0xD（AIV->AIC：H 已写入 arena slot0）、
//                  0xF（AIC->AIV：QdwhFinalGemm 已完成）
//   Phase 2 SDC  : 0x7（AIC 内部 barrier）、0x8（AIV 组内 barrier）、
//                  0x9/0xA（split 的 QR）、0xB/0xC/0xE（节点握手）
//   说明：提示词里的“0x8 全核 barrier”在本实现里展开成 0xD/0xF 这一对双向
//   握手——0x8 这个 flag id 已被 qdwh_vec::AivBarrier（AIV 组内 barrier）
//   占用，再按字面复用会与 AIV 内部 barrier 的计数混在一起。
// ============================================================================
#include "sdc_node.hpp"      // 含 qr_custom.cpp（qdwh_vec / qdwh_cube）+ SDC 单节点 split
#include "fused_layout.hpp"
#include "jacobi_leaf.hpp"

// Phase 1 -> Phase 2 全核 barrier 的两个方向（见文件头注释）
static constexpr uint16_t kFlagPhaseAiv2Aic = 0xD;
// SDC 节点握手
static constexpr uint16_t kFlagSdcDispatch = 0xB;   // AIC -> AIV : Desc 就绪
static constexpr uint16_t kFlagSdcCopyInfo = 0xE;   // AIC -> AIV : CopyInfo 就绪
static constexpr uint16_t kFlagSdcCopied   = 0xC;   // AIV -> AIC : 搬运完成

class QdwhSdcFused {
public:
    __aicore__ inline void Init(__gm__ uint8_t *wk, int mIn, int nIn, int nRootIn, int thresholdIn,
                                int maxIterIn, int maxNodesIn, int maxLeavesIn,
                                int leafCapIn, float alphaIn, float l0In)
    {
        wk_ = wk;
        n_ = nIn;               // padded 宽度（128 的倍数）
        nRoot_ = (nRootIn < nIn) ? nRootIn : nIn;
        threshold_ = thresholdIn;
        maxIter_ = maxIterIn;
        maxNodes_ = maxNodesIn;
        maxLeaves_ = maxLeavesIn;
        leafCap_ = leafCapIn;
        alpha_ = alphaIn;
        l0_ = l0In;
        L.Build(mIn, nIn, maxNodesIn, maxLeavesIn, leafCapIn);
        qd_ = L.qd;

        // ---- SDC 节点队列 / 元数据（int32 视图；索引以 4 字节为单位）----
        Desc.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.desc)));
        CopyInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.copyInfo)));
        NodeH.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.nodeH)));
        NodeN.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.nodeN)));
        NodeLd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.nodeLd)));
        LeafCount.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.cnt + 3)));
        Overflow.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.cnt + 5)));
        GrpCnt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.cnt + 6)));
        GrpTotal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.cnt + 7)));
        LeafN.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.leafN)));
        LeafLd.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.leafLd)));
        LeafOff.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.leafOff)));
        Sigma.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Addr(L.sigma)));
        SigmaOff.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.sigmaOff)));
        SigmaCount.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.sigmaCount)));
        SigmaLeafCount.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Addr(L.sigmaLeafCount)));

        // ---- AIV 侧矩阵视图 ----
        HArena.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Addr(L.arena)));
        LeafH.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Addr(L.leaf)));
        HTmpP.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Addr(L.hplus)));
        HTmpM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Addr(L.hminus)));
    }

    // =====================================================================
    // Phase 1 : QDWH  A -> U, H     （AIV 侧）
    //   InitX -> [wait Xt -> Prepare -> QR panel -> wait Q -> scale Q1] x I
    //         -> wait Xt -> final Xt scale -> wait restored X/H -> CopyH
    // =====================================================================
    __aicore__ inline void QdwhAiv()
    {
#ifdef __DAV_C220_VEC__
        TPipe pipe;
        TBufPool<TPosition::VECCALC, 16> tbufPool;
        pipe.InitBufPool(tbufPool, 192 * 1024);

        qdwh_vec op;
        op.Init(&tbufPool);
        // hwork = APad（n x qd）、Q = qbuf（qd x qd）
        op.SetMatrix(Addr(L.hwork), Addr(L.tau), Addr(L.tmat), Addr(L.w), Addr(L.y),
                     Addr(L.shapes), 1);
        op.SetDims(qd_, n_);                      // QR 输入 (m+n) x n
        op.SetQdwhBuffers(Addr(L.a), Addr(L.a0), Addr(L.qbuf),
                          Addr(L.arena) /* H 直接落 arena slot0 */, Addr(L.coef), alpha_,
                          Addr(L.pt), Addr(L.xt));
        op.InitShapes();
        op.SetL0(l0_);

        // ---- arena + leaf 缓冲一次性清 0（必须在 QdwhCopyH 写 slot0 之前）----
        // split 尾部用一次 Qᵀ·H 同时得到 Q+ᵀH / Q-ᵀH：K 维跑满 nPad（Q 的 padding
        // 行是 0），最后一个 K 块会读到各节点 H 块之后最多 127*nPad 个 float。
        // 清 0 保证那些“多读”是 0 而不是未初始化位型（0*NaN=NaN 会污染结果）。
        {
            const int blockNum0 = GetBlockNum() << 1;
            const int blockIdx0 = GetBlockIdx();
            op.ZeroRange(Addr(L.arena), (long)(L.desc - L.arena), blockIdx0, blockNum0);
            op.AivBarrier();
        }

        op.QdwhInitX();                           // X = A0 / alpha
        op.AivBarrier();
        // Publish initial X for the AIC tiled transpose. Flag 0x5 is also used
        // after each Q1 scale and after the final Xt scale, in that order.
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);

        for (int it = 0; it < maxIter_; ++it) {
            CrossCoreWaitFlag(0x4);               // AIC 完成 Pmat（it==0 是初始放行）
            op.QdwhPrepare();
            op.AivBarrier();
            op.ComputeAll(GetBlockNum() << 1, GetBlockIdx());
            CrossCoreWaitFlag(0x6);
            op.AivBarrier();
            op.QdwhScaleQ1(GetBlockIdx(), GetBlockNum()<<1);
            op.AivBarrier();
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
        }

        CrossCoreWaitFlag(0x4);                   // 最后一轮 Pmat 完成
        op.QdwhFinalizeScale();                  // Xt = c_last * Yt
        op.QdwhDebugScale(Addr(L.dbg));
        op.AivBarrier();
        // AIC restores X from Xt before the final H GEMM.
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);    // 通知 AIC：Xt 已定稿，可以恢复 X 并计算 H
        CrossCoreWaitFlag(0x6);                   // AIC final GEMM 完成
        op.QdwhCopyH();                           // Q 的前 n*n -> arena slot0（H）
#endif
    }

    // =====================================================================
    // Phase 1 : QDWH  A -> U, H     （CUBE 侧）
    //   [QR -> publish Q -> wait scaled Q1 -> atomic Xt update] x I
    //   -> wait final Xt scale -> restore X -> FinalGemm(H) -> 0x6
    // =====================================================================
    __aicore__ inline void QdwhCube()
    {
#ifdef __DAV_C220_CUBE__
        qdwh_cube op;
        op.Init(Addr(L.hwork), Addr(L.tau), Addr(L.tmat), Addr(L.w), Addr(L.y),
                Addr(L.qbuf), Addr(L.shapes), Addr(L.eye), 1);
        // 不读 host 的 P 数组，直接指定单矩阵的 (m, n)
        op.ResetMatrixDims(Addr(L.hwork), Addr(L.qbuf), qd_, n_, 1);
        op.SetQdwhBuffers(Addr(L.a), Addr(L.a0), Addr(L.coef), Addr(L.pt), Addr(L.xt));

        // Xᵀ = I·Xᵀ（AIV 写完初值 X 后的一次 GEMM，替代 AIV 的 scalar 转置）
        CrossCoreWaitFlag(0x5);
        op.QdwhInitXt(Addr(L.xt), Addr(L.eye));
        op.AicBarrier();                          // 所有 AIC 的 Xt 都写完
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);     // 放行第一轮 AIV prepare

        for (int it = 0; it < maxIter_; ++it) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x2);   // QR 起始
            op.qr_update();
            op.AicBarrier();
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);
            CrossCoreWaitFlag(0x5);
            op.AicBarrier();
            op.QdwhPmat(true, true);                 // Xt += Q2 * scaledQ1^T
            op.AicBarrier();                         // all Xt updates complete
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);    // 通知 AIV：Xt 就绪
        }

        CrossCoreWaitFlag(0x5);                  // AIV completed final Xt scale
        op.QdwhRestoreX(Addr(L.eye));
        op.AicBarrier();
        op.QdwhFinalGemm();                      // Q = U^T A0 = H
        op.AicBarrier();
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);    // 放行 AIV 写 H
#endif
    }

    // =====================================================================
    // Phase 2 : device 内 SDC 递归   H -> leaves   （AIV 侧）
    //
    //   等 0xB -> 读本组 Desc（最多 kSdcBatchMax 个节点）
    //          -> SdcGroupAiv：λ/A0 -> batch QDWH -> 逐节点 tail + 搬运
    // =====================================================================
    __aicore__ inline void SdcAiv()
    {
#ifdef __DAV_C220_VEC__
        TPipe pipe;
        TBuf<TPosition::VECCALC> ubBuf;
        pipe.InitBuffer(ubBuf, kCopyBufFloats * sizeof(float));
        ubCopy_ = ubBuf.Get<float>();
        evM2M3_ = static_cast<int>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        evM3M2_ = static_cast<int>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

        // root H：QDWH 写在 offset 0、行宽 n_（padded）。若 nRoot<n_，先压成
        // ld=nRoot 的紧凑块放到 offset n_²；nRoot==n_ 时直接复用 offset 0。
        if (nRoot_ != n_) {
            const int blockNum = GetBlockNum() << 1;
            const int blockIdx = GetBlockIdx();
            CopyRowsPad(HArena, (long)n_ * n_, nRoot_, HArena, 0, n_,
                        nRoot_, nRoot_, blockIdx, blockNum);
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
            CrossCoreWaitFlag(0x8);
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0xD);   // 紧凑 root H 已就绪
        }

        while (true) {
            CrossCoreWaitFlag(kFlagSdcDispatch);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(GrpCnt[0]);
            const int cnt = GrpCnt.GetValue(0);

            int gSlot[kSdcBatchMax], gN[kSdcBatchMax], gLd[kSdcBatchMax];
            for (int b = 0; b < cnt && b < kSdcBatchMax; ++b) {
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(Desc[4 * b]);
                gSlot[b] = Desc.GetValue(4 * b + 0);
                gN[b] = Desc.GetValue(4 * b + 1);
                gLd[b] = Desc.GetValue(4 * b + 2);
            }

            if (cnt <= 0) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagSdcCopied);
                break;
            }

            SdcGroupAiv(cnt, gSlot, gN, gLd);
        }
#endif
    }

    // =====================================================================
    //   Phase 2 : device 内 SDC 递归   H -> leaves   （CUBE 侧）
    //
    //   节点/叶块计数器保存在每个 AIC 的本地寄存器里同步推进；AIC0 只用本地
    //   计数器把队列头的最多 kSdcBatchMax 个节点打包成 Desc，SetFlag(0xB) 启动
    //   AIV。组内逐节点的 CopyInfo / Node* / Leaf* GM 写由第 b 个 AIC 分摊，
    //   不再全部压在一个核上。
    // =====================================================================
    __aicore__ inline void SdcCube()
    {
#ifdef __DAV_C220_CUBE__
        const int blockIdx = GetBlockIdx();
        const int blockNum = GetBlockNum();

        // 节点/叶块计数器保存在每个 AIC 的寄存器里（所有核同步推进），GM 只在
        // 最终写一次给 host；逐节点的 GM 写由组内不同 owner 核分摊，不再全部压
        // 在 blockIdx==0 上。
        int leafCount = 0;
        int leafTop = 0;
        int nodeCount = 1;      // root 已登记
        int hTop = 1;
        int idx = 0;
        int grpTotal = 0;
        int overflow = 0;

        // nRoot<n_ 时等 AIV 把 root H 压成 ld=nRoot 的紧凑块；对齐时直接用 offset 0
        const bool rootCompact = (nRoot_ != n_);
        if (rootCompact) {
            CrossCoreWaitFlag(0xD);
        }
        const long hCompactOff = rootCompact ? (long)n_ * n_ : 0L;
        const int rootLd = rootCompact ? nRoot_ : n_;

        // ---- 初始化 device 节点队列：root H（ld=rootLd）----
        if (blockIdx == 0) {
            NodeH.SetValue(0, static_cast<int32_t>(hCompactOff));
            NodeN.SetValue(0, nRoot_);     // 只取原始 n
            NodeLd.SetValue(0, rootLd);    // 对齐时 n_，否则紧凑 nRoot_
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(NodeH[0]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(NodeN[0]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(NodeLd[0]);
        }
        AicSync();

        long arenaTop = hCompactOff + (long)nRoot_ * nRoot_;

        while (true) {
            // ---- 派发下一组（AIC0 读本核的 idx/nodeCount 并写 Desc/GrpCnt）----
            if (blockIdx == 0) {
                int g = nodeCount - idx;
                if (g > kSdcBatchMax) {
                    g = kSdcBatchMax;
                }
                for (int b = 0; b < g; ++b) {
                    // Node* 由组内 owner 核写入，读之前先 invalidate 本核 cache。
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(NodeH[idx + b]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(NodeN[idx + b]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(NodeLd[idx + b]);
                    Desc.SetValue(4 * b + 0, NodeH.GetValue(idx + b));
                    Desc.SetValue(4 * b + 1, NodeN.GetValue(idx + b));
                    Desc.SetValue(4 * b + 2, NodeLd.GetValue(idx + b));
                    Desc.SetValue(4 * b + 3, 1);
                }
                GrpCnt.SetValue(0, g);
                idx += g;
                if (g > 0) {
                    ++grpTotal;
                }
                // Desc 是 4*B 个 int32：B>4 时跨 2 条 64B cacheline，必须逐线 clean，
                // 否则第 5+ 个节点的表项留在 AIC 的 cache 里没刷回 GM，AIV 读到旧值，
                // 两侧 plan/maxK 不一致 → QR 的 panel 数不同 → 死锁。
                for (int o = 0; o < 4 * g; o += 8) {
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(Desc[o]);
                }
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(GrpCnt[0]);
            }
            AicSync();
            CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagSdcDispatch);

            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(GrpCnt[0]);
            const int cnt = GrpCnt.GetValue(0);

            int gSlot[kSdcBatchMax], gN[kSdcBatchMax], gLd[kSdcBatchMax];
            for (int b = 0; b < cnt && b < kSdcBatchMax; ++b) {
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(Desc[4 * b]);
                gSlot[b] = Desc.GetValue(4 * b + 0);
                gN[b] = Desc.GetValue(4 * b + 1);
                gLd[b] = Desc.GetValue(4 * b + 2);
            }

            if (cnt <= 0) {
                CrossCoreWaitFlag(kFlagSdcCopied);
                break;
            }

            SdcGroupCube(cnt, gSlot, gN, gLd,
                         leafCount, leafTop, nodeCount, hTop, overflow, arenaTop);
        }

        // ---- 叶块元数据刷回 GM，供 host 下载（所有 AIC 按 64B 行分摊）----
        const int leafLines = (leafCount + 15) >> 4;
        for (int l = blockIdx; l < leafLines; l += blockNum) {
            const int i = l << 4;
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(LeafN[i]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(LeafLd[i]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(LeafOff[i]);
        }
        if (blockIdx == 0) {
            LeafCount.SetValue(0, leafCount);
            Overflow.SetValue(0, overflow);
            GrpTotal.SetValue(0, grpTotal);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(LeafCount[0]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(Overflow[0]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(GrpTotal[0]);
        }
        // Mode 2 only joins one AIC and its two AIVs. Publish all metadata
        // before any AIC releases its local AIV pair into Jacobi.
        AicSync();
#endif
    }

    // =====================================================================
    // 一组的 AIV 流程：
    //   (1) 每个内部节点 λ = trace(H)/n，A0 = pad(H-λI) 写到 a0 的第 i 个切片
    //   (2) 这 cntI 个矩阵跑一次 batch QDWH（sign-QDWH 4 轮）
    //   (3) 逐节点 tail（P/P_ + split QR）-> 等 CopyInfo -> 搬运 H+/H-/leaf
    // =====================================================================
    __aicore__ inline void SdcGroupAiv(int cnt, const int *gSlot, const int *gN, const int *gLd)
    {
#ifdef __DAV_C220_VEC__
        const int blockNum = GetBlockNum() << 1;
        const int blockIdx = GetBlockIdx();

        // 本组专用的 QDWH UB 池（与 SdcAiv 的搬运缓冲时段错开，和旧版每节点
        // 新建一套 TPipe 的做法一致）
        TPipe pipe;
        TBufPool<TPosition::VECCALC, 16> tbufPool;
        pipe.InitBufPool(tbufPool, 192 * 1024);
        qdwh_vec op;
        op.Init(&tbufPool);

        SdcBatchPlan plan;
        sdc_build_plan(gN, cnt, threshold_, plan);

        if (plan.cntI > 0) {
            // ---- (1) λ / A0（逐节点；行归属与 QdwhInitX 完全一致）----
            for (int b = 0; b < cnt; ++b) {
                const int i = plan.idx[b];
                if (i < 0) {
                    continue;
                }
                __gm__ uint8_t *h = Addr(L.arena) + (long)gSlot[b] * 4;
                const float lambda = op.NodeLambda(h, gLd[b], gN[b], blockIdx, blockNum,
                                                    Addr(L.lambdaTmp));
                op.BuildA0FromH(h, Addr(L.xsdc) + (long)plan.aOff[i] * 4, gLd[b], gN[b],
                                plan.nPad[i], lambda, blockIdx, blockNum);
            }

            // ---- (2) 一次 batch QDWH ----
            // 【Xt】split 也打开 Xᵀ 缓存（和 phase 1 一样）：QdwhPrepare 改用连续行读
            // Xᵀ 建 H，不再按列 scalar gather X；代价是 QdwhPmat 多算一块 Pᵀ（AIC）。
            sdc_qdwh_batch_aiv(op, Addr(L.hwork), Addr(L.tau), Addr(L.tmat), Addr(L.w),
                               Addr(L.y), Addr(L.shapes), Addr(L.xsdc), Addr(L.a0),
                               Addr(L.qbuf), Addr(L.arena), Addr(L.coef),
                               Addr(L.pt), Addr(L.xt), true, plan, maxIter_, alpha_, l0_);
        }

        // ---- (3) 逐节点 tail + 搬运 ----
        for (int b = 0; b < cnt; ++b) {
            const int i = plan.idx[b];
            if (i >= 0) {
                sdc_tail_aiv(op, (long)plan.aOff[i], gN[b], plan.nPad[i],
                             Addr(L.tmp), Addr(L.qt), Addr(L.hwork), Addr(L.tau),
                             Addr(L.tmat), Addr(L.w), Addr(L.y), Addr(L.shapes) + 8, Addr(L.lambdaTmp),
                             blockIdx, blockNum);
            }

            CrossCoreWaitFlag(kFlagSdcCopyInfo);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(CopyInfo[0]);

            for (int part = 0; part < 2; ++part) {
                const int off = part * 3;
                const int kind = CopyInfo.GetValue(off + 0);   // -1 skip, 0 node, 1 leaf(tmp), 2 leaf(arena)
                if (kind < 0) {
                    continue;
                }
                const int dst = CopyInfo.GetValue(off + 1);
                const int srcLd = CopyInfo.GetValue(off + 2);
                const int childN = (part == 0) ? CopyInfo.GetValue(6) : CopyInfo.GetValue(7);
                if (childN <= 0) {
                    continue;
                }
                if (kind == 2) {
                    // 顶层 leaf：直接从 arena 偏移搬到 leafH
                    const int srcSlot = CopyInfo.GetValue(8);
                    CopyRowsPad(LeafH, (long)dst, childN,
                                HArena, (long)srcSlot, srcLd,
                                childN, childN, blockIdx, blockNum);
                } else if (kind == 1) {
                    // 本节点 split 出来的 leaf：从 padded 临时缓冲搬到 leafH
                    if (part == 0) {
                        CopyRowsPad(LeafH, (long)dst, childN,
                                    HTmpP, 0, srcLd, childN, childN, blockIdx, blockNum);
                    } else {
                        CopyRowsPad(LeafH, (long)dst, childN,
                                    HTmpM, 0, srcLd, childN, childN, blockIdx, blockNum);
                    }
                } else {
                    // 非叶节点：compact 到 arena 的新偏移
                    CopyRowsPad(HArena, (long)dst, childN,
                                (part == 0) ? HTmpP : HTmpM, 0, srcLd,
                                childN, childN, blockIdx, blockNum);
                }
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagSdcCopied);
        }
#endif
    }

    // =====================================================================
    // 一组的 AIC 流程：
    //   (1) cntI 个矩阵的 batch QDWH（qr_update + QdwhPmat，与 AIV 严格配对）
    //   (2) 逐节点 split tail（k / UΩ / 转置 / QR / H±）+ 孩子登记 + CopyInfo
    // =====================================================================
    __aicore__ inline void SdcGroupCube(int cnt, const int *gSlot, const int *gN, const int *gLd,
                                        int &leafCount, int &leafTop, int &nodeCount, int &hTop,
                                        int &overflow, long &arenaTop)
    {
#ifdef __DAV_C220_CUBE__
        const int blockNum = GetBlockNum();
        const int blockIdx = GetBlockIdx();

        SdcBatchPlan plan;
        sdc_build_plan(gN, cnt, threshold_, plan);

        qdwh_cube op;   // 每组建一次（与旧版每节点建一次的粒度一致）

        if (plan.cntI > 0) {
            sdc_qdwh_batch_cube(op, Addr(L.hwork), Addr(L.tau), Addr(L.tmat), Addr(L.w),
                                Addr(L.y), Addr(L.qbuf), Addr(L.shapes), Addr(L.eye),
                                Addr(L.xsdc), Addr(L.a0), Addr(L.coef),
                                Addr(L.pt), Addr(L.xt), Addr(L.eyeN), n_, plan, maxIter_);
        }

        for (int b = 0; b < cnt; ++b) {
            const int n = gN[b];
            const int i = plan.idx[b];

            // ---------------- 顶层 leaf：直接从 arena slot 出手 ----------------
            if (i < 0) {
                const int lc = leafCount;
                const int lt = leafTop;
                if (lc >= maxLeaves_ || lt + n * n > leafCap_) {
                    overflow = 1;
                }
                // 本节点的 GM 写由 AIC0 负责（元数据块很小，单写者保证 cacheline 一致）
                if (blockIdx == 0) {
                    CopyInfo.SetValue(0, 2);
                    CopyInfo.SetValue(1, lt);
                    CopyInfo.SetValue(2, gLd[b]);
                    CopyInfo.SetValue(8, gSlot[b]);
                    CopyInfo.SetValue(3, -1);
                    CopyInfo.SetValue(6, n);
                    if (lc < maxLeaves_) {
                        LeafN.SetValue(lc, n);
                        LeafLd.SetValue(lc, n);
                        LeafOff.SetValue(lc, lt);
                        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                                 DcciDst::CACHELINE_OUT>(LeafN[lc]);
                        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                                 DcciDst::CACHELINE_OUT>(LeafLd[lc]);
                        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                                 DcciDst::CACHELINE_OUT>(LeafOff[lc]);
                    }
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(CopyInfo[0]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(CopyInfo[3]);
                }
                leafCount = lc + 1;
                leafTop = lt + n * n;
                AicSync();
                CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagSdcCopyInfo);
                CrossCoreWaitFlag(kFlagSdcCopied);
                AicSync();
                continue;
            }

            // ---------------- 内部节点：split tail ----------------
            const int nPad = plan.nPad[i];
            const int hOff = gSlot[b];
            __gm__ uint8_t *h = Addr(L.arena) + (long)hOff * 4;
            const int k = sdc_tail_cube(op, h, gLd[b], n, nPad, n_, (long)plan.aOff[i],
                                        Addr(L.xsdc),
                                        Addr(L.tmp), Addr(L.qt), Addr(L.hwork), Addr(L.qbuf),
                                        Addr(L.uomega), Addr(L.hplus), Addr(L.hminus),
                                        Addr(L.tmat), Addr(L.om1), Addr(L.om2), Addr(L.eyeN),
                                        Addr(L.dbg), Addr(L.lambdaTmp), blockIdx, blockNum);

            const int m2 = n - k;
            const int kPad = ((k + 127) / 128) * 128;
            const int m2Pad = ((m2 + 127) / 128) * 128;

            // 孩子登记：所有核用相同的本地计数器算索引；只有第 b 个 AIC 写 GM。
            const int lc = leafCount;
            const int lt = leafTop;
            const int nc = nodeCount;
            const bool pLeaf = (k > 0 && k <= threshold_);
            const bool mLeaf = (m2 > 0 && m2 <= threshold_);
            const int pLeafIdx = lc;
            const int mLeafIdx = lc + (pLeaf ? 1 : 0);
            const int pLeafOff = lt;
            const int mLeafOff = lt + (pLeaf ? k * k : 0);
            const int pNodeIdx = nc;
            const int mNodeIdx = nc + ((k > 0 && !pLeaf) ? 1 : 0);

            // H 偏移：先所有孩子都重新分配（不复用父块），排除复用/覆盖类问题。
            // root 的 H（偏移 0）要留给 host 校验，不能覆盖。
            int pNodeOff = 0;
            int mNodeOff = 0;
            if (k > 0 && !pLeaf) {
                if (arenaTop + (long)k * k > L.arenaFloats) {
                    overflow = 1;
                    pNodeOff = static_cast<int>(L.arenaFloats);
                } else {
                    pNodeOff = static_cast<int>(arenaTop);
                    arenaTop += k * k;
                }
            }
            if (m2 > 0 && !mLeaf) {
                if (arenaTop + (long)m2 * m2 > L.arenaFloats) {
                    overflow = 1;
                    mNodeOff = static_cast<int>(L.arenaFloats);
                } else {
                    mNodeOff = static_cast<int>(arenaTop);
                    arenaTop += m2 * m2;
                }
            }

            if (nc + 2 > maxNodes_ || lc + 2 > maxLeaves_) {
                overflow = 1;
            }
            if (pLeaf && lt + k * k > leafCap_) {
                overflow = 1;
            }
            if (mLeaf && mLeafOff + m2 * m2 > leafCap_) {
                overflow = 1;
            }
            if ((long)arenaTop > L.arenaFloats) {
                overflow = 1;
            }

            // CopyInfo 是跨核控制线：固定 AIC0 写，避免同一 cacheline 多核写回。
            if (blockIdx == 0) {
                // H+ 孩子
                if (k > 0) {                    if (pLeaf) {
                        CopyInfo.SetValue(0, 1);
                        CopyInfo.SetValue(1, pLeafOff);
                        CopyInfo.SetValue(2, kPad);
                        CopyInfo.SetValue(6, k);
                    } else {
                        CopyInfo.SetValue(0, 0);
                        CopyInfo.SetValue(1, pNodeOff);
                        CopyInfo.SetValue(2, kPad);
                        CopyInfo.SetValue(6, k);
                    }
                } else {
                    CopyInfo.SetValue(0, -1);
                    CopyInfo.SetValue(6, 0);
                }
                // H- 孩子
                if (m2 > 0) {
                    if (mLeaf) {
                        CopyInfo.SetValue(3, 1);
                        CopyInfo.SetValue(4, mLeafOff);
                        CopyInfo.SetValue(5, m2Pad);
                        CopyInfo.SetValue(7, m2);
                    } else {
                        CopyInfo.SetValue(3, 0);
                        CopyInfo.SetValue(4, mNodeOff);
                        CopyInfo.SetValue(5, m2Pad);
                        CopyInfo.SetValue(7, m2);
                    }
                } else {
                    CopyInfo.SetValue(3, -1);
                    CopyInfo.SetValue(7, 0);
                }
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(CopyInfo[0]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(CopyInfo[3]);
            }

            // Node*/Leaf* 元数据：固定由 AIC0 写。多个核写同一 64B cacheline 时，
            // scalar cache 的整行回写会互相覆盖，n 稍大（叶子阈值 32）就会打乱节点表。
            if (pLeaf && pLeafIdx < maxLeaves_ && blockIdx == 0) {
                LeafN.SetValue(pLeafIdx, k);
                LeafLd.SetValue(pLeafIdx, k);
                LeafOff.SetValue(pLeafIdx, pLeafOff);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(LeafN[pLeafIdx]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(LeafLd[pLeafIdx]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(LeafOff[pLeafIdx]);
            }
            if (mLeaf && mLeafIdx < maxLeaves_ && blockIdx == 0) {
                LeafN.SetValue(mLeafIdx, m2);
                LeafLd.SetValue(mLeafIdx, m2);
                LeafOff.SetValue(mLeafIdx, mLeafOff);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(LeafN[mLeafIdx]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(LeafLd[mLeafIdx]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(LeafOff[mLeafIdx]);
            }
            if (k > 0 && !pLeaf && pNodeIdx < maxNodes_ && blockIdx == 0) {
                NodeH.SetValue(pNodeIdx, pNodeOff);
                NodeN.SetValue(pNodeIdx, k);
                NodeLd.SetValue(pNodeIdx, k);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(NodeH[pNodeIdx]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(NodeN[pNodeIdx]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(NodeLd[pNodeIdx]);
            }
            if (m2 > 0 && !mLeaf && mNodeIdx < maxNodes_ && blockIdx == 0) {
                NodeH.SetValue(mNodeIdx, mNodeOff);
                NodeN.SetValue(mNodeIdx, m2);
                NodeLd.SetValue(mNodeIdx, m2);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(NodeH[mNodeIdx]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(NodeN[mNodeIdx]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(NodeLd[mNodeIdx]);
            }

            // 所有核同步推进本地计数器（顺序与旧版 lc/lt/nc/ht 的递增一致）。
            if (k > 0) {
                if (pLeaf) {
                    ++leafCount;
                    leafTop += k * k;
                } else {
                    ++nodeCount;
                    ++hTop;
                }
            }
            if (m2 > 0) {
                if (mLeaf) {
                    ++leafCount;
                    leafTop += m2 * m2;
                } else {
                    ++nodeCount;
                    ++hTop;
                }
            }

            AicSync();
            CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagSdcCopyInfo);
            CrossCoreWaitFlag(kFlagSdcCopied);
            AicSync();
        }
#endif
    }

    // =====================================================================
    // 一次 kernel 内的 Phase 1 -> Phase 2 全核 barrier
    //   AIV 先 set（H 已落 arena slot0）再等 AIC 的 final GEMM 完成信号；
    //   AIC 反之。两个方向都用已验证的 mode 2 跨核 flag 原语。
    // =====================================================================
    __aicore__ inline void PhaseBarrier(uint16_t flag)
    {
#ifdef __DAV_C220_VEC__
        CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);
        CrossCoreWaitFlag(static_cast<uint16_t>(flag + 2));
#else
        CrossCoreWaitFlag(flag);
        CrossCoreSetFlag<0x2, PIPE_FIX>(static_cast<uint16_t>(flag + 2));
#endif
    }

    __aicore__ inline void JacobiAiv()
    {
#ifdef __DAV_C220_VEC__
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(LeafCount[0]);
        const int leafCount = LeafCount.GetValue(0);
        const int cid = GetBlockIdx();
        const int logicalAiv = GetBlockNum() << 1;
        TPipe pipe;
        fused_jacobi::JacobiLeafWorker worker;
        worker.Init(&pipe);

        // Publish immutable leaf metadata before parallel Jacobi work.
        // The final AIV/AIC phase barrier guarantees all sigma slots are
        // complete before host code observes these entries.
        if (cid == 0) {
            for (int i = 0; i < leafCount; ++i) {
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(LeafN[i]);
                SigmaOff.SetValue(i, i * 64);
                SigmaCount.SetValue(i, LeafN.GetValue(i));
            }
            for (int i = 0; i < leafCount; i += 16) {
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(SigmaOff[i]);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(SigmaCount[i]);
            }
            SigmaLeafCount.SetValue(0, leafCount);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(SigmaLeafCount[0]);
        }

        for (int i = cid; i < leafCount; i += logicalAiv) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(LeafN[i]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(LeafLd[i]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(LeafOff[i]);
            const int n = LeafN.GetValue(i);
            const int ld = LeafLd.GetValue(i);
            const int off = LeafOff.GetValue(i);
            if (n <= 0 || n > 64 || ld < n) continue;
            worker.Run(reinterpret_cast<__gm__ float *>(Addr(L.leaf)), n, ld, off,
                       reinterpret_cast<__gm__ float *>(Addr(L.sigma)), i * 64, 30);
        }
#endif
    }

    __aicore__ inline void RunAiv()
    {
        QdwhAiv();
        PhaseBarrier(kFlagPhaseAiv2Aic);
#ifdef AB1
        return;   // ????? Phase 1?QDWH?
#endif
        SdcAiv();
        PhaseBarrier(kFlagPhaseAiv2Aic);
#ifdef AB2
        return;   // ???Phase 1 + SDC??? Jacobi
#endif
        JacobiAiv();
        PhaseBarrier(kFlagPhaseAiv2Aic);
    }

    __aicore__ inline void RunCube()
    {
        QdwhCube();
        PhaseBarrier(kFlagPhaseAiv2Aic);
#ifdef AB1
        return;
#endif
        SdcCube();
        PhaseBarrier(kFlagPhaseAiv2Aic);
#ifdef AB2
        return;
#endif
        PhaseBarrier(kFlagPhaseAiv2Aic);
    }

private:
    static constexpr int kCopyBufFloats = 2048;   // 行搬运 UB 缓冲

    __aicore__ inline __gm__ uint8_t *Addr(long off) const
    {
        return wk_ + off * 4;
    }

    // 所有 AIC 的 barrier（mode 0）。用 0x7，避开 QR 内部的 0x0~0x3/0x9/0xA。
    __aicore__ inline void AicSync()
    {
#ifdef __DAV_C220_CUBE__
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);
#endif
    }

    // ---------------------------------------------------------------------
    // 所有矩阵搬运的唯一入口（AIV DataCopyPad，GM -> UB -> GM）。
    //   src[srcBase + r*srcLd .. +cols]  ->  dst[dstBase + r*dstLd .. +cols]
    // 行按 AIV 编号轮转。禁止 AIC 用 scalar 读写 GEMM/Fixpipe 刚写出的矩阵。
    // ---------------------------------------------------------------------
    __aicore__ inline void CopyRowsPad(GlobalTensor<float> &dst, long dstBase, int dstLd,
                                       GlobalTensor<float> &src, long srcBase, int srcLd,
                                       int rows, int cols, int blockIdx, int blockNum)
    {
#ifdef __DAV_C220_VEC__
        DataCopyPadExtParams<float> pp{false, 0, 0, 0};
        // 按 UB 容量分列块，支持 nRoot/childN > kCopyBufFloats 的大矩阵
        for (int c0 = 0; c0 < cols; c0 += kCopyBufFloats) {
            const int cc = (cols - c0 < kCopyBufFloats) ? (cols - c0) : kCopyBufFloats;
            DataCopyExtParams cp{1, static_cast<uint32_t>(cc * 4), 0, 0, 0};
            for (int r = blockIdx; r < rows; r += blockNum) {
                DataCopyPad(ubCopy_, src[srcBase + (long)r * srcLd + c0], cp, pp);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evM2M3_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evM2M3_);
                DataCopyPad(dst[dstBase + (long)r * dstLd + c0], ubCopy_, cp);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2_);
            }
        }
#endif
    }

    FusedLayout L;
    __gm__ uint8_t *wk_ = nullptr;
    int n_ = 0, qd_ = 0;       // n_ = padded 宽度；nRoot_ = 原始 n（SDC root）
    int nRoot_ = 0;
    int threshold_ = 64, maxIter_ = 15;
    int maxNodes_ = 0, maxLeaves_ = 0, leafCap_ = 0;
    float alpha_ = 1.0f, l0_ = 1e-6f;

    // ---- metadata / 矩阵视图 ----
    GlobalTensor<int32_t> Desc, CopyInfo, NodeH, NodeN, NodeLd;
    GlobalTensor<int32_t> LeafCount, Overflow, LeafN, LeafLd, LeafOff, GrpCnt, GrpTotal;
    GlobalTensor<int32_t> SigmaOff, SigmaCount, SigmaLeafCount;
    GlobalTensor<float> HArena, LeafH, HTmpP, HTmpM, Sigma;

    // ---- AIV 行搬运 ----
    LocalTensor<float> ubCopy_;
    int evM2M3_ = 0, evM3M2_ = 0;
};
