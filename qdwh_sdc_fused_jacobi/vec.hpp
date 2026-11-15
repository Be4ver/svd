
// ============================================================================
// qdwh_vec：QDWH 的 AIV（vector）侧实现
//   * X 的初值（X = A0/alpha）；初始 Xt 由 AIC 分块生成
//   * Y 迭代的权重记账与 Q1 缩放；Yt 由 AIC atomic-add 更新
//   * H = [sqrt(c)*cPrev*Yt | I] 与 Q = I 的构造、最终 Xt 缩放
//   * split 的 A0 = pad(H - lambda*I)、P/P_ = U +/- I
//   * QR 的 Householder 面板计算（ComputeAll/ComputeBatch）
// ============================================================================

__aicore__ inline int align_to(int a, int b) {
    return (a + b - 1) / b * b;
}
namespace AscendC
{
    // ------------------------------------------------------------------------
    // 批量实现（多个矩阵的同一个 panel 合并成一批，一趟跑完所有 phase）
    // ------------------------------------------------------------------------
    class qdwh_vec
    {
    public:
        __aicore__ inline qdwh_vec() : M(0), N(0), padM(0), padN(0) {}

        __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
        {
            pipe->InitBuffer(aQueueIn, 1, 2 * chunkSize * sizeof(float));
            pipe->InitBuffer(bQueueIn, 1, chunkSize * sizeof(float));
            pipe->InitBuffer(aQueueOut, 1, chunkSize * sizeof(float));
            // 【修正3】T_local 现在要装 kMaxG 个矩阵的 T，每个 blockDim*columns*columns；
            //          原来只给了 1024 float，blockDim>32 时（910B3 常见 40/48）直接越界。
            //          这里按 blockDim <= 64、columns == 2 (=> blockDim*4 <= 256) 预留。
            pipe->InitBuffer(tQueueDst, 1, kMaxG * 64 * 4 * sizeof(float)); // 2048 float = 8KB
            pipe->InitBuffer(tauQueueOut, 2, 32 * sizeof(float));
            // 点积区容量 = columns*curM，curM 最大 = kQrMaxQd（fused_batch_cfg.hpp）。
            // 原来这里是 chunkSize+1024，并把 tau/v0/tauUb 也塞在尾部
            // （workLocal[kSpan..]）；一旦 columns*curM > kSpan 就会覆写 tau。
            pipe->InitBuffer(workQueue, kQrColumns * kQrMaxQd * sizeof(float));
        }

        // Hwork 的 padding 清零（AIV 向量版，放在 sdc_tail_aiv 的 0xD 之前，
        // 正好与 AIC 的 UΩ GEMM 重叠）：行 ≥ n 整行清 0，行 < n 只清列 [n, nPad)。
        // 原来由 AIC 逐元素 scalar 写（每节点 ~12k 次）+ 每 16 float 一次 DCCI。
        __aicore__ inline void clear_pad(GM_ADDR hwork, int n, int nPad, int blockIdx, int blockNum)
        {
            GlobalTensor<float> Hw;
            Hw.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hwork));
            LocalTensor<float> buf = aQueueOut.AllocTensor<float>();
            int32_t evVM3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            int32_t evM3M2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::Duplicate(buf, 0.0f, nPad);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVM3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVM3);
            DataCopyExtParams cpFull{1, static_cast<uint32_t>(nPad * 4), 0, 0, 0};
            DataCopyExtParams cpTail{1, static_cast<uint32_t>((nPad - n) * 4), 0, 0, 0};
            for (int j = blockIdx; j < nPad; j += blockNum) {
                if (j >= n) {
                    AscendC::DataCopyPad(Hw[(long)j * nPad], buf, cpFull);
                } else if (nPad > n) {
                    AscendC::DataCopyPad(Hw[(long)j * nPad + n], buf, cpTail);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            aQueueOut.FreeTensor(buf);
        }

        __aicore__ inline void SetMatrix(__gm__ uint8_t *APad_, __gm__ uint8_t *tau_, __gm__ uint8_t *T_, __gm__ uint8_t *W_, __gm__ uint8_t *Y_, __gm__ uint8_t *P_,int batch)
        {
            APad.SetGlobalBuffer((__gm__ float *)APad_);
            tau.SetGlobalBuffer((__gm__ float *)tau_);
            T.SetGlobalBuffer((__gm__ float *)T_);
            W.SetGlobalBuffer((__gm__ float *)W_);
            Y.SetGlobalBuffer((__gm__ float *)Y_);
            P.SetGlobalBuffer((__gm__ int32_t *)P_);
            this->batch = batch;
            shapesForced = false;   // 默认仍从 P 读 shapes
            shapesBatch = false;
        }

        // 单矩阵、已知 m/n 时直接指定 shapes，避免读 host 的 P 数组。
        __aicore__ inline void SetDims(int mm, int nn)
        {
            forcedM = mm;
            forcedN = nn;
            shapesForced = true;
            shapesBatch = false;
        }

        // ------------------------------------------------------------------
        // 【batch QR】直接给定批内每个矩阵的 QR 输入形状：
        //   mArr[b] = 行数（QDWH 里是 2*nPad[b]），nArr[b] = 列数（nPad[b]）。
        // 一次设置好 b 的 m/n/size/tauOff/yOff/maxK/panelYSize，之后 InitShapes()
        // 直接返回（不再从 P 读、也不再被 forcedM/forcedN 覆盖）。
        // ------------------------------------------------------------------
        __aicore__ inline void SetDimsBatch(const int *mArr, const int *nArr, int cnt)
        {
            batch = cnt;
            int tauSizeAll = 0, ySizeAll = 0;
            max_M = 0;
            maxK = 0;
            for (int b = 0; b < cnt; ++b) {
                m[b] = mArr[b];
                n[b] = nArr[b];
                size[b] = (b == 0) ? 0 : m[b - 1] * n[b - 1];
                tauOff[b] = tauSizeAll;
                tauSizeAll += n[b];
                yOff[b] = ySizeAll;
                ySizeAll += 64 * m[b];
                max_M = max(max_M, m[b]);
                maxK = max(maxK, min(m[b], n[b]));
            }
            M = max_M;
            panelYSize = ySizeAll;
            shapesForced = false;
            shapesBatch = true;
        }

        // ------------------------------------------------------------------
        // 以下接口用于单 kernel 内的 batch QDWH：所有步骤都在 device 侧完成。
        // QdwhA  = 工作矩阵 X（m x n 行主序）
        // QdwhA0 = 原始矩阵 A0（m x n 行主序）
        // QdwhQ  = QR 输出的完整 Q（(m+n) x (m+n) 行主序）
        // APad   = H 工作区（n x (m+n) 行主序）
        // ------------------------------------------------------------------
        __aicore__ inline void SetQdwhBuffers(__gm__ uint8_t *a_, __gm__ uint8_t *a0_,
                                             __gm__ uint8_t *q_, __gm__ uint8_t *hout_,
                                             __gm__ uint8_t *coef_, float alpha_,
                                             __gm__ uint8_t *pt_ = nullptr,
                                             __gm__ uint8_t *xt_ = nullptr)
        {
            QdwhA.SetGlobalBuffer((__gm__ float *)a_);
            QdwhA0.SetGlobalBuffer((__gm__ float *)a0_);
            QdwhQ.SetGlobalBuffer((__gm__ float *)q_);
            QdwhHOut.SetGlobalBuffer((__gm__ float *)hout_);
            QdwhCoef.SetGlobalBuffer((__gm__ float *)coef_);
            alpha = alpha_;
            // P^T / X^T 是可选加速缓冲：不传就退回“按列 scalar gather”的老路径
            hasPt = (pt_ != nullptr);
            hasXt = (xt_ != nullptr);
            if (hasPt) QdwhPt.SetGlobalBuffer((__gm__ float *)pt_);
            if (hasXt) QdwhXt.SetGlobalBuffer((__gm__ float *)xt_);
        }

        __aicore__ inline void SetL0(float l0)
        {
            // float32 下 l^4 在 l < 1e-10 附近会下溢，权重公式不再可用；
            // 这里按 1e-6 做下限保护（默认值就是 1e-6）。
            float lv = l0;
            if (lv < 1e-6f) lv = 1e-6f;
            if (lv > 1.0f) lv = 1.0f;
            for (int b = 0; b < batch; ++b) {
                lArr[b] = lv;
                cProd[b] = 1.0f;
            }
        }

        // 只同步所有 AIV（mode 0）；不要与 QR 内部使用的 0x0~0x3 混用。
        __aicore__ inline void AivBarrier()
        {
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
            CrossCoreWaitFlag(0x8);
        }

        // 把 [base, base+count) 的每个 64B cache line 刷到 GM。
        // 按“行”分配写任务，保证一条 cache line 只由一个核写，
        // 避免多核 scalar 写同一行的 false sharing。
        __aicore__ inline void CleanRange(GlobalTensor<float> &t, int base, int count)
        {
            for (int off = 0; off < count; off += 16) {
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(t[base + off]);
            }
        }

        // scalar 读别的核写的 X 之前，先 invalidate 全部 X cache line。
        __aicore__ inline void InvalidateXAll()
        {
            int aOff = 0;
            for (int b = 0; b < batch; ++b) {
                const int qd = m[b];
                const int N = n[b];
                const int M = qd - N;
                const int total = M * N;
                for (int off = 0; off < total; off += 16) {
                    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(QdwhA[aOff + off]);
                }
                aOff += total;
            }
        }

        // X = A0 / alpha
        //
        // 全 MTE/向量实现，**没有任何 scalar 兜底**：
        //   GM -> UB (DataCopyPad) -> Muls(1/alpha) 原地缩放 -> UB -> GM (DataCopyPad)。
        //   * 原实现逐元素 scalar GetValue/SetValue（1024x1024 时 1M 次读 + 1M 次写），
        //     而且每写一行 DCCI 一次，是 QDWH 里的标量热点；
        //   * 用 DataCopyPad 而不是 DataCopy：blockLen 按字节算，不再要求 N 是 32B 的
        //     整数倍、也不需要 8-float 对齐，所以可以彻底删掉原来的标量回退分支；
        //   * MTE3 的写不经过 scalar cache，不需要逐行 DCCI；后续 QdwhPrepare 的
        //     InvalidateXAll() 仍然负责 invalidate 整块 X；
        //   * **行归属必须和调用方写 A0 的划分完全一致**（r = blockIdx; r += blockNum）：
        //     BuildA0FromH 是按这个划分写 A0 并逐行 clean 的，若这里改成按连续区间
        //     切块，就会读到别的核还没刷回 GM 的行（跨核竞争，eig_err 会退到 1e-2）；
        //   * 一次搬 nRows 行（带 srcStride/dstStride），MTE 指令数是 O(行数/批)。
        // 约束：N <= chunkSize（8192），本题 shape 都满足。
        __aicore__ inline void QdwhInitX()
        {
            const int blockNum = GetBlockNum() << 1;
            const int blockIdx = GetBlockIdx();
            const float invAlpha = 1.0f / alpha;

            const int32_t evM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            const int32_t evV2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            const int32_t evM3M2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

            LocalTensor<float> ub = aQueueOut.AllocTensor<float>();
            DataCopyPadExtParams<float> noPad{false, 0, 0, 0};

            int aOff = 0;
            for (int b = 0; b < batch; ++b) {
                const int N = n[b];
                const int M = m[b] - N;
                const int total = M * N;
                int nRows = chunkSize / N;
                if (nRows < 1) {
                    nRows = 1;
                }

                const uint32_t rowBytes = static_cast<uint32_t>(N * 4);           // 每行字节数
                const uint32_t rowGap = static_cast<uint32_t>((blockNum - 1) * N * 4);  // 相邻行间隔
                DataCopyExtParams inP{static_cast<uint16_t>(1), rowBytes, rowGap, 0, 0};
                DataCopyExtParams outP{static_cast<uint16_t>(1), rowBytes, 0, rowGap, 0};
                for (int r0 = blockIdx; r0 < M; r0 += blockNum * nRows) {
                    int rows = (M - 1 - r0) / blockNum + 1;     // 本核剩余支行数
                    if (rows > nRows) {
                        rows = nRows;
                    }
                    inP.blockCount = static_cast<uint16_t>(rows);
                    outP.blockCount = static_cast<uint16_t>(rows);
                    DataCopyPad(ub, QdwhA0[aOff + r0 * N], inP, noPad);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
                    Muls(ub, ub, invAlpha, rows * N);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2M3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2M3);
                    DataCopyPad(QdwhA[aOff + r0 * N], ub, outP);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
                }
                aOff += total;
            }

            aQueueOut.FreeTensor(ub);
            // [barrier removed: 跨核握手/pipe 序已覆盖]

            // X^T 的初值（只做一次）：X^T = I · Xᵀ。
            // 【不再由 AIV scalar 转置】原来是逐元素 scalar 读 X 的列 + 写 X^T 的行
            // （每节点 nPad² 次 GM 标量读写 + DCCI，全树 ≈1e7 次，是 AIV scalar 的
            // 第三大块）。现在改由 AIC 的 QdwhInitXt 用 128x128 单位阵分块生成，
            // 每块 Xt 使用局部 identity-times-transpose GEMM。
            // 顺序：AIV 写完 X 后 SetFlag(0x5)，AIC 等它 -> 转置 -> AicBarrier ->
            // SetFlag(0x4)，AIV 的第一轮 QdwhPrepare 才读 Xt。见调用方。
        }

        // Fold beta into Q1, then AIC atomically accumulates Yt.
        __aicore__ inline void QdwhScaleQ1(int bi, int bn)
        {
            const int e2v = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            const int ev3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            const int e32 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            LocalTensor<float> ub = aQueueOut.AllocTensor<float>();
            DataCopyPadExtParams<float> np{false,0,0,0};
            int qo=0;
            for (int b=0; b<batch; ++b) {
                const int qd=m[b], N=n[b], M=qd-N;
                DataCopyExtParams cp{1,static_cast<uint32_t>(N*4),0,0,0};
                for(int r=bi; r<M; r+=bn) {
                    DataCopyPad(ub,QdwhQ[qo+r*qd],cp,np);
                    SetFlag<HardEvent::MTE2_V>(e2v); WaitFlag<HardEvent::MTE2_V>(e2v);
                    Muls(ub,ub,kappaArr[b],N);
                    SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3);
                    DataCopyPad(QdwhQ[qo+r*qd],ub,cp);
                    SetFlag<HardEvent::MTE3_MTE2>(e32); WaitFlag<HardEvent::MTE3_MTE2>(e32);
                }
                qo+=qd*qd;
            }
            aQueueOut.FreeTensor(ub);
        }

        // Debug slot 112 occupies a separate cache line from progress slots 0..19.
        __aicore__ inline void QdwhDebugScale(GM_ADDR dbg_)
        {
            if (GetBlockIdx() != 0) return;
            GlobalTensor<float> dbg;
            dbg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbg_));
            dbg.SetValue(112, cProd[0]);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(dbg[112]);
        }

        // Xt = c_last * Yt, once after the final iteration.
        __aicore__ inline void QdwhFinalizeScale()
        {
            const int bi = GetBlockIdx(), bn = GetBlockNum() << 1;
            const int e2v = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            const int ev3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            const int e32 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            LocalTensor<float> ub = aQueueOut.AllocTensor<float>();
            DataCopyPadExtParams<float> np{false, 0, 0, 0};
            int off = 0;
            for (int b = 0; b < batch; ++b) {
                const int total = (m[b] - n[b]) * n[b];
                for (int j = bi * chunkSize; j < total; j += bn * chunkSize) {
                    const int cnt = min(chunkSize, total-j);
                    DataCopyExtParams cp{1, static_cast<uint32_t>(cnt*4), 0, 0, 0};
                    if (hasXt) {
                        DataCopyPad(ub, QdwhXt[off+j], cp, np);
                        SetFlag<HardEvent::MTE2_V>(e2v); WaitFlag<HardEvent::MTE2_V>(e2v);
                        Muls(ub, ub, cProd[b], cnt);
                        SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3);
                        DataCopyPad(QdwhXt[off+j], ub, cp);
                        SetFlag<HardEvent::MTE3_MTE2>(e32); WaitFlag<HardEvent::MTE3_MTE2>(e32);
                    }
                }
                off += total;
            }
            aQueueOut.FreeTensor(ub);
        }

        // A0 = pad(H - λI)（split 的 QDWH 输入，AIV 侧逐行构造）
        // 1~2 个向量指令 + 2 次 UB 标量 + 1 次 MTE3。
        __aicore__ inline void BuildA0FromH(__gm__ uint8_t *h, __gm__ uint8_t *a0, int ldH,
                                            int n, int nPad, float lambda,
                                            int blockIdx, int blockNum)
        {
            GlobalTensor<float> H, A0G;
            H.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(h));
            A0G.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a0));

            const int32_t evM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            const int32_t evV2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            const int32_t evS2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
            const int32_t evM3M2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

            LocalTensor<float> rowAll = aQueueIn.AllocTensor<float>();  // 2*chunkSize
            LocalTensor<float> rowH = rowAll;                          // [0, nPad)：读 H 的行
            LocalTensor<float> rowO = rowAll[chunkSize];               // [chunkSize, +nPad)：写 A0 的行
            DataCopyExtParams cpRowN{1, static_cast<uint32_t>(n * 4), 0, 0, 0};
            DataCopyExtParams cpRowP{1, static_cast<uint32_t>(nPad * 4), 0, 0, 0};
            DataCopyPadExtParams<float> noPad{false, 0, 0, 0};

            for (int r = blockIdx; r < nPad; r += blockNum) {
                // 注意：向量指令要求 UB 地址 32B 对齐，而 n 不一定是 8 的倍数，
                // 所以不能对 row[n] 直接 Duplicate——先整行清零（偏移 0，对齐），
                // 再用 Adds 把 H 的 n 个元素复制到行首。
                Duplicate(rowO, 0.0f, nPad);
                if (r < n) {
                    DataCopyPad(rowH, H[r * ldH], cpRowN, noPad);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
                    Adds(rowO, rowH, 0.0f, n);
                    rowO.SetValue(r, rowO.GetValue(r) - lambda);
                } else {
                    // padding 维：取一个比真实负特征值更负的常数，保证 sign-QDWH
                    // 在 n<128、padding 占比很大时仍能稳定收敛出 -1。
                    rowO.SetValue(r, -(lambda + 1.0f));
                }
                //AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2M3);
                //AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2M3);
                //AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evS2M3);
                //AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evS2M3);
                DataCopyPad(A0G[r * nPad], rowO, cpRowP);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            }

            aQueueIn.FreeTensor(rowAll);
        }

        // P = U + I、P_ = U - I（split 的两块 GEMM 操作数）
        //
        // 原来在 AIC 上用 scalar 逐元素写 2*n^2 个元素 + 每行 DCCI，是 AIC 侧
        // 最大的标量块之一。现在改成 AIV 侧用 DataCopyPad 构造：
        //   * DataCopyPad 把 U 的第 r 行（前 n 列，行步长 ldU）读进 UB；
        //   * Adds(..., 0.0f) 复制一份到 ubQ（= U）；
        //   * 两个缓冲各用一次 UB 标量 SetValue 把对角元 ±1（每行只有 2 次标量操作）；
        //   * DataCopyPad 把两行分别写回 P（ldOut）与 P_（ldOut）。
        // 只写前 n 行 n 列（与旧实现一致，尾部/多余行留给后面的 kernel 逻辑）。
        // ------------------------------------------------------------------
        // lambda = trace(H)/n
        //   * 行按 blockNum 平均切成 blockNum 段，第 blockIdx 个 AIV 只读自己
        //     那一段的对角元（原来 40 个核各读完整对角线，40x 重复）。
        //   * 各核把自己的部分和写进一条连续的 GM 数组（第 blockIdx 个 float）。
        //   * 一次 AivBarrier 后所有核把数组整段读回 UB，用 WholeReduceSum 归约。
        // ------------------------------------------------------------------
        __aicore__ inline float NodeLambda(GM_ADDR h, int ldH, int n, int blockIdx, int blockNum,
                                           GM_ADDR trBuf)
        {
            GlobalTensor<float> H, Tr;
            H.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(h));
            Tr.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(trBuf));
            if (n <= 0 || blockNum <= 0) return 0.0f;

            const int32_t evM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            const int32_t evM3M2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            const int32_t evS2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
            const int32_t evV2S  = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));

            const int per = (n + blockNum - 1) / blockNum;
            const int r0  = blockIdx * per;
            const int cnt = (r0 < n) ? ((n - r0 < per) ? (n - r0) : per) : 0;

            float acc = 0.0f;
            for (int c = 0; c < cnt; ++c) {
                const int r = r0 + c;
                acc += H.GetValue((long)r * ldH + r);
            }

            LocalTensor<float> ub = aQueueIn.AllocTensor<float>();     // 2*chunkSize float
            DataCopyPadExtParams<float> noPad{false, 0, 0, 0};

            // 本核分片 -> GM（连续排布：第 blockIdx 个 float）
            ub.SetValue(0, acc);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evS2M3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evS2M3);
            DataCopyExtParams cp1{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(Tr[blockIdx], ub, cp1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);

            AivBarrier();                                   // 所有核的分片都落 GM

            for (int c = 0; c < blockNum; c += 16) {
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(Tr[c]);
            }
            LocalTensor<float> part = ub[512];              // blockNum float
            DataCopyExtParams cpN{1, static_cast<uint32_t>(blockNum * sizeof(float)), 0, 0, 0};
            DataCopyPad(part, Tr[0], cpN, noPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);

            LocalTensor<float> red = ub[768];
            WholeReduceSum(red, part, blockNum, 1, 1, 1, 8);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(evV2S);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(evV2S);
            float tot = red.GetValue(0);

            AivBarrier();                                   // 下一节点可以覆写分片区
            aQueueIn.FreeTensor(ub);
            return tot / (float)n;
        }

        // Build P+/- and accumulate trace from rows already resident in UB.
        // Each AIV publishes one partial by MTE3, without scattered GM scalar reads.
        __aicore__ inline void BuildPmats(__gm__ uint8_t *pm, __gm__ uint8_t *pmMinus,
                                         int n, int ldU, int ldOut, int blockIdx, int blockNum,
                                         long uOff, GM_ADDR trBuf)
        {
            GlobalTensor<float> Pm, PmM, Tr;
            Pm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pm));
            PmM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pmMinus));
            Tr.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(trBuf));
            const int evM2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            const int evM2S = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
            const int evV2S = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            const int evS2M3 = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
            const int evM3M2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            const int evM3S = GetTPipePtr()->FetchEventID(HardEvent::MTE3_S);
            LocalTensor<float> ubP = aQueueIn.AllocTensor<float>();
            LocalTensor<float> ubQ = aQueueOut.AllocTensor<float>();
            DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
            float trace = 0.0f;
            const int begin = n * blockIdx / blockNum;
            const int end = n * (blockIdx + 1) / blockNum;
            const int cap = chunkSize / ldU;
            // ldU == ldOut == nPad at the only callsite. Copy whole padded
            // rows so both GM and UB addressing stay contiguous and aligned.
            // Only the real diagonal participates in trace and gets +/-1.
            for (int r0 = begin; r0 < end; r0 += cap) {
                const int rows = min(cap, end - r0);
                const int count = rows * ldU;
                DataCopyExtParams cp{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
                DataCopyPad(ubP, QdwhA[uOff + (long)r0 * ldU], cp, noPad);
                SetFlag<HardEvent::MTE2_V>(evM2V);
                WaitFlag<HardEvent::MTE2_V>(evM2V);
                SetFlag<HardEvent::MTE2_S>(evM2S);
                WaitFlag<HardEvent::MTE2_S>(evM2S);
                Adds(ubQ, ubP, 0.0f, count);
                SetFlag<HardEvent::V_S>(evV2S);
                WaitFlag<HardEvent::V_S>(evV2S);
                for (int rr = 0; rr < rows; ++rr) {
                    const int pos = rr * ldU + r0 + rr;
                    const float diag = ubP.GetValue(pos);
                    trace += diag;
                    ubP.SetValue(pos, diag + 1.0f);
                    ubQ.SetValue(pos, diag - 1.0f);
                }
                SetFlag<HardEvent::S_MTE3>(evS2M3);
                WaitFlag<HardEvent::S_MTE3>(evS2M3);
                DataCopyPad(Pm[(long)r0 * ldOut], ubP, cp);
                DataCopyPad(PmM[(long)r0 * ldOut], ubQ, cp);
                SetFlag<HardEvent::MTE3_MTE2>(evM3M2);
                WaitFlag<HardEvent::MTE3_MTE2>(evM3M2);
            }
            // Scalar reuse must also wait for the last row's DMA reads.
            SetFlag<HardEvent::MTE3_S>(evM3S);
            WaitFlag<HardEvent::MTE3_S>(evM3S);
            ubP.SetValue(0, trace);
            SetFlag<HardEvent::S_MTE3>(evS2M3);
            WaitFlag<HardEvent::S_MTE3>(evS2M3);
            DataCopyExtParams cp1{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(Tr[blockIdx], ubP, cp1);
            SetFlag<HardEvent::MTE3_S>(evM3S);
            WaitFlag<HardEvent::MTE3_S>(evM3S);
            aQueueOut.FreeTensor(ubQ);
            aQueueIn.FreeTensor(ubP);
        }

        // H = [sqrt(c)*cPrev*Yt | I], Q = I; update l, cProd, betaPrime.
        // Rows are distributed over AIVs. Only the legacy scalar gather path
        // requires invalidating X; the fused Xt path reads through MTE2.
        __aicore__ inline void QdwhPrepare()
        {
            const int blockNum = GetBlockNum() << 1;
            const int blockIdx = GetBlockIdx();

            // 只有 gather 路径（hasXt=false）用 scalar 读 X，才需要 invalidate
            if (!hasXt) {
                InvalidateXAll();
            }

            const int32_t evM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            const int32_t evV2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            const int32_t evS2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
            const int32_t evM3M2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

            // 行缓冲：hRow 存 H 的一行，qRow 存 Q 的一行（都要求 >= qd 个 float）
            LocalTensor<float> hRow = aQueueOut.AllocTensor<float>();
            LocalTensor<float> qRow = aQueueIn.AllocTensor<float>();

            // ================= 【负载均衡 A.1】批内展平行分配 =================
            // 原来每个矩阵各自 `for (r = blockIdx; r < N; r += blockNum)`：
            // N=128、blockNum=40 时有 8 个核 4 行、32 个核 3 行（33% 不均衡），
            // 而墙钟≈最忙核。这里把 batch 内所有矩阵的"要写的行"展平成一条序列，
            // 每核取 [blockIdx*rpc, +rpc) —— 每核行数差 ≤1。
            int prefH[16], prefQ[16];
            int offA[16], offH[16], offQ[16], offT[16];
            float sqcArr[16];
            prefH[0] = 0;
            prefQ[0] = 0;
            int aOff = 0;
            int hOff = 0;
            int qOff = 0;
            int tOff = 0;
            for (int b = 0; b < batch; ++b) {
                const int qd = m[b];
                const int N = n[b];
                const int M = qd - N;

                float l = lArr[b];
                if (l > 1.0f) l = 1.0f;
                float a, bb, c;
                compute_weight(a, bb, c, l);
                const float sqrt_c = sqrt(c);
                const float theta = bb / c;
                const float kappa = (a - bb / c) / sqrt_c;
                QdwhCoef.SetValue(b * 2 + 0, theta);
                QdwhCoef.SetValue(b * 2 + 1, kappa);
                const float cPrev = cProd[b];
                cProd[b] = cPrev * theta;
                kappaArr[b] = kappa / cProd[b];
                if (b == 0) QdwhCoef.SetValue(100, cProd[b]);
                l = l * (a + bb * l * l) / (1.0f + c * l * l);
                if (l > 1.0f) l = 1.0f;
                lArr[b] = l;

                sqcArr[b] = sqrt_c * cPrev;
                offA[b] = aOff;
                offH[b] = hOff;
                offQ[b] = qOff;
                offT[b] = tOff;
                prefH[b + 1] = prefH[b] + N;
                prefQ[b + 1] = prefQ[b] + qd;
                aOff += M * N;
                hOff += N * qd;
                qOff += qd * qd;
                tOff += M * N;
            }

            // ---- H 行：前 M 列 sqrt(c)*X^T，后 N 列单位阵 ----
            {
                const int totalH = prefH[batch];
                const int rpc = (totalH + blockNum - 1) / blockNum;
                const int gBeg = blockIdx * rpc;
                const int gEnd = min(totalH, gBeg + rpc);
                for (int g = gBeg; g < gEnd; ++g) {
                    int bb2 = 0;
                    while (prefH[bb2 + 1] <= g) {
                        ++bb2;
                    }
                    const int r = g - prefH[bb2];
                    const int qd = m[bb2];
                    const int N = n[bb2];
                    const int M = qd - N;
                    DataCopyExtParams cpRow{1, static_cast<uint32_t>(qd * 4), 0, 0, 0};
                    if (hasXt) {
                        DataCopyExtParams cpRowM{1, static_cast<uint32_t>(M * 4), 0, 0, 0};
                        DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
                        DataCopyPad(hRow, QdwhXt[offT[bb2] + r * M], cpRowM, noPad);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
                        Muls(hRow, hRow, sqcArr[bb2], M);
                    } else {
                        for (int col = 0; col < M; ++col) {
                            hRow.SetValue(col, QdwhA.GetValue(offA[bb2] + col * N + r));
                        }
                        Muls(hRow, hRow, sqcArr[bb2], M);
                    }
                    Duplicate(hRow[M], 0.0f, qd - M);
                    hRow.SetValue(M + r, 1.0f);
                    //AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2M3);
                    //AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2M3);
                    //AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evS2M3);
                    //AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evS2M3);
                    DataCopyPad(APad[offH[bb2] + r * qd], hRow, cpRow);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
                }
            }

            // ---- Q 行：单位阵 ----
            {
                const int totalQ = prefQ[batch];
                const int rpc = (totalQ + blockNum - 1) / blockNum;
                const int gBeg = blockIdx * rpc;
                const int gEnd = min(totalQ, gBeg + rpc);
                const int evVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                const int evM3ToV = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
                // Duplicate -> scalar diagonal -> DMA -> next Duplicate.
                // Keep explicit events for the producer/consumer pipes.
                for (int bb2 = 0; bb2 < batch; ++bb2) {
                    const int qd = m[bb2];
                    const int rBeg = max(0, gBeg - prefQ[bb2]);
                    const int rEnd = min(qd, gEnd - prefQ[bb2]);
                    const int cap = max(1, chunkSize / qd);
                    for (int r0 = rBeg; r0 < rEnd; r0 += cap) {
                        const int rows = min(cap, rEnd-r0);
                        Duplicate(qRow, 0.0f, rows * qd);
                        SetFlag<HardEvent::V_S>(evVToS);
                        WaitFlag<HardEvent::V_S>(evVToS);
                        for (int rr = 0; rr < rows; ++rr) {
                            qRow.SetValue(rr * qd + r0 + rr, 1.0f);
                        }
                        //SetFlag<HardEvent::S_MTE3>(evS2M3);
                        //WaitFlag<HardEvent::S_MTE3>(evS2M3);
                        DataCopyExtParams cpQ{1, static_cast<uint32_t>(rows * qd * 4), 0, 0, 0};
                        DataCopyPad(QdwhQ[offQ[bb2] + r0 * qd], qRow, cpQ);
                        SetFlag<HardEvent::MTE3_V>(evM3ToV);
                        WaitFlag<HardEvent::MTE3_V>(evM3ToV);
                    }
                }
            }
            aQueueIn.FreeTensor(qRow);
            aQueueOut.FreeTensor(hRow);
            // 系数只有 batch*2 <= 128 个 float；整块刷一次即可
            CleanRange(QdwhCoef, 0, 128);
        }

        // split 的 QR 需要 Q 的初值 = 块对角 diag(I_nIdent, 0)（行宽 = n = nPad）：
        // 只给左上 nIdent x nIdent 写单位阵，其余（padding 区）连对角也写 0。
        // 这样 Q 的 padding 行在整个 QR 里恒为 0（Householder 只从右边乘、行的支撑
        // 不变），于是 split 尾部可以用一次 Qᵀ·H（K 维直接跑到 nPad、K 块恒为满 128）
        // 同时得到 Q+ᵀH 与 Q-ᵀH，而 padding 行的贡献严格是 0。
        // 原来由 cube 用一次 GEMM（Q = eyeN·eyeN）生成，那是 cube 关键路径上的一次
        // nPad^3 乘加 + 整块 DCCI；现在改由 AIV 在 0xD（P/P_ 就绪）之前顺手写掉
        //（和 QdwhPrepare 里写 Q 行完全同构：整行 Duplicate 清 0 -> 标量置对角 ->
        // DataCopyPad 写回）。
        // 注意：不能"整块 DataCopy 清零 + 标量补对角"混着写——MTE3 与 scalar
        // 之间没有同步时零会把 1 覆盖掉（实测 leaves 从 23 变 24）。
        __aicore__ inline void BuildIdentityQ(int nIdent, int n, int blockIdx, int blockNum)
        {
            const int evV2S = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            const int evS2M3 = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
            const int evM3V = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            LocalTensor<float> qRows = aQueueOut.AllocTensor<float>();
            const int begin = n * blockIdx / blockNum;
            const int end = n * (blockIdx + 1) / blockNum;
            const int cap = chunkSize / n;
            for (int r0 = begin; r0 < end; r0 += cap) {
                const int rows = min(cap, end - r0);
                Duplicate(qRows, 0.0f, rows * n);
                SetFlag<HardEvent::V_S>(evV2S);
                WaitFlag<HardEvent::V_S>(evV2S);
                for (int rr = 0; rr < rows && r0 + rr < nIdent; ++rr) {
                    qRows.SetValue(rr * n + r0 + rr, 1.0f);
                }
                SetFlag<HardEvent::S_MTE3>(evS2M3);
                WaitFlag<HardEvent::S_MTE3>(evS2M3);
                DataCopyExtParams cp{1, static_cast<uint32_t>(rows * n * 4), 0, 0, 0};
                DataCopyPad(QdwhQ[(long)r0 * n], qRows, cp);
                SetFlag<HardEvent::MTE3_V>(evM3V);
                WaitFlag<HardEvent::MTE3_V>(evM3V);
            }
            aQueueOut.FreeTensor(qRows);
        }

        // arena（以及紧随其后的 leaf 缓冲）一次性清 0。
        // split 尾部为了把 Qᵀ·H 的 K 维跑满 128 块，会读到各节点 H 块之后最多
        // 127*nPad 个 float（乘数是 Q padding 行的 0）；先把这片区域清 0，
        // 保证那些“多读”是 0 而不是未初始化的位型（0*NaN = NaN，会污染结果）。
        __aicore__ inline void ZeroRange(GM_ADDR base, long floats, int blockIdx, int blockNum)
        {
            const int32_t evV2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            const int32_t evM3M2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            if (floats <= 0) {
                return;
            }
            GlobalTensor<float> Dst;
            Dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base));
            const int chunk = chunkSize;
            LocalTensor<float> buf = aQueueOut.AllocTensor<float>();
            Duplicate(buf, 0.0f, chunk);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2M3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2M3);
            for (long o = (long)blockIdx * chunk; o < floats; o += (long)blockNum * chunk) {
                const int cc = (floats - o < chunk) ? static_cast<int>(floats - o) : chunk;
                DataCopyExtParams cp{1, static_cast<uint32_t>(cc * 4), 0, 0, 0};
                DataCopyPad(Dst[o], buf, cp);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            }
            aQueueOut.FreeTensor(buf);
        }

        // H 前 n*n 个元素 = Q 中 final GEMM 的结果（连续 n*n，行 stride = n）
        __aicore__ inline void QdwhCopyH()
        {
            const int blockNum = GetBlockNum() << 1;
            const int blockIdx = GetBlockIdx();

            // [DCCI removed] 后续全走 DataCopyPad(MTE2)，不经 scalar cache，无需 invalidate

            // 纯搬运：Q 的前 N 列（行宽 qd）-> H（行宽 N）。
            // 用 DataCopyPad 批量搬：每核按 blockNum 步长取行，一次最多 chunkSize/N 行，
            // 靠 blockCount + srcStride/dstStride 一次搬多行（原来每个元素一次标量读写）。
            const int32_t evM2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            const int32_t evM3M2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

            LocalTensor<float> ub = aQueueIn.AllocTensor<float>();   // 2*chunkSize
            DataCopyPadExtParams<float> noPad{false, 0, 0, 0};

            int hOutOff = 0;
            int qOff = 0;
            for (int b = 0; b < batch; ++b) {
                const int qd = m[b];
                const int N = n[b];
                int cap = chunkSize / N;
                if (cap < 1) {
                    cap = 1;
                }
                const uint32_t rowBytes = static_cast<uint32_t>(N * 4);
                // 源是 H（FinalGemm 把它紧凑写在 Q 的前 N*N 个元素里，行距 = N）
                const uint32_t srcGap = static_cast<uint32_t>((blockNum - 1) * N * 4);
                const uint32_t dstGap = static_cast<uint32_t>((blockNum - 1) * N * 4);
                DataCopyExtParams inQ{1, rowBytes, srcGap, 0, 0};
                DataCopyExtParams outH{1, rowBytes, 0, dstGap, 0};
                for (int r0 = blockIdx; r0 < N; r0 += blockNum * cap) {
                    int rows = (N - 1 - r0) / blockNum + 1;
                    if (rows > cap) {
                        rows = cap;
                    }
                    inQ.blockCount = static_cast<uint16_t>(rows);
                    outH.blockCount = static_cast<uint16_t>(rows);
                    DataCopyPad(ub, QdwhQ[qOff + r0 * N], inQ, noPad);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evM2M3);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evM2M3);
                    DataCopyPad(QdwhHOut[hOutOff + r0 * N], ub, outH);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
                }
                hOutOff += N * N;
                qOff += qd * qd;
            }
            aQueueIn.FreeTensor(ub);
        }

        // 【修正2】分批方案只与 (m[], offsetM, panelSize) 有关，所有核算出的结果完全一致。
        //         抽成独立函数后，空转核也能按“同样的批数”发出同样多的跨核 flag，
        //         否则批数 > 1 时（batch > kMaxG，或 ΣcurM > chunkSize）计算核多等了
        //         (批数-1)*(blockDim+2) 次同步，flag 计数错位 -> 死锁/结果错乱。
        __aicore__ inline void BuildPlan(int offsetM, int panelSize,
                                         int *planS, int *planE, int &planNum)
        {
            planNum = 0;
            int s = 0;
            while (s < batch) {
                // 行、列任一不足 offset+panel_size 的矩阵不再参与本列块
                while (s < batch && min(m[s], n[s]) < offsetM + panelSize) { ++s; }
                if (s >= batch) { break; }
                int e = s;
                int tot = 0;
                while (e < batch && min(m[e], n[e]) >= offsetM + panelSize) {
                    int len = m[e] - offsetM;      // 行方向长度（向量长度）
                    if (e > s && (tot + len > chunkSize || e - s + 1 > kMaxG)) { break; }
                    tot += len;
                    ++e;
                }
                planS[planNum] = s;
                planE[planNum] = e;
                ++planNum;
                s = e;
            }
        }

        // 从 P 读入形状。P 描述的是“QR 输入矩阵”的 shapes：
        //   m[k] = P[2k+1]（QR 输入行数），n[k] = P[2k]（QR 输入列数）。
        // 对 QDWH，P 传的是 (N, M+N)，因此 m=qd、n=N；对普通 QR，
        // P 传的是 (M, N)，因此 m=N、n=M。两种情况下本类都按 m x n 处理。
        __aicore__ inline void InitShapes()
        {
            if (shapesBatch) {
                // SetDimsBatch 已经把 m[]/n[]/size[]/tauOff[]/yOff[]/maxK/panelYSize
                // 全部就位；这里必须**原样返回**（下面的 maxK=0 会把 panel 循环清空）。
                return;
            }
            int tauSizeAll = 0, ySizeAll = 0;
            max_M = 0;
            maxK = 0;
            if (shapesForced) {
                m[0] = forcedM;
                n[0] = forcedN;
                size[0] = 0;
                tauOff[0] = 0;
                yOff[0] = 0;
                max_M = forcedM;
                maxK = min(forcedM, forcedN);
                M = forcedM;
                panelYSize = 64 * forcedM;
                return;
            }
            for (int k = 0; k < batch; ++k) {
                m[k] = P.GetValue(2 * k + 1);
                n[k] = P.GetValue(2 * k);
                size[k] = (k == 0) ? 0 : m[k - 1] * n[k - 1];
                tauOff[k] = tauSizeAll;
                tauSizeAll += n[k];
                yOff[k] = ySizeAll;
                ySizeAll += 64 * m[k];
                max_M = max(max_M, m[k]);
                maxK = max(maxK, min(m[k], n[k]));
            }
            M = max_M;
            panelYSize = ySizeAll;
        }

        __aicore__ inline void ComputeAll(int blockDim, int blockIdx, int qrFlag = 0x1)
        {
            int offset = 0;
            int flag = 0;
            int columns = kQrColumns;      // 点积区容量与 host guard 都以此为准
            int align_blockdim = min(32, blockDim);
            // 与 cube 侧保持一致：T 按 32x32x16 预留（host 也是 batch*32*32*32）
            panelTSize = 32 * 32 * 16;
            int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

            InitShapes();

            int panel_size = align_blockdim * columns;
            int planS[64], planE[64], planNum = 0;
            for (; offset + panel_size <= maxK;) {
                CrossCoreWaitFlag(qrFlag + 1);
                // mode2 only joins one AIC and its two AIVs. Complete the
                // group rendezvous before any AIV loads the next panel.
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                CrossCoreWaitFlag(0x0);
                BuildPlan(offset, panel_size, planS, planE, planNum);

                if (blockIdx < align_blockdim) {
                    ComputeBatch(align_blockdim, blockIdx, offset, flag, columns,
                                 planS, planE, planNum);
                }
                else {
                    // 空转核：每个“批”都要陪计算核走完一轮 (blockDim + 2) 次同步
                    for (int c = 0; c < planNum; ++c) {
                        syn_all(align_blockdim, 0);
                    }
                }

                //AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                //AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

                CrossCoreSetFlag<0x2, PIPE_MTE3>(qrFlag);

                flag = flag ^ 1;
                offset += panel_size;
            }
        }

        __aicore__ inline void syn_all(int blockDim, int blockIdx)
        {
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);
            for(int k = blockIdx + 1; k != blockDim; k++) {
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                // 读取中间结果
                CrossCoreWaitFlag(0x0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);
        }

        __aicore__ inline void ComputeBatch(int blockDim, int blockIdx, int offsetM, int flag,
                                            int columns, int *planS, int *planE, int nPlan)
        {
            const int kSpan      = chunkSize;                       // 8192：a 的列间距 = 单列容量
            const int WStride    = 2048;                            // ★ W 中每矩阵占用（host 需 batch*WStride）
            const int TStride    = blockDim * columns * columns;     // 每矩阵 T_local 步长
            const int TRowStride = blockDim * columns * 8;           // 每矩阵“分散 T 行”步长

            LocalTensor<float> workLocal = workQueue.Get<float>();
            LocalTensor<float> a = aQueueIn.AllocTensor<float>();
            LocalTensor<float> b = bQueueIn.AllocTensor<float>();
            LocalTensor<float> householder_v = aQueueOut.AllocTensor<float>();
            LocalTensor<float> T_local = tQueueDst.AllocTensor<float>();

            int32_t eventIDVToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
            int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));

            // 小缓冲：tau/tauUb 放在 T_local 的**空闲后半段**。
            // 【修正】这几块原来在 workLocal[kSpan..]，而点积结果写在
            // workLocal[0 .. columns*curM)。只要 columns*curM > kSpan（即 qd > 4096）
            // 两者就重叠：tau 被点积结果覆写 -> Householder 反射子失效，
            // H 全错而 kernel 不报错（qd=4352 就已经 h_err≈5e-2）。
            // 之后一度挪到 householder_v 尾部，但那里与 wgAll 争地方，会把
            // batch 上限压到 14。T_local 每矩阵只用 blockDim*columns^2(=128) 个，
            // 而缓冲按每矩阵 256 个分配，后半段 [TStride*kMaxG, 256*kMaxG) 恒空闲，
            // 正好放 2*8*kMaxG 个（128+16=144 <= 256）。
            const int scBase = TStride * kMaxG;
            LocalTensor<float> tauLocal = T_local[scBase];                      // [gi][8] 未归一化 tau
            // （原来这里还有一个 [gi][8] 的 v0 缓冲：它只喂给已删除的两处死代码
            //   —— tau*v0^2 与 APad 的 Householder 向量归一化 —— 所以一并去掉。）
            // 【优化】tau 的跨核交换改成 MTE2 读：每 (k,矩阵) 一次 columns 个 float 的
            // DataCopyPad（取代 per-i 的 scalar GM 读 + 每 k 一次 DCCI）。
            // scalar GM 读要 ~170 ns/次（DCCI 后又强制 cache miss），MTE2 一次只需一次 DMA。
            LocalTensor<float> tauUb   = T_local[scBase + 8 * kMaxG];           // [gi][8] 本 panel 的 tau
            LocalTensor<float> hvW      = householder_v;                         // [gi][columns][8]
            LocalTensor<float> wgAll    = householder_v[8 * kMaxG * columns];    // [gi][TRowStride]
            LocalTensor<float> tmp8     = householder_v[8 * kMaxG * columns + kMaxG * TRowStride]; // 标量暂存

            int lineOffset[64];
            // 【修正1】原为 int tau_k[64]：tau 一般 < 1，赋值时被截断成 0，
            //         Phase2 的 Axpy 全部退化成“乘 0”，前序反射子根本没作用到本核列上。
            float tau_k[64];
            float wdot[8];        // Phase 2 批量归约：columns 个点积结果（columns <= 8）
            // BuildPlan allows up to kMaxG matrices in a chunk.
            int degFlag[kSdcBatchMax];
            float alpha[8 * 4];

            // APad 中各矩阵基址（等价于原 ComputeAll 里 batchOffset 的累加，含自身 size）
            int baseOf[64];
            baseOf[0] = size[0];
            for (int t = 1; t < batch; ++t) {
                baseOf[t] = baseOf[t - 1] + size[t];
            }
            Duplicate<float>(hvW, 0.0f, kMaxG * columns * 8);

            // ================= 切批：每批完整跑一遍所有 phase =================
            for (int pc = 0; pc < nPlan; ++pc) {
                const int chunkStart = planS[pc];
                const int chunkEnd   = planE[pc];
                const int Gnum       = chunkEnd - chunkStart;
                if (Gnum <= 0) { continue; }

                // 批内打包偏移（对应草稿里的 lineOffset / totallen / shapeNum）
                int totallen = 0;
                for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                    lineOffset[bidx] = totallen;
                    totallen += m[bidx] - offsetM;
                }
                for (int gi = 0; gi < Gnum; ++gi) {
                    degFlag[gi] = 0;
                    /*for (int i = 0; i < 8; ++i) {
                        tauLocal.SetValue(gi * 8 + i, 0.0f);
                    }*/
                    Duplicate<float>(tauLocal[gi * 8], 0.0f, 8);
                }
                Duplicate<float>(hvW, 0.0f, kMaxG * columns * 8);

                // ==================== Phase 1: 载入整批 ====================
                // 每矩阵每列各一次 DataCopyPad（blockCount=1，绕开 DataCopyPad 的
                // dstStride 在新旧 CANN 版本里单位不一致的问题）
                for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                    const int curM = m[bidx] - offsetM;
                    const int gPad = m[bidx];
                    for (int j = 0; j < columns; ++j) {
                        DataCopyParams cp{1, static_cast<uint16_t>(curM * sizeof(float)), 0, 0};
                        DataCopyPadParams pp{true, 0, 0, 0};
                        DataCopyPad(a[kSpan * j + lineOffset[bidx]],
                                    APad[baseOf[bidx] + gPad * offsetM + offsetM +
                                         gPad * (blockIdx * columns + j)], cp, pp);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

                // ========== Phase 2: 应用前序核的 Householder（批量 Mul）==========
                for (int k = 0; k != blockIdx; k++) {
                    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                    CrossCoreWaitFlag(0x0);
                    if (Gnum == 1) {
                        // ---- 单矩阵（本项目的情形）：一个 k 的 columns 列用一次多行 copy ----
                        const int curM1 = m[chunkStart] - offsetM;
                        const int gPad1 = m[chunkStart];
                        const int offsetY1 = flag * panelYSize + yOff[chunkStart];
                        DataCopyPadExtParams<float> noPad2{false, 0, 0, 0};
                        DataCopyExtParams cpY{static_cast<uint16_t>(columns),
                                              static_cast<uint32_t>(curM1 * 4),
                                              static_cast<uint32_t>((gPad1 - curM1) * 4),
                                              0, 0};
                        DataCopyPad(b, Y[offsetY1 + gPad1 * k * columns], cpY, noPad2);
                        // tau 也走 MTE2（同一批）
                        DataCopyExtParams cpTau1{1, static_cast<uint32_t>(columns * 4), 0, 0, 0};
                        DataCopyPad(tauUb, tau[tauOff[chunkStart] + k * columns + offsetM], cpTau1, noPad2);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
                        for (int i = 0; i != columns; i++) {
                            tau_k[chunkStart] = tauUb.GetValue(i);
                            // 【批量归约】同一个 Householder 向量 i 对 columns 列的点积
                            // 先把 Mul+ReduceSum 全部发出去（每个 j 占一个 workLocal slot），
                            // 再一次性读回标量，最后统一 Axpy —— 原来每个 (i,j) 都要一次
                            // GetValue 等向量流水（锁步），columns² 次 → columns 次。
                            // 注意不能跨 i 批：i 的 Axpy 会改 a，i+1 的点积必须用更新后的 a。
                            for (int j = 0; j != columns; j++) {
                                const int off = j * curM1;
                                Mul(workLocal[off], a[kSpan * j], b[i * curM1], curM1);
                                ReduceSum(workLocal[off], workLocal[off], workLocal[off], curM1);
                            }
                            for (int j = 0; j != columns; j++) {
                                wdot[j] = workLocal.GetValue(j * curM1);
                            }
                            for (int j = 0; j != columns; j++) {
                                Axpy(a[kSpan * j], b[i * curM1],
                                     -tau_k[chunkStart] * wdot[j], curM1);
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
                    } else {
                    // tau: 每 (k, 矩阵) 一次 MTE2（columns 个 float），替代 per-i scalar 读
                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        DataCopyExtParams cpTau{1, static_cast<uint32_t>(columns * 4), 0, 0, 0};
                        DataCopyPadExtParams<float> ppTau{false, 0, 0, 0};
                        DataCopyPad(tauUb[bidx * 8], tau[tauOff[bidx] + k * columns + offsetM],
                                    cpTau, ppTau);
                    }
                    for (int i = 0; i != columns; i++) {
                        for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                            const int curM = m[bidx] - offsetM;
                            const int gPad = m[bidx];
                            int offsetY = flag * panelYSize + yOff[bidx];
                            DataCopy(b[lineOffset[bidx]], Y[offsetY + gPad * k * columns + gPad * i], curM);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
                        for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                            tau_k[bidx] = tauUb.GetValue(bidx * 8 + i);
                        }
                        for (int j = 0; j != columns; j++) {
                            Mul(workLocal, a[kSpan * j], b, totallen);       // 批量
                            for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                                const int curM = m[bidx] - offsetM;
                                const int lo = lineOffset[bidx];
                                ReduceSum(workLocal[lo], workLocal[lo], workLocal[lo], curM);
                                Axpy(a[kSpan * j + lo], b[lo],
                                     -tau_k[bidx] * workLocal.GetValue(lo), curM);
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
                    }
                    }
                }

                // ====== Phase 3: Householder 形成 + 本核两列互相更新（批量 Mul）======
                for (int i = 0; i != columns; i++) {
                    const int m0 = blockIdx * columns + i;

                    // --- 列范数平方：批量 Mul + 逐矩阵分段规约 ---
                    Mul(workLocal, a[kSpan * i], a[kSpan * i], totallen);
                    if (m0 > 0) {
                        for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                            Duplicate<float>(workLocal[lineOffset[bidx]], 0, m0);
                        }
                    }
                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        const int curM = m[bidx] - offsetM;
                        ReduceSum(workLocal[lineOffset[bidx]], workLocal[lineOffset[bidx]],
                                  workLocal[lineOffset[bidx]], curM);
                    }

                    // --- 逐矩阵标量处理（sqrt / tau），与原实现逐条对应 ---
                    // v0（= x0 + alpha，Householder 向量的首元）只用在那两处已删除的
                    // 死代码里（tau*v0^2 与 APad 归一化），所以整块 v0 缓冲也一起去掉。
                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        const int gi = bidx - chunkStart;
                        const int lo = lineOffset[bidx];
                        float x0 = a.GetValue(kSpan * i + lo + m0);
                        float norm_2 = workLocal.GetValue(lo);
                        if (norm_2 == x0 * x0) {                       // 退化：该列已是 [-alpha, 0,...]
                            alpha[gi * columns + i] = -x0;
                            tauLocal.SetValue(gi * 8 + i, 0.0f);
                            degFlag[gi] = 1;
                        } else {
                            degFlag[gi] = 0; // The flag belongs to this column.
                            float norm_x = sqrt(norm_2);
                            float al = x0 < 0 ? -norm_x : norm_x;
                            alpha[gi * columns + i] = al;
                            a.SetValue(kSpan * i + lo + m0, x0 + al);
                            tauLocal.SetValue(gi * 8 + i, 1.0f / (norm_2 + x0 * al));
                        }
                    }

                    // --- hv：复用 b，= a 本列，前 m0 个清 0；退化矩阵整段清 0 ---
                    Adds(b, a[kSpan * i], 0.0f, totallen);
                    if (m0 > 0) {
                        for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                            Duplicate<float>(b[lineOffset[bidx]], 0, m0);
                        }
                    }
                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        const int gi = bidx - chunkStart;
                        if (degFlag[gi]) {
                            Duplicate<float>(b[lineOffset[bidx]], 0, m[bidx] - offsetM);
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        const int curM = m[bidx] - offsetM;
                        int offsetY = flag * panelYSize + yOff[bidx];
                        DataCopy(Y[offsetY + m[bidx] * (blockIdx * columns + i)],
                                 b[lineOffset[bidx]], curM);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);

                    // --- 用本列更新后面的列：a_j -= tau * <a_j, hv> * hv ---
                    for (int j = i + 1; j != columns; j++) {
                        Mul(workLocal, a[kSpan * j], b, totallen);
                        if (m0 > 0) {
                            for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                                Duplicate<float>(workLocal[lineOffset[bidx]], 0, m0);
                            }
                        }
                        for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                            const int curM = m[bidx] - offsetM;
                            ReduceSum(workLocal[lineOffset[bidx]], workLocal[lineOffset[bidx]],
                                      workLocal[lineOffset[bidx]], curM);
                        }
                        for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                            const int gi = bidx - chunkStart;
                            if (degFlag[gi]) { continue; }           // tau=0，等价于跳过
                            const int curM = m[bidx] - offsetM;
                            const int lo = lineOffset[bidx];
                            Axpy(a[kSpan * j + lo], b[lo],
                                 -tauLocal.GetValue(gi * 8 + i) * workLocal.GetValue(lo), curM);
                        }
                    }
                }
                // Partial DMA stores avoid scalar cache-line writeback collisions
                // between AIVs owning adjacent tau values.
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
                for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                    const int gi = bidx - chunkStart;
                    DataCopyParams cpTauOut{1, static_cast<uint16_t>(columns * sizeof(float)), 0, 0};
                    DataCopyPad(tau[tauOff[bidx] + blockIdx * columns + offsetM],
                                tauLocal[gi * 8], cpTauOut);
                }

                // 等待本核 v 向量拷贝过去
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                CrossCoreWaitFlag(0x0);

                // ========== Phase 4a: 本核 w_ij 计算 + 写 W（批量 Mul）==========
                for (int i = columns - 1; i >= 0; i--) {
                    for (int j = i - 1; j >= 0; j--) {
                        Mul(workLocal, a[kSpan * i], a[kSpan * j], totallen);
                        if (blockIdx * columns + i > 0) {
                            for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                                Duplicate<float>(workLocal[lineOffset[bidx]], 0, blockIdx * columns + i);
                            }
                        }
                        for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                            const int curM = m[bidx] - offsetM;
                            ReduceSum(workLocal[lineOffset[bidx]], workLocal[lineOffset[bidx]],
                                      workLocal[lineOffset[bidx]], curM);
                        }
                        for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                            const int gi = bidx - chunkStart;
                            hvW.SetValue((gi * columns + i) * 8 + j, workLocal.GetValue(lineOffset[bidx]));
                        }
                    }
                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        const int gi = bidx - chunkStart;
                        DataCopy(W[bidx * WStride + blockDim * 8 * i + blockIdx * 8 + (blockIdx % 2) * 1024],
                                 hvW[(gi * columns + i) * 8], 8);
                    }
                }
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                CrossCoreWaitFlag(0x0);

                // ============ Phase 4b: T 初始化（逐矩阵，标量为主）============
                for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                    const int gi = bidx - chunkStart;
                    LocalTensor<float> Tg = T_local[gi * TStride];
                    LocalTensor<float> wg = wgAll[gi * TRowStride];
                    Duplicate<float>(Tg, 0, blockDim * columns * columns);
                    Duplicate<float>(wg, 0, TRowStride);
                    // 设置为负，从而算出 -T；wg 存储“分散”的当前 T 的一行
                    for (int i = 0; i != columns; i++) {
                        float ti = tauLocal.GetValue(gi * 8 + i);
                        Tg.SetValue(blockDim * columns * i + i + blockIdx * columns, -ti);
                        wg.SetValue((blockDim * i * 8) + blockIdx * 8 + i, -ti);
                        for (int j = i + 1; j != columns; j++) {
                            Mul(tmp8, wg[blockDim * i * 8 + blockIdx * 8], hvW[(gi * columns + j) * 8], 8);
                            ReduceSum(tmp8, tmp8, tmp8, 8);
                            float tv = -tauLocal.GetValue(gi * 8 + j) * tmp8.GetValue(0);
                            Tg.SetValue(blockDim * columns * i + j + blockIdx * columns, tv);
                            wg.SetValue(blockDim * i * 8 + blockIdx * 8 + j, tv);
                        }
                    }
                }
                Duplicate<float>(hvW, 0.0f, kMaxG * columns * 8);

                // ========== Phase 4c: W 交换 + T 累积（barrier 每 k 整批一次）==========
                for (int k = blockIdx + 1; k != blockDim; k++) {
                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        const int gi = bidx - chunkStart;
                        const int curM = m[bidx] - offsetM;
                        const bool pairLoad = (512 + columns * curM <= chunkSize);
                        const int offsetY = flag * panelYSize + yOff[bidx];
                        if (pairLoad) {
                            DataCopyExtParams cpPair{static_cast<uint16_t>(columns),
                                static_cast<uint32_t>(curM * 4),
                                static_cast<uint32_t>((m[bidx] - curM) * 4), 0, 0};
                            DataCopyPadExtParams<float> noPadPair{false, 0, 0, 0};
                            DataCopyPad(b[512], Y[offsetY + m[bidx] * k * columns],
                                        cpPair, noPadPair);
                            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                        }
                        for (int i = 0; i != columns; i++) {
                            if (!pairLoad) {
                                DataCopy(b, Y[offsetY + m[bidx] * k * columns + m[bidx] * i], curM);
                                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                            }
                            const int vOff = pairLoad ? 512 + i * curM : 0;
                            for (int j = 0; j != columns; j++) {
                                const int off = j * curM;
                                Mul(workLocal[off], a[kSpan * j + lineOffset[bidx]], b[vOff], curM);
                                ReduceSum(workLocal[off], workLocal[off], workLocal[off], curM);
                            }
                            // 【批量归约】columns 个点积已经全部算完，再一次性读回（原来每个
                            // j 一次 GetValue，锁步 columns 次 → 1 次）
                            for (int j = 0; j != columns; j++) {
                                // 写回中间结果 w（只能分开写，连续地址写冲突严重）
                                hvW.SetValue((gi * columns + i) * 8 + j, workLocal.GetValue(j * curM));
                            }
                            // 【修正4】b 下一步会被覆盖，先等 V 读完，否则 MTE2 与 V 竞争
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
                            DataCopy(W[bidx * WStride + blockDim * 8 * i + blockIdx * 8 + (k % 2) * 1024],
                                     hvW[(gi * columns + i) * 8], 8);
                        }
                    }
                    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                    CrossCoreWaitFlag(0x0);

                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        DataCopyExtParams cpTau2{1, static_cast<uint32_t>(columns * 4), 0, 0, 0};
                        DataCopyPadExtParams<float> ppTau2{false, 0, 0, 0};
                        DataCopyPad(tauUb[bidx * 8], tau[tauOff[bidx] + k * columns + offsetM],
                                    cpTau2, ppTau2);
                    }
                    for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                        const int gi = bidx - chunkStart;
                        LocalTensor<float> Tg = T_local[gi * TStride];
                        LocalTensor<float> wg = wgAll[gi * TRowStride];
                        const int wCount = align_to(k * 8 + 8, 64);
                        DataCopyExtParams cpWPair{static_cast<uint16_t>(columns),
                            static_cast<uint32_t>(wCount * 4),
                            static_cast<uint32_t>((blockDim * 8 - wCount) * 4), 0, 0};
                        DataCopyPadExtParams<float> noPadW{false, 0, 0, 0};
                        DataCopyPad(b, W[bidx * WStride + (k % 2) * 1024], cpWPair, noPadW);
                        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                        for (int i = 0; i != columns; i++) {
                            // Preserve the sequential i dependency through wg.
                            // （nT = k*8+8 <= 256，所以每个 j 占 workLocal 的 256 float slot）
                            const int nT = k * 8 + 8;
                            for (int j = 0; j != columns; j++) {
                                const int off = j * 256;
                                Mul(workLocal[off], b[i * wCount], wg[blockDim * 8 * j], nT);
                                ReduceSum(workLocal[off], workLocal[off], workLocal[off], nT);
                            }
                            const float tauv = tauUb.GetValue(bidx * 8 + i);
                            for (int j = 0; j != columns; j++) {
                                float tv = -tauv * workLocal.GetValue(j * 256);
                                Tg.SetValue(blockDim * columns * j + i + k * columns, tv);
                                wg.SetValue(blockDim * 8 * j + k * 8 + i, tv);
                            }
                            // 【修正4】(第二处) 同上，b 下一轮会被覆盖
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
                        }
                    }
                }

                // ============ Phase 5: T 写回（逐矩阵）============
                for (int bidx = chunkStart; bidx < chunkEnd; ++bidx) {
                    const int gi = bidx - chunkStart;
                    DataCopy(T[columns * columns * blockDim * blockIdx + flag * panelTSize + bidx * 256 * 128],
                             T_local[gi * TStride], blockDim * columns * columns);
                }
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                CrossCoreWaitFlag(0x0);

                // 【删除】原来这里把 tau 再乘 v0^2 并存回 GM 一遍：
                //   tau <- tau * v0 * v0     （2 次 8-float Mul）
                //   DataCopyPad(tau[...], tauLocal, columns)
                // 但 tau 的唯一读者是本 panel 内 Phase 2 / Phase 4c 的跨核交换
                //（读的是 Phase 3 写进去的**未归一化** tau），AIC 侧（qr_custom.cpp）
                // 从不读 tau（只用 A / Y / W / T），后续 panel 也用不到这个槽位。
                // 所以这两次 Mul + 这次写回是死代码，删掉（结果逐位不变，已实测）。


                // The fused pipeline consumes Q only. Completed APad columns
                // (including R) have no readers; reflectors are published in Y/T.
                // Do not copy the private panel back to GM.

            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);

            tQueueDst.FreeTensor<float>(T_local);
            aQueueOut.FreeTensor<float>(householder_v);
            bQueueIn.FreeTensor<float>(b);
            aQueueIn.FreeTensor<float>(a);
            // 等待下一个列块更新完毕
        }
    private:
        int M, N, padM, padN, panelYSize, panelTSize;
        int forcedM = 0, forcedN = 0;
        bool shapesForced = false;
        bool shapesBatch = false;      // SetDimsBatch 已就位（InitShapes 直接返回）
        const int chunkSize = kQrChunkSize;  // 8192：见 fused_batch_cfg.hpp
        // 批内矩阵数上限：必须与 kSdcBatchMax 一致，tQueueDst / householder_v
        // 的尺寸都由它决定（上限已由 UB 预算在 fused_batch_cfg.hpp 里推导并 static_assert）
        const int kMaxG = kSdcBatchMax;
        TQue<QuePosition::VECIN, 1> aQueueIn;
        TQue<QuePosition::VECIN, 1> bQueueIn;
        TQue<QuePosition::VECOUT, 1> aQueueOut;
        TBuf<QuePosition::VECCALC> workQueue;
        TQue<QuePosition::VECOUT, 1> tQueueDst;
        TQue<QuePosition::VECOUT, 2> tauQueueOut;

        GlobalTensor<float> APad;
        GlobalTensor<float> tau;
        GlobalTensor<float> T;
        GlobalTensor<float> W;
        GlobalTensor<float> Y;
        GlobalTensor<int32_t> P;
        GlobalTensor<float> QdwhA;
        GlobalTensor<float> QdwhA0;
        GlobalTensor<float> QdwhQ;
        GlobalTensor<float> QdwhHOut;
        GlobalTensor<float> QdwhCoef;
        GlobalTensor<float> QdwhPt;    // P^T（cube 侧 Pmat 顺带产出）
        GlobalTensor<float> QdwhXt;    // X^T（AIV 侧维护，Prepare 连续读）
        bool hasPt = false;
        bool hasXt = false;
        float alpha = 1.0f;
        float lArr[64];
        float cProd[64];
        float kappaArr[64];      // betaPrime for Q1 scaling and legacy AIV update
        int max_M;
        int maxK;
        int batch;
        int size[64], m[64], n[64];
        int tauOff[64], yOff[64];
    };
}