#ifdef DUMMY
#define __CCE_AICORE__ 220
#define __DAV_C220_CUBE__
#define __DAV_C220_VEC__
#include <__clang_cce_aicore_functions.h>
#include <tikicpulib.h>
#endif

#include <kernel_operator.h>
#include "fused_batch_cfg.hpp"     // 同层 batch 的 W/T/Y 布局常量（host/device 共用）

using namespace AscendC;
//static constexpr int M_threshold = 2176;
static constexpr int M_threshold = -1;

// QDWH 权重需要在 vec.hpp / gemm.cpp 之前定义，方便 qdwh_vec 直接调用。
__aicore__ inline float cube_root(float x) {
    if (x == 0.0f) return 0.0f;
    
    // 处理负数
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float abs_x = abs(x);
    
    // 更好的初始猜测。QDWH 首次迭代的 d 可到 1e24 量级，
    // 用 abs_x/3 需要近百次牛顿迭代；sqrt(abs_x) 可把迭代次数降到 ~20。
    float guess = (abs_x > 1.0f) ? sqrt(abs_x) : abs_x;
    
    // 牛顿迭代：y = (2*y + x/y^2) / 3
    for (int i = 0; i < 200; i++) {
        float prev = guess;
        // 分步计算 x / (guess * guess)，避免 guess*guess 下溢为 0
        float tmp = abs_x / guess;
        tmp /= guess;
        guess = (2.0f * guess + tmp) / 3.0f;
        
        // 检查收敛
        if (abs(guess - prev) < 1e-7f * abs(guess)) {
            break;
        }
    }
    
    return sign * guess;
}

__aicore__ inline void compute_weight(float &a, float &b, float &c, float l) {
    
    float l2 = l * l;//10 8
    float l4 = l2 * l2;//10 16
    float d = 4 * (1 - l2) / l4;//4 10 16
    d = cube_root(d);//3 10 5
    float t = sqrt(1 + d);//50
    a = 8 * (2 - l2) / l2;//16 10 8
    a = a / t;//3 10 7
    a = a + 8 - 4 * d;//3 10 7
    a = t + 0.5f * sqrt(a);//
    //a = t + 0.5f * sqrt(8 * (2 - l2) / (l2 * t) - 4 * d + 8);
    b = (a - 1) * (a - 1) / 4;
    c = a + b - 1;
}

#include "gemm.cpp"
#include "vec.hpp"

__aicore__ inline int gcd(int a, int b) {
    while (b) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a < 0 ? -a : a;
}

namespace AscendC
{
    class qdwh_cube{
        public:
            __aicore__ inline void Init(GM_ADDR a, GM_ADDR tau, GM_ADDR T, GM_ADDR w, GM_ADDR y, GM_ADDR q, GM_ADDR p, GM_ADDR eye, int batch){
                this->batch = batch;
                A.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
                Tmat.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(T));
                // W 的低位是 AIV（qdwh_vec）的交换区（batchMax*2048），之后留 gap 再给 AIC
                W.SetGlobalBuffer((reinterpret_cast<__gm__ float *>(w)) + kQdwhWVecTotal + kQdwhWGap);
                Y.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y));
                Q.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(q));
                P.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p));
                matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eye));
                opmm.Init(&pipe);
                ReadShapes();
            }

            // 不读 P (host int 数组可能 cache 不一致)，直接指定单矩阵的 m/n。
            __aicore__ inline void ResetMatrixDims(GM_ADDR a, GM_ADDR q, int mm, int nn, int batch_)
            {
                this->batch = batch_;
                A.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
                Q.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(q));
                m[0] = mm;
                n[0] = nn;
                size[0] = 0;
                max_M = mm;
                max_N = nn;
                maxK = min(mm, nn);
                yOff[0] = 0;
                qOff[0] = 0;
                panelYSize = 64 * mm;
            }

            __aicore__ inline MatmulCustom<float, float, float> &Mm()
            {
                return opmm;
            }

            // ------------------------------------------------------------------
            // 【batch QR】直接给定批内每个矩阵的 QR 输入形状（m[b] = 2*nPad[b]，
            // n[b] = nPad[b]），并重算 size[]/yOff[]/qOff[]/max_M/max_N/maxK/
            // panelYSize。Init 里 ReadShapes() 读到的 P 值会被完全覆盖。
            // ------------------------------------------------------------------
            __aicore__ inline void SetDimsBatch(const int *mArr, const int *nArr, int cnt)
            {
                batch = cnt;
                int ySizeAll = 0;
                int qSizeAll = 0;
                max_M = 0;
                max_N = 0;
                maxK = 0;
                for (int k = 0; k < cnt; ++k) {
                    m[k] = mArr[k];
                    n[k] = nArr[k];
                    size[k] = (k == 0) ? 0 : m[k - 1] * n[k - 1];
                    yOff[k] = ySizeAll;
                    ySizeAll += 64 * m[k];
                    qOff[k] = qSizeAll;
                    qSizeAll += m[k] * m[k];
                    max_M = max(max_M, m[k]);
                    max_N = max(max_N, n[k]);
                    maxK = max(maxK, min(m[k], n[k]));
                }
                panelYSize = ySizeAll;
            }

            __aicore__ inline void ReadShapes()
            {
                // 与 vec 侧完全一致：P[2k+1] 为 QR 输入行数，P[2k] 为列数
                m[0] = P.GetValue(1);
                n[0] = P.GetValue(0);
                size[0] = 0;
                max_M = m[0];
                max_N = n[0];
                maxK = min(m[0], n[0]);
                yOff[0] = 0;
                qOff[0] = 0;
                int ySizeAll = 64 * m[0];
                int qSizeAll = m[0] * m[0];
                for (int k = 1; k < batch; ++k) {
                    m[k] = P.GetValue(2 * k + 1);
                    n[k] = P.GetValue(2 * k);
                    size[k] = m[k - 1] * n[k - 1];
                    yOff[k] = ySizeAll;
                    ySizeAll += 64 * m[k];
                    qOff[k] = qSizeAll;
                    qSizeAll += m[k] * m[k];
                    max_M = max(max_M, m[k]);
                    max_N = max(max_N, n[k]);
                    maxK = max(maxK, min(m[k], n[k]));
                }
                panelYSize = ySizeAll;
            }
            // ---------------- QDWH device 侧方法 ----------------
            __aicore__ inline void SetQdwhBuffers(GM_ADDR x_, GM_ADDR a0_, GM_ADDR coef_,
                                                GM_ADDR pt_ = nullptr,
                                                GM_ADDR xt_ = nullptr)
            {
                X.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x_));
                A0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a0_));
                Coef.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(coef_));
                hasPt = (pt_ != nullptr);
                if (hasPt) {
                    Pt.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pt_));
                }
                hasXt = (xt_ != nullptr);       // 与 hasPt 同开同关（都是 X^T 缓存）
                if (hasXt) {
                    Xt.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xt_));
                }
            }

            // ------------------------------------------------------------------
            // Initialize Xt using 128x128 identity-times-transpose tiles.
            // Eye is the compact 128x128 identity; X has ld=N and Xt has ld=M.
            // The GEMM C stride is M while blockN is the tile's true column count.
            // 调用前必须保证 AIV 已把 X 写完（0x5），调用后由调用方 AicBarrier 再放 0x4。
            // ------------------------------------------------------------------
            __aicore__ inline void QdwhInitXt(GM_ADDR xt_, GM_ADDR eye128)
            {
                Xt.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xt_));
                Eye.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eye128));
                int off=0;
                const int bi=GetBlockIdx(), bn=GetBlockNum();
                for(int b=0; b<batch; ++b) {
                    const int N=n[b], M=m[b]-N;
                    const int nr=(N+127)/128, nc=(M+127)/128;
                    opmm.SetMatrixLd(Eye, X[off], Xt[off], M, 128, 128, N);
                    for(int t=bi; t<nr*nc; t+=bn) {
                        const int r=t/nc*128, c=t%nc*128;
                        const int rows=min(128,N-r), cols=min(128,M-c);
                        opmm.Process(0, c*N+r, r*M+c,
                                     rows, cols, rows, -1, true, false);
                    }
                    off+=M*N;
                }
            }

            // Compatibility mode produces P/Pt. The fused path calls (true,true):
            // Xt += Q2 * scaledQ1^T, with scaledQ1 = betaPrime * Q1.
            // Q has ld=qd; Xt has ld=M, where M=qd-N.
            __aicore__ inline void QdwhPmat(bool atomicY = false, bool onlyT = false)
            {
                const int blockNum = GetBlockNum();
                const int blockIdx = GetBlockIdx();

                int qOffArr[64], hOffArr[64], MArr[64], NArr[64], tileOff[64];
                int xOffArr[64], tOffArr[64];
                int totalTiles = 0, qAcc = 0, hAcc = 0, xAcc = 0;
                for (int b = 0; b < batch; ++b) {
                    const int qd = m[b];
                    const int N = n[b];
                    const int M = qd - N;
                    MArr[b] = M;
                    NArr[b] = N;
                    qOffArr[b] = qAcc;
                    hOffArr[b] = hAcc;
                    xOffArr[b] = xAcc;      // X（M x N, ld = N）
                    tOffArr[b] = xAcc;      // Xt（N x M, ld = M），与 X 同大小
                    tileOff[b] = totalTiles;
                    totalTiles += (M + 127) / 128;
                    qAcc += qd * qd;
                    hAcc += N * qd;
                    xAcc += M * N;
                }

                if (!onlyT) {
                for (int t = blockIdx; t < totalTiles; t += blockNum) {
                    int b = 0;
                    while (b + 1 < batch && t >= tileOff[b + 1]) {
                        ++b;
                    }
                    const int qd = m[b];
                    const int N = NArr[b];
                    const int M = MArr[b];
                    const int rowStart = (t - tileOff[b]) * 128;
                    const int rows = min(128, M - rowStart);

                    opmm.SetMatrixLd(Q[qOffArr[b]],
                                     Q[qOffArr[b] + M * qd],
                                     atomicY ? X[xOffArr[b]] : A[hOffArr[b]],
                                     N, N, qd, qd);
                    // K 方向按 128 分块，后续块 atomic add 累加。
                    for (int kc = 0; kc < N; kc += 128) {
                        const int kk = min(128, N - kc);
                        opmm.Process(rowStart * qd + kc, kc, rowStart * N,
                                     rows, N, kk, -1, true, atomicY || kc != 0);
                    }
                }

                }

                // Pt = Q2*Q1^T, or direct atomic accumulation into Xt.
                // C is N x M with stride M.
                // 【fix】Pt 和 X 一样是"按批切片"的（每矩阵 M*N 个 float），原来写死
                // Pt[pAcc] 会让 batch>1 时所有矩阵都写第 0 块；这里改用 xOffArr[b]
                // （与 X 的切片偏移一致，AIV 侧的 QdwhPt[Xt] 也是按同样偏移读的）。
                if (hasPt) {
                    int tileOffT[64];
                    int totalTilesT = 0;
                    for (int b = 0; b < batch; ++b) {
                        tileOffT[b] = totalTilesT;
                        totalTilesT += (NArr[b] + 127) / 128;
                    }
                    for (int t = blockIdx; t < totalTilesT; t += blockNum) {
                        int b = 0;
                        while (b + 1 < batch && t >= tileOffT[b + 1]) {
                            ++b;
                        }
                        const int qd = m[b];
                        const int N = NArr[b];
                        const int M = MArr[b];
                        const int rowStart = (t - tileOffT[b]) * 128;
                        const int rows = min(128, N - rowStart);

                        opmm.SetMatrixLd(Q[qOffArr[b] + M * qd],
                                         Q[qOffArr[b]],
                                         atomicY ? Xt[xOffArr[b]] : Pt[xOffArr[b]],
                                         M, N, qd, qd);
                        for (int kc = 0; kc < N; kc += 128) {
                            const int kk = min(128, N - kc);
                            opmm.Process(rowStart * qd + kc, kc, rowStart * M,
                                         rows, M, kk, -1, true, atomicY || kc != 0);
                        }
                    }
                }
            }

            // Reconstruct X once per QDWH group from the final Xt.
            // Each 128x128 output tile is an identity-times-transpose GEMM:
            // A is the reusable local 128x128 identity, B is an Xt tile.
            // No A-transpose path and no full-width reduction are needed.
            __aicore__ inline void QdwhRestoreX(GM_ADDR eye128)
            {
                Eye.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eye128));
                int off=0;
                const int bi=GetBlockIdx(), bn=GetBlockNum();
                for(int b=0; b<batch; ++b) {
                    const int N=n[b], M=m[b]-N;
                    const int nr=(M+127)/128, nc=(N+127)/128;
                    opmm.SetMatrixLd(Eye, Xt[off], X[off], N, 128, 128, M);
                    for(int t=bi; t<nr*nc; t+=bn) {
                        const int r=t/nc*128, c=t%nc*128;
                        const int rows=min(128,M-r), cols=min(128,N-c);
                        opmm.Process(0, c*M+r, r*N+c,
                                     rows, cols, rows, -1, true, false);
                    }
                    off+=M*N;
                }
            }

            // 所有 AIC 的全局 barrier（mode 0）。用 0x7，避开 QR 内部的 0x0~0x3。
            __aicore__ inline void AicBarrier()
            {
                CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
                CrossCoreWaitFlag(0x7);
            }

            __aicore__ inline void QdwhFinalGemm()
            {
                const int blockNum = GetBlockNum();
                const int blockIdx = GetBlockIdx();

                int qOffArr[64], xOffArr[64], aOffArr[64], MArr[64], NArr[64], tileOff[64];
                int totalTiles = 0, qAcc = 0, xAcc = 0, aAcc = 0;
                for (int b = 0; b < batch; ++b) {
                    const int qd = m[b];
                    const int N = n[b];
                    const int M = qd - N;
                    MArr[b] = M;
                    NArr[b] = N;
                    qOffArr[b] = qAcc;
                    xOffArr[b] = xAcc;      // U 本体（X）里的偏移
                    aOffArr[b] = aAcc;
                    tileOff[b] = totalTiles;
                    totalTiles += (N + 127) / 128;
                    qAcc += qd * qd;
                    xAcc += M * N;
                    aAcc += M * N;
                }

                for (int t = blockIdx; t < totalTiles; t += blockNum) {
                    int b = 0;
                    while (b + 1 < batch && t >= tileOff[b + 1]) {
                        ++b;
                    }
                    const int N = NArr[b];
                    const int M = MArr[b];
                    const int rowStart = (t - tileOff[b]) * 128;
                    const int rows = min(128, N - rowStart);

                    // A = U 本体（M x N，ld = N），isTransposeA 让 cube 转置成 U^T
                    opmm.SetMatrix2(X[xOffArr[b]], A0[aOffArr[b]],
                                    Q[qOffArr[b]], N, N, M);
                    for (int kc = 0; kc < M; kc += 128) {
                        const int kk = min(128, M - kc);
                        // 转置形态下 offsetA 是“列偏移 + k 块行偏移”：rowStart + kc * M
                        opmm.Process(rowStart + kc * N, kc * N, rowStart * N,
                                     rows, N, kk, -1, false, kc != 0, /*isTransposeA=*/true);
                    }
                }
            }

            // Physical row-major update: C += (C*Y^T)*(T*Y).
            // Cache is beyond both AIV Y slots and has the same row strides.
            __aicore__ inline void BuildQrTy(int qpoff, int pflag)
            {
                int tileBase=0;
                for(int b=0;b<batch;++b) {
                    if(qpoff>=min(m[b],n[b])) continue;
                    const int rows=m[b], curM=rows-qpoff;
                    const int nt=(curM+127)/128;
                    const int first=(blockIdx-tileBase%blockDim+blockDim)%blockDim;
                    opmm.SetMatrix(Tmat[pflag*128*128+b*256*128],
                                   Y[pflag*panelYSize+yOff[b]],
                                   Y[2*panelYSize+yOff[b]],rows,64);
                    for(int t=first;t<nt;t+=blockDim) {
                        const int rc=t*128, rr=min(128,curM-rc);
                        opmm.Process(0,rc,rc,64,rr,64,-1,false,false);
                    }
                    tileBase+=nt;
                }
                CrossCoreSetFlag<0x0,PIPE_FIX>(0x3);
                CrossCoreWaitFlag(0x3);
            }

            // The next-panel projection partitions output rows, never K.
            // A core owns an 8x64 result and retains its ordered sum in L0C.
            __aicore__ inline void UpdateNearProjection(int offset,int pflag)
            {
                const int ws=128*max(max_M,max_N);int abase=0,taskBase=0;
                for(int b=0;b<batch;++b) {
                    abase+=size[b];if(m[b]<=offset || n[b]<=offset) continue;
                    const int rows=m[b],rem=n[b]-offset,curM=rows-offset+64;
                    const int near=(rem%128)?64:min(128,rem),nt=(near+7)/8;
                    const int first=(blockIdx-taskBase%blockDim+blockDim)%blockDim;
                    opmm.SetMatrix(A[abase+offset*rows+offset-64],Y[pflag*panelYSize+yOff[b]],W[b*ws],64,rows);
                    for(int t=first;t<nt;t+=blockDim) {
                        const int c0=t*8;opmm.ProjectLongK(c0*rows,0,c0*64,min(8,near-c0),curM);
                    }
                    taskBase+=nt;
                }
            }

            // All independent row bands are projected first, then updated.
            // Both passes use identical ownership and disjoint A/Q halves of W.
            // A single local FIX->MTE2 fence publishes every private projection;
            // only completion of the entire panel needs an all-AIC barrier.
            __aicore__ inline void UpdateOwnedPanels(int offset,int pflag,bool updateA)
            {
                const int ws=128*max(max_M,max_N),qs=64*max(max_M,max_N);
                const int ev=static_cast<int>(pipe.FetchEventID(HardEvent::FIX_MTE2));
                for(int stage=0;stage<2;++stage) {
                    int taskBase=0;
                    for(int kind=updateA?0:1;kind<2;++kind) {
                        int abase=0;
                        for(int b=0;b<batch;++b) {
                            abase+=size[b];
                            const int qpoff=offset-64;
                            if(qpoff>=min(m[b],n[b])) continue;
                            if(kind==0 && (m[b]<=offset || n[b]<=offset)) continue;
                            const int rows=m[b],curM=rows-qpoff;
                            const int rem=max(0,n[b]-offset);
                            const int near=(rem%128)?64:min(128,rem);
                            const int cols=kind==0?rem-near:rows;
                            if(cols<=0) continue;
                            const int band=kind==0?64:128;
                            const int nt=(cols+band-1)/band;
                            const int first=(blockIdx-taskBase%blockDim+blockDim)%blockDim;
                            // Near GEMM consumers may still be reading W's prefix.
                            // Keep tail projections behind it in the A half;
                            // the final panel barrier protects next-panel reuse.
                            const int wbase=b*ws+(kind==0?near*64:qs);
                            const int cbase=kind==0?abase+(offset+near)*rows+qpoff:qOff[b]+qpoff;
                            GlobalTensor<float> C=kind==0?A[cbase]:Q[cbase];
                            for(int t=first;t<nt;t+=blockDim) {
                                const int c0=t*band,bn=min(band,cols-c0);
                                if(stage==0) {
                                    opmm.SetMatrix(C[c0*rows],Y[pflag*panelYSize+yOff[b]],
                                                   W[wbase+c0*64],64,rows);
                                    const int firstK=(kind==0 && curM%128)?64:min(128,curM);
                                    opmm.Process(0,0,0,bn,64,firstK,-1,true,false);
                                    for(int kc=firstK;kc<curM;kc+=128) {
                                        opmm.Process(kc,kc,0,bn,64,min(128,curM-kc),-1,true,true);
                                    }
                                } else {
                                    opmm.SetMatrix(W[wbase+c0*64],Y[2*panelYSize+yOff[b]],
                                                   C[c0*rows],rows,64);
                                    for(int rc=0;rc<curM;rc+=256) {
                                        opmm.Process(0,rc,rc,bn,min(256,curM-rc),64,-1,false,true);
                                    }
                                }
                            }
                            taskBase+=nt;
                        }
                    }
                    if(stage==0) {
                        SetFlag<HardEvent::FIX_MTE2>(ev);
                        WaitFlag<HardEvent::FIX_MTE2>(ev);
                    }
                }
                // Shared TY and W may be reused by the next panel only after
                // every core has finished reading them and writing A/Q.
                CrossCoreSetFlag<0x0,PIPE_FIX>(0x3);
                CrossCoreWaitFlag(0x3);
            }

            __aicore__ inline void UpdateQPanel(int qpoff,int pflag)
            {
                UpdateOwnedPanels(qpoff+64,pflag,false);
            }

            __aicore__ inline void qr_update(int qrFlag = 0x1){
                int panelTSize = 128 * 128;
                // W 中每矩阵要容纳 u_cubes*64 行（u_cubes 可远大于 2），
                // 不能只按 128 行预留，否则剩余列更新会写到下一个矩阵的 W。
                const int WStride = 128 * max(max_M, max_N);
                //int panelYSize = 128 * max_M;
                int vector_blockdim, loop, change_loop, M_tail, Cstart, remainingCols,MtoCompute;
                int batchOffset;
                int updateCols;
                int updateCols2,remainingCols2,MtoCompute2, Cstart2;
                int M;
                int stride = 128;

                vector_blockdim = 64;
                int flag = 0;
                int offset = vector_blockdim;

                for(; offset < maxK; offset += vector_blockdim) {
                    CrossCoreWaitFlag(qrFlag);
                    // Near projection and TY have independent outputs. The
                    // completion barrier inside BuildQrTy publishes both.
                    UpdateNearProjection(offset,flag);
                    BuildQrTy(offset-vector_blockdim,flag);

                    // 计算C = C - Y * W
                    int totalTiles = 0;
                    for (int b = 0; b < batch; ++b) {
                        M = m[b];
                        if (M <= offset || n[b] <= offset) {
                            continue;                       // 该 batch 不产生 tile
                        }
                        int MtoComputeB = M - offset + 64;
                        totalTiles += (MtoComputeB + 63) / 64;
                    }

                    // ---- 按核号连续均分全局任务流 ----
                    int baseTiles   = totalTiles / blockDim;
                    int remainTiles = totalTiles % blockDim;
                    int myStartTile = blockIdx * baseTiles + min(blockIdx, remainTiles);
                    int myNumTiles  = baseTiles + (blockIdx < remainTiles ? 1 : 0);

                    // ---- Pass 2：只处理落在 [myStartTile, myStartTile + myNumTiles) 内的 tile ----
                    batchOffset = 0;
                    int globalTile = 0;   // 当前 batch 的第一个 tile 在全局任务流中的位置
                    for (int batchIdx = 0; batchIdx < batch; ++batchIdx) {
                        M = m[batchIdx];
                        batchOffset += size[batchIdx];
                        if (M <= offset || n[batchIdx] <= offset) {
                            continue;                       // 与原先一致
                        }

                        Cstart        = offset * M + offset - 64 + batchOffset;
                        remainingCols = n[batchIdx] - offset;
                        MtoCompute    = M - offset + 64;
                        updateCols    = (remainingCols > 128) ? 128 : remainingCols;
                        if (remainingCols % 128) {
                            updateCols = 64;
                        }
                        int m_cubes = (MtoCompute + 63) / 64;

                        // 本 batch 占据全局 tile 区间 [globalTile, globalTile + m_cubes)
                        int tileBegin = max(myStartTile, globalTile);
                        int tileEnd   = min(myStartTile + myNumTiles, globalTile + m_cubes);

                        if (tileBegin < tileEnd) {
                            // 对应原先的 my_start_m_block / my_num_m_blocks
                            int localStart = tileBegin - globalTile;
                            int localNum   = tileEnd - tileBegin;

                            // 计算 W = Y^T * C：每个核可能横跨多个 batch，切换 batch 时重设矩阵
                            opmm.SetMatrix(W[batchIdx * WStride],
                                        Y[2 * panelYSize + yOff[batchIdx]],
                                        A[Cstart], M, vector_blockdim);
                            int my_blockM = min(localNum * 64, MtoCompute - 64 * localStart);
                            opmm.Process(0, localStart * 64, localStart * 64, updateCols, my_blockM,
                                        vector_blockdim, -1, false, true);
                        }

                        globalTile += m_cubes;
                    }

                    // Each AIC publishes its completed near-panel writes.
                    // All AIVs rendezvous after their paired mode2 wait, before
                    // loading APad; no all-AIC publication barrier is needed.
                    CrossCoreSetFlag<0x2, PIPE_FIX>(qrFlag + 1);

                    UpdateOwnedPanels(offset,flag,true);
                    flag ^= 1;
                }
                CrossCoreWaitFlag(qrFlag);
                BuildQrTy(maxK-vector_blockdim,((maxK/vector_blockdim)-1)&1);

                // ------------------------------------------------------------------
                // 尾列收尾：对 M>N 的矩阵，B=A^T 的列数 M 大于行数 N=k，
                // 最后一个 panel（列 k-64..k-1）的 Householder 还需要作用到
                // 列 k..M-1 上。主循环 offset<maxK 不会执行 offset=k 这一趟，
                // 因此这里等所有 vec panel 完成后，用最后一个 panel 的 Y/T 补一次。
                // ------------------------------------------------------------------
                // Both phases cover the whole batch. The global task index is
                // continuous across matrices so short tails use all AICs.
                int tailTasks = 0;
                for (int b = 0; b < batch; ++b)
                    if (n[b] > m[b])
                        tailTasks += (n[b] - m[b] + vector_blockdim - 1) / vector_blockdim;
                if (tailTasks > 0) {
                    for (int stage = 0; stage < 2; ++stage) {
                        int abase = 0, taskBase = 0;
                        for (int b = 0; b < batch; ++b) {
                            abase += size[b];
                            const int rows = m[b], cols = n[b];
                            if (cols <= rows) continue;
                            const int k = rows, upd = cols - k;
                            const int nt = (upd + vector_blockdim - 1) / vector_blockdim;
                            const int pflag = ((k / vector_blockdim) - 1) & 1;
                            const int cbase = abase + k * rows + (k - vector_blockdim);
                            const int first = (blockIdx - taskBase % blockDim + blockDim) % blockDim;
                            for (int t = first; t < nt; t += blockDim) {
                                const int c0 = t * vector_blockdim;
                                const int bn = min(vector_blockdim, upd - c0);
                                if (stage == 0) {
                                    opmm.SetMatrix(A[cbase + c0 * rows],
                                                   Y[pflag * panelYSize + yOff[b]],
                                                   W[b * WStride + c0 * vector_blockdim],
                                                   vector_blockdim, rows);
                                    opmm.Process(0, 0, 0, bn, vector_blockdim, vector_blockdim,
                                                 -1, true, false);
                                } else {
                                    opmm.SetMatrix(W[b * WStride + c0 * vector_blockdim],
                                                   Y[2 * panelYSize + yOff[b]],
                                                   A[cbase + c0 * rows], rows, vector_blockdim);
                                    opmm.Process(0, 0, 0, bn, vector_blockdim, vector_blockdim,
                                                 -1, false, true);
                                }
                            }
                            taskBase += nt;
                        }
                        // Publish all W before any A update; then retire all
                        // updates before the subsequent Q phase reuses W/Y.
                        CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
                        CrossCoreWaitFlag(0x3);
                    }
                }

                // ------------------------------------------------------------------
                // Q 的最后一个 panel：主循环 offset<maxK 只到 p=maxK/64-2，
                // 对 k==maxK 的矩阵，最后一个 panel 的 Q 更新不会执行。
                // 这里等 vec 全部完成后，用最后一个 panel 的 Y/T 补一次。
                // ------------------------------------------------------------------
                UpdateQPanel(maxK - vector_blockdim,
                             ((maxK / vector_blockdim) - 1) & 1);
            }
        private:
            TPipe pipe;
            MatmulCustom<float, float, float> opmm;
            GlobalTensor<float> A;
            GlobalTensor<float> Tmat;
            GlobalTensor<float> W;
            GlobalTensor<float> W2;
            GlobalTensor<float> Y;
            GlobalTensor<float> Q;
            GlobalTensor<float> X;
            GlobalTensor<float> A0;
            GlobalTensor<float> Coef;
            GlobalTensor<float> Pt;      // P^T（缓冲保留，原子加方案下不再写）
            GlobalTensor<float> Xt;      // X^T（原子加目标）
            GlobalTensor<float> Eye;     // 单位阵（QdwhInitXt 的 A 操作数）
            bool hasPt = false;
            bool hasXt = false;
            GlobalTensor<int32_t> P;
            GlobalTensor<float> matrix;
            int batch;
            int blockDim = GetBlockNum();
            int blockIdx = GetBlockIdx();
            int size[64],m[64],n[64];
            int max_M;
            int max_N;
            int maxK;
            int panelYSize;
            int yOff[64];
            int qOff[64];

    };
}

#ifndef QDWH_BUILD
extern "C" __global__ __aicore__ void qr_custom(GM_ADDR aPad, GM_ADDR tau, GM_ADDR T, GM_ADDR W, GM_ADDR Y, GM_ADDR Q, GM_ADDR H, GM_ADDR P,GM_ADDR matrix,int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    //batch = 1;
    #ifdef __DAV_C220_VEC__
        TPipe pipe;
        TBufPool<TPosition::VECCALC, 16> tbufPool;
        pipe.InitBufPool(tbufPool, 192 * 1024);
        AscendC::qdwh_vec op;
        op.Init(&tbufPool);
        op.SetMatrix(aPad, tau, T, W, Y, P,batch);
        op.ComputeAll(GetBlockNum() << 1, GetBlockIdx());
    #elif  __DAV_C220_CUBE__
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x2);
        AscendC::qdwh_cube op;
        op.Init(aPad, tau, T, W, Y, Q, P, matrix, batch);
        op.qr_update();
        //update_c(aPad, tau, T, W, Y, Q, matrix, batch);
    #endif
}
#endif // QDWH_BUILD

// ============================================================================
// 单 kernel 完成的 batch QDWH 极分解。
//
// 所有步骤（alpha 缩放、H 构造、Q 重置、l/权重递推、QR、P=Q1Q2^T、
// A 更新、最终 H=U^T A0）都在 device 侧完成；host 只做一次 launch。
//
// 同步协议（全部使用 CrossCoreSetFlag/WaitFlag，适配 kernel 直调工程）：
//   0x4 (AIC -> AIV, mode2)：允许 AIV 做下一轮 prepare / 最后的 transpose
//   0x5 (AIV -> AIC, mode2)：AIV 的 transpose 完成
//   0x6 (AIC -> AIV, mode2)：AIC 的 final GEMM 完成
//   0x7 (AIC 内部, mode0)  ：AIC 全核 barrier
//   0x8 (AIV 内部, mode0)  ：AIV 全核 barrier
//   QR 内部继续使用 qr_custom 原有的 0x0/0x1/0x2/0x3。
