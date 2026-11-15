#ifdef DUMMY
#define __CCE_AICORE__ 220
#include <__clang_cce_aicore_functions.h>
#include <tikicpulib.h>
#endif

#include <kernel_operator.h>

__aicore__ inline constexpr int ceil_div(int x, int y)
{
    return (x - 1) / y + 1;
}

__aicore__ inline constexpr int ceil_round(int x, int y)
{
    return ceil_div(x, y) * y;
}

template<typename A_T, typename B_T, typename C_T>
class MatmulCustom
{
    AscendC::GlobalTensor<A_T> a_global_tensor;
    AscendC::GlobalTensor<B_T> b_global_tensor;
    AscendC::GlobalTensor<C_T> c_global_tensor;
    AscendC::TQue<AscendC::QuePosition::A1, 1> a1_t_que;
    AscendC::TQue<AscendC::QuePosition::A2, 1> a2_t_que;
    AscendC::TQue<AscendC::QuePosition::B1, 1> b1_t_que;
    AscendC::TQue<AscendC::QuePosition::B2, 1> b2_t_que;
    AscendC::TQue<AscendC::QuePosition::CO1, 1> co1_t_que;
    static constexpr int L0_M = 128;
    static constexpr int L0_N = 128;
    static constexpr int L0_K = 64;
    static constexpr int L1_M_BLOCKS = 2;
    static constexpr int L1_N_BLOCKS = 2;
    static constexpr int L1_K_BLOCKS = 2;
    static constexpr int L1_M = L0_M * L1_M_BLOCKS;
    static constexpr int L1_N = L0_N * L1_N_BLOCKS;
    static constexpr int L1_K = L0_K * L1_K_BLOCKS;
    int offsetA, offsetB, offsetC;
    int singleCoreM, singleCoreN, singleCoreK;
    int M, N, K;
    // A 按行主序 M x K、B 的存储步长；默认与 K/N 相同，保持原 QR 调用行为不变。
    // QDWH 里 Q 是 (m+n) x (m+n) 的行主序矩阵，取 Q1/Q2 子块时行步长是 m+n，
    // 而不是子块自己的列数，所以需要把 ldA/ldB 单独传进来。
    int ldA, ldB;
    bool useCustomLd;
    bool isTransposeB, isTransposeA, if_atomicadd;
public:
    // 三个 flag 必须全部显式初始化：原来 isTransposeA 没初始化，而 Process2 的
    // L1->L0 拆分要读它，漏赋值就会随寄存器残留变化（历史 bug：同一份输入三次
    // 跑出三个结果）。现在 Process / Process2 都从自己的参数写入它，不再有隐式状态。
    __aicore__ inline MatmulCustom() : ldA(1), ldB(1), useCustomLd(false),
                                       isTransposeB(false), isTransposeA(false),
                                       if_atomicadd(false) {}
    __aicore__ inline void Init(AscendC::TPipe *t_pipe) {
        t_pipe->InitBuffer(a1_t_que, 2, L1_M * L1_K * sizeof(A_T));
        t_pipe->InitBuffer(a2_t_que, 2, L0_M * L0_K * sizeof(A_T));
        t_pipe->InitBuffer(b1_t_que, 2, L1_K * L1_N * sizeof(B_T));
        t_pipe->InitBuffer(b2_t_que, 2, L0_K * L0_N * sizeof(B_T));
        t_pipe->InitBuffer(co1_t_que, 2, L0_M * L0_N * sizeof(C_T));
    }
    __aicore__ inline void SetMatrix(AscendC::GlobalTensor<A_T> aGlobal, AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<C_T> cGlobal, int N, int K) {
        this->N = N;
        this->K = K;
        this->ldA = K;
        this->ldB = N;
        this->useCustomLd = false;
        this->a_global_tensor = aGlobal;
        this->b_global_tensor = bGlobal;
        this->c_global_tensor = cGlobal;
    }
    // 显式传入 M/N/K（A 转置时 srcDValue = M，即 A_T 的紧凑行步长 = 逻辑 M）。
    __aicore__ inline void SetMatrix2(AscendC::GlobalTensor<A_T> aGlobal, AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<C_T> cGlobal, int M, int N, int K) {
        this->M = M;
        this->N = N;
        this->K = K;
        this->ldA = K;
        this->ldB = N;
        this->useCustomLd = false;
        this->a_global_tensor = aGlobal;
        this->b_global_tensor = bGlobal;
        this->c_global_tensor = cGlobal;
    }
    // 带自定义 leading dimension 的版本：
    //   A 为 M x K 行主序，ldA = A 的每行元素数；
    //   B 非转置时是 K x N 行主序，ldB = 每行元素数；
    //   B 转置时存储为 N x K 行主序，ldB = 每行元素数（即 N 方向的原矩阵行步长）。
    __aicore__ inline void SetMatrixLd(AscendC::GlobalTensor<A_T> aGlobal, AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<C_T> cGlobal, int N, int K, int ldA, int ldB) {
        SetMatrix(aGlobal, bGlobal, cGlobal, N, K);
        this->ldA = ldA;
        this->ldB = ldB;
        this->useCustomLd = true;
    }
    // 带自定义 ld，并显式给出 A 的逻辑 M。
    // 只有 isTransposeA 通路会用到 M：那时 A 物理是 K x M 的**紧凑**行主序
    //（srcDValue = M，见 Process 里 GM->L1 的 A 分支），M 既是逻辑行数也是它的
    // 真实行步长，所以不能再靠 ldA 表达。N/K/ldA/ldB 的语义与 SetMatrixLd 相同。
    // 现有调用不受影响（M 在非转置通路里没被读过）。
    __aicore__ inline void SetMatrixLdM(AscendC::GlobalTensor<A_T> aGlobal, AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<C_T> cGlobal, int M, int N, int K, int ldA, int ldB) {
        SetMatrixLd(aGlobal, bGlobal, cGlobal, N, K, ldA, ldB);
        this->M = M;
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int cubeidx, bool isTransposeB, bool if_atomicadd, bool isTransposeA = false) {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;
        this->isTransposeB = isTransposeB;
        this->isTransposeA = isTransposeA;
        this->if_atomicadd = if_atomicadd;

        // 未显式指定 ld 时，恢复原行为：
        //   A: M x K 行主序 (ld = K)
        //   B: 非转置为 K x N 行主序 (ld = N)，转置存储为 N x K 行主序 (ld = K)
        if (!useCustomLd) {
            this->ldA = K;
            this->ldB = isTransposeB ? K : N;
        }

        static constexpr uint8_t pad_list[4]{};

        int l2_m_blocks = ceil_div(singleCoreM, L1_M);
        int l2_n_blocks = ceil_div(singleCoreN, L1_N);
        int l2_k_blocks = ceil_div(singleCoreK, L1_K);

        for(int i = 0; i < l2_k_blocks; i++) 
        {
            int l1_k = min(singleCoreK - i * L1_K, L1_K);
            int l1_k_aligned = ceil_round(l1_k, 8);
            AscendC::SetFmatrix(1, l1_k_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_RIGHT);
            for(int j = 0; j < l2_n_blocks; j++) 
            {
                int l1_n = min(singleCoreN - j * L1_N, L1_N);
                int l1_n_aligned = ceil_round(l1_n, 16);
                // copy in b
                if(isTransposeB)
                {
                    AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.AllocTensor<B_T>();
                    AscendC::Nd2NzParams nd2_nz_params;
                    nd2_nz_params.ndNum = 1;
                    nd2_nz_params.nValue = l1_n;
                    nd2_nz_params.dValue = l1_k;
                    nd2_nz_params.srcDValue = ldB;
                    nd2_nz_params.dstNzC0Stride = l1_n_aligned;
                    nd2_nz_params.dstNzNStride = 1;
                    AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K + j * L1_N * ldB], nd2_nz_params);
                    b1_t_que.EnQue(b1_local_tensor);
                }
                else
                {
                    AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.AllocTensor<B_T>();
                    AscendC::Nd2NzParams nd2_nz_params;
                    nd2_nz_params.ndNum = 1;
                    nd2_nz_params.nValue = l1_k;
                    nd2_nz_params.dValue = l1_n;
                    nd2_nz_params.srcDValue = ldB;
                    nd2_nz_params.dstNzC0Stride = l1_k_aligned;
                    nd2_nz_params.dstNzNStride = 1;
                    AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K * ldB + j * L1_N], nd2_nz_params);
                    b1_t_que.EnQue(b1_local_tensor);
                }
                AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.DeQue<B_T>();
                for(int k = 0; k < l2_m_blocks; k++)
                {
                    int l1_m = min(singleCoreM - k * L1_M, L1_M);
                    int l1_m_aligned = ceil_round(l1_m, 16);
                    AscendC::SetFmatrix(1, l1_m_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_LEFT);
                    // copy in a
                    {
                        AscendC::LocalTensor<A_T> a1_local_tensor = a1_t_que.AllocTensor<A_T>();
                        AscendC::Nd2NzParams nd2_nz_params;
                        nd2_nz_params.ndNum = 1;
                        nd2_nz_params.dstNzNStride = 1;
                        if (isTransposeA) {
                            // A 以 K x M 紧凑行主序存储（转置形态），srcDValue = M
                            nd2_nz_params.nValue = l1_k;
                            nd2_nz_params.dValue = l1_m;
                            nd2_nz_params.srcDValue = M;
                            nd2_nz_params.dstNzC0Stride = l1_k_aligned;
                            AscendC::DataCopy(a1_local_tensor,
                                              a_global_tensor[offsetA + k * L1_M + i * L1_K * M],
                                              nd2_nz_params);
                        } else {
                            nd2_nz_params.nValue = l1_m;
                            nd2_nz_params.dValue = l1_k;
                            nd2_nz_params.srcDValue = ldA;
                            nd2_nz_params.dstNzC0Stride = l1_m_aligned;
                            AscendC::DataCopy(a1_local_tensor,
                                              a_global_tensor[offsetA + k * L1_M * ldA + i * L1_K],
                                              nd2_nz_params);
                        }
                        a1_t_que.EnQue(a1_local_tensor);
                    }
                    AscendC::LocalTensor<A_T> a1_local_tensor = a1_t_que.DeQue<A_T>();

                    int l1_m_blocks = ceil_div(l1_m, L0_M);
                    for (int m = 0; m < l1_m_blocks; m++)
                    {
                        int l0_m = min(l1_m - m * L0_M, L0_M);
                        int l0_m_aligned = ceil_round(l0_m, 16);
                        int l1_n_blocks = ceil_div(l1_n, L0_N);
                        for (int n = 0; n < l1_n_blocks; n++)
                        {
                            int l0_n = min(l1_n - n * L0_N, L0_N);
                            int l0_n_aligned = ceil_round(l0_n, 16);
                            AscendC::LocalTensor<C_T> co1_local_tensor = co1_t_que.AllocTensor<C_T>();
                            int l1_k_blocks = ceil_div(l1_k, L0_K);
                            for (int o = 0; o < l1_k_blocks; o++) 
                            {
                                int l0_k = min(l1_k - o * L0_K, L0_K);
                                int l0_k_aligned = ceil_round(l0_k, 8);
                                // split a
                                {
                                    AscendC::LocalTensor<A_T> a2_local_tensor = a2_t_que.AllocTensor<A_T>();
                                    AscendC::LoadData3DParamsV2Pro load_data_3d_params_v2_pro;
                                    load_data_3d_params_v2_pro.channelSize = isTransposeA ? l1_m_aligned : l1_k_aligned;
                                    short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
                                    if (isTransposeA) {
                                        AscendC::SetFmatrix(1, l1_m_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_LEFT);
                                        extConfig[0] = l0_m_aligned;
                                        extConfig[1] = l0_k_aligned;
                                        extConfig[2] = m * L0_M;
                                        extConfig[3] = o * L0_K;
                                        load_data_3d_params_v2_pro.enTranspose = true;
                                    } else {
                                        extConfig[0] = l0_k_aligned;
                                        extConfig[1] = l0_m_aligned;
                                        extConfig[2] = o * L0_K;
                                        extConfig[3] = m * L0_M;
                                    }
                                    AscendC::LoadData(a2_local_tensor, a1_local_tensor, load_data_3d_params_v2_pro);
                                    a2_t_que.EnQue(a2_local_tensor);
                                }
                                // split b
                                if(isTransposeB)
                                {
                                    AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.AllocTensor<B_T>();
                                    AscendC::LoadData2dParams loadDataParams;
                                    int loopCount = l0_k_aligned / 8;
                                    loadDataParams.startIndex = (l1_n_aligned * o * L0_K + n * L0_N * 8) / 128;
                                    loadDataParams.repeatTimes = l0_n_aligned / 16;
                                    loadDataParams.srcStride = 1;
                                    uint64_t l0bOffset = 0;
                                    if(l1_n_blocks == 1) {
                                        loadDataParams.repeatTimes *= loopCount;
                                        AscendC::LoadData(b2_local_tensor, b1_local_tensor, loadDataParams);
                                    }
                                    else {
                                        for(int s = 0; s != loopCount; s++) {
                                        AscendC::LoadData(b2_local_tensor[l0bOffset], b1_local_tensor, loadDataParams);
                                        loadDataParams.startIndex += (l1_n_aligned * 8) / 128;
                                        l0bOffset += 8 * l0_n_aligned;
                                        } 
                                    }
                                    b2_t_que.EnQue(b2_local_tensor);
                                }
                                else
                                {
                                    AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.AllocTensor<B_T>();
                                    AscendC::LoadData3DParamsV2Pro load_data_3d_params_v2_pro;
                                    load_data_3d_params_v2_pro.channelSize = l1_n_aligned;
                                    load_data_3d_params_v2_pro.fMatrixCtrl = true;
                                    short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
                                    extConfig[0] = l0_n_aligned;
                                    extConfig[1] = l0_k_aligned;
                                    extConfig[2] = n * L0_N;
                                    extConfig[3] = o * L0_K;
                                    AscendC::LoadData(b2_local_tensor, b1_local_tensor, load_data_3d_params_v2_pro);
                                    b2_t_que.EnQue(b2_local_tensor);
                                }
                                
                                // compute
                                {
                                    AscendC::LocalTensor<A_T> a2_local_tensor = a2_t_que.DeQue<A_T>();
                                    AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.DeQue<B_T>();
                                    AscendC::MmadParams mmad_params;
                                    mmad_params.m = max(l0_m, 2);
                                    mmad_params.n = l0_n;
                                    mmad_params.k = l0_k;
                                    mmad_params.cmatrixInitVal = o == 0;
                                    AscendC::Mmad(co1_local_tensor, a2_local_tensor, b2_local_tensor, mmad_params);
                                    if (l0_m * l0_n < 2560)
                                        AscendC::PipeBarrier<PIPE_M>();
                                    a2_t_que.FreeTensor(a2_local_tensor);
                                    b2_t_que.FreeTensor(b2_local_tensor);
                                }
                            }
                            co1_t_que.EnQue(co1_local_tensor);
                            // copy out
                            {
                                co1_local_tensor = co1_t_que.DeQue<C_T>();
                                AscendC::FixpipeParamsV220 fixpipe_params_v220;
                                fixpipe_params_v220.nSize = l0_n;
                                fixpipe_params_v220.mSize = l0_m;
                                fixpipe_params_v220.srcStride = l0_m_aligned;
                                fixpipe_params_v220.dstStride = N;
                                if(if_atomicadd) {
                                    if(cubeidx == -1) {
                                        AscendC::SetAtomicAdd<float>();
                                        AscendC::Fixpipe(c_global_tensor[offsetC + (k * L1_M + m * L0_M )* N + n * L0_N + j * L1_N], co1_local_tensor, fixpipe_params_v220);
                                        AscendC::SetAtomicNone();
                                    }
                                    else if(cubeidx == 0) {
                                        AscendC::Fixpipe(c_global_tensor[offsetC + (k * L1_M + m * L0_M )* N + n * L0_N + j * L1_N], co1_local_tensor, fixpipe_params_v220);
                                        AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
                                        AscendC::CrossCoreWaitFlag(0x3);
                                        cubeidx = -1;
                                    } else if(cubeidx > 0 && cubeidx < AscendC::GetBlockNum()){
                                        AscendC::SetAtomicAdd<float>();
                                        AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
                                        AscendC::CrossCoreWaitFlag(0x3);
                                        AscendC::Fixpipe(c_global_tensor[offsetC + (k * L1_M + m * L0_M )* N + n * L0_N + j * L1_N], co1_local_tensor, fixpipe_params_v220);
                                        AscendC::SetAtomicNone();
                                        cubeidx = -1;
                                    } 
                                } else {
                                    AscendC::Fixpipe(c_global_tensor[offsetC + (k * L1_M + m * L0_M )* N + n * L0_N + j * L1_N], co1_local_tensor, fixpipe_params_v220);
                                }
                                co1_t_que.FreeTensor(co1_local_tensor);
                            }
                        }
                    }
                    a1_t_que.FreeTensor(a1_local_tensor);
                }
                b1_t_que.FreeTensor(b1_local_tensor);
            }
        }
    }

    // QR near-panel projection: one output stripe owns the complete K reduction.
    // Keep FP32 sums in L0C until K is exhausted; no GM/cross-core atomic sums.
    // Only used with non-transposed A, physically transposed B, M<=8 and N=64.
    __aicore__ inline void ProjectLongK(int oa,int ob,int oc,int bm,int bk)
    {
        const int ma=ceil_round(bm,16), nn=64;
        static constexpr uint8_t padding[4]{};
        auto acc=co1_t_que.AllocTensor<C_T>();
        for(int kc=0;kc<bk;kc+=L1_K) {
            const int kk=min(L1_K,bk-kc),ka=ceil_round(kk,8);
            AscendC::SetFmatrix(1,ka,padding,AscendC::FmatrixMode::FMATRIX_RIGHT);
            AscendC::SetFmatrix(1,ma,padding,AscendC::FmatrixMode::FMATRIX_LEFT);
            auto bt=b1_t_que.AllocTensor<B_T>();
            AscendC::Nd2NzParams cb;
            cb.ndNum=1;cb.nValue=nn;cb.dValue=kk;cb.srcDValue=K;
            cb.dstNzC0Stride=nn;cb.dstNzNStride=1;
            AscendC::DataCopy(bt,b_global_tensor[ob+kc],cb);b1_t_que.EnQue(bt);
            auto at=a1_t_que.AllocTensor<A_T>();
            AscendC::Nd2NzParams ca;
            ca.ndNum=1;ca.nValue=bm;ca.dValue=kk;ca.srcDValue=K;
            ca.dstNzC0Stride=ma;ca.dstNzNStride=1;
            AscendC::DataCopy(at,a_global_tensor[oa+kc],ca);a1_t_que.EnQue(at);
            bt=b1_t_que.DeQue<B_T>();at=a1_t_que.DeQue<A_T>();
            for(int o=0;o<ceil_div(kk,L0_K);++o) {
                const int ko=min(L0_K,kk-o*L0_K),koa=ceil_round(ko,8);
                auto b2=b2_t_que.AllocTensor<B_T>();
                AscendC::LoadData2dParams lb;
                lb.startIndex=nn*o*L0_K/128;lb.repeatTimes=nn/16*(koa/8);lb.srcStride=1;
                AscendC::LoadData(b2,bt,lb);b2_t_que.EnQue(b2);
                auto a2=a2_t_que.AllocTensor<A_T>();
                AscendC::LoadData3DParamsV2Pro la;la.channelSize=ka;
                short (&ext)[4]=*reinterpret_cast<short (*)[4]>(&la.extConfig);
                ext[0]=ko;ext[1]=bm;ext[2]=o*L0_K;ext[3]=0;
                AscendC::LoadData(a2,at,la);a2_t_que.EnQue(a2);
                a2=a2_t_que.DeQue<A_T>();b2=b2_t_que.DeQue<B_T>();
                AscendC::MmadParams mm;mm.m=max(bm,2);mm.n=nn;mm.k=ko;
                mm.cmatrixInitVal=(kc==0 && o==0);
                AscendC::Mmad(acc,a2,b2,mm);AscendC::PipeBarrier<PIPE_M>();
                a2_t_que.FreeTensor(a2);b2_t_que.FreeTensor(b2);
            }
            a1_t_que.FreeTensor(at);b1_t_que.FreeTensor(bt);
        }
        co1_t_que.EnQue(acc);acc=co1_t_que.DeQue<C_T>();
        AscendC::FixpipeParamsV220 cp;cp.nSize=nn;cp.mSize=bm;cp.srcStride=ma;cp.dstStride=N;
        AscendC::Fixpipe(c_global_tensor[oc],acc,cp);co1_t_que.FreeTensor(acc);
    }

    // ------------------------------------------------------------------------
    // Process2：与 Process 的区别是 **A 只能以 M x K 行主序给出**（A 的 GM->L1 读取
    // 不做转置），A 的"转置"靠 L1->L0 的 enTranspose 完成。
    //
    // 两个 bool 的含义（已用独立的 GEMM 定点试验验证，试验算子已删除）：
    //   bTransposed : **B** 以 N x K 存储（取 Bᵀ）；C = A·Bᵀ
    //   aTransposed : **A** 按 Aᵀ 参与乘加（A 物理仍是 M x K）；C = Aᵀ·B
    // 同时为 true 时 C = Aᵀ·Bᵀ。默认 aTransposed = false，与 Process 的第 10 个参数
    // 一致；两者都从参数写成员，不再依赖任何"隐式状态 / setter"。
    //
    // 历史陷阱：这里第 8 个参数原来叫 isTransposeB，同时它还被写进成员；而 A 的转置
    // 只由成员 isTransposeA 决定（Process2 原来没有 A 的参数，只能靠 SetTransposeA
    // 打开，且该成员在构造函数里没初始化）——调用处 Process2(..., false, true) 里
    // 那个 false 到底是谁的转置、A 侧到底转不转，用签名完全看不出来，极易误读。
    // ------------------------------------------------------------------------
    __aicore__ inline void Process2(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK,
                                    int cubeidx, bool bTransposed, bool if_atomicadd,
                                    bool aTransposed = false) {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;
        this->isTransposeB = bTransposed;      // 管 B
        this->isTransposeA = aTransposed;      // 管 A（L1->L0 的 enTranspose）
        this->if_atomicadd = if_atomicadd;

        // 未显式指定 ld 时，恢复原行为：
        //   A: M x K 行主序 (ld = K)
        //   B: 非转置为 K x N 行主序 (ld = N)，转置存储为 N x K 行主序 (ld = K)
        if (!useCustomLd) {
            this->ldA = K;
            this->ldB = isTransposeB ? K : N;
        }

        static constexpr uint8_t pad_list[4]{};

        int l2_m_blocks = ceil_div(singleCoreM, L1_M);
        int l2_n_blocks = ceil_div(singleCoreN, L1_N);
        int l2_k_blocks = ceil_div(singleCoreK, L1_K);

        for(int i = 0; i < l2_k_blocks; i++) 
        {
            int l1_k = min(singleCoreK - i * L1_K, L1_K);
            int l1_k_aligned = ceil_round(l1_k, 8);
            AscendC::SetFmatrix(1, l1_k_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_RIGHT);
            for(int k = 0; k < l2_m_blocks; k++)
            {
                int l1_m = min(singleCoreM - k * L1_M, L1_M);
                int l1_m_aligned = ceil_round(l1_m, 16);
                AscendC::SetFmatrix(1, l1_m_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_LEFT);
                // copy in a
                {
                    AscendC::LocalTensor<A_T> a1_local_tensor = a1_t_que.AllocTensor<A_T>();
                    AscendC::Nd2NzParams nd2_nz_params;
                    nd2_nz_params.ndNum = 1;
                    nd2_nz_params.nValue = l1_m;
                    nd2_nz_params.dValue = l1_k;
                    nd2_nz_params.srcDValue = ldA;
                    nd2_nz_params.dstNzC0Stride = l1_m_aligned;
                    nd2_nz_params.dstNzNStride = 1;
                    AscendC::DataCopy(a1_local_tensor, a_global_tensor[offsetA + k * L1_M * ldA + i * L1_K], nd2_nz_params);
                    a1_t_que.EnQue(a1_local_tensor);
                }
                AscendC::LocalTensor<A_T> a1_local_tensor = a1_t_que.DeQue<A_T>();
                
                for(int j = 0; j < l2_n_blocks; j++) 
                {
                    int l1_m_blocks = ceil_div(l1_m, L0_M);
                    int l1_n = min(singleCoreN - j * L1_N, L1_N);
                    int l1_n_aligned = ceil_round(l1_n, 16);
                    // copy in b
                    if(isTransposeB)
                    {
                        AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.AllocTensor<B_T>();
                        AscendC::Nd2NzParams nd2_nz_params;
                        nd2_nz_params.ndNum = 1;
                        nd2_nz_params.nValue = l1_n;
                        nd2_nz_params.dValue = l1_k;
                        nd2_nz_params.srcDValue = ldB;
                        nd2_nz_params.dstNzC0Stride = l1_n_aligned;
                        nd2_nz_params.dstNzNStride = 1;
                        AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K + j * L1_N * ldB], nd2_nz_params);
                        b1_t_que.EnQue(b1_local_tensor);
                    }
                    else
                    {
                        AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.AllocTensor<B_T>();
                        AscendC::Nd2NzParams nd2_nz_params;
                        nd2_nz_params.ndNum = 1;
                        nd2_nz_params.nValue = l1_k;
                        nd2_nz_params.dValue = l1_n;
                        nd2_nz_params.srcDValue = ldB;
                        nd2_nz_params.dstNzC0Stride = l1_k_aligned;
                        nd2_nz_params.dstNzNStride = 1;
                        AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K * ldB + j * L1_N], nd2_nz_params);
                        b1_t_que.EnQue(b1_local_tensor);
                    }

                    AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.DeQue<B_T>();
          
                    for (int m = 0; m < l1_m_blocks; m++)
                    {
                        int l0_m = min(l1_m - m * L0_M, L0_M);
                        int l0_m_aligned = ceil_round(l0_m, 16);
                        int l1_n_blocks = ceil_div(l1_n, L0_N);
                        for (int n = 0; n < l1_n_blocks; n++)
                        {   
                            int l0_n = min(l1_n - n * L0_N, L0_N);
                            int l0_n_aligned = ceil_round(l0_n, 16);
                            AscendC::LocalTensor<C_T> co1_local_tensor = co1_t_que.AllocTensor<C_T>();
                            int l1_k_blocks = ceil_div(l1_k, L0_K);

                            for (int o = 0; o < l1_k_blocks; o++) 
                            {
                                int l0_k = min(l1_k - o * L0_K, L0_K);
                                int l0_k_aligned = ceil_round(l0_k, 8);
                                
                                // split b
                                if(isTransposeB)
                                {
                                    AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.AllocTensor<B_T>();
                                    AscendC::LoadData2dParams loadDataParams;
                                    int loopCount = l0_k_aligned / 8;
                                    loadDataParams.startIndex = (l1_n_aligned * o * L0_K + n * L0_N * 8) / 128;
                                    loadDataParams.repeatTimes = l0_n_aligned / 16;
                                    loadDataParams.srcStride = 1;
                                    uint64_t l0bOffset = 0;
                                    if(l1_n_blocks == 1) {
                                        loadDataParams.repeatTimes *= loopCount;
                                        AscendC::LoadData(b2_local_tensor, b1_local_tensor, loadDataParams);
                                    }
                                    else {
                                        for(int s = 0; s != loopCount; s++) {
                                        AscendC::LoadData(b2_local_tensor[l0bOffset], b1_local_tensor, loadDataParams);
                                        loadDataParams.startIndex += (l1_n_aligned * 8) / 128;
                                        l0bOffset += 8 * l0_n_aligned;
                                        } 
                                    }
                                    b2_t_que.EnQue(b2_local_tensor);
                                }
                                else
                                {
                                    AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.AllocTensor<B_T>();
                                    AscendC::LoadData3DParamsV2Pro load_data_3d_params_v2_pro;
                                    load_data_3d_params_v2_pro.channelSize = l1_n_aligned;
                                    load_data_3d_params_v2_pro.fMatrixCtrl = true;
                                    short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
                                    extConfig[0] = l0_n_aligned;
                                    extConfig[1] = l0_k_aligned;
                                    extConfig[2] = n * L0_N;
                                    extConfig[3] = o * L0_K;
                                    AscendC::LoadData(b2_local_tensor, b1_local_tensor, load_data_3d_params_v2_pro);
                                    b2_t_que.EnQue(b2_local_tensor);
                                }
                                // split a
                                {
                                    AscendC::LocalTensor<A_T> a2_local_tensor = a2_t_que.AllocTensor<A_T>();
                                    AscendC::LoadData3DParamsV2Pro load_data_3d_params_v2_pro;
                                    load_data_3d_params_v2_pro.channelSize = isTransposeA ? l1_m_aligned : l1_k_aligned;
                                    short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
                                    if (isTransposeA) {
                                        AscendC::SetFmatrix(1, l1_m_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_LEFT);
                                        extConfig[0] = l0_m_aligned;
                                        extConfig[1] = l0_k_aligned;
                                        extConfig[2] = m * L0_M;
                                        extConfig[3] = o * L0_K;
                                        load_data_3d_params_v2_pro.enTranspose = true;
                                    } else {
                                        extConfig[0] = l0_k_aligned;
                                        extConfig[1] = l0_m_aligned;
                                        extConfig[2] = o * L0_K;
                                        extConfig[3] = m * L0_M;
                                    }
                                    AscendC::LoadData(a2_local_tensor, a1_local_tensor, load_data_3d_params_v2_pro);
                                    a2_t_que.EnQue(a2_local_tensor);
                                }
                                // compute
                                {
                                    AscendC::LocalTensor<A_T> a2_local_tensor = a2_t_que.DeQue<A_T>();
                                    AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.DeQue<B_T>();
                                    AscendC::MmadParams mmad_params;
                                    mmad_params.m = max(l0_m, 2);
                                    mmad_params.n = l0_n;
                                    mmad_params.k = l0_k;
                                    mmad_params.cmatrixInitVal = o == 0;
                                    AscendC::Mmad(co1_local_tensor, a2_local_tensor, b2_local_tensor, mmad_params);
                                    if (l0_m * l0_n < 2560)
                                        AscendC::PipeBarrier<PIPE_M>();
                                    a2_t_que.FreeTensor(a2_local_tensor);
                                    b2_t_que.FreeTensor(b2_local_tensor);
                                }
                            }
                            co1_t_que.EnQue(co1_local_tensor);
                            // copy out
                            {
                                co1_local_tensor = co1_t_que.DeQue<C_T>();
                                AscendC::FixpipeParamsV220 fixpipe_params_v220;
                                fixpipe_params_v220.nSize = l0_n;
                                fixpipe_params_v220.mSize = l0_m;
                                fixpipe_params_v220.srcStride = l0_m_aligned;
                                fixpipe_params_v220.dstStride = N;
                                if(if_atomicadd) {
                                    AscendC::SetAtomicAdd<float>();
                                    AscendC::Fixpipe(c_global_tensor[offsetC + (k * L1_M + m * L0_M )* N + n * L0_N + j * L1_N], co1_local_tensor, fixpipe_params_v220);
                                    AscendC::SetAtomicNone();
                                } else {
                                    AscendC::Fixpipe(c_global_tensor[offsetC + (k * L1_M + m * L0_M )* N + n * L0_N + j * L1_N], co1_local_tensor, fixpipe_params_v220);
                                }
                                co1_t_que.FreeTensor(co1_local_tensor);
                            }
                        }
                    }
                    b1_t_que.FreeTensor(b1_local_tensor);
                }
                a1_t_que.FreeTensor(a1_local_tensor);
            }
        }
    }
};