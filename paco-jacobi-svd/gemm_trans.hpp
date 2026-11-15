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
    int N, K;
    bool isTransposeB, if_atomicadd;
public:
    __aicore__ inline MatmulCustom() {}
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
        this->a_global_tensor = aGlobal;
        this->b_global_tensor = bGlobal;
        this->c_global_tensor = cGlobal;
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int cubeidx, bool isTransposeB, bool if_atomicadd) {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;
        this->isTransposeB = isTransposeB;
        this->if_atomicadd = if_atomicadd;

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
                    nd2_nz_params.srcDValue = K;
                    nd2_nz_params.dstNzC0Stride = l1_n_aligned;
                    nd2_nz_params.dstNzNStride = 1;
                    AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K + j * L1_N * K], nd2_nz_params);
                    b1_t_que.EnQue(b1_local_tensor);
                }
                else
                {
                    AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.AllocTensor<B_T>();
                    AscendC::Nd2NzParams nd2_nz_params;
                    nd2_nz_params.ndNum = 1;
                    nd2_nz_params.nValue = l1_k;
                    nd2_nz_params.dValue = l1_n;
                    nd2_nz_params.srcDValue = N;
                    nd2_nz_params.dstNzC0Stride = l1_k_aligned;
                    nd2_nz_params.dstNzNStride = 1;
                    AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K * N + j * L1_N], nd2_nz_params);
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
                        nd2_nz_params.nValue = l1_m;
                        nd2_nz_params.dValue = l1_k;
                        nd2_nz_params.srcDValue = K;
                        nd2_nz_params.dstNzC0Stride = l1_m_aligned;
                        nd2_nz_params.dstNzNStride = 1;
                        AscendC::DataCopy(a1_local_tensor, a_global_tensor[offsetA + k * L1_M * K + i * L1_K], nd2_nz_params);
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
                                    load_data_3d_params_v2_pro.channelSize = l1_k_aligned;
                                    short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
                                    extConfig[0] = l0_k_aligned;
                                    extConfig[1] = l0_m_aligned;
                                    extConfig[2] = o * L0_K;
                                    extConfig[3] = m * L0_M;
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
    __aicore__ inline void Process2(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int cubeidx, bool isTransposeB, bool if_atomicadd) {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;
        this->isTransposeB = isTransposeB;
        this->if_atomicadd = if_atomicadd;

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
                    nd2_nz_params.srcDValue = K;
                    nd2_nz_params.dstNzC0Stride = l1_m_aligned;
                    nd2_nz_params.dstNzNStride = 1;
                    AscendC::DataCopy(a1_local_tensor, a_global_tensor[offsetA + k * L1_M * K + i * L1_K], nd2_nz_params);
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
                        nd2_nz_params.srcDValue = K;
                        nd2_nz_params.dstNzC0Stride = l1_n_aligned;
                        nd2_nz_params.dstNzNStride = 1;
                        AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K + j * L1_N * K], nd2_nz_params);
                        b1_t_que.EnQue(b1_local_tensor);
                    }
                    else
                    {
                        AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.AllocTensor<B_T>();
                        AscendC::Nd2NzParams nd2_nz_params;
                        nd2_nz_params.ndNum = 1;
                        nd2_nz_params.nValue = l1_k;
                        nd2_nz_params.dValue = l1_n;
                        nd2_nz_params.srcDValue = N;
                        nd2_nz_params.dstNzC0Stride = l1_k_aligned;
                        nd2_nz_params.dstNzNStride = 1;
                        AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K * N + j * L1_N], nd2_nz_params);
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
                                    load_data_3d_params_v2_pro.channelSize = l1_k_aligned;
                                    short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
                                    extConfig[0] = l0_k_aligned;
                                    extConfig[1] = l0_m_aligned;
                                    extConfig[2] = o * L0_K;
                                    extConfig[3] = m * L0_M;
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