// ============================================================================
// 融合 kernel：QDWH 极分解（Phase 1） + device 内 SDC 递归（Phase 2）
//
//   一次 launch 内完成：A -> (QDWH) -> U, H -> (SDC device 递归) -> leaves
//
// 全部实现都在 QdwhSdcFused（fused_qdwh_sdc.hpp）里；本文件只做 Init + Run，
// 不再 include qdwh_custom.cpp / sdc_dev_custom.cpp（会重复定义）。
//
// 参数：wk = workspace 基址（A/H/U/Q/scratch/meta 全在里面，见 fused_layout.hpp）
// ============================================================================
#include "fused_qdwh_sdc.hpp"

extern "C" __global__ __aicore__ void qdwh_sdc_fused_custom(
    GM_ADDR wk,
    int m, int n, int nRoot, int threshold, int maxIter,
    int maxNodes, int maxLeaves, int leafCap,
    float alpha, float l0)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    QdwhSdcFused op;
    op.Init(wk, m, n, nRoot, threshold, maxIter, maxNodes, maxLeaves, leafCap, alpha, l0);

#ifdef __DAV_C220_VEC__
    op.RunAiv();
#else
    op.RunCube();
#endif
}
