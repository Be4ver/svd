#pragma once
// ============================================================================
// 融合 kernel 的 workspace 布局（host 与 device 共用同一份计算）。
//
// host 只 aclrtMalloc 一次 total * 4 字节，device 用同一个 Build() 算出全部
// 子缓冲的偏移。这样 launch 参数只有 1 个基址 + 若干标量（远小于 55 个指针），
// 也避免 host/device 两处手写偏移不一致。
//
// 所有 offset 的单位是 float32（4 字节）；int32 元数据数组也放在同一块
// workspace 里，用 reinterpret_cast 取 int32 视图。
// ============================================================================
#include "fused_batch_cfg.hpp"

#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__) || defined(__CCE_AICORE__)
#define FUSED_INL __aicore__ inline
#else
#define FUSED_INL inline
#endif

struct FusedLayout {
    int m, n;          // A_pad : s x s（s = ceil128(max(M,N))，补零方阵）
    int qd;            // m + n : QDWH 的 QR 输入行数
    int nPad;          // ceil(n/128)*128（split 内部所有矩阵的行步长）
    int maxNodes;      // device 节点队列 / arena slot 上限
    int maxLeaves;     // 叶块个数上限
    int leafCap;       // 叶块缓冲 float 容量
    int sigmaCap;      // 每叶 64 槽的 cache-line 独占 scratch 容量
    int batchMax;      // Phase 2 同层 batch 上限（= kSdcBatchMax）

    long a;            // QDWH X（迭代收敛后即 U），max(m*n, nPad^2)
    long a0;           // QDWH A0（= 原始 A）/ split 的 A0（batchMax*nPad^2）
    long xsdc;         // split 的 QdwhA（batchMax*nPad^2）：避免覆盖 Phase 1 的 U
    long pt;           // P^T（QDWH 的 Pmat 顺带算出来，n*m）：给 XT 的向量更新用
    long xt;           // X^T（n*m）：QdwhPrepare 直接读它的行，省掉按列 scalar gather
    long hwork;        // QDWH APad（n*qd）/ split 的 hwork（batchMax*2*nPad^2）
    long qbuf;         // QDWH 的完整 Q（qd^2，split 时 batchMax*4*nPad^2）
    long tau;          // QR 的 tau（split 的 batch 需要 sum(nPad[b])）
    long tmat;         // QR 的 T（batchMax*128*256 + panel + 余量）
    long w;            // QR 的 W：前 batchMax*2048 给 AIV，之后 gap，再给 AIC
    long y;            // QR 的 Y（batchMax 个矩阵的 2 个 panel）
    long coef;         // QDWH theta/kappa（128）
    long eye;          // gather 单位阵（128 x 256）
    long eyeN;         // 完整 n x n 单位阵（用于 GEMM 生成 Q=I）
    long uomega;       // PΩ1 | P_Ω2（2*nPad^2）
    long qt;           // Q^T 工作区（nPad^2）
    long tmp;          // 通用临时矩阵（nPad^2）
    long hplus;        // H+ padded 输出（nPad^2）
    long hminus;       // H- padded 输出（nPad^2）
    long om1;          // Ω1（nRoot^2）
    long om2;          // Ω2（nRoot^2）
    long arena;        // 节点 H 缓冲：slot0 = QDWH 的 H，其余按 n² 紧凑分配
    long arenaFloats;  // arena 的 float 容量（4*nPad²，见 SdcGroupCube 的复用策略）
    long leaf;         // 叶块缓冲（leafCap）
    long desc;         // 节点描述（16 int32）
    long copyInfo;     // 子节点/叶块搬运信息（16 int32）
    long nodeH;        // device 节点队列：arena slot
    long nodeN;        // device 节点队列：n
    long nodeLd;       // device 节点队列：ld
    long cnt;          // 8 个 int32：[0..2] 预留, leafCount, [4] 预留,
                       //              overflow, grpCnt(本组节点数), grpTotal(组数)
    long leafN;        // 叶块 n
    long leafLd;       // 叶块 ld
    long leafOff;      // 叶块在 leaf 缓冲中的偏移
    long sigma;        // 每个叶块贡献的奇异值，连续 float32 输出
    long sigmaOff;     // 每叶 sigma 在 sigma 缓冲中的偏移（int32）
    long sigmaCount;   // 每叶 sigma 数量（int32）
    long sigmaLeafCount; // 最终叶块数量（int32）
    long kTmp;         // 预留（旧 split 输出的 k，已不再使用）
    long lambdaTmp;    // 预留（旧 split 的 λ 调试槽）
    long dbg;          // 调试（128 int32：split 用前 48 个，1..80 留给融合层）
    long shapes;       // 8 int32：pQdwh[2] | pQr[2] | spare
    long total;        // 总 float 数

    FUSED_INL void Build(int m_, int n_, int maxNodes_, int maxLeaves_, int leafCap_);
};

FUSED_INL void FusedLayout::Build(int m_, int n_, int maxNodes_, int maxLeaves_, int leafCap_)
{
    m = m_;
    n = n_;
    qd = m + n;
    nPad = ((n + 127) / 128) * 128;
    maxNodes = maxNodes_;
    maxLeaves = maxLeaves_;
    leafCap = leafCap_;
    // sigma 每叶固定 64 槽（SigmaOff[i] = i*64，见 fused_qdwh_sdc.hpp），
    // 所以容量就是 maxLeaves*64，不需要再以 leafCap 兜底
    //（旧的 floor 在 4096² 上会白占 130 MB）。
    sigmaCap = maxLeaves_ * 64;
    batchMax = kSdcBatchMax;

    const long mn = (long)m * (long)n;
    const long np2 = (long)nPad * (long)nPad;     // nPad^2
    // 节点 H 缓冲：root 的 nPad²（留给 host）+ root 紧凑 nRoot² + 子节点分配，
    // 子节点总量实测约 root 的 0.7~2.4 倍，6 倍留足余量。
    arenaFloats = 6 * np2;
    const long qd2 = (long)qd * (long)qd;
    const int B = kSdcBatchMax;

    // ------------------------------------------------------------------
    // Phase 2 同批切片的真实上界（替代原来一律按 root nPad² 预留的 B 倍）
    //
    // 队列里的节点是 root 谱区间的互不相交划分，因此任意一批都满足
    //     Σ_b n_b ≤ nRoot ≤ nPad
    // 而 split 的 nPad_b = ceil128(n_b) ≤ n_b + 127，于是
    //     Σ_b nPad_b ≤ nPad + 127*B
    //     Σ_b nPad_b² ≤ max_b(nPad_b) * Σ_b nPad_b ≤ nPad * (nPad + 127*B)
    //
    // 每个矩阵的占用：x / a0 / pt / xt 是 m_b*n_b = 2*nPad_b²，
    // qbuf 是 (2*nPad_b)² = 4*nPad_b²，hwork 是 n_b*qd_b = 2*nPad_b²。
    // 原来按 B*np2（= B*nPad²）预留，对 4096² 是 6.4 倍浪费。
    // ------------------------------------------------------------------
    const long batchF = (long)nPad * ((long)nPad + 127L * B);

    // QDWH APad：n x qd；split 的 hwork：Σ 2*nPad_b²
    long hw = (long)n * (long)qd;
    if (2 * batchF > hw) hw = 2 * batchF;

    // QDWH Q：(m+n)^2；split 的 QDWH 用 Σ (2*nPad_b)^2
    long qb = qd2;
    if (4 * batchF > qb) qb = 4 * batchF;

    // W：AIV 侧 batchMax*2048，AIC 侧 batchMax*128*max(M,N)
    long wAic = 128 * (long)qd;
    if (128 * 2 * (long)nPad > wAic) wAic = 128 * 2 * (long)nPad;
    if (3 * (long)n * 128 > wAic) wAic = 3 * (long)n * 128;
    long wTotal = kQdwhWVecTotal + kQdwhWGap + (long)B * wAic;

    // Y：panelYSize = 64*qd（Phase 1）/ 64*2*nPad（split 每个矩阵），两块 ping-pong
    long yNeed = 2 * 64 * (long)qd;
    if (2 * 64 * 2 * (long)nPad * B > yNeed) yNeed = 2 * 64 * 2 * (long)nPad * B;
    yNeed += 4096;   // 尾部余量（列偏移最大到 blockDim*columns*qd）

    long o = 0;
    long mnOrPad = (mn > np2) ? mn : np2;
    // Phase 1 的 A0 是完整 mPad x nPad；Phase 2 的 A0 只需 Σ 2*nPad_b²
    const long mnBatch = (mnOrPad > 2 * batchF) ? mnOrPad : 2 * batchF;
    a = o;          o += mnOrPad;
    a0 = o;         o += mnBatch;
    xsdc = o;       o += 2 * batchF;
    // P^T / X^T：QDWH 的 M*N（= (qd-N)*N），split 时 = Σ 2*nPad_b²
    const long mnQdwh = ((long)qd - n) * (long)n;
    const long ptNeed = (mnQdwh > 2 * batchF) ? mnQdwh : 2 * batchF;
    pt = o;         o += ptNeed;
    xt = o;         o += ptNeed;
    hwork = o;      o += hw;
    qbuf = o;       o += qb;
    tau = o;        o += (long)B * nPad;
    tmat = o;       o += (long)B * kQdwhTStride + kQdwhTPanel + kQdwhTSlack;
    w = o;          o += wTotal;
    y = o;          o += 2 * yNeed; // AIV ping-pong Y followed by AIC T*Y cache
    coef = o;       o += 128;
    eye = o;        o += 128 * 256;
    eyeN = o;       o += (long)n * n;
    uomega = o;     o += 2 * np2;
    qt = o;         o += np2;
    tmp = o;        o += np2;
    hplus = o;      o += np2;
    hminus = o;     o += np2;
    om1 = o;        o += (long)n * n;
    om2 = o;        o += (long)n * n;
    arena = o;      o += arenaFloats;
    leaf = o;       o += leafCap;
    // Scalar DCCI operates on whole cache lines. Keep every control array
    // and sigma scratch aligned and isolated, including odd input sizes.
    // Otherwise one 4-int descriptor can straddle two lines and retain
    // a stale ld; the final partial metadata line can also stay dirty.
    o = (o + 31) / 32 * 32;
    desc = o;       o += 4 * B;
    o = (o + 31) / 32 * 32;
    copyInfo = o;   o += 16;
    o = (o + 31) / 32 * 32;
    nodeH = o;      o += maxNodes;
    o = (o + 31) / 32 * 32;
    nodeN = o;      o += maxNodes;
    o = (o + 31) / 32 * 32;
    nodeLd = o;     o += maxNodes;
    o = (o + 31) / 32 * 32;
    cnt = o;        o += 8;
    o = (o + 31) / 32 * 32;
    leafN = o;      o += maxLeaves;
    o = (o + 31) / 32 * 32;
    leafLd = o;     o += maxLeaves;
    o = (o + 31) / 32 * 32;
    leafOff = o;    o += maxLeaves;
    o = (o + 31) / 32 * 32;
    sigma = o;      o += sigmaCap;
    o = (o + 31) / 32 * 32;
    sigmaOff = o;   o += maxLeaves;
    o = (o + 31) / 32 * 32;
    sigmaCount = o; o += maxLeaves;
    o = (o + 31) / 32 * 32;
    sigmaLeafCount = o; o += 1;
    o = (o + 31) / 32 * 32;
    kTmp = o;       o += 1;
    o = (o + 31) / 32 * 32;
    lambdaTmp = o;  o += 256;   // AIV 逐核的 lambda 分片（连续 blockNum 个 float）
    o = (o + 31) / 32 * 32;
    dbg = o;        o += 128;
    o = (o + 31) / 32 * 32;
    shapes = o;     o += 8;
    total = o;
}
