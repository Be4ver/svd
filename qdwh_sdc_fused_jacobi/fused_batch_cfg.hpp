#pragma once
// ============================================================================
// 同一 BFS 层 batch 执行 QDWH 的公共常量（host 的 fused_layout.hpp 与 device 的
// qr_custom.cpp / vec.hpp / sdc_node.hpp 共用一份，保证两边算出的偏移完全一致）。
//
// 语义：Phase 2 每轮从 device 节点队列头取最多 kSdcBatchMax 个节点，把它们的
//       sign-QDWH 迭代环（QdwhInitX -> maxIter x [QdwhPrepare /
//       ComputeAll(QR)] ）打包成**一次 batch QDWH**，从而把每轮固定的
//       逐行搬运 + 跨核 flag 成本按批大小摊薄。
// ============================================================================

// ---- QR panel 的点积区 / 支持的最大 qd ----
// ComputeBatch 把点积结果写在 workLocal[0 .. columns*curM)，Phase 1 的
// curM = qd = mPad + nPad。workLocal 的容量 / columns 就是这个 QR 实现
// 真正能支持的 qd 上限，host 的 shape guard 必须用它，否则会出现
// “guard 放行、device 静默算错”的情况（旧的 4096 就是这样来的）。
static constexpr int kQrColumns = 2;
static constexpr int kQrMaxQd = 6144;

// QR 的 chunkSize：单列的 float 容量，也是 aQueueIn（2*chunkSize）的一半。
static constexpr int kQrChunkSize = 8192;

// ============================================================================
// kSdcBatchMax 由 AIV 的 UB 预算推导，不再硬编码。
//
// qdwh_vec::Init 一共分配 6 个 UB 缓冲，只有 tQueueDst 随 batch 线性增长，
// 其余 5 个与 batch 无关。另外 householder_v(= aQueueOut, 8192 float) 内部
// 按 batch 切片（hvW / wgAll / tmp8 / tau·v0·tauUb），这是更紧的那个约束。
//
// 因此上界 = min(UB 总量约束, householder_v 约束, 定长数组约束)，
// 下面的 static_assert 会在编译期兜住它。
// ============================================================================
static constexpr int kAivUbBytes = 192 * 1024;

// 与 batch 无关的固定占用：aQueueIn / bQueueIn / aQueueOut / tauQueueOut
static constexpr int kUbFixedBytes =
    2 * kQrChunkSize * 4      // aQueueIn  = 2 * chunkSize float
  +     kQrChunkSize * 4      // bQueueIn  =     chunkSize float
  +     kQrChunkSize * 4      // aQueueOut =     chunkSize float
  +     2 * 32 * 4;           // tauQueueOut = 2 * 32 float
// 点积区：columns * kQrMaxQd float
static constexpr int kUbWorkFloats = kQrColumns * kQrMaxQd;

// householder_v(= aQueueOut, 8192 float) 内每个矩阵占的 float 数：
//   hvW 8*columns + wgAll TRowStride
// TRowStride = align_blockdim(<= 32) * columns * 8 = 512（vec.hpp:735 已把
// blockDim 夹到 32，所以这是与运行参数无关的常量）。
// tau/v0/tauUb 已挪进 T_local 的空闲后半段，不再占 householder_v。
static constexpr int kQdwhTRowStrideMax = 32 * kQrColumns * 8;               // 512
static constexpr int kPerBatchInHouseholder = 8 * kQrColumns                  // hvW
                                             + kQdwhTRowStrideMax;            // wgAll

// 约束①：householder_v 能装下 B 个矩阵（另留 tmp8 的 8 float）
static constexpr int kBatchCapHouseholder =
    (kQrChunkSize - 8) / kPerBatchInHouseholder;                              // 15

// 约束②：tQueueDst 每个 batch 256 float（blockDim<=64 时 blockDim*4<=256），
//        其中后 128 float 还给 tau/v0/tauUb 当 scratch（128 + 3*8 = 152 <= 256）
static constexpr int kBatchCapTQueue =
    (kAivUbBytes - kUbFixedBytes - kUbWorkFloats * 4) / (256 * 4);            // 15

// 约束③：QdwhPrepare 里的 prefH/prefQ/offA/offH/offQ/offT/sqcArr 都是 16，
//        且用到 prefH[batch]，故 B + 1 <= 16
static constexpr int kBatchCapArrays = 15;
static_assert(128 + 3 * 8 <= 256, "tau/v0/tauUb 放不进 T_local 的空闲后半段");

static constexpr int kSdcBatchMaxUb =
    (kBatchCapHouseholder < kBatchCapTQueue)
        ? ((kBatchCapHouseholder < kBatchCapArrays) ? kBatchCapHouseholder : kBatchCapArrays)
        : ((kBatchCapTQueue < kBatchCapArrays) ? kBatchCapTQueue : kBatchCapArrays);

#ifndef SDC_BATCH_MAX
#define SDC_BATCH_MAX kSdcBatchMaxUb
#endif
static constexpr int kSdcBatchMax = SDC_BATCH_MAX;

static_assert(kSdcBatchMax >= 1, "batch 上限必须 >= 1");
static_assert(kSdcBatchMax <= kBatchCapHouseholder,
              "batch 超出 householder_v 的切片容量（改 vec.hpp 的布局或减小 batch）");
static_assert(kUbFixedBytes + kUbWorkFloats * 4 + kSdcBatchMax * 256 * 4 <= kAivUbBytes,
              "batch 超出 AIV 的 192KB UB 预算");
static_assert(kSdcBatchMax + 1 <= 16,
              "batch 超出 QdwhPrepare 的 16 项定长数组（需同步扩大 vec.hpp 里的数组）");

// ---- QDWH 的 W 区 ----
// AIV 侧每个矩阵一段（w 交换，固定 2048 float）；AIC 侧每个矩阵 128*max(M,N)。
// 两段之间留 kQdwhWGap 的空档（保证 AIC 的 W 基址仍在 128B 对齐的高位）。
static constexpr int kQdwhWVecStride = 2048;
static constexpr int kQdwhWVecTotal = kQdwhWVecStride * kSdcBatchMax;
static constexpr int kQdwhWGap = 16384;

// ---- QR 的 T 区 ----
// cube 侧 panelTSize = 128*128，vec 侧 panelTSize = 32*32*16（两者等值）；
// 每个矩阵 128*256，另有 AIV 侧的 blockDim*columns*columns 偏移余量。
static constexpr int kQdwhTPanel = 128 * 128;
static constexpr int kQdwhTStride = 128 * 256;
static constexpr int kQdwhTSlack = 8 * 1024;
