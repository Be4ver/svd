# 融合 QDWH 极分解 + device 内谱分治（SDC）—— 一次 kernel launch

> **当前版本（2026-11-13）**：单次 kernel 已包含叶块 Jacobi，输出原始输入矩阵的奇异值 `output/sigma.bin`；奇异值按叶块组织，整体未排序。1024 与 1999 方阵的修复、精度、kernel 时间和复现命令见 [回归报告](regression_20261113/repair_1024_1999_20261113.md)。以下历史记录中关于仅输出 leaves 的描述保留作开发记录。


把原来“两次 main / 两个 kernel”的流程

```
qdwh_main:     A -> qdwh_custom        -> U, H        （写 h.bin/u.bin）
sdc_dev_main:  H -> sdc_dev_custom     -> leaves      （读 h.bin）
```

融合成**一个 host main、一次 kernel launch**：

```
fused_main:    上传 A 一次 -> launch qdwh_sdc_fused_custom 一次 -> 下载 leaves
```

* 计算期间 host 与 device **没有任何数据传输**：不落 `h.bin`/`u.bin`，
  不读节点元数据；
* `H` 直接由 QDWH 的最后一步写进 SDC arena 的 `slot0`（偏移 0），SDC 递归
  的第一个节点就从这里读；
* 递归完全在一个 kernel 内跑完（节点队列 / H± 搬运 / 叶块收集都在 device 上）；
* kernel 结束后 host 只下载 `leaves`（本实现另外可选下载 `H`/`U` 供
  `verify.py` 校验，这部分在计算结束之后，不影响“计算期间零传输”）。

## 任意形状输入（行列分别对齐，H 边长为 min 维）

`fused_main` 不再要求 M、N 是 128 的倍数，也不要求二者相等：

* 行列分别对齐：`r=max(M,N)`、`c=min(M,N)`，`mPad=ceil128(r)`、`nPad=ceil128(c)`；
* 若 `M<N`，host 先把 A 转置（kernel 始终按“行数 >= 列数”算），
  因此 QDWH 的 H 是 `nPad × nPad`，**边长对应原始 min(M,N)**，不会引入 max 维的零块；
* QDWH 在 `mPad × nPad` 上算 `U/H`；SDC root 取 H 左上 `c × c`（`nRoot=c`）；
  `c<nPad` 时先把 root H 压成 `ld=c` 的紧凑块，再递归；
* host 输出：`M>=N` 时 `A=U H`；`M<N` 时 kernel 算的是 A^T 的极分解，输出
  `U=U_T^T`、`H=(A A^T)^{1/2}`（`c × c`），满足 `A=H U`；
* **Phase 2 的 batch 上限由 AIV 的 UB 预算推导**（不再硬编码）：`fused_batch_cfg.hpp`
  用 `constexpr` 算出 `kSdcBatchMax`，并有 `static_assert` 兜底。当前 AIV UB 用量：

  | 缓冲 | 字节 | 随 batch |
  |---|---:|---|
  | `aQueueIn` (2×8192 float) | 65,536 | 否 |
  | `bQueueIn` (8192 float) | 32,768 | 否 |
  | `aQueueOut` (8192 float) | 32,768 | 否 |
  | `tauQueueOut` (2×32 float) | 256 | 否 |
  | `workQueue` (2×6144 float) | 49,152 | 否 |
  | `tQueueDst` | 1,024·B | **是** |

  三个上界取最小：`tQueueDst` → B≤15；`householder_v` 内部切片
  （`hvW 8·columns + wgAll TRowStride` = 528 float/矩阵，`tau/v0/tauUb` 已挪进
  `T_local` 的空闲后半段 `[128·B, 256·B)`）→ B≤15；`QdwhPrepare` 的
  `prefH/prefQ/...` 定长 16 → B≤15。
  故 **`kSdcBatchMax = 15`**（UB 实际用量 195,840 B / 196,608 B，仅剩 768 B）。
  `vec.hpp` 里原来的 `const int kMaxG = 8;` **已绑到 `kSdcBatchMax`**——它才是给
  `tQueueDst`/`householder_v` 定尺寸的常量，只改 `SDC_BATCH_MAX` 会直接越界。
* **batch 已到 UB 天花板，继续加大收益已饱和**（实测，同参数、各 2 次取小）：

  | 输入 | batch=8 | batch=15 | 分组数 8→15 |
  |---|---:|---:|---|
  | 1024×1024 | 50.15 ms | 49.96 ms | 6→6 |
  | 1999×1999 | 135.92 ms | **132.60 ms** | 9→8 |
  | 2048×2048 | 136.09 ms | 134.97 ms | 9→8 |
  | 3072×3072 | 401.7 ms | 403.5 ms | 11→8 |

  3072² 的分组数从 11 降到 8，时间却没有变化 —— 说明**每组固定开销已经被摊薄，
  继续合并分组不再是有效杠杆**。要再往上只能放开 UB：固定占用里 `aQueueIn`
  65,536 B（2×chunkSize）和 `workQueue` 49,152 B（columns×kQrMaxQd）是大头，
  压缩它们会直接牺牲 QR 的 panel 宽度或支持的 qd。
* **lambda 计算改成跨 AIV 分布式归约**（已落地）：原来 40 个 AIV 核**各自**把
  H 的整条对角线读一遍（`n` 次 scalar GM 读 × 40），现在行按 `blockNum` 平均切段、
  每核只读自己那一段，把部分和连续写进 GM，一次 `AivBarrier` 后全体读回 UB 用
  `WholeReduceSum` 归约。实测 2048²：`aiv_scalar` 21007 → **19917 µs（−5.2%）**，
  kernel 132.76 → 132.29 ms；1999² 连跑 8 次 132.5~133.1 ms（原 133.0~135.4）。
  注意：求和顺序改变会让 lambda 末位不同，从而改变分裂树（2048² 的 leaves 47→46），
  结果仍在 1.5e-06 量级。
* **已实测证伪的优化方向（别再重做）**：
  * 去掉 UΩ1/UΩ2 之间那道冗余 `AicBarrier`（两者写区不相交）：1999² 4 次
    133.7/131.9/133.9/132.2 → 133.7/132.1/133.9/132.0 ms，**零收益**。
  * 把 λ 的 trace 从"两趟循环（n 次 DCCI + n 次 scalar 读）"合成一趟：
    同样**零收益**。
  * 结论：这两类 scalar/同步操作实际都命中 cache，量级远小于"170 ns/次"的
    估算值；kernel 已经不是"调度开销"受限，而是计算（QR 串行链 / GEMM）受限。
    下一步要先 `msprof` 定位，而不是继续猜。
* 叶子阈值默认 **64**（`run.sh` 的 `THRESHOLD=64`），即每个叶子 `n <= 64`；
* 当前 QDWH/QR 管线限制 **`mPad + nPad <= kQrMaxQd = 6144`**（`fused_batch_cfg.hpp`）。
  这个上限就是 AIV QR panel 点积区 `workLocal[0 .. columns*curM)` 的容量除以
  `columns`，host guard 直接引用同一个常量，避免两边脱节。可跑的例子：
  `2048×2048`(qd=4096)、`2560×2560`(5120)、`3072×3072`(6144)、`4096×2048`(6144)、
  `5120×1024`(6144)；`3200×3072`(6272) 会被 host 拒绝，而不是静默算错。
* **大 n 需要更多 sign-QDWH 迭代**：`maxIter=4` 在 `n <= 2560` 够用，
  `n >= 2816` 需要 `maxIter >= 6`（实测 2816/3072 在 4 次时 `rel_l2` 约 5e-6~2e-5，
  6 次回到 6e-7；8 次与 6 次同level，说明 6 已收敛）。`run.sh` 默认仍是 4。
* **已知既有缺陷：特定矩阵上 SDC 结果不可复现且超差**（与尺寸对齐无关）。
  实测 `1000×800`（`mPad=1024`、`nPad=896`、`cOrig=800`）在**固定种子 42** 下
  奇异值误差 `1.5e-05 ~ 3.1e-04` 且逐次运行不同；但**同一个形状换随机种子**
  （seed 1/2/3）三次都稳定在 `~2e-06`，`1024×800` 同样正常。
  两组的叶子树（18 叶，n/ld/off 逐项一致）、谱、`lambda=trace(H)/n`、
  最近特征值距离（4.670e-04）**完全相同**，唯一差别是 H 的特征向量基。
  已排除：未初始化显存（整块 memset 0 无效）、`QdwhCopyH` 后缺 AIV 屏障（补上无效）、
  `blockDim`、`maxIter`（4/6/8/12 均重现）、`l0`（1e-6~1e-9 只部分改善）、
  `Hplus/Hminus` 的 padding 污染（只拷左上 `childN²`，padding 从不被读）。
  现有证据指向：Phase 1 的 `QdwhPmat` 用 atomic 累加，使 H 每次有 ~1e-06 的
  最低位差异，而 SDC 的 `H± = Q±ᵀ H Q±` 在 `lambda` 贴近某个特征值时把这点噪声
  放大 `~||H||/gap ≈ 2000` 倍，是否触发取决于噪声与被放大方向的夹角。
  这是**算法层面的条件数问题，不是回归**；修它要么消除 Phase 1 的 atomic 非确定性，
  要么改 SDC 的分裂数值路径。128 对齐尺寸同样可能触发（与对齐无关）。

## 当前状态（第 10 轮优化结束时的实测）

> 本目录只保留算子必需文件。下文历史章节出现的 `QdwhUpdateX`、
> `sdc_split_node`、`QdwhTransposeU`、`gemm_test_*` 等是已删除的旧实现/试验代码，
> 仅用于说明优化过程；当前生效的接口以本表与源码为准。

| 项 | 值 | 提示词目标 |
| --- | --- | --- |
| kernel（th=64，壁钟） | **53.5 ~ 55.8 ms**（均值 ~54.7；基线 67.9） | ≤50（未达，-19.4% 已达） |
| kernel（msprof） | **54.8 ms** | — |
| AIV scalar | **9.69 ms**（基线 22.48） | ≤10 ✓ |
| AIC scalar | 12.81 ms（基线 12.96） | ≤6 ✗ |
| 四阈值 | **20.2 / 28.9 / 46.6 / 55.7 ms 全 PASS** | ✓ |
| 精度 | `u_err 7.59e-07`、`eig_err` 连跑 5 次 1.92~2.45e-07、`sym_max 0` | ✓ |
| 叶块 | `leaves=23 / max_leaf=63 / all_leaves_small=True` | ✓ |

**已做的 5 个大改动**（详见 `NO_SCALAR_REPORT.md`）：
1. `QdwhInitX` 的 `Xᵀ` 初值从 AIV 逐元素 scalar 转置改成 AIC 一次 `I·Xᵀ` GEMM（**-11.1 ms**）；
2. `ComputeBatch` 的 tau 从「scalar 读 + DCCI」改成 MTE2（-0.5~0.8 ms）；
3. `Hwork` padding 清零从 AIC scalar 搬到 AIV 向量（-0.3~0.4 ms）；
4. **修掉 `SDC_BATCH_MAX≥5` 死锁**（根因：`Desc` 跨两条 cacheline 只刷了一条）并把 batch 开到 8；
5. split 的 `A0 → X` 直写（省一趟 `nPad²` 拷贝，壁钟中性）。

**为什么停在 54.7**：AIV 侧的工作几乎都被掩盖（`BuildPmats` 双写 +0 ms、`QdwhPrepare`
双写仅 +2.2 ms、省掉整趟拷贝 0 ms），剩下的 ~5 ms 全在**串行化的 QR 链**上；
而"每核 4 列 + 16 核（把链步数减半）"被 **192KB UB 预算**卡死（`a` 4 列 64KB ✓ +
点积槽 4×chunkSize 需 +32KB → 204KB > 192KB）。下一步只有两条路：
* **批量 split QR**（每组 ≤8 个节点的 QR 合并成一次 batch QR，-1.5~2 ms，需重排尾部三段）；
* 把 `a`/点积槽改 fp16 或降 `chunkSize`（改数值路径 / 大矩阵分块变多，风险高）。

## 目录

| 文件 | 作用 |
| --- | --- |
| `qdwh_sdc_fused_custom.cpp` | 融合 kernel 入口：只有 `KERNEL_TYPE_MIX_AIC_1_2` + `Init` + `RunAiv/RunCube` |
| `fused_qdwh_sdc.hpp` | **`QdwhSdcFused` 类**：Phase 1/2 的阶段推进、跨核 flag、`CopyRowsPad`（唯一的矩阵搬运入口） |
| `sdc_node.hpp` | `sdc_build_plan` / `sdc_qdwh_batch_*` / `sdc_tail_*`：**一个节点**的 split（λ / sign-QDWH / PΩ / QR / H±），AIV 与 CUBE 两个分支 |
| `qr_custom.cpp` | `compute_weight`（QDWH 权重）+ **`qdwh_cube`**（AIC 侧：`qr_update` / `QdwhPmat` / `QdwhFinalGemm`） |
| `vec.hpp` | **`qdwh_vec`**（AIV 侧：`QdwhInitX` / `QdwhPrepare` / `ComputeAll` / `BuildPmats` 等） |
| `gemm.cpp` | `MatmulCustom`：分块 GEMM（L1 256×256×128、L0 128×128×64），`Process` 支持 Aᵀ/Bᵀ，`Process2` 支持 Bᵀ |
| `fused_layout.hpp` | 工作区布局：host/device 共用同一份 `Build()`，launch 只传 1 个基址 |
| `fused_batch_cfg.hpp` | Phase 2 同层 batch 的公共常量（`kSdcBatchMax`、W/T 布局） |
| `fused_main.cpp` | host：读数据 → 1 次 `aclrtMalloc` → H2D → **1 次 launch** → D2H(leaves) → 写文件 |
| `utils.h` | host 侧 `Timer` / `ReadFile` |
| `CMakeLists.txt` | `fused_kernels` + `fused_main` |
| `gen_data.py` / `verify.py` | 数据生成 / 结果验证 |
| `run.sh` | 一键：生成数据 → 编译 → 运行 → verify（`tee output/verify.txt`） |

## 类划分（`QdwhSdcFused`）

```cpp
class QdwhSdcFused {
public:
    __aicore__ void Init(GM_ADDR wk, int m, int n, int threshold, int maxIter,
                         int maxNodes, int maxLeaves, int leafCap,
                         float alpha, float l0);

    // Phase 1：QDWH（只调用现有 Qdwh* 方法，不重写算法）
    __aicore__ void QdwhAiv();   // QdwhInitX -> (QdwhPrepare + ComputeAll)×maxIter
                                 //   -> QdwhTransposeU -> 0x5 -> 等 0x6 -> QdwhCopyH
    __aicore__ void QdwhCube();  // (qr_update + QdwhPmat + QdwhUpdate)×maxIter
                                 //   -> 等 0x5 -> QdwhFinalGemm -> 0x6

    // Phase 2：SDC（把 sdc_dev_custom 的节点循环搬进类；split 仍调 sdc_split_node）
    __aicore__ void SdcAiv();    // 等 0xB -> 读 Desc -> split/leaf -> 等 0xE -> 搬 H±/leaf -> 0xC
    __aicore__ void SdcCube();   // 节点队列 + Desc 广播 + split/分配孩子 + 0xE/0xC 握手

    __aicore__ void RunAiv()  { QdwhAiv();  PhaseBarrier(0xD); SdcAiv();  }
    __aicore__ void RunCube() { QdwhCube(); PhaseBarrier(0xD); SdcCube(); }

private:
    __aicore__ void CopyRowsPad(...);   // 所有矩阵搬运的唯一入口（AIV DataCopyPad）
    __aicore__ void PhaseBarrier(uint16_t flag);
    __aicore__ void AicSync();          // 16 个 AIC 的 barrier（0x7）
    FusedLayout L;  ...                 // GM 视图 / shapes / node 队列 / 阈值
};
```

kernel 文件只剩：

```cpp
QdwhSdcFused op;
op.Init(wk, m, n, threshold, maxIter, maxNodes, maxLeaves, leafCap, alpha, l0);
#ifdef __DAV_C220_VEC__
    op.RunAiv();
#else
    op.RunCube();
#endif
```

与旧 `qdwh_sdc_batch` 的对应关系：

| 融合算子里的位置 | 旧实现 |
| --- | --- |
| `QdwhAiv` / `QdwhCube` | `qdwh_custom.cpp` 的 AIV/CUBE 分支（逐行对应，flag 0x1~0x6 完全一致） |
| `SdcAiv` / `SdcCube` | `sdc_dev_custom.cpp` 的 AIV/CUBE 分支（节点循环 + 0x7/0xB/0xC/0xE 握手） |
| `sdc_split_node(...)` | 原样复用 `sdc_node.hpp`（λ / sign(QDWH) / PΩ / QR / H± / 对称化） |
| `CopyRowsPad` | `sdc_dev_custom.cpp` 里 DataCopyPad 搬 H+/H-/leaf 的循环 |
| `PhaseBarrier` | 新增：一个 kernel 内 Phase 1 → Phase 2 的全核同步 |

### 优化记录 10：去掉 device 侧的显式对称化（-26.5 ms）

`sdc_split_node` 里 H+ / H− 算完后原来各做一次**标量两遍对称化**：
`H ← 0.5(H + Hᵀ)`（`sdc_node.hpp`，`Tmp` 暂存 + `AicBarrier` + 拷回，每节点
约 `4 × nPad²` 次 scalar 读写 + DCCI，是 Phase 2 里 AIC scalar 的大头）。

但数学上 `H+ = Q₊ᵀ H Q₊` **本来就对称**（H 对称、Q₊ 正交归一），GEMM 只引入
~1e-7 的舍入不对称。用 `static constexpr bool kSymmetrize = false;` 关掉后：

| | kernel | `eig_err` | `sym_max` | 结果 |
| --- | --- | --- | --- | --- |
| 打开（旧） | 160.8 ms | 2.2e-07 | 0.0e+00 | PASS |
| **关闭（现在）** | **134.3 ms** | **2.0e-07** | 0.0e+00 | PASS |

阈值回归（bd=20）：1024 → 50.7 ms、512 → 70.9 ms、128 → 116.2 ms、64 → 134.0 ms，
`eig_err` 3.3e-08 ~ 2.0e-07 全部 PASS；1280×1024 → 135.3 ms PASS。

### 优化记录 11：Phase 1 的 P / Pᵀ 是什么，能不能只留一次矩阵乘加

**现在（两段式）**：`QdwhPmat`（AIC）算两次 GEMM：
* `P = Q1·Q2ᵀ`（m×n）写进 `APad` —— 唯一消费者是 AIV 的 `QdwhUpdateX`：
  `X ← θX + κP`（`Muls` + `Axpy`，读一遍 P）；
* `Pᵀ = Q2·Q1ᵀ`（n×m）写进 `pt` —— 唯一用途是维护 **Xᵀ 缓存**（`xt`），
  好让 `QdwhPrepare` 用连续行读 Xᵀ 而不是按列 scalar gather X（优化记录 6）。
  也就是说 `Pt` 不是算法需要，纯属那条 gather 优化的副产物；phase 1 用它，
  split 侧目前仍是 gather。

**可以按你说的改成一次原子加**（vec 缩放 + AIC 一次矩阵乘加）：

```
AIV: 等「QR 完成」 → X ← θ·X（vec）、Q1 ← κ·Q1（vec） → 置「已缩放」
AIC: 等「已缩放」 → QdwhPmat 用 atomic-add 直接 X += Q1·Q2ᵀ（Q1 已带 κ）
```
* 这样 `P`/`APad` 那块 scratch 与 AIV 读 P 的 MTE2 全部消失；
* 若还想保留 Xᵀ 缓存，**不需要额外缩放**：`Q2·(κQ1)ᵀ = κ·Pᵀ`，
  第二个 atomic-add GEMM 直接写 `xt`，`Pt` 缓冲也可以删掉；
* 代价：每轮多两次跨核握手（AIC→AIV「Q 就绪」、AIV→AIC「已缩放」），
  需要从 QR 内部那对 `flag^1`（0/1 交替）里腾一个 flag 出来，这是一次独立的回归项。
* 注意原子加要求"先缩放、后累加"，所以缩放必须在 `QdwhPmat` **之前**完成——
  也就是 `qr_update` 之后、`QdwhPmat` 之前插这两个握手（不能像现在这样把更新放在
  下一轮开头）。

### 优化记录 12：Phase 2 的 SDC 节点改成「同层 batch QR」（-11.8 ms）

**问题**：Phase 2 原来一个节点调一次 `sdc_split_node`（λ / A0 / sign-QDWH 4 轮 /
k / P,P_ / UΩ / 转置 / QR / H±）。QDWH 的 4 轮迭代每轮都要付一次「逐行
`DataCopyPad` + 跨核 flag + QR panel 循环」的固定成本，23 个节点就是 23 份；
消融拟合 `kernel ≈ 42.6 + 3.75 ms/节点`，节点占 66%。

**做法**：把 device 节点队列**队头连续的最多 4 个节点**打包成一组（队列本身就是
BFS 序，所以天然同层；跨层混批也无害，batch QR 的 panel 数 = 组内 `maxK/64`）：

1. 每个内部节点各自算 `λ = trace(H)/n`，`A0 = pad(H−λI)` 写到 `a0` 的第 b 个切片；
2. 这 B 个矩阵跑**一次 batch QDWH**：新增的 `qdwh_vec::SetDimsBatch(mArr,nArr,B)` /
   `qdwh_cube::SetDimsBatch(...)` 直接给定 `m[b]=2*nPad[b]`、`n[b]=nPad[b]`，
   `QdwhInitX / QdwhUpdateX / QdwhPrepare / ComputeAll / qr_update / QdwhPmat`
   全部按 b 切片跑（这些接口本来就是 batch 实现，内部按 `m[]/n[]/size[]/yOff[]/
   tauOff[]/qOff[]` 递推）；
3. 之后仍是**逐节点** tail（k / P,P_ / UΩ / 转置 / QR / H±）与逐节点 `0xE/0xC` 搬运。

**配套改动**：

* `fused_layout.hpp`：`a0/xsdc/pt/xt/hwork/qbuf/tau/tmat/w/y` 从「单节点」放大到
  「batchMax 份」，workspace 611 → 739 MB（arena 512 MB 不变）；`desc` = `4*batchMax`。
* 新增 `fused_batch_cfg.hpp`：`kSdcBatchMax`(=4)、W/T 的布局常量（host/device 共用；
  原来 `qdwh_cube::Init` 里硬编码的 `W + 2048 + 16384` 改成 `kQdwhWVecTotal + kQdwhWGap`）。
* flag **逐位不变**，只是 `0x4/0x6/0x2/0x1` 由「每节点一次」变成「每组一次」；
  `0xB` 改成一次派发一整组（组大小在 `cnt[6]`，组数统计在 `cnt[7]`）。
* **新增 `0x5` 握手（AIV→AIC「X 定稿」）**：`k = (trace(U)+nPad)/2` 是 AIC 用 scalar
  读 AIV 的 `QdwhUpdateX` 结果，而原实现里 AIC `SetFlag(0x6)` 之后**没有等** AIV 的
  最后一轮 X 更新。单节点时窗口小；batch 后窗口是 `Σ_b nPad[b]²`，偶发读到旧 U →
  k 偏 1 → `eig_err` 从 2.4e-07 跳到 9.4e-04（复现率 ~1/5）。补上 0x5 后稳定。
* `Q = I` 从「AIC 一次 `eyeN·eyeN` GEMM + 整块 DCCI」搬到 AIV
  （`qdwh_vec::BuildIdentityQ`，按行 `Duplicate(0)+SetValue(对角)+DataCopyPad`），
  省掉 cube 关键路径上一次 `nPad³` 乘加（-0.6 ms）。注意**不能**「整块 DataCopy 清零
  + 标量补对角」——MTE3 与 scalar 无同步会把 1 覆盖成 0（实测 leaves 23→24）。

**实测**（bd=20、maxIter=4、1024×1024、cond=1e2）：

| threshold | 改动前 | 改动后 | 组数 | leaves |
| --- | --- | --- | --- | --- |
| 1024 | 47.9 ms | 48.0 ms | 1 | 2 |
| 512 | 67.7 ms | 68.1 ms | 2 | 3 |
| 128 | 111.5 ms | **104.7 ms** | 5 | 12 |
| **64（验收）** | **128.8 ms** | **117.6 ms** | **8** | **23** |

精度：`u_err` 7.56e-07、`eig_err` 1.79e-07（比改动前的 2.20e-07 还好）、`sym_max` 0、
`leaves=23 / max_leaf=63`，四个 threshold 全 PASS。

msprof（th=64）：**算术没变**（AIC cube 3.7 → 3.73 ms、AIV vec 6.6 → 6.57 ms），
省下来的是标量簿记与指令数：**AIV scalar 49.0 → 45.0 ms、AIV mte2 9.9 → 8.0 ms**、
AIC scalar 39.2 → 38.8 ms。

**为什么没到 110 ms**：截距里的 Phase 1(6.7 ms) + **根节点 split(~36 ms) 是一个
只有 1 个节点的组，batch 摊不到它**；剩下的 AIC scalar 38.8 + AIV scalar 45.0 ms
是 `Hᵀ/Q±ᵀ` 的 O(n²) 逐元素 DCCI（提示词 §5.3 明确不能批量，试过会退到 0.4~0.8）
和 O(n³) GEMM 的 invoke 开销（在 `MatmulCustom::Process` 里，要求不改 `gemm.cpp`）。
`batchMax=8` 与 `blockDim≥24` 都会卡死（kMaxG 边界 / 只有 20 个 AIC），没有余量。

踩坑与完整数据见 `BATCH_QR_REPORT.md`。

### 优化记录 13：split 的三处标量转置改成 GEMM 隐式转置（-26.6 ms）

`BATCH_QR_REPORT.md` 的瓶颈分析里，AIC scalar 38.5 ms 的大头是 `sdc_node.hpp` split 里的
**三处逐元素 scalar 转置**（`Hwork = Aᵀ`、`QT = Q₊ᵀ`、`QT = Q₋ᵀ`）：每处 `nPad²` 量级的
`GetValue` + `SetValue` + **每元素一次 `DataCacheCleanAndInvalid`**。

三处全部改成 **GEMM 的隐式转置**（`C = I·Bᵀ`，`isTransposeB`）：

```cpp
mm.SetMatrixLd(eyeN, B, C, /*N=*/nPad, /*K=*/k, /*ldA=*/ldOmega, /*ldB=*/nPad);
mm.Process(rowStart*ldOmega + kc, kc, rowStart*nPad, rows, /*blockN=*/n, kk, -1,
           /*isTransposeB=*/true, kc != 0);
```

三个要点：

* **`dstStride` 恒等于 `Process` 的 `N` 参数**（Fixpipe 不走 dstStride），所以 `N` 必须给
  目标矩阵的行宽 `nPad`，而**输出列数**用 `blockN=n` 单独限制 —— 这样既对上 `ld`，
  又不会把 padding 列写成 `UOmega` 的残留；
* `A` 操作数直接用 host 生成的 `eyeN`（`ld=ldOmega`），`K` 只取 `k`/`m2`，所以
  `k..kPad-1` 的补齐行**自动是 0**（原来那两处 `if (t < k) ... else 0` 的补零可以删）；
* padding（行 ≥ n 的整行、列 ≥ n）在 **GEMM 之前**清 0 —— GEMM 只写 `[0:n, 0:n]`，
  清 0 不会被覆盖；放前面还能避免「Fixpipe 写列 < n」与「scalar 清列 ≥ n」落在
  同一条 cache line 上的竞争。

顺带删掉：`qr_update` 之后那圈 **`Q` 的整块 DCCI**（`Q` 的 consumer 现在全是 MTE）。

| threshold | 改动前 | 改动后 | Δ |
| --- | --- | --- | --- |
| 1024 | 48.3 ms | **35.6 ms** | -12.7 |
| 512 | 68.1 ms | **49.8 ms** | -18.3 |
| 128 | 104.7 ms | **79.6 ms** | -25.1 |
| **64** | **117.5 ms** | **90.9 ms** | **-26.6** |

msprof（th=64）：**AIC scalar 38.53 → 12.94 ms（-2/3）**，而 cube/vec/mte 全都不变
（算术没变，省的就是 cache 维护 + scalar 读写）。精度还变好了：`eig_err`
2.0e-07 → **1.4e-07~1.9e-07**，`leaves=23 / max_leaf=63 / sym_max 0`，四阈值全 PASS。

⚠️ **坑**：`Hwork` 现在由 **Fixpipe** 写，`0xA`（通知 AIV「Hwork 可以读了」）必须挂在
**`PIPE_FIX`** 上；原来 `PIPE_MTE3` 只保证 MTE3 序，实测 4 次里有 1 次 `eig_err`
跳到 **3.1e-03**（`leaves` 仍是 23，很容易漏掉）。改成 `PIPE_FIX` 后连跑 13 次全 ≤2.9e-07。

余下未做的项（P1/P2/P4/P5/P6）与原因见 `PENDING_OPTS_REPORT.md` §5：
瓶颈已经换到 **AIV scalar 43.8 ms**（QR 面板锁步，属"不改 QR 算法"的边界），
其余项都要先做 flag 账本重排，单项收益 1~4 ms。

### 优化记录 14：split 打开 `Xᵀ` 缓存 + QR 面板批量归约（-22.8 ms）

**问题**（优化记录 13 之后的 profile）：`AIC scalar` 已从 38.5 降到 12.9 ms，
**AIV scalar 43.8 ms** 成了唯一大头。最大的一笔是 `QdwhPrepare` 建 `H = [√c·Xᵀ | I]`
时的**按列 scalar gather X**（`hasXt=false` 的路径）：
`hRow.SetValue(c, QdwhA.GetValue(c*N + r))`，每轮 `O(nPad²)` 次 GM 标量读，
再加 `InvalidateXAll()` 的 `O(nPad²/16)` 次 DCCI，4 轮下来非常可观。

**做法**：split 的 batch QDWH 也打开 `hasXt`（phase 1 一直在用这条路径）：

* `QdwhInitX` 建 `Xt = Xᵀ` 初值（一次性）；
* `QdwhUpdateX` 用向量维护 `Xt ← θXt + κPt`；
* `QdwhPrepare` 变成**连续行 `DataCopyPad` 读 Xᵀ**（一条指令取 M 个 float）；
* 代价：`QdwhPmat` 多算一块 `Pᵀ = Q2·Q1ᵀ`（AIC 侧 fixpipe 只 +0.04 ms）。

**必须同时修的三个 bug**（少修一个结果就是 `eig_err 5.5e-01`）：

1. `qdwh_cube::QdwhPmat` 的 `Pt` 偏移写死 `pAcc = 0` → batch>1 时所有矩阵的 `Pᵀ`
   都写第 0 块；改成 `Pt[xOffArr[b]]`（与 `X` 的切片偏移一致）；
2. `sdc_qdwh_batch_cube` 调 `SetQdwhBuffers(x, a0, coef)` **漏了 `pt`** → cube 侧
   `hasPt=false` → **`Pᵀ` 的 GEMM 根本没跑**；改成传 `(x, a0, coef, pt, xt)`；
3. `QdwhInitX` 建 `Xt` 初值时要**跨核 scalar 读 X**（别的核刚 MTE3 写的行）→ 前面
   必须补 `AivBarrier()`，否则读到旧值（`leaves` 23→25）。

**另外**：`ComputeBatch` 的 Phase 2 / Phase 4c 把每个 `(i,j)` 的
`Mul → ReduceSum → GetValue → Axpy` 改成「`columns` 个点积先全部 `Mul+ReduceSum`，
再一次性读回标量」（Phase 2 不能跨 i 批：i 的 `Axpy` 会改 `a`）。收益 1.1 ms
——反过来说明 AIV 的 scalar 里大部分其实是 flag 自旋等待。

| threshold | 改动前（记录 13） | 改动后 | Δ |
| --- | --- | --- | --- |
| 1024 | 35.4 ms | **27.2 ms** | -8.2 |
| 512 | 49.7 ms | **37.0 ms** | -12.7 |
| 128 | 79.5 ms | **58.4 ms** | -21.1 |
| **64** | **90.7 ms** | **67.9 ms** | **-22.8** |

msprof（th=64，66.7 ms）：**AIV scalar 43.81 → 22.48 ms**、`vec` 6.58 → 6.31、
`mte2` 8.22 → 8.34；AIC `scalar` 12.95 → 12.96、`cube` 3.73 → 3.94（多算了 `Pᵀ`）。
精度还更好：`eig_err` 1.9e-07 → **1.7e-07**（连跑 5 次 2.05~2.51e-07），
`leaves=23 / max_leaf=63 / sym_max 0`，四阈值全 PASS。

未做的项（把迭代搬 AIC 的 Y+原子加、panel 加宽、Phase 4c 向量化）与原因见
`AIC_ITER_REPORT.md` §3（其中 panel 加宽经推导会**增加**总点积数，不建议做）。

### 优化记录 15：`Xᵀ` 初值改成 GEMM（-11.1 ms，67.9 → 56.9）

**问题**（记录 14 之后的 profile）：`AIV scalar` 已降到 22.5 ms，但 `QdwhInitX` 里
`Xᵀ` 的初值仍是**逐元素 scalar 转置**：

```cpp
for r: for c: QdwhXt[tOff + r*M + c] = QdwhA.GetValue(xOff + c*N + r);   // 每节点 N*M 次
```

外加 `AivBarrier()` 和 `InvalidateXAll()`（`nPad²/16` 次 DCCI）。它在**每个节点**的
QDWH 前都做一次（phase 1 + split 全部 23 个节点），是全树 ≈1e7 次 scalar 读写。

**做法**：改成 AIC 上一次 `I·Xᵀ` 的 GEMM（与 `Pᵀ = Q2·Q1ᵀ` 同款）：

* `qdwh_cube::QdwhInitXt(xt, eyeN, ldEye)`：
  `C = Xt (N×M, ld=M) = EyeN (N×N) · Xᵀ`，`isTransposeB=true`，
  `Process(rowStart*ldEye + kc, kc, rowStart*M, rows, M, kk, -1, true, kc != 0)`，K 方向 128 分块；
* `QdwhInitX` 删掉整块 scalar 转置（含 `AivBarrier`/`InvalidateXAll`）；
* 同步复用已有 flag，不新增 id：AIV 写完 X → `SetFlag<0x2,PIPE_MTE3>(0x5)`
  （沿用 "AIV→AIC：X 定稿" 语义，一 set 对一 wait）→ AIC `Wait(0x5)` →
  `QdwhInitXt` → `AicBarrier` → `SetFlag<0x2,PIPE_FIX>(0x4)`。AIV 第一轮
  `QdwhPrepare` 本来就在等 `0x4`，所以只是把"第一个 0x4 的放行点"挪到转置之后，
  其余 flag 账本完全不动；
* phase 1 与 split 两处都改（`ldEye = n_`，`L.eyeN` 是 root 的 `n×n` 单位阵，
  `nPad ≤ n` 所以任意节点都能当 A 操作数）。

**代价**：AIC 多一次 `nPad³` 转置 GEMM —— cube +0.08 / fixpipe +0.07 / mte2 +0.28 ms。

| threshold | 记录 14 | 记录 15 | Δ |
| --- | --- | --- | --- |
| 1024 | 27.2 ms | **20.8 ms** | -6.4 |
| 512 | 37.0 ms | **29.1 ms** | -7.9 |
| 128 | 58.4 ms | **47.7 ms** | -10.7 |
| **64** | **67.9 ms** | **56.9 ms** | **-11.0** |

msprof（th=64，55.4 ms）：**AIV scalar 22.48 → 12.00 ms**，`vec/mte2/mte3` 基本不变；
AIC `cube` 3.94 → 4.02、`scalar` 12.96 → 13.21（转置 GEMM 的 invoke 开销）。
精度不变：`eig_err` **1.51~2.25e-07（11 次里 1 次 3.21e-07，属 atomic 累加顺序的
run-to-run 波动，见 NO_SCALAR_REPORT.md §4.1）**、`u_err 7.59e-07`、
`leaves=23 / max_leaf=63 / sym_max 0`，四阈值全 PASS，
th=64 连跑 5 次 55.7~56.1 ms（另一时段 56.4~57.6，随机器负载 ±1.5 ms 漂移）。

消融（"同一幂等操作做两遍看壁钟增量"）与否定结论见 `NO_SCALAR_REPORT.md` §3，要点：
`BuildPmats/IdentityQ` 双写 **+0 ms**（§2.2 拆两段无收益）、AIC tail 全部 GEMM 双写 +1.5 ms、
`QdwhPmat` +1.9 ms、`QdwhPrepare` +2.2 ms、`QdwhUpdateX` +0.5 ms；**每多一轮 QDWH ≈10.9 ms**
（QR 迭代占 44/56.9）；`maxIter=3` 时 `eig_err 2.6e-03`（不能减轮）；
提示词 §3 说的“悬空 0x2”**不是悬空**（删掉后 B=4 也死锁，它对账 AIV 的 panel 0）。

### 优化记录 16：修掉 `SDC_BATCH_MAX≥5` 死锁 + 开 batch 8（-0.4 ms，且解锁后续优化）

**根因**（排查了 4 轮才定位）：`L.desc` 是 `4*B` 个 int32。B=4 时正好 64B（一条 cacheline），
派发处一次 `DCCI(Desc[0])` 就能整块刷回 GM；**B≥5 时是 80B+（跨两条 cacheline）**，
而派发只刷了 `Desc[0]` 那一条 —— 第 5+ 个节点的表项留在 AIC 的 scalar cache 里没写回。
AIV 读到第 5 个节点的 `gN` 是**上一组的旧值** → 两侧 `sdc_build_plan` 算出不同的
`cntI/maxK` → QR 的 panel 数不一致 → `0x1/0x2` 握手死锁。

**修复**（`fused_qdwh_sdc.hpp`，`SdcCube` 派发处）：把单次 DCCI 改成逐线刷：
```cpp
for (int o = 0; o < 4 * g; o += 8) {   // 32B 粒度，保守覆盖 B>=5 / B>=8
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(Desc[o]);
}
```
验证：512×512/th=48（会出现 5 节点组）修复前卡死、修复后 24.7 ms；1024 下 B=5/B=8
均可跑；`SDC_BATCH_MAX` 提到 **8**（组数 8→**6**）。

| 版本 | th=64 壁钟（各 3 次） | 组数 |
| --- | --- | --- |
| `.bak15`（B=4） | 55.3 / 55.7 / 55.0 · 55.3 / 55.1 / 55.3 | 8 |
| **`.bak16`（B=8）** | 53.8 / 55.7 / 53.8 · 55.1 / 56.4 / 54.8 | **6** |

四阈值 21.4 / 28.7 / **48.8** / **54.9**（th=64 与 th=512 改善，th=128 反而 +2 ms——
因为该档组数没变，只是每 panel 的活变多）。`leaves=23 / max_leaf=63 / sym_max 0`，
`eig_err` 1.60~3.27e-07。收益小（-0.4 ms）但**死锁修好本身解锁了"更大的 batch / 批量 split QR"**
这条路。备份 `.bak16`。

## 算子流程（阶段 → 函数 → 同步）

一次 `fused_main` = **host 侧 6 步 + device 侧 1 个 kernel**，kernel 里 AIV（40 个）与 AIC（20 个）
并行跑各自的分支，靠 flag 对齐。**计算期间 host 不参与**。

### 0. Host（`fused_main.cpp`）

| # | 动作 |
| --- | --- |
| 1 | 读 `input/shapes.txt` + `input/matrix.bin`（单个 m×n 矩阵） |
| 2 | `FusedLayout::Build(m, n, maxNodes=128, maxLeaves=512, leafCap)` → 一块 607 MB workspace 的偏移表 |
| 3 | **1 次** `aclrtMalloc`（整块 workspace）+ 几个小缓冲（leafCount / leafN / leafLd / leafOff / kTmp / dbg） |
| 4 | H2D：A、Ω1/Ω2（LCG 种子 20261026）、eye、shapes、cnt=0 —— **全部在计算前** |
| 5 | **1 次** `ACLRT_LAUNCH_KERNEL(qdwh_sdc_fused_custom)(blockDim, stream, wk, m, n, threshold, maxIter, maxNodes, maxLeaves, leafCap, alpha, l0)` |
| 6 | `aclrtSynchronizeStream` → D2H(leaves + meta) → 写 `output/leaf_h.bin`、`leaf_meta.txt`（+ H/U） |

### 1. Kernel 入口（`qdwh_sdc_fused_custom.cpp`）

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // blockDim=N ⇒ N 个 AIC + 2N 个 AIV
QdwhSdcFused op; op.Init(wk, m, n, threshold, maxIter, maxNodes, maxLeaves, leafCap, alpha, l0);
#ifdef __DAV_C220_VEC__   op.RunAiv();   // = QdwhAiv() -> SdcAiv()
#else                     op.RunCube();  // = QdwhCube() -> SdcCube()
#endif
```
`RunAiv`/`RunCube` 之间用 `PhaseBarrier(0xD)` 做 **全核同步**（Phase 1 → Phase 2）。

### 2. Phase 1：QDWH  `A -> U, H`（`QdwhAiv` / `QdwhCube`）

| 阶段 | AIV（`qdwh_vec`） | AIC（`qdwh_cube`） | 同步 |
| --- | --- | --- | --- |
| 准备 | **`QdwhInitX`**：X = A0/α（MTE2→`Muls`→MTE3）；同时 scalar 建 Xᵀ | — | — |
| 迭代 it=0…maxIter-1 | 等 `0x4` → （it>0）**`QdwhUpdateX`**：X = θX+κP → `AivBarrier` → **`QdwhPrepare`**：逐行建 H=[√c·Xᵀ \| I]、Q=I、递推 θ/κ/l、写 coef → `AivBarrier` → **`ComputeAll`**：Householder 面板（norm/反射/T/W 的批量向量计算） | **`qr_update`**：吃 AIV 面板、更新 Q（GEMM）→ `AicBarrier` → **`QdwhPmat`**：P=Q1Q2ᵀ→APad、Pᵀ=Q2Q1ᵀ→Pt → `AicBarrier` → 置 `0x4` | `0x4`：AIC→AIV「P 就绪」；`0x2`：AIC→AIV「QR 起始」；`0x1/0x3`：QR 内部 |
| 收尾 | 等 `0x4` → **`QdwhUpdateX`**（最后一轮）→ `AivBarrier` → 置 `0x5` → 等 `0x6` → **`QdwhCopyH`**（Q 的前 n×n → arena slot0 = H） | 等 `0x5` → **`QdwhFinalGemm`**（H = Uᵀ·A0，`isTransposeA` 隐式转置）→ `AicBarrier` → 置 `0x6` | `0x5`：AIV→AIC；`0x6`：AIC→AIV |

### 3. Phase 2：device 内 SDC 递归  `H -> leaves`（`SdcAiv` / `SdcCube`）

节点循环（每轮一个节点，`Desc = (slot, n, ld, work)`；`work=0` 表示结束）：

| 谁 | 动作 | 同步 |
| --- | --- | --- |
| AIC `SdcCube` | 所有 AIC 用相同的本地计数器；AIC0 写 `Desc` → 置 `0xB` → 读回 `Desc`；组内第 b 个 AIC 负责该节点的 `CopyInfo`/`Node*`/`Leaf*` GM 写（顶层 leaf 或 split 后的两个子节点）→ 置 `0xE` → 等 `0xC` | `0xB` 派发、`0xE` CopyInfo、`0xC` 搬运完成 |
| AIV `SdcAiv` | 等 `0xB` → 读 `Desc` → 调 **`sdc_split_node(...)`（AIV 分支）**（内部自己跑 QDWH+QR）→ 等 `0xE` → 按 `CopyInfo` 用 `DataCopyPad` 搬 H+/H−/leaf → 置 `0xC` | 同上 |

`sdc_split_node`（一个节点内部，AIV 与 AIC 并行）：

| 步 | AIV 分支 | AIC 分支 | 同步 |
| --- | --- | --- | --- |
| 1 | λ = trace(H)/n（scalar 读对角线） | — | |
| 2 | **`BuildA0FromH`**：A0 = pad(H−λI)（逐行 DataCopyPad+Adds+对角−λ+补零） | — | |
| 3 | **`QdwhInitX`**（X = A0/α，dims=(2nPad, nPad)） | — | |
| 4 | 迭代 maxIter 次：**`QdwhUpdateX`** / **`QdwhPrepare`** / **`ComputeAll`** | 迭代 maxIter 次：**`qr_update`** / **`QdwhPmat`** | `0x4`（每轮）、`0x2`（QR 起始） |
| 5 | 等 `0x6` 后 **`BuildPmats`**：P=U+I、P_=U−I（逐行 DataCopyPad+对角 ±1）→ 置 `0xD` | 等 `0x6` → **k = (trace(U)+nPad)/2**（scalar 读 U 对角线）→ 等 `0xD` | `0x6`、`0xD` |
| 6 | — | **UΩ1 = Tmp·Ω1**、**P_Ω2 = QT·Ω2**（两组 `mm.Process`）→ `AicBarrier` | |
| 7 | — | 构造 QR 输入 **Aᵀ → Hwork**（scalar，行 j = A 的列 j）+ **Q = I**（scalar） | |
| 8 | **`ComputeAll(..., 0x9)`**（QR 的 AIV 面板） | **`qr_update(0x9)`**（同一 QR 的 AIC 部分）→ `AicBarrier` | `0x9/0xA` |
| 9 | — | **Q₊ᵀ → QT**（kPad×n，k..kPad−1 补 0）→ **Tmp = QT·H** → **H+ = Tmp·Q₊ᵀ**（`isTransposeB`）→ 对称化（0.5(H+Hᵀ)）→ `Hplus` | `AicBarrier` |
| 10 | — | **Q₋ᵀ → QT**（m2Pad×n）→ **Tmp = QT·H** → **H− = Tmp·Q₋ᵀ** → 对称化 → `Hminus` | `AicBarrier` |

### 4. flag 一览（mode 2 = 跨 AIC/AIV，mode 0 = 同类型内屏障）

| flag | 方向 | 用途 |
| --- | --- | --- |
| `0x1` / `0x9` | 面板级 | Phase 1 QR / split QR 的 AIV↔AIC 面板握手 |
| `0x2` | AIC→AIV | QR 起始 |
| `0x3` | AIC 内 | GEMM 的 atomic-add 顺序化 |
| `0x4` | AIC→AIV | 「P 就绪，可以更新 X / 准备下一轮」 |
| `0x5` | AIV→AIC | 「X 定稿，可以 final GEMM」 |
| `0x6` | AIC→AIV | 「QDWH 完成（可以写 H / 造 P/P_）」 |
| `0x7` / `0x8` | 同类内 | AIC barrier / AIV barrier |
| `0xA` | AIC→AIV | split QR 起始 |
| `0xB` / `0xE` / `0xC` | AIC→AIV / AIC→AIV / AIV→AIC | 节点派发 / CopyInfo 就绪 / 搬运完成 |
| `0xD` | AIV→AIC | P/P_ 已落 GM（也是 Phase 1→2 的全核屏障） |

## phase / flag 表

| flag | 方向 | 用途 |
| --- | --- | --- |
| 0x1 / 0x2 | AIV→AIC / AIC→AIV | QR panel 握手（QR 内部，`ComputeAll`/`qr_update`） |
| 0x3 | AIC 组内 | QR 的 W/T 阶段 barrier |
| 0x4 | AIC→AIV | 一轮 `QdwhUpdate` 完成，放行下一轮 `QdwhPrepare`（最后再放行 `QdwhTransposeU`） |
| 0x5 | AIV→AIC | `QdwhTransposeU` 完成（`H` 前 n·m 列 = Xᵀ 就绪） |
| 0x6 | AIC→AIV | `QdwhFinalGemm` 完成（`Q=UᵀA0` 就绪） |
| 0x7 | AIC 组内 | `AicBarrier` / `AicSync` |
| 0x8 | AIV 组内 | `qdwh_vec::AivBarrier`（split 内部也用） |
| 0x9 / 0xA | AIC→AIV / AIV panel | split 里那次 QR 的 flag（避开 QDWH 残留的 0x1/0x2） |
| **0xD** | **AIV→AIC** | **Phase 1→2：`QdwhCopyH` 已把 H 写进 arena slot0** |
| **0xF** | **AIC→AIV** | **Phase 1→2：`QdwhFinalGemm` 已完成** |
| 0xB | AIC→AIV | SDC：`Desc`（slot/n/ld/work）就绪，可开始本轮 |
| 0xE | AIC→AIV | SDC：`CopyInfo`（孩子/leaf 的去向）就绪 |
| 0xC | AIV→AIC | SDC：H±/leaf 搬运完成，可以进入下一节点 |

> 关于提示词里的“0x8 全核 barrier”：`0x8` 这个 flag id 已经被
> `qdwh_vec::AivBarrier`（AIV 组内 barrier）占用，按字面再用会和 AIV 内部
> barrier 的计数混在一起。所以 Phase 分界用 **0xD/0xF 一对双向握手**实现同样
> 的语义：AIV 置 0xD（H 已落 GM）→ AIC 等 0xD；AIC 置 0xF（final GEMM 完成）
> → AIV 等 0xF。0xD/0xF 在这套代码里没有别的使用者。

## 数据流与 workspace

`fused_layout.hpp` 的 `FusedLayout::Build(m, n, maxNodes, maxLeaves, leafCap)`
被 host 与 device 各算一次，保证偏移完全一致；host 只 `aclrtMalloc` 一次，
launch 参数只有 **1 个基址 + 9 个标量**（避免 55 个指针的超长参数表）。

| 区 | 大小（1024×1024） | 说明 |
| --- | --- | --- |
| `a` / `a0` | 4 MB ×2 | QDWH 的 X（收敛即 U）/ A0 |
| `xsdc` | 4 MB | split 自己的 QdwhA（**独立一块，否则会覆盖 U**） |
| `hwork` / `qbuf` | 8 / 20 MB | QDWH 的 APad（n×qd）与 Q（qd²） |
| `uomega` / `qt` / `tmp` / `hplus` / `hminus` | 16 MB 合计 | split 的 scratch（H± 是 Fixpipe 输出，padded） |
| `om1` / `om2` | 4 MB ×2 | host 生成的 Ω（ld = n） |
| `arena` | 512 MB | 节点 H 的 slot 池（128 × n²），**slot0 = QDWH 的 H** |
| `leaf` | 16 MB | 叶块堆（capacity 4·n²） |
| meta | < 20 KB | Desc / CopyInfo / node 队列 / leaf 元数据 / k / λ / dbg |

## 关键点（对应提示词里的“坑”）

1. **AIC 不做矩阵搬运**：H+ / H- 是 Fixpipe 刚写出的矩阵，任何 scalar 读都会
   读到半可见值（旧版 `eig_err≈9e-2` 的根因）。融合版所有搬运（H± → arena
   slot、H± → leafH、顶层 leaf → leafH）都只走 `CopyRowsPad`
   （AIV 的 `DataCopyPad` GM→UB→GM），AIC 只写元数据。
2. **所有 GEMM 的 C 行步长 128 对齐**：`nPad`/`kPad`/`m2Pad` 都向上取整到
   128；子节点一律 compact（`ld = n`），因为 AIV 搬运时已经 compact。
3. **arena 深拷贝**：split 的 `ldH` 用的是节点真实的行步长（root = n，
   子节点 = n），所以 QDWH 的 H 可以原地当 root 用，不需要 host 中转。
4. **flag 配平**：QDWH 的 0x1~0x6 在 Phase 1 结束时全部消费干净；Phase 2 用
   0x7/0x9/0xA/0xB/0xC/0xE，两者不重叠（0x8 只在各自组内使用）。
5. **kernel 代码量**：Phase 1 与 split 各自实例化一份 `qdwh_vec/qdwh_cube`，
   融合后 AIC 代码约 354 KB、AIV 约 90 KB（基线 145 KB / 33 KB），可正常
   load/执行（见下方实测）。
6. **workspace 偏移必须 4 字节对齐**：`Addr(off)` 以 float 为单位，元数据数组
   的索引要写 `Addr(L.cnt + i)` 而不是 `Addr(L.cnt) + i`（后者是字节偏移，
   会让 `SetValue` 落在非对齐地址上，直接 aicore exception）。
7. 单矩阵：`P2 = (N, M+N)`、`batch = 1`；`shapes.txt` 只支持一行，多矩阵
   需要扩 arena 的 slot 下标（本版按题面要求只做单矩阵融合）。

## 编译与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/project/svd/qdwh_sdc_fused

python3 gen_data.py --shapes 1024x1024 --cond 1e2 --out-dir input
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fused_main -j 16

./build/fused_main 0 16 64 15 1.0 1e-6      # device blockDim threshold maxIter alpha l0
python3 verify.py --threshold 64
```

或一键（会 `tee output/verify.txt`）：

```bash
./run.sh                      # 默认 1024x1024 / cond=1e2 / threshold=64
SHAPES=1280x1024 ./run.sh
```

`fused_main` 参数：`deviceId blockDim threshold [maxIter=15] [alpha=1.0] [l0=1e-6] [--no-hu]`。
`--no-hu` 表示不下载 H/U（只剩 leaves 的 D2H）。

## 实测（Ascend 910B3，device 0，blockDim 16）

### 1024×1024，cond=1e2，threshold=64（题面验收用例）

```
[fused] device 端完成: leaves=23 overflow=0, kernel=767.8 ms
[fused] 汇总: leaves=23 max_leaf=63 threshold=64 |
        host 传输: h2d 6 次/12.1 MB, d2h 8 次/24.0 MB（全部在计算前后）

QDWH: A = U H
  M     N    u_err      orth      h_err     h_sym    min_eig       rec   status
1024  1024  1.294e-06  3.894e-07  1.151e-06 1.918e-06 9.000e-03 4.062e-07  PASS

SDC leaves
root    N  #leaf max_leaf  eig_err   trace_err  frob_err  sym_max  min_eig  status
   0 1024     23       63  1.772e-07  6.948e-09  7.651e-09  0.000e+00  0.000e+00  PASS

roots=1, leaves=23, threshold=64, all_leaves_small=True
RESULT: PASS
```

（`h2d 6 次` = A + Ω1 + Ω2 + eye + shapes + 计数器清零，全部在 launch 之前；
A0 是 device 内 `ACL_MEMCPY_DEVICE_TO_DEVICE` 复制，不占 host 带宽。）

* `leaves = 23`、`max_leaf = 63`、`all_leaves_small = True`；
* `eig_err = 1.75e-07`（≤1e-6）、`trace_err = 5.8e-09`；
* 全部叶块对称（`sym_max = 0`）、半正定（`min_eig = 0`）。

### 1280×1024（非方阵，同一 kernel）

```
[fused] device 端完成: leaves=23 overflow=0, kernel=804.9 ms
1024  1024  ... （root 是 1024×1024 的 H）
u_err=1.449e-06 orth=3.441e-07 h_err=1.033e-06 rec=7.425e-07  PASS
leaves=23 max_leaf=63 eig_err=2.173e-07 trace_err=1.073e-08     PASS
RESULT: PASS
```

### device 侧耗时：`msprof op`（历史记录；`msprof_op_report.py` 等分析脚本已删除）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# msprof 需要 driver 的 dcmi/hal 库，否则报 "Fail to get soc platform"
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

mkdir -p ~/prof && chmod 750 ~/prof     # 注意：--output 的父目录不能对 group/other 可写
msprof op --output=~/prof/fused --kernel-name=qdwh_sdc_fused_custom \
          --launch-count=1 --warm-up=0 ./build/fused_main 0 16 64 15 1.0 1e-6

python3 msprof_op_report.py output/msprof_op_report.txt    # 汇总报告
```

| 任务 | Task Duration |
| --- | --- |
| **融合 `qdwh_sdc_fused_custom`（四步都优化完）** | **462.6 ~ 482.7 ms（2 次，均值 472.6 ms）** |
| 融合（+ `P=U±I` 移到 AIV 之前） | 588.0 ~ 591.2 ms（2 次，均值 589.6 ms） |
| 融合（`InitX`+`Prepare` 向量化，`QdwhUpdate` 仍在 AIC） | 596.7 ~ 602.9 ms（2 次，均值 599.8 ms） |
| 融合（仅 `QdwhInitX` 向量化） | 747.6 ~ 766.7 ms（2 次，均值 757.1 ms） |
| 融合（全 scalar 原始版） | 753.9 ~ 778.1 ms（4 次，均值 767.7 ms） |
| 基线 `qdwh_custom`（只做 QDWH） | 161.8 ms |
| 基线 `sdc_dev_custom`（只做 SDC） | 624.0 ms |
| 基线两次 launch 之和 | 785.8 ms |

融合 kernel 内部 pipe 占用（16 AIC / 32 AIV 均值，`PipeUtilization.csv`）：

| pipe | 忙碌时间（优化后） | 占 aic/aiv 时间 | 原始 |
| --- | --- | --- | --- |
| AIV scalar | **256.0 ms** | 53.0% | 419.6 ms |
| AIC scalar | **117.6 ms** | 24.4% | 237.8 ms |
| AIV mte2 / mte3 | 38.7 / 13.1 ms | 8.0% / 2.7% | 37.4 / 11.7 ms |
| AIC mte2 / mte1 / fixpipe | 34.8 / 11.5 / 31.1 ms | 7.2% / 2.4% / 6.4% | 34.3 / 11.5 / 31.3 ms |
| AIV vec | 28.6 ms | 5.9% | 28.2 ms |
| AIC cube | 15.6 ms | 3.2% | 15.6 ms |

* device 侧单次 ≈ **473 ms**（原始 768 ms，-38.4%），与 host `aclrtSynchronizeStream`
  量到的 473 ms 一致 → launch 开销可忽略；比基线两次 launch 之和（785.8 ms）快 40%；
* AIV scalar 419.6 → 256.0 ms（-39%）、AIC scalar 237.8 → **117.6 ms（-51%）**；
* AIC 侧已经很轻（scalar 24%、cube 3.2%、fixpipe 6.4%），瓶颈转到 AIV 的 scalar（53%）；
* 16 AIC / 32 AIV 忙碌时间极差 < 0.01 ms（barrier 全程对齐，没有核掉队）；
* `msprof op` 的 7 项指标是分 7 遍 replay 采的，所以被 profile 的那次进程内
  wall-clock 会涨到 ~5 s（`kernel=5477 ms` 那种），`Task Duration` 才是真实单次耗时。

原始数据：`output/msprof_op_report.txt`、`output/msprof_fused_OpBasicInfo.csv`、
`output/msprof_fused_PipeUtilization.csv`、`output/msprof_fused_Memory.csv`。

### 优化记录 1：`QdwhInitX` 改成 `DataCopyPad` + `Muls`（无 scalar）

`vec.hpp::qdwh_vec::QdwhInitX`（`X = A0 / alpha`）原来是逐元素
`GetValue/SetValue` 双重循环 + 每行一次 DCCI，现改为：

```cpp
DataCopyExtParams inP {rows, N*4, (blockNum-1)*N*4, 0, 0};   // 一次搬 rows 行
DataCopyExtParams outP{rows, N*4, 0, (blockNum-1)*N*4, 0};
DataCopyPad(ub, QdwhA0[aOff + r0*N], inP, noPad);   // MTE2（pad 版本，长度按字节）
Muls(ub, ub, invAlpha, rows * N);                   // V：原地缩放
DataCopyPad(QdwhA[aOff + r0*N], ub, outP);          // MTE3
```

* 用 **`DataCopyPad`**（`blockLen` 按字节）而不是 `DataCopy`：不再要求 `N` 是 32B 的
  整数倍，所以**彻底删掉了原来的标量兜底分支**；
* **行划分必须和调用方写 A0 的划分一致**（`r = blockIdx; r += blockNum`）：
  `sdc_split_node` 按这个划分写 `A0 = pad(H−λI)` 并逐行 `DCCI`；若改成“每核一段
  连续区间”，会读到别的核还没刷回 GM 的行 —— 第一版就是这样把 `eig_err` 从
  1.7e-7 打到 4.8e-2 的（`max_leaf` 63→59）；
* MTE3 写不经过 scalar cache，不再需要逐行 `DCCI`；`QdwhPrepare` 的
  `InvalidateXAll()` 仍负责 invalidate 整块 X。

收益：AIV scalar 419.6 → 406.2 ms，kernel 767.7 → 757.1 ms（-10.6 ms / -1.4%）。

### 优化记录 2：`QdwhPrepare` 的 H/Q 行写改成 `DataCopyPad` + `Muls` + `SetValue(1)`

`QdwhPrepare` 每轮要重建 `H = [sqrt(c)·Xᵀ | I]`（`n×qd`）和 `Q = I`（`qd×qd`），
原来每行都是 `qd` 次 scalar `SetValue` + 一次 `CleanRange`，1024 方阵下每轮
就是 2M + 4M 次 GM 标量写。现在按行在 UB 里拼好、一次 `DataCopyPad` 写回：

```cpp
// 行缓冲 hRow: 前 M 个 = sqrt(c)*X[:, r]，后 N 个 = e_r
for (int r = blockIdx; r < N; r += blockNum) {
    for (int col = 0; col < M; ++col) hRow.SetValue(col, QdwhA.GetValue(aOff + col*N + r));
    Muls(hRow, hRow, sqrt_c, M);            // 前 M 个乘 sqrt(c)
    Duplicate(hRow[M], 0.0f, qd - M);       // 尾部清零（比 pad 的 rightPadding 可靠：
                                            // rightPadding 按 32B 块计数且只有 8bit）
    hRow.SetValue(M + r, 1.0f);             // 对角位置置 1
    DataCopyPad(APad[hOff + r*qd], hRow, cpRow);   // 整行一次 MTE3，替代 qd 次 scalar 写
}
// Q = I：qRow 先 Duplicate 成 0，每行只 SetValue 一个 1（写完清掉上一行的 1）
Duplicate(qRow, 0.0f, qd);
for (int r = blockIdx; r < qd; r += blockNum) {
    if (prevDiag >= 0) qRow.SetValue(prevDiag, 0.0f);
    qRow.SetValue(r, 1.0f); prevDiag = r;
    DataCopyPad(QdwhQ[qOff + r*qd], qRow, cpRow);
}
```

**为什么列 gather 还是 scalar（没有用 DataCopyPad）**：H 的第 r 行前 M 列取自 X 的
第 r *列*（列方向是 4 字节间隔）。多 block 的 `DataCopyPad` 要求 `blockLen` 按 32B
对齐，用 `blockCount=M, blockLen=4B, srcStride=(N-1)*4B` 去 gather 实测会读到错位
数据（kernel 跑完但 `u_err=1.0`、`leafCount` 变成随机数），所以这一步保留
“逐列标量读进 UB”，只把 **GM 标量写** 彻底去掉。

收益：AIV scalar 406.2 → **250.7 ms**，kernel 757.1 → **599.8 ms**
（**-157 ms / -20.8%**）；精度不变（`eig_err 1.7e-07`、`trace_err 6.5e-10`、PASS）。

**下一步**（预计还能再降 20-25%）：让 AIC 在 `QdwhUpdate` 里顺带写一份 `Xᵀ`
（它本来就在按元素算 `X = θX + κP`），AIV 就能对 `Xᵀ` 做**连续行**
`DataCopyPad` + `Muls`，把剩下的 250 ms 里最大的一块（按列 gather 的 M 次读）也消掉。

### 优化记录 3：`P = U+I` / `P_ = U-I` 从 AIC 搬到 AIV，用 `DataCopyPad` 构造

`sdc_split_node` 里这两块 GEMM 操作数原来是 **AIC** 上 `2*n²` 次 scalar 写 +
每行 DCCI（注释还写着“跨核 AIV→AIC 的 scalar 可见性不可靠，所以在 AIC 本地构造”）。
现在改成 AIV 的 `qdwh_vec::BuildPmats`：

```cpp
// 每个节点 n 行，每行一次 MTE2 + 一次向量复制 + 2 次 UB 标量 + 2 次 MTE3
for (int r = blockIdx; r < n; r += blockNum) {
    DataCopyPad(ubP, QdwhA[r * ldU], cpRow, noPad);   // U 的第 r 行（前 n 列）
    Adds(ubQ, ubP, 0.0f, n);                          // ubQ = U（复制一份）
    ubP.SetValue(r, ubP.GetValue(r) + 1.0f);          // 对角 +1 -> U+I
    ubQ.SetValue(r, ubQ.GetValue(r) - 1.0f);          // 对角 -1 -> U-I
    DataCopyPad(Pm[r * ldOut],  ubP, cpRow);          // -> Tmp（ld = nPad）
    DataCopyPad(PmM[r * ldOut], ubQ, cpRow);          // -> QT （ld = nPad）
}
```

配套的跨核握手：AIV 写完两块矩阵后 `CrossCoreSetFlag<0x2,PIPE_MTE3>(0xD)`，
AIC 在 GEMM 之前 `CrossCoreWaitFlag(0xD)`（AIC 侧原来的 `AicBarrier` 一起删掉）。
`0xD` 与 Phase 1→2 分界用的是同一个 id，语义也一致：“AIV 已把 AIC 要读的数据写进
GM”；分界只用一次、每节点再各一次，收发严格配平。

收益：AIC scalar **236.8 → 222.5 ms**（少掉的就是 `2*n²` 的 P/P_ 构造），
AIV scalar +4 ms（多了 2 次 UB 标量/行），kernel **599.8 → 589.6 ms**
（-10.2 ms / -1.7%），精度不变（`eig_err 2.4e-07`、PASS）。
收益比预期小是因为 SDC 的耗时主要在“每个节点一次的 QDWH 迭代”上，
P/P_ 构造是每节点一次的活（22 个节点、Σn² ≈ 3.5M 次写）。

**下一步最大的两块**（预计合计还能降 ~35%）：
1. AIC 的 `QdwhUpdate`（`X = θX + κP`）：每轮迭代 `n²` 次 scalar 读写 × 15 轮 × 23 次调用，
   而且两个操作数都是行主序 —— 可以直接 `Muls` + `Axpy` 向量化，不需要转置；
2. AIV 的列 gather：让 AIC 在 `QdwhUpdate` 时顺带写一份 `Xᵀ`，AIV 就能连续行处理。

### 优化记录 4：`X = θX + κP` 从 AIC 的 scalar 搬到 AIV 的 `Muls`+`Axpy`

`qdwh_cube::QdwhUpdate` 原来是 AIC 上逐元素 `X = theta*X + kappa*P`
（`M*N` 次 scalar 读 + 写，外加每行两次 `DCCI`，15 轮 × 23 次调用）。
**AIC 没有向量单元**，所以不是"在 AIC 上换指令"，而是把这一步整体搬到 AIV：

```cpp
// qdwh_vec::QdwhUpdateX(blockIdx, blockNum)：theta/kappa 用上一轮 QdwhPrepare 存下的值
for (int r0 = blockIdx; r0 < M; r0 += blockNum * nRows) {   // nRows = chunkSize/N，一次搬 8 行
    DataCopyPad(ubX, QdwhA[aOff + r0*N], inP, noPad);       // X 行（ld = N，连续）
    DataCopyPad(ubP, APad [hOff + r0*N], inP, noPad);       // P 行（Pmat 写在 APad 前 M*N，ld = N）
    Muls(ubX, ubX, theta, rows * N);                        // theta * X
    Axpy(ubX, ubP, kappa, rows * N);                        // + kappa * P
    DataCopyPad(QdwhA[aOff + r0*N], ubX, outP);
}
```

配套改动：
* `qdwh_vec::QdwhPrepare` 把每轮的 `theta/kappa` 额外存到 `thetaArr/kappaArr`；
* **AIC 不再调 `QdwhUpdate`**：`qr_update -> AicBarrier -> QdwhPmat -> AicBarrier -> SetFlag(0x4)`
  （0x4 的含义变成"P 就绪"）；
* **AIV** 在每次 `CrossCoreWaitFlag(0x4)` 之后先做 `QdwhUpdateX`，**再** `QdwhPrepare`，
  中间加一个 `AivBarrier()`：`QdwhPrepare` 要跨核读整列 X，必须等所有 AIV 都更新完；
  循环结束后再补一次 `QdwhUpdateX`（最后一轮的 P）然后 `QdwhTransposeU`；
* 顺序安全性：`QdwhPrepare` 会整块覆盖 `APad`（H），但它在同一核上排在
  `QdwhUpdateX` 之后，P 一定已经被读完；AIC 的下一轮 `QdwhPmat` 又要等 AIV 的
  panel flag（0x1），所以也不会提前覆盖 P。

收益（`msprof op`）：**AIC scalar 222.5 → 117.6 ms（-47%）**，kernel
**589.6 → 472.6 ms（-117 ms / -19.9%）**；AIV scalar 254.9 → 256.0 ms（基本不变）。
精度：`eig_err 1.9e-07`、`trace_err 1.4e-08`、`leaves=23 / max_leaf=63`、PASS；
1280×1024 也 PASS（`eig_err 1.88e-07`）。
累计：768 → **473 ms（-38.4%）**，比基线两次 launch 之和快 **40%**。

**下一步**：AIV scalar 现在又成了唯一瓶颈（256 ms，占 53%），其中最大一块是
`QdwhPrepare` 里"按列 gather X"的读（列方向 4 字节间隔，多 block `DataCopyPad`
要求 32B 对齐做不了），可以靠"AIC 顺带写一份 `Xᵀ`"或把 H 的列块按 32B 分组来解，
预计还能再降 ~15%。

### GEMM 试验算子：`Process` / `Process2` 的转置能力（历史记录；试验算子与脚本已删除）

`gemm.cpp` 里的 `MatmulCustom` 有 `Process` 和 `Process2` 两个入口，都能做
`C = op(A)·op(B)` 的分块 GEMM（L1/L0 三级分块 + Fixpipe 写回）。为了确认它们到底
支持哪些转置组合，我加了一个**独立的分块 GEMM 试验算子**：

```bash
cmake --build build --target gemm_test_main -j 16
python3 gemm_test.py            # 15 个用例：形状 × mode × Process/Process2 × atomicAdd
python3 gemm_test_extra.py       # 定位 isTransposeA 的失败条件
```

- `gemm_test_custom.cpp`：按 C 的行分 128 一块发给 16 个 AIC，`mode = tA*2 + tB`
  决定 A/B 的物理存储形态，`proc2` 选 `Process`/`Process2`，`atomicAdd` 测 `C += A·B`；
- `gemm_test_main.cpp`：生成随机 A/B（**padding 区填 5.0**，能一眼看出是否读了 padding），
  跑一次后把 `a.bin/b.bin/c.bin/meta.txt` 落盘；
- `gemm_test.py`：用 numpy 按物理存储还原 `op(A)·op(B)` 对拍。

实测结论（`output/gemm_test_report.txt`，13/15 通过；两个 FAIL 正是下面第 3 条的约束）：

| 能力 | 结论 |
| --- | --- |
| `Process`，`C = A·B` | ✅ 任意 M/N/K，**非 128 倍数也对**（200×136×100 通过） |
| `Process`，`C = A·Bᵀ` | ✅ `isTransposeB=true`，任意 K |
| `Process`，`C = Aᵀ·B` | ⚠️ `isTransposeA=true`，**要求 M 和每个 k 块都是 128 的倍数**（K=192/320/64、M=200 全部算错） |
| `Process`，`C = Aᵀ·Bᵀ` | ✅ 两个 flag 同时给（1024³ 通过） |
| `Process2`，`C = A·B / A·Bᵀ` | ✅ 与 `Process` 数值一致 |
| `Process2` + A 转置 | ❌ 函数体里**没有** A 转置分支；而且它不重置 `isTransposeA` 成员，若之前调过 `Process(isTransposeA=true)` 会污染它的 L0 段 |
| 自定义 `ldA/ldB`（`SetMatrixLd`） | ✅ 只对非转置方向生效（512³ + 640 padding 通过） |
| 转置 A 的物理布局 | 必须 `(K × M)` 且行步长 **恰好等于 M**（代码用 `this->M` 当 Nd2Nz 的 `srcDValue`，必须走 `SetMatrix2` 把 M 设进去） |
| C 的 leading dimension | ❌ 固定 `dstStride = N`（Fixpipe 写回没有 ldC 参数，C 必须紧凑） |
| `atomicAdd` | ✅ `SetAtomicAdd` 实现 `C += A·B`（512³ mode0/mode2 都通过） |

**已落地的应用**：`QdwhFinalGemm`（`Q = Uᵀ·A0`）原来要 AIV 先用 `QdwhTransposeU`
把 `Xᵀ` 物理写进 H（n×m 次 scalar 读+写），现在直接

```cpp
opmm.SetMatrix2(X[xOff], A0[aOff], Q[qOff], /*M=*/n, /*N=*/n, /*K=*/m);
opmm.Process(rowStart + kc * n, kc * n, rowStart * n, rows, n, kk, -1, false, kc != 0,
             /*isTransposeA=*/true);
```

由 cube 的 `enTranspose` 完成转置，`QdwhTransposeU` 在融合 kernel 和基线里都不再调用。
收益：AIV scalar 256.0 → 249.3 ms，kernel ≈ -3 ms（它是每次 QDWH 调用才做一次，
不是每轮迭代，所以量级不大），PASS。

**还没能去掉的两个转置**（原因都来自上表的约束）：
1. `sdc_split_node` 里 `Tmp = Q₊ᵀ·H`、`Hplus = (Q₊ᵀH)·Q₊` 前的 `QT = Q₊ᵀ`
   物理构造：Q₊ 是 `Q` 的前 k 列，行步长是 `nPad`，而 `isTransposeA` 的 `srcDValue`
   被硬编码成 `M`（= k），**除非把 `srcDValue` 改成可传 `ldA`**（`gemm.cpp` 加一个
   `ldAT`，再补一个 gemm_test 用例）才能用；
2. `Hwork = Aᵀ`（split 的 QR 输入）：QR 的 `ComputeAll`/`qr_update` 按列读 `APad`，
   要求输入是 A 的转置形态；改成非转置要同时改 QR 两边的索引，风险大，暂不动。
3. 另外 `C` 的 `dstStride` 固定为 `N`，所以「让 cube 直接把 `sqrt(c)·Xᵀ` 写进
   `H`（行步长 qd）」这条路必须先给 Fixpipe 加 `dstLd` 参数。

### QDWH 到底迭代几次

**每次极分解迭代 `maxIter` 次，现在的默认值是 4**（原来是 15；`fused_main` 的第 4 个参数，
`for (it = 0; it < maxIter; ++it)` 跑满、不判收敛）。

- AIV：`QdwhUpdateX`（第 2 轮起）→ `QdwhPrepare`（建 `H=[sqrt(c)Xᵀ|I]`、`Q=I`、递推 l）
  → `ComputeAll`（QR 的 Householder 面板）；
- AIC：`qr_update` → `QdwhPmat`（`P=Q1Q2ᵀ`）→ 通知 AIV 更新 `X`。

一次 1024×1024 / threshold=64 的融合运行里：

| 项 | 数量 |
| --- | --- |
| Phase 1（`A -> U,H`）的 QDWH | 1 次 |
| Phase 2 每个内部节点 split 都要算一次 sign(H−λI) | 22 次（23 个叶块 ⇒ 22 次 split） |
| **QDWH 调用总数** | **23 次** |
| 每次 QDWH 的 QR 迭代 | **15 轮** |
| **QR 迭代总数** | **23 × 15 = 345 轮** |
| 每轮 QR 的 Householder 面板数 | `ceil(min(m+n,n)/64)`：phase 1 = 16，split 根节点 = 16 |

实际的收敛速度（`qdwh_iters.py` 复刻 `compute_weight` 的 l 递推，l0=1e-6）：

```
iter      l(开始)    theta       kappa      l(结束)
   1   1.0000e-06  0.99984  1.9998e+00  2.5194e-02
   2   2.5194e-02  0.87649  1.8806e+00  6.4891e-01
   3   6.4891e-01  0.40876  1.5521e+00  9.9875e-01
   4   9.9875e-01  0.33354  1.5396e+00  1.0000e+00
   5   1.0000e+00  0.33333  1.5396e+00  1.0000e+00
```

即 **第 4 轮 l 就到 1（数值上已收敛）**，之后每轮只是 θ=1/3、κ=1.5396 的常规
"抛光"迭代。所以 `maxIter` 默认改成 **4**（见「优化记录 6」的实测表：4 轮比 15 轮
还准一点，耗时只有 42%）；需要更保守时可以 `./build/fused_main 0 16 64 6`。

### 优化记录 5：`Xᵀ` 缓冲 + 向量化 `A0`/`H` 构造（部分生效）

沿用 gemm_test 里验证过的「转置交给 cube」的思路，进一步把 `QdwhPrepare` 里
**按列 scalar gather X** 这一步也干掉：

1. `QdwhPmat`（cube）在算完 `P = Q1Q2ᵀ` 之后，**顺带再算一次 `Pᵀ = Q2Q1ᵀ`**
   （只是把两个操作数换一下，仍是 `isTransposeB`，成本约 +1 次 GEMM）写进 `pt`；
2. `QdwhInitX`（AIV）在初始化 `X = A0/alpha` 的同时，用一次 scalar 转置建好 `Xᵀ`（`xt`，
   每次 QDWH 调用只做一次，而不是每轮 15 次）；
3. `QdwhUpdateX`（AIV）在更新 `X` 的同时用 `Muls + Axpy` 更新 `Xᵀ = θXᵀ + κPᵀ`；
4. `QdwhPrepare` 于是只需要 `DataCopyPad` 读 `Xᵀ` 的**连续行**，不再有按列 gather；
5. 顺带把 split 的 `A0 = pad(H−λI)` 也改成逐行 `DataCopyPad + Adds`（`BuildA0FromH`）。

配套：`fused_layout.hpp` 新增 `pt`/`xt` 两块（各 `m*n`）；`sdc_split_node` 增加
`pt`/`xt` 两个可选参数（`nullptr` 时退回旧的按列 gather）。

结果：
* **phase 1 的 `Xᵀ` 路径生效**（`h_err` 仍是 1.1e-06）；
* **`BuildA0FromH` 生效**：AIV scalar 249.3 → 207.4 ms，kernel **474.5 → 431.7 ms（-9%）**，
  PASS（`eig_err 1.8e-07`、`leaves=23 / max_leaf=63`）；
* **split 的 `Xᵀ` 路径暂未启用**：打开后 kernel 能到 ~331 ms，但 `eig_err` 退到 1.4e-2
  （phase 1 用同一套代码是对的，split 里却是错的）。已经二分到：
  - 不是 `BuildA0FromH`（关掉 split 的 `pt/xt` 后 PASS）；
  - 不是 `Pt`（`Pᵀ`）那个 GEMM：**只开 `pt`、不开 `xt` 时仍然 PASS**（`eig_err 1.8e-07`），
    说明 `QdwhPmat` 里那次 `Pᵀ` GEMM 的结果和写入范围都是对的；
  - 不是 `A0` 指针/初始化顺序、不是布局重叠（`pt`/`xt` 各 `m*n`，正好等于 `nPad²`）；
  - 因此问题锁定在 **split 里 `Xᵀ` 的维护/读取链**（`QdwhInitX` 的标量转置建初值、
    `QdwhUpdateX` 的 `θXᵀ+κPᵀ`、`QdwhPrepare` 读 `Xᵀ` 行），而这三段在 phase 1 是同
    一份代码却能过。下一步建议：用 `dbg` 把某个节点 `Xᵀ[r][c]` 与 `X[c][r]` 逐点对比
    （或改用 cube 的转置 GEMM 建初值，顺带去掉最后那 1M 次标量转置）。代码路径保留
    （`hasPt/hasXt` 开关），验证通过后再打开。

进度小结（`msprof op`，device 侧单次）：

| 阶段 | 累计优化后 |
| --- | --- |
| 原始 | 767.7 ms |
| InitX 向量化 | 757.1 ms |
| + Prepare H/Q 行向量化 | 599.8 ms |
| + P=U±I 移到 AIV | 589.6 ms |
| + X=θX+κP 移到 AIV | 472.6 ms |
| + 去掉 QdwhTransposeU | 474.5 ms |
| + Xᵀ 缓冲（phase 1）/ 向量化 A0 构造 | 431.7 ms |
| **+ maxIter 15 → 4** | **178.6 ms（-76.7%）** |

### 优化记录 6：`maxIter` 15 → 4（最大的一笔）

`maxIter` 是每次 QDWH 的 QR 迭代轮数（默认原来是 15，`for (it...)` 跑满、不判收敛）。
`qdwh_iters.py` 复刻 `compute_weight` 的 l 递推可见：**l 在第 4 轮就到 1**
（1e-6 → 2.5e-2 → 0.65 → 0.999 → 1.0），后面 11 轮只是 θ=1/3、κ=1.54 的抛光。

`iter_sweep.py` 实测（1024×1024，cond=1e2，threshold=64，23 次 QDWH 调用）：

| maxIter | kernel | u_err | orth | h_err | leaves / max | eig_err | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 158.3 ms | 7.85e-04 | 1.57e-03 | 8.16e-04 | 23 / 63 | — | PASS（余量小） |
| **4** | **178.7 ms** | **7.59e-07** | **4.38e-07** | **4.63e-07** | **23 / 63** | **3.21e-07** | **PASS** |
| 5 | 206.6 ms | 7.81e-07 | 4.08e-07 | 5.02e-07 | 23 / 63 | 2.09e-07 | PASS |
| 6 | 222.5 ms | 8.16e-07 | 3.90e-07 | 5.54e-07 | 23 / 63 | 1.68e-07 | PASS |
| 8 | 265.7 ms | 9.08e-07 | 3.92e-07 | 6.82e-07 | 23 / 63 | 1.85e-07 | PASS |
| 10 | 323.4 ms | 2.79e-02 | 3.91e-07 | 1.59e-03 | 23 / 63 | 4.09e-03 | FAIL ※ |
| 15 | 422.6 ms | 1.29e-06 | 3.95e-07 | 1.15e-06 | 23 / 63 | 1.82e-07 | PASS |

※ 该次 10 轮跑出了偶发错误（u_err 2.8e-2），重跑 4/5/6 各 3~5 次全部 PASS，
说明**每多跑一轮就多一次放大 float32 舍入的机会**；4 轮不仅最快，精度还比 15 轮好
（u_err 7.6e-07 vs 1.3e-06）。

* 默认值已改成 **4**（`fused_main.cpp` 与 `run.sh`）；`./build/fused_main 0 16 64 15` 仍可回到 15 轮；
* 稳定性：`maxIter=4` 连跑 3 次、`=5` 3 次、`=6` 5 次，全部 PASS（`iter_repeat.py`）；
* cond=1e3 的数据也复测：`maxIter=4` → 178.3 ms、`u_err 3.53e-06`、`eig_err 1.53e-07`、PASS
  （比 15 轮的 3.69e-06 还好）。

### 优化记录 7（未完成）：split 的 `Xᵀ` 路径

最后一块没打掉的是 split 里 `QdwhPrepare` 的按列 gather（约 100 ms）。这一轮继续二分：

* **`Pᵀ` GEMM 验证 OK**：split 里只开 `pt`（`Pᵀ` 由 cube 算）、不开 `xt` → `eig_err 1.78e-07` **PASS**，
  说明 `QdwhPmat` 里那次 `Pᵀ = Q2Q1ᵀ` 正确；
* **把 `Xᵀ` 的向量更新换成 scalar 重建 → 直接崩坏**（`overflow` 乱码、7 s、eig_err 7e-2）。
  这反证了：**AIV 用 scalar 读"被 MTE3 覆盖过的 X"会命中过期 cache line**
  （`QdwhInitX` 里读 `X` 前有 `InvalidateXAll()` 所以没事，但更新路径里没有）；
* 于是改成 **cube 用 host 传入的 `eye`（128×128 单位阵 + 128×128 零阵，ld=128）做转置 GEMM**：
  `qdwh_cube::QdwhInitXt()` 按 128 列块算 `Xᵀ`，每个块 `C = Xᵀ_block`、`A = X 的 128 行 slab`、
  `B = eye`，正好满足 gemm_test 测出的"`isTransposeA` 要求 M、K 都是 128 的倍数"；
  C 的行步长是 `M`（≠ N），为此给 `gemm.cpp` 加了 `SetLdC()`（Fixpipe `dstStride` 可配）。
  握手也理清了：AIV 的 `QdwhInitX` 完成后置 `0x5`，cube 等 `0x5` → `QdwhInitXt` → 置 `0x4`
  放行第一轮 prepare（`0x5` 与"X 定稿"共用，收配平）。
* **结果：`SetLdC` 这条路径报错**（`errStr: When the D-cache reads and writes data to the UB,
  the response value returned by the bus is a non-zero value`，即 Fixpipe 的 `dstStride != N`
  写到了非法 GM 地址），所以该方案**先回退、代码保留但未启用**（`hasXt=false`）。

**下一步（两条都可，建议先做 a）**：
a) 在 `gemm_test` 里加一个 `ldC != N` 的用例，先把 `SetLdC` 验证/修好（怀疑 `dstStride` 的单位
   或对齐要求，可能是 128 的倍数、或需要 `nSize`/`mSize` 配合），再打开 `QdwhInitXt`；
b) 换一种 `Xᵀ` 布局避开 `ldC`：按 128 列块存成 `(m/128)` 个 `N×128` 的块（`C` 的 ld 正好 = 128 = N），
   `QdwhPrepare`/`QdwhUpdateX` 按块访问（每行 8 次 `DataCopyPad` 而不是 1 次，可接受）。

#### 实测结果（单测已通过：转置 GEMM 本身没问题）

**关键结论：分块转置 GEMM 已经用单测验证通过（误差 0.000e+00）。**

给 `gemm_test` 加了 mode 10/11「分块转置」用例（`QdwhInitXt` 的最小复现）：
`A` 是 `M×N`（ld=N），`B` 是 host 传入的 `eye` 里的 128×128 单位阵，
输出按 **A 的 128 行 slab** 分块：`C_block[kc] = A[kc:kc+128, :]ᵀ`（`N×128`，ld=128）。

```
分块转置 1024² mode10（Process+isTransposeA）       max_abs=0.000e+00  PASS
分块转置 512²  mode10                              max_abs=0.000e+00  PASS
分块转置 256x512 mode10（A 是 256x512）             max_abs=0.000e+00  PASS
分块转置 1024² mode11（Process2+SetTransposeA）     max_abs=1.997e+00  FAIL
```

**踩到并修掉的两个关键点**（都写进单测了）：

1. 分块要按 **A 的行 slab（K 方向）** 分，不是按 A 的列分块 ——
   转置 A 的 GEMM 要求 A 以 `(K × M_gemm)` 的转置形态给出，行步长 = `M_gemm`，
   所以 A 的 slab 只能是「若干整行」（行步长连续），列方向切片就破坏了这个前提。
2. **M 方向的 tile 偏移必须通过 `offsetA` 传**（`offsetC` 只管 C 的起始行）；
   只传 `offsetC` 会让所有核写同一块 M tile，结果全错。
3. `isTransposeA` 是 `MatmulCustom` 的成员，只传 9 个参数的 `Process` 调用会**继承**它，
   所以 `QdwhInitXt` 结束时必须 `SetTransposeA(false)` 复位。

用这套修法后：**phase 1 的 `Xᵀ` 已完全走 GEMM 转置**（`u_err 7.59e-07`、`orth 4.43e-07`、
`h_err 4.67e-07`，PASS），说明 `QdwhInitXt` + 分块 `QdwhPrepare`/`QdwhUpdateX` 这条链是通的。

**但 split 打开 `xt` 后仍然不对**（`eig_err 0.55`、`leaves=28`，kernel 反而更快：
**143 ms**）。phase 1（同样的代码、同样的分块布局）是对的，所以问题只在
split 的 QDWH 调用序列里；已排除：`BuildA0FromH`、`A0` 指针、`Pᵀ` GEMM、
布局重叠、`isTransposeA` 残留（已加复位）。下一步建议用 `dbg` 把某个 split 节点的
`Xt[kc][r][c]` 与 `X[kc*128+c][r]` 逐点比对，或在 split 里把 `QdwhInitXt`
换成「cube 算完 + AIV 用 scalar 重算一遍」做二选一。

⚠️ **两套 `Xᵀ` 布局是"成套"的，不能混用**：`QdwhInitXt`（cube）写的是**分块布局**
（`Xt[kc*N + r*128]`），而 `QdwhPrepare`/`QdwhUpdateX` 的**连续版**读的是
`Xt[r*M + c]`。当前发布版用的是「AIV scalar 转置 init + 连续版消费者」（180 ms，
`u_err 7.58e-07`、`eig_err 1.49e-07`，1024² 与 1280×1024 都 PASS）；如果只把
cube 的 `InitXt` 打开、消费者仍用连续版，会互相覆盖导致 `u_err≈0.98`（踩过）。

**成套切到分块模式（`#define FUSED_XT_BLOCKED`）之后的实测**：

| 配置 | kernel | phase-1 `u_err` | `eig_err` |
| --- | --- | --- | --- |
| 连续版（默认，发布） | 180.0 ms | 7.58e-07 | **1.49e-07** ✓ |
| 分块版（`FUSED_XT_BLOCKED`） | **151.9 ms** | 7.58e-07 | 7.0e-03（残差） |

分块版的 phase-1 是精确的，残差只出现在**最深层节点**。用阈值把树缩小定位：

| threshold | 叶块数 | `eig_err` |
| --- | --- | --- |
| 1024（只 split 1 次） | 2 | 3.89e-08 ✓ |
| 512 | 3 | 2.81e-08 ✓ |
| 128 | 12 | 1.06e-07 ✓ |
| **64** | 23 | **7.02e-03** ✗ |

即：**`nPad = 128` 的节点被继续 split 时**才出错（threshold=128 时这类节点还是叶块，
QDWH 只算不 split，所以没事）。把 `maxIter` 加到 8/12 残差仍是 1.0e-2 / 9.9e-3，
说明是**数据 bug** 而不是收敛不够。

**进一步的定位（本轮）**：

* 分块转置 GEMM 在 **128×128 / 128×256 / 256×128 / 256×512 / 512² / 1024²** 上单测全部
  `max_abs=0.000e+00`，所以 `QdwhInitXt` 本身没问题（`gemm_test.py` 的 `BLK` 用例）；
* 分块版的 `leaf_h.bin` **三次跑出三个不同结果**（连续版也是 bitwise 不确定，但
  `eig_err` 稳定在 2e-07；分块版抖动放大到 1e-2 量级，直接把 k 边界翻掉）；
* 定位到一个可疑点：blocked 模式下 AIV 在 `QdwhInitX` 之后用一个 **`0x5`** 通知 cube
  做转置 GEMM，而 `0x5` 同时又是"X 定稿"的信号 → **同一个 flag 一节点两次、跨节点会串号**
  （连续版每节点只发一次 `0x5`，没有这个问题）。修法是给 InitX 用**专用 flag**。
* 另外把 `qr_update` 里那个 `Process2`（A 转置）的 `isTransposeA` 显式设为 `false`
  （它原来依赖未初始化成员 = UB）。

结论：分块版要先把 flag 拆开再验证，之后才能拿到 ~145 ms。

**(a) `SetLdC` 这条路是死的。** 给 `gemm_test` 加了 4 个 `ldC != N` 的用例
（`512³ ldC=640`、`512³ mode1 ldC=576`、`512x256 ldC=320`、`512x128 mode2 ldC=256`），
**全部 FAIL**，而且 padding 区被写进了数据（`pad=40`）——说明 Fixpipe 并没有按
`dstStride` 走非紧凑存储（本版本 CANN 的 `FixpipeParamsV220.dstStride` 不是"行步长(元素)"）。
完整结果见 `output/gemm_test_report.txt`（16/24 通过，失败的是 4 个 ldC + 3 个 mode11）。

**(b) 128 列分块布局：GEMM 侧已通过单测，集成到 split 还差最后一步**（见上）。

### 优化记录 8：blockDim 16 → 20（20 AIC + 40 AIV）

`KERNEL_TYPE_MIX_AIC_1_2` 下 blockDim 就是 AIC 数（AIV = 2×），所以只要把
`fused_main` 的第二个参数改成 20，就是 **20 个 cube + 40 个 vector**：

| blockDim | AIC / AIV | kernel | `u_err` | `eig_err` |
| --- | --- | --- | --- | --- |
| 16 | 16 / 32 | 178.9 ms | 7.60e-07 | 2.0e-07 |
| 18 | 18 / 36 | 167.2 ms | 7.57e-07 | PASS |
| **20** | **20 / 40** | **158.2 ~ 160.5 ms** | 7.57e-07 | 2.1e-07 |
| 22 | 22 / 44 | **挂住（timeout）** | — | — |

* kernel 里所有"按核分配"用的都是 `GetBlockNum()`（AIC 数）/ `GetBlockNum() << 1`
  （AIV 数），所以 16 → 20 **不需要改 kernel**，只要改启动参数；默认值已改为 20
  （`run.sh` 的 `BLOCKDIM`）。
* **QR 的 vec 侧仍然只用 32 个 AIV**（`ComputeAll` 里 `align_blockdim = min(32, blockDim)`，
  多出来的 8 个 AIV 走"空转核"分支陪同步）；QR 的 cube 侧 `m_cubes = ceil(M/64)`，
  1024² 下只有 16 块，所以 AIC 16..19 在 QR 段也基本空转。这也是 blockDim=22 会挂住的原因
  （QR 的同步模式只覆盖到 32 个 AIV 参与）。
* 回归（bd=20）：threshold 512 → 85.1 ms、128 → 136.1 ms、64 → 159.6 ms；
  1280×1024 → 163.9 ms，全部 PASS。
* 新瓶颈（bd=20 的 msprof）：**AIC scalar 66.5 ms（42%）**、AIV scalar 49.8 ms（31%）、
  AIC cube 3.7 ms、AIV vec 6.9 ms、AIC mte2 8.4 ms、AIV mte2 9.0 ms —— 剩下的时间主要在
  **标量代码**（QR 的控制流/索引、split 的标量片段），继续堆核数收益不大。

#### 为什么"让 4 个空转 AIC 也参与 QR"没有收益（实测）

* `threshold=1024`（只剩 1 次 split，QR 占绝对主导）：bd=16 **68.7 ms** → bd=20 **60.9 ms**
  （**-11%**，说明 QR 已经在吃多出来的核）；`threshold=300`：123.3 → 107.4 ms（-13%）；
  `threshold=64`：179 → 160 ms（-11%）。核数从 16 加到 20（+25%）整体只快 11%，
  因为很大一部分时间是同步/标量关键路径。
* QR 主导配置（bd=20、threshold=1024、59.2 ms）的 pipe 拆分：
  **AIC scalar 29.5 ms（50%）**、AIV scalar 19.5 ms（33%）、AIC cube 2.65 ms（4.5%）、
  AIC fixpipe 5.1 ms、AIC mte2 5.9 ms、AIV vec 2.42 ms。
  也就是说 **QR 是"标量受限"的，不是 cube 受限**：cube 侧只占 4.5%，
  再怎么往 QR 里塞 AIC 也没有空间；要提速必须减少标量指令（面板控制的索引/gcd/取模、
  `SetMatrix`/`Process` 的调用开销、AIV 侧 Householder 的逐元素 `SetValue`）。
* 转置这块已经没有物理转置可省了：`QdwhFinalGemm` 用 `isTransposeA`（Uᵀ 隐式），
  `QdwhPmat`/`Pᵀ` 用 `isTransposeB`（Q2ᵀ 隐式）；整个 kernel 里剩下的唯一物理转置是
  split 里 `Hwork = Aᵀ` 的标量构造（`sdc_node.hpp`），实测量级只有 ~1 ms，不值得动。

### 优化记录 9：`X = αX + βQ₁ᵀQ₂` 的"原子加"写法（结构已备好，未启用）

目标结构（比现在的 `Muls(θ)+Axpy(κ)` 更贴公式）：
1. **vec 侧**只做缩放：`X ← α·X`（copy→UB→`Muls`→copy 回）；
2. **cube 侧**用开启原子加的 GEMM 把 `β·Q₁ᵀQ₂` **直接累加进 X**，AIV 不再读 P。

这条流水线里有一个**顺序死结**：现在的时序是

```
AIC: qr_update(it) -> QdwhPmat(it) -> 0x4        // Pmat 在 AIV 的缩放之前！
AIV: 等 0x4 -> (it>0) X<-θX+κP -> Prepare(it+1) -> ComputeAll(it+1)
```

把 Pmat 改成"原子加进 X"之后，`X += P` 会发生在 `θX` **之前**，所以缩放必须提前到
**`QdwhPrepare` 末尾**（那里正好在 Pmat 之前，且与下一轮 Pmat 之间被 QR 面板天然隔开）。
但 `it=0` 的 Pmat(0) 在 Prepare(0) 之前，所以 θ₀/κ₀ 必须**折进 `QdwhInitX`**。

需要维护的不变量（已推导并写成注释）：

| 量 | 含义 |
| --- | --- |
| `X_stored` | 存的是 `X_true / κ_last` |
| `H` 的缩放 | `sqrt(c) · κ_prev`（回推 `X_true`） |
| `Prepare(it)` 末尾的预缩放 | `θ_it · κ_prev / κ_it`（随后 Pmat 原子加 P(it)） |
| 收尾 | phase 1：`QdwhFinalizeScale()` 乘回 `κ_last`（供 FinalGemm）；split：直接在 trace 里乘回 |

我按这套改了两版，第一版错在"原子加在缩放之前"，第二版把缩放移到 `Prepare` 末尾、
并把 θ₀/κ₀ 折进 `InitX` 之后仍有 ~0.3 的误差（说明还有一处因子没对上，最可能在
多矩阵/收尾那一段）。**已全部回滚**，只留下了推导与代码痕迹（`ScaleXAt`、
`QdwhFinalizeScale`、`kappaPrevArr` 都在，但默认 `kappaPrevArr = 1`，等价于旧行为）。

⚠️ 提醒：这条改写的**性能收益约 0.1~1 ms**（AIV 的 `vec` 总忙碌只有 6.9 ms，P/Pᵀ 的
总读取量 ≈ 77 MB ≈ 0.08 ms），它主要是让代码结构贴合公式；真正的瓶颈仍是
**AIC scalar 66.5 ms + AIV scalar 49.8 ms**。

### 与旧两次 launch 流程的对比

| | 旧（qdwh_main + sdc_dev_main） | 融合（fused_main） |
| --- | --- | --- |
| host main / kernel launch | 2 / 2 | **1 / 1** |
| 计算期间 h2d/d2h | 有（H 需要落盘再上传） | **0** |
| kernel 间 host 中转 | `h.bin` 写 4 MB + 读 4 MB + D2H 4 MB + H2D 4 MB | **无** |
| 结果 | leaves=23 / max 63 / eig 1.9e-7 | leaves=23 / max 63 / eig 1.77e-7 |

日志：`output/verify.txt`（1024×1024）、`output/verify_1280x1024.txt`（1280×1024）、
`output/fused_run.txt`（kernel 侧打印）。
