# 提示词：把 SDC split 的 6 次 GEMM 合并成 3 次（利用 `isTransposeA` + 自定义 ldB）

> 把本文件全文作为任务描述交给执行者即可。它自带全部上下文，不依赖之前的会话。

---

## 1. 目标

在昇腾 910B3 上的融合 QDWH+SDC SVD 算子里，`sdc_tail_cube`（SDC 分裂的单节点尾部）每个节点当前要发 **6 组 GEMM**。利用 `MatmulCustom` 已实现的 **A 转置通路（`isTransposeA`）** 与 **自定义 leading dimension**，把它压成 **3 组**：

```
现状（每节点 6 组）                    目标（每节点 3 组）
  GEMM1  QT   = Q+ᵀ            ┐
  GEMM2  Tmp  = QT · H         ┘→  TmpAll = Qᵀ · H          （一次，M=nPad）
  GEMM3  H+   = Tmp · Q+ᵀ           H+   = TmpAll[0:k] · Q+
  GEMM4  QT   = Q-ᵀ            ┐
  GEMM5  Tmp  = QT · H         ┘→   （已合并进上面的 TmpAll）
  GEMM6  H-   = Tmp · Q-ᵀ           H-   = TmpAll[k:n] · Q-
```

核心洞察：把 A 的"逻辑 M"设成 **`nPad`** 而不是 `kPad`，则 A 物理恰好是 `K×M = n×nPad` 紧凑行主序 —— **就是 `Q` 本身，零拷贝、零物化**；一次 GEMM 得到的 `TmpAll` 前 `k` 行是 `Q+ᵀH`、第 `k..n-1` 行是 `Q-ᵀH`。

**收益的准确预期**（不要期待 MAC 下降）：合并前后 MAC 基本持平（`kPad+m2Pad ≈ nPad`）。真正的收益是
- 少 2 次 GEMM 启动 + 2 处 `AicBarrier`；
- 不再写 `QT`（省掉 `(kPad+m2Pad)×n` 个 float 的 Fixpipe 写回）；
- `qt` 整块 `nPad²` 空间可以删掉（2048² 省 16 MB）；
- `Q+ᵀ`/`Q-ᵀ` 的写后读假依赖消失，`H+`/`H-` 天然解耦。

预期收益落在 `aic_fixpipe` / `aic_mte1` / `aic_scalar` 三项上，`aic_mac` 基本不变。

---

## 2. 环境与工作路径

| 项 | 值 |
|---|---|
| 本地工程 | `C:\Users\dell\Desktop\qdwh_sdc_fused_jacobi` |
| 远端（编译/运行） | `ssh server` → `/home/zhushengrui/project/svd/qdwh_sdc_fused_jacobi` |
| 硬件 | Ascend 910B3 ×2（NPU 6/7），aarch64，容器 host 内存仅 ~2 GB |
| 工具链 | CANN 8.2.RC1，`/usr/local/Ascend/ascend-toolkit/latest` |

**构建**（在远端工程目录）：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake --build build --target fused_main -j2     # 首次或改 CMakeLists 时先 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

**运行**（必须在含 `input/` 的用例目录里跑，kernel 只接受单矩阵）：
```bash
cd <case_dir> && <proj>/build/fused_main 0 20 64 4 1.0 1e-6
#                                    设备 blockDim 阈值 maxIter alpha l0
```

### ⚠️ 三条纪律（都踩过坑）

1. **先 `scp` 再 build**。编译器在远端；只改本地不传 = 编译的是旧代码，会得出完全错误的结论（本项目曾因此白测一轮）。
2. **传输带中文的脚本必须用 base64**。PowerShell 的 `Set-Content -Encoding ascii` 会把中文变成 `????`，导致补丁锚点匹配不上、文件只改了一半。
3. **改完立刻 `md5sum` 核对**本地与远端一致，再继续。

（若 `pwsh` 报 `sandbox backend is not usable ... windows-acl-run --temp ...`，是本机沙箱后端故障，需要 `danger-full-access` 才能执行命令。）

---

## 3. 当前已验证状态（**必须保持**，改动前后都要能回到这里）

远端这 5 个文件的 md5 是"已验证通过"的基线：

```
28c6bc6f62dac072f9d75d748562c00e  vec.hpp
a2cf122e1014f9e5b2cf79a0773c4386  fused_layout.hpp
c5797a11475ebc01987d0b91593159ed  fused_qdwh_sdc.hpp
4033097ea55805b121bf561a81a5f129  sdc_node.hpp
0eb157d935178dd39ea77c4fcd8718c3  fused_batch_cfg.hpp
```

### 性能基线（2048×2048，`blockDim=20 / threshold=64 / maxIter=4`）

| 指标 | 值 |
|---|---|
| kernel | **133.6 ms**（1999² 约 132.4 ms） |
| workspace | 802.3 MB（1999² 为 800.8 MB） |
| leaves | 46（1999² 为 45，1024² 为 23） |
| `max_abs` | 1.4e-06 ~ 2.5e-06，`rel_l2` ≈ 5e-07 |

### 部件分解（msprof `--aic-metrics=PipeUtilization`）

| 流水 | 2048×2048 | 占比 |
|---|---:|---:|
| **aic scalar** | **45.43 ms** | 34.0% |
| aic mte2 | 34.72 ms | 26.0% |
| aic fixpipe | 33.05 ms | 24.7% |
| aic mac | 25.58 ms | 19.1% |
| aic mte1 | 17.23 ms | 12.9% |
| AIC 合计 | 156.01 ms | 重叠 **1.18×** |
| aiv mte2 | 21.70 ms | 16.2% |
| aiv scalar | 20.02 ms | 15.0% |
| aiv vec | 18.72 ms | 14.0% |
| aiv mte3 | 6.46 ms | 4.8% |
| AIV 合计 | 66.90 ms | 重叠 **0.50×**（AIV 一半空闲） |

**结论：AIC 是瓶颈，`aic scalar` 最大；cube 花在标量上的时间是 MAC 的 1.78 倍。**

---

## 4. 需要改的代码：精确事实

### 4.1 `MatmulCustom` 接口（`gemm.cpp`）

```cpp
__aicore__ inline void SetMatrix(AscendC::GlobalTensor<A_T> a, ... , int N, int K);
__aicore__ inline void SetMatrixLd(AscendC::GlobalTensor<A_T> a, AscendC::GlobalTensor<B_T> b,
                                   AscendC::GlobalTensor<C_T> c, int N, int K, int ldA, int ldB);
__aicore__ inline void Process(int offsetA, int offsetB, int offsetC,
                               int blockM, int blockN, int blockK, int cubeidx,
                               bool isTransposeB, bool if_atomicadd, bool isTransposeA = false);
```

要点：
- **第 4 个参数 `N` 兼作 C 的行步长**；实际写入列数由 `Process` 的 `blockN` 限制。这正是能保持 `C ld = kPad/nPad`（128 对齐）的关键。
- `Process` 里若 `!useCustomLd`，会按 `ldA=K`、`ldB=(isTransposeB?K:N)` 重设。
- 类里有成员 `int M, N, K;`（`gemm.cpp:42`），但 `SetMatrix` / `SetMatrixLd` **都不设 `M`**。

### 4.2 A 转置通路的真实语义（`gemm.cpp:162-176`，这是本任务的关键依据）

```cpp
if (isTransposeA) {
    // A 以 K x M 紧凑行主序存储（转置形态），srcDValue = M
    nd2_nz_params.nValue      = l1_k;
    nd2_nz_params.dValue      = l1_m;
    nd2_nz_params.srcDValue   = M;                                  // 用成员 M
    AscendC::DataCopy(a1_local_tensor,
                      a_global_tensor[offsetA + k * L1_M + i * L1_K * M],   // 也用 M
                      nd2_nz_params);
}
```
并在 L1→L0 设 `enTranspose = true`（`gemm.cpp:211` 附近）。

**含义**：A 转置时物理布局 = `K×M` 行主序、**行步长必须等于 `M`**（紧凑）。由于我们的 `SetMatrix*` 从不设 `M`，这条分支目前**不可用** —— 必须先加 M 入口。

（参考实现 `~/project/paco-qdwhsvd/gemm.cpp` 的 `Process2` 是靠 `this->isTransposeB = isTransposeA;` 复用 B 通路实现 A 转置的，语义一致。）

### 4.3 现状：`sdc_node.hpp::sdc_tail_cube` 里的 6 组 GEMM

```cpp
    const int m2 = n - k;
    // ... UΩ1 / UΩ2 / Hwork 三次 GEMM + QR（qr_update）...

    const int kPad = ((k + 127) / 128) * 128;
    const int m2Pad = ((m2 + 127) / 128) * 128;

    // GEMM1: QT = Q+ᵀ  (kPad x n)
    {   GlobalTensor<float> EyeN; EyeN.SetGlobalBuffer((__gm__ float *)eyeN);
        for (int t = blockIdx; t < (kPad + 127) / 128; t += blockNum) {
            const int rowStart = t * 128;
            const int rows = min(128, kPad - rowStart);
            mm.SetMatrixLd(EyeN, QdwhQ, QT, nPad, k, ldOmega, nPad);
            for (int kc = 0; kc < k; kc += 128) {
                const int kk = min(128, k - kc);
                mm.Process(rowStart * ldOmega + kc, kc, rowStart * nPad,
                           rows, n, kk, -1, true, kc != 0);
            }
        }
        op.AicBarrier();
    }
    // GEMM2: Tmp = QT · H  (kPad x n)
    int tilesK = (kPad + 127) / 128;
    mm.SetMatrixLd(QT, H, Tmp, nPad, n, nPad, ldH);
    for (int t = blockIdx; t < tilesK; t += blockNum) {
        int rowStart = t * 128;
        int rows = min(128, kPad - rowStart);
        for (int kc = 0; kc < n; kc += 128) {
            int kk = min(128, n - kc);
            mm.Process(rowStart * nPad + kc, kc * n, rowStart * nPad,
                       rows, n, kk, -1, false, kc != 0);
        }
    }
    op.AicBarrier();
    // GEMM3: H+ = Tmp · QTᵀ  -> Hplus (ld = kPad)
    mm.SetMatrixLd(Tmp, QT, Hplus, kPad, n, nPad, nPad);
    for (int t = blockIdx; t < tilesK; t += blockNum) { ... 
        mm.Process(rowStart * nPad + kc, kc, rowStart * kPad,
                   rows, kPad, kk, -1, true, kc != 0); }
    op.AicBarrier();
    // GEMM4: QT = Q-ᵀ (m2Pad x n，offsetB = k + kc，K = m2)   ← 结构同 GEMM1
    // GEMM5: Tmp = QT · H                                     ← 结构同 GEMM2
    // GEMM6: H- = Tmp · QTᵀ -> Hminus (ld = m2Pad)            ← 结构同 GEMM3
```

相关布局（`fused_layout.hpp`）：`qt = o; o += np2;`、`tmp = o; o += np2;`（`np2 = nPad²`）。
调用方（`fused_qdwh_sdc.hpp:~540`）传入 `Addr(L.qt)`、`Addr(L.tmp)`。

---

## 5. 目标改法

### 步骤 1：给 GEMM 增加 M 入口（`gemm.cpp`）

只新增，不改现有调用：

```cpp
    // 带自定义 ld 且显式指定 A 的逻辑 M —— isTransposeA 时用于 A 的转置寻址
    __aicore__ inline void SetMatrixLdM(AscendC::GlobalTensor<A_T> a, AscendC::GlobalTensor<B_T> b,
                                        AscendC::GlobalTensor<C_T> c,
                                        int M, int N, int K, int ldA, int ldB) {
        SetMatrixLd(a, b, c, N, K, ldA, ldB);
        this->M = M;
    }
```

### 步骤 2：用一次 GEMM 同时得到 `Q+ᵀH` 与 `Q-ᵀH`

```cpp
    // TmpAll = Qᵀ · H   (nPad x n)：A 物理 = K×M = n×nPad 紧凑 → 就是 Q 的前 n 行
    // 第 [0, k)   行 = Q+ᵀ H
    // 第 [k, n)   行 = Q-ᵀ H
    mm.SetMatrixLdM(QdwhQ, H, Tmp,
                    /*M*/ nPad, /*N(=C ld)*/ nPad, /*K*/ n, /*ldA*/ nPad, /*ldB*/ ldH);
    for (int t = blockIdx; t < (nPad + 127) / 128; t += blockNum) {
        const int rowStart = t * 128;
        const int rows = min(128, nPad - rowStart);
        mm.Process(0, /*offsetB*/ 0, rowStart * nPad,
                   rows, n, kk, -1, /*isTransposeB*/ false, /*atomic*/ false, /*isTransposeA*/ true);
        // 注意：K 维需按 128 分块，offsetA/offsetB 的分块增量见下方"待实测确认"
    }
    op.AicBarrier();
```
> `TmpAll` 复用现有 `tmp`（`nPad×n ≤ nPad²` ✓，不需要扩空间）。

### 步骤 3：`H±` 用 `TmpAll` 的行块 + `Q` 的列块

```cpp
    // H+ = TmpAll[0:k] · Q+   ：B = Q 的 n×k 列块，ldB = nPad（自定义），不转置
    mm.SetMatrixLd(Tmp, QdwhQ, Hplus, /*N(=C ld)*/ kPad, /*K*/ n, /*ldA*/ nPad, /*ldB*/ nPad);
    for (int t = blockIdx; t < (kPad + 127) / 128; t += blockNum) {
        const int rowStart = t * 128;
        const int rows = min(128, kPad - rowStart);
        for (int kc = 0; kc < n; kc += 128) {
            const int kk = min(128, n - kc);
            mm.Process(rowStart * nPad + kc, /*B base*/ kc * nPad, rowStart * kPad,
                       rows, k, kk, -1, false, kc != 0);      // blockN = k（只写前 k 列）
        }
    }
    op.AicBarrier();

    // H- = TmpAll[k:n] · Q-  ：B 基址偏移 + k，N 维 = m2，C ld = m2Pad
    mm.SetMatrixLd(Tmp, QdwhQ, Hminus, m2Pad, n, nPad, nPad);
    for (int t = blockIdx; t < (m2Pad + 127) / 128; t += blockNum) {
        const int rowStart = t * 128;
        const int rows = min(128, m2Pad - rowStart);
        for (int kc = 0; kc < n; kc += 128) {
            const int kk = min(128, n - kc);
            mm.Process((k + rowStart) * nPad + kc, k + kc * nPad, rowStart * m2Pad,
                       rows, m2, kk, -1, false, kc != 0);
        }
    }
    op.AicBarrier();
```

### 步骤 4：删掉 `qt`

`fused_layout.hpp` 里 `qt = o; o += np2;` → 改成 `qt = o; o += 0;`（或直接删除并把调用方参数去掉）。注意 host 与 device 共用同一份 `Build()`，两边一致即可。

---

## 6. **必须先实测确认的两个 offset 语义**（否则会静默算错）

现有代码里有一处可疑：GEMM2 写 `mm.Process(..., kc * n, ...)` 而 `SetMatrixLd(..., ldB=ldH)`——K 维偏移按 `n` 而非 `ldH` 递增，只在 `n == nPad` 时两者才相等。实施前请先确认：

1. **非转置 B 的 `offsetB` 语义**：是"逻辑左上角 (行, 列) 的线性偏移"（即 `kc*ldB + colOff`），还是"按块计数"？
2. **A 转置时 `offsetA` 的语义**：`gemm.cpp` 内部已含 `offsetA + k*L1_M + i*L1_K*M`，所以 `offsetA` 应只给 A 的基址 + 列偏移（`Q-` 时为 `k`）。

**验证方式**：先用小尺寸（`128×128`、`192×192`、`256×192`）对比 CPU SVD，一次即可判断语义是否对。不要在大尺寸上摸索。

---

## 7. 验证协议（**分级、可回退**）

### 7.0 准备
```bash
mkdir -p ~/dsh_tmp/v4 && cd <proj> && cp vec.hpp fused_layout.hpp fused_qdwh_sdc.hpp sdc_node.hpp gemm.cpp ~/dsh_tmp/v4/
```

### 7.1 小尺寸语义验证（最重要，先做）
对 `128×128`、`192×192`、`256×192` 生成数据并跑，要求 `np.abs(sigma - cpu_svd).max() < 1e-5`。
生成数据：`python3 gen_data.py --shapes 128x128 --cond 1e2 --out-dir <case>/input`（工程自带；参考 `run.sh`）。

### 7.2 中尺寸正确性（各 6 次）
| 用例 | 期望 leaves | 期望 `max_abs` |
|---|---:|---:|
| 1024×1024 | 23 | ≤ 3e-06 |
| 1999×1999 | 45 | ≤ 1e-05 |
| 2048×2048 | 46 | ≤ 3e-06 |

`leaves` 必须与基线一致（差 1 以内可接受，差更多说明 `k` 或分裂点错了）。

### 7.3 性能与部件（关键判据）
```bash
msprof --application="<proj>/build/fused_main 0 20 64 4 1.0 1e-6" \
       --output=<out> --ai-core=on --aic-metrics=PipeUtilization --task-time=on
```
解析 `op_summary_*.csv` 的 `qdwh_sdc_fused_custom` 行，对比：

| 指标 | 基线 | 目标 |
|---|---:|---|
| kernel | 133.6 ms | 下降 |
| `aic_fixpipe_time` | 33.05 ms | 下降（少写 QT） |
| `aic_mte1_time` | 17.23 ms | 下降或持平 |
| `aic_scalar_time` | 45.43 ms | 下降（少 2 次 GEMM 启动） |
| `aic_mac_time` | 25.58 ms | **基本不变**（MAC 没省） |

**只有 kernel 时间真的下降才保留；只降部件不降 kernel 说明被掩盖了。**

### 7.4 权威回归
```bash
cd <proj> && source /usr/local/Ascend/ascend-toolkit/set_env.sh
OPENBLAS_NUM_THREADS=4 python3 regression_20261113/run_final_regression.py
```
期望 `PASS: both sizes passed ...`（20 次）。**已知既有问题**：1999² 有约 2~3% 概率出现单个特征值 `max_abs ≈ 1.2~1.4e-05` 越线（`rel_l2` 仍 ~1e-06），与本次改动无关，见 `README.md` 的"已知既有缺陷"。

### 7.5 失败时
立刻 `cp ~/dsh_tmp/v4/* .` 回退、重建、确认 4 次 PASS，再分析原因。**绝不把间歇性错误留在树上。**

---

## 8. 不要重做的事（本项目已验证为负结果）

| 尝试 | 结果 |
|---|---|
| 去掉 UΩ1/UΩ2 之间的冗余 `AicBarrier` | 0 收益 |
| λ 的 trace 两趟循环合成一趟 | 0 收益 |
| 把 `Q+ᵀ`/`Q-ᵀ` 拆到 `qt` 的两半、提前合并 | **正确但慢 0.6~1.0 ms**（打乱了与 H+ 链的重叠） |
| 把 `k=(trace(U)+n)/2` 搬到 AIV（4 种发布变体） | **全部间歇性错误**，已放弃 |
| 加大 `batch` 到 14 / 15 | 已落地（UB 推导上限），但**继续加大已饱和**（3072² 分组 11→8，时间不变） |

结论：**当前 GEMM 调度已有不错的流水重叠，"去掉 barrier"不等于变快**。任何改动都必须用 msprof + kernel 时间双重验证。

---

## 9. 验收标准

1. `sdc_tail_cube` 每节点的 GEMM 组数从 **6 → 3**，`qt` 空间删除；
2. 1024²/1999²/2048² 各 6 次全部 PASS，`leaves` 与基线一致（±1），`max_abs` 不劣于基线；
3. `kernel` 时间较基线（2048²: 133.6 ms，1999²: 132.4 ms）**下降 ≥1%**，且 `aic_fixpipe` / `aic_scalar` 至少一项明显下降；
4. 权威回归 20/20（允许 1999² 的既有尾部抖动）；
5. 本地与远端 5+1 个文件 md5 一致，README 记录改动与实测数据。

---

## 10. 实施结果（2026-11-13，已落地，详见 README §“优化记录 17”）

- **6 → 3 组 GEMM 已实现并上线**：`TmpAll = Qᵀ·H`（`isTransposeA`，A 物理就是 `Q`），
  `H+ = TmpAll[0:k]·Q+`、`H- = TmpAll[k:n]·Q-`。
- kernel 时间（同机交替跑 10 次中位数）：2048² **133.89 → 131.09 ms（-2.09%）**、
  1999² **131.83 → 129.63（-1.67%）**、1024² 50.22 → 48.77（-2.89%）。
  `mac` 基本不变（符合 §1 的预期）；`fixpipe`/`scalar` 只降 ~1%，收益来自
  "少 2 次启动 + 2 次 `AicBarrier`" 而不是流水负载。
- 权威回归 20/20；128²~2048² 的 `verify_end_to_end.py`（对 CPU SVD）全部 PASS，
  `leaves` 与基线一致。
- **两处与本文档原方案不同，务必按实测来**：
  1. §4.2 说的"先加 M 入口"是对的（已加 `SetMatrixLdM`），但 A 转置通路
     **只在 M/N/K 全是满 128 tile 时可用**：K 维残块会让 kernel 卡死，
     M/N 残块会算错 → 必须 `K = M = N = nPad`，并让 `Q` 的 padding 行/列恒为 0
     （`BuildIdentityQ(nIdent, n, …)`），同时把 arena 清 0（`ZeroRange`）以吃掉
     K 维多读的那几行 `H`。
  2. **§5 步骤 4"删掉 qt"不能做**：`P_ = U-I` 放进 `hplus` 会让 n ≥ 256 的用例错到
     `max_abs ≈ 0.2~0.3`（小尺寸看不出来）。`qt` 保留，workspace 因此没有变小。
- 验收口径提醒：`verify_sigma.py`（叶子内部一致性）**不能**用来验收本改动，
  它会漏掉整块 H± 算错的情况；必须用 `verify_end_to_end.py`。

