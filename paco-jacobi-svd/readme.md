# paco-jacobi-svd

昇腾 910B3（Ascend C）上的**批量单边雅可比 SVD**：输入一批矩阵，同时做单边雅可比旋转，
使每个矩阵各行两两正交，行范数即奇异值。

## 当前实现（小 shape + 大 batch）

**调度（多层划分）**

1. **pad（device 侧）**：`A_raw(rows×cols)` → `A_pad(rowsPad×colsPad)`，补 0；
   行按 **64** 对齐（AIC 分块），列按 **32** 对齐（cube fractal 是 16，AIV 只需 8）。
2. **每个 AIV 平均分到 `perCore = ceil(batch/coreNum)` 个矩阵**：
   - `perCore == 1`：一矩阵一核，逐矩阵旋转（块内 + 块对）；
   - `perCore > 1`：再按 UB 容量切块，一次装 **K** 个矩阵（按行交错 `tile[(i*K+k)*colsW]`），
     同一对 (i,j) 上：
     - 3 条 `Mul` 覆盖 K 个矩阵；
     - 3 条 `WholeReduceSum(repeat=K)` 分别得到 a/b/c（结果落在 `red_[0]/red_[r8]/red_[2*r8]`，
       `r8 = ceil(K/8)*8` 保证 32B 对齐）；
     - 角度逐矩阵用标量算（矢量角度只在 K=64 时才正确，见下）；
     - 旋转仍用 `Muls/Axpy/Adds`（每对只留一次 `Adds` 拷回）。
3. **AIC**：等全部 AIV 完成后，每个 AIC 分摊若干矩阵，用 `MatmulCustom(isTransposeB)` 算
   `G = B·Bᵀ`；主机取对角线开根得到奇异值（AIC 的 Gram 只占 ~0.5% 时间）。

**sweep 停止**：一个 sweep 内没有任何旋转（`|c| <= 1e-6·sqrt(a·b)`）即收敛；`maxSweeps` 兜底。
K 合并路径下是"这一块 K 个矩阵全部收敛"才停。

## UB 分配（192KB 上限）

| 缓冲 | 大小 | 说明 |
| --- | --- | --- |
| `tile_` | 128KB | K 个矩阵的交错数据 |
| `tempBuf_` | 4×2048 float | rowA/B/C + ReduceSum work |
| `prod3_` | K×colsW | 乘积缓冲（三次 Mul 复用） |
| `red_` | 256 float | 3×r8 ≤ 192 |
| `angBuf_`/`maskBuf_` | 4KB | 矢量角度/mask |

K 由 `120KB / ((rows+4)·colsW·4)` 决定，上限 64。实测 K 再大（140KB）收益为负。

## 已验证的坑

* **矢量角度链只在 count=64 时正确**：`CompareScalar/Select/Sqrt/Div` 在 K=1/4/10/13/26 下
  结果都不可信（会让 sigma 错或 NaN），K=64（早期的 128 步版本里验证过）才对。所以角度用标量循环。
* `WholeReduceSum` 的 `dstRepStride=1`（元素）、`srcRepStride=colsW/8`（32B 块）在 repeat=K 下正确。
* UB 超限（>192KB）会直接 `aclError 507015`、输出全 0。
* 列 pad 到 64（而不是 32）会让 32 列矩阵的带宽翻倍，batch 大了以后明显变慢。
* 每次旋转必须"先算 v' 到临时行 → 就地更新 u' → 拷回 v'"，只保留一次 `Adds`。

## 性能（全部 PASS）

| 形状 | batch | 耗时 |
| --- | --- | --- |
| 13×21 | 32 | 0.92 ms |
| 20×20 | 16 | 1.06 ms |
| 32×32 | 16 | 1.71 ms |
| 64×64 | 8 | 4.93 ms |
| 128×128 | 4 | 27.80 ms |
| 20×20 | 512 | 3.68 ms |
| 32×32 | 512 | 7.69 ms |
| 32×32 | 3200 | 42.98 ms（算子内 ~36 ms） |
| 64×64 | 256 | 18.06 ms |

## 使用

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/project/paco-jacobi-svd/cmake-build-debug      # 需 -DCMAKE_BUILD_TYPE=Debug 构建
python3 ../scripts/gen_jacobi_svd_data.py --rows 32 --cols 32 --batch 3200
./jacobi_svd_main 0 32 32 3200                      # device rows cols batch [coreNum] [maxSweeps]
python3 ../scripts/verify_jacobi_svd.py --skip-recon
```
