# AIV 批量向量化与 4096×4096 实测

在 server 的 Ascend 910B3 / CANN 8.2.RC1 上实现和验证，2026-11-17 整理。
本轮重点重构 `block_rotation.hpp` 的 AIV 小 Gram 求解；仍为一次
mixed kernel launch，20 AIC + 40 AIV。全部数值计算、配对、循环和收敛控制
在 device。**对原矩阵面板执行旋转的仍是 AIC Cube：`X_new = R X`。**

## AIV 改动

- 将 64×64 Gram 从每元素复制 8 份改成紧凑存储。
- 用 circle schedule 将 2016 个 pivot 分成 63 轮，每轮同时处理 32 对
  不相交 pivot。存储次序为 `[P rows, Q rows]`，每轮用固定 permutation
  推进配对，63 轮后回到原行标签。
- 32 对角度一起执行 Gather、Sqrt、Div、Compare、Select；没有逐 pivot
  的 UB 标量读取、角度分支或 Scalar/Vector 握手。角度向量含 32 个有效
  元素，行更新向量含 2048 个有效 FP32 元素，即 32 个完整 vector repeat。
- 每轮完整执行 `J G J^T` 和 `J R`，包含不同 pivot 之间的交叉项。
  两次 Gram 转置/排列 Gather 和一次 R 行排列复用每核初始化一次的映射。
  不将更新后的行简单镜像到列。
- G/trans 和 R/rNext 交替作为独立源/目标；每次行变换用六条连续向量
  算术指令，去掉旧的第七条行复制，不增加 UB。
- 局部 sweep 末用 WholeReduceMax 后仅一次显式 V→S 读取 active；不使用
  会在内部引入额外标量复制和事件的高层 ReduceMax。
- 所有 R 行的 norm 用一次 WholeReduceSum 批量归约，再做向量 Newton
  行长度校正；不再逐行标量归约和同步。

block16 使用相同算法，d=32，每轮 16 对 pivot，行更新为 512 个有效元素。
UB 总分配 180 KiB；前 128 KiB 仍供不同阶段的预处理/后处理复用，静态
Gather 映射放在其外。完整矩阵、面板、任务间的 Gram/R 均常驻 GM。

## 同步与部件分工

输入 Gram DMA 前的 `V_MTE2` 是必要的缓冲区复用依赖。低核数回归发现，
同一 AIV 连续处理多个任务时，下一任务的 MTE2 可能覆盖上一任务仍在使用
的 g_。原本结尾的 `MTE3_V` 只保护后续 V，不能保护下一次 MTE2。补上
依赖后，2/4 AIV 和重复调用测试均恢复正确，未增加全核 device barrier。

同一个参数任务内部的各 circle round 持续执行向量计算，没有逐 pivot
的 Scalar/Vector 握手。跨 AIC/AIV 仍沿用两组流水：

| 阶段 | AIC | AIV |
|---|---|---|
| 0 | 等待 | Pack 全部面板 |
| 1 | Gram(A) | 等待 |
| 2 | Gram(B) | 求解参数(A) |
| 3 | 完整面板旋转 `R_A X_A` | 求解参数(B) |
| 4 | 完整面板旋转 `R_B X_B` | 等待 |

阶段之间仍有 device barrier，轮末全部写回后再重新配对。AIV 操作小 G/R，
AIC 执行 Gram 和完整面板 GEMM；Cube 实现及 mixed kernel 调度没有修改。

## 4096×4096 耗时

同机、同 seed=20261116 的 FP32 标准正态输入，batch1、block32、
local-sweeps1、tol=1e-6、max-sweeps30。各配置顺序运行，四次完整分解，
舍弃首次，取其余三次中位数。修改前程序单独保存在
`artifacts/pre-aiv/jacobi_svd_main`，本轮重新测量。

| 实现/调度 | 设备时间中位数 | Host launch 至同步中位数 | sweep |
|---|---:|---:|---:|
| 修改前，40+24 | 2642.737 ms | 2642.991 ms | 18 |
| AIV 向量化，串行阶段 | 2031.171 ms | 2031.396 ms | 18 |
| AIV 向量化，32+32 | 1756.200 ms | 1756.457 ms | 18 |
| AIV 向量化，40+24 | 1729.624 ms | 1729.888 ms | 18 |

默认配置耗时减少 34.55%，加速约 1.528 倍。计时包括 device 归一化、补零、
全部 sweep、收敛、norm/排序及输出生成；不含文件 I/O、分配、初始 H2D、
最终 D2H 和独立 CPU 校验。输出沿用 PACO 的 B_rot/sigma/Vt 范围，
不累计或输出 U。

保留 40+24 默认分组；本轮比 32+32 少约 1.5% 耗时，比新求解器的串行
阶段少约 14.85%。四配置输入 SHA256 相同，均完成独立 FP64 数值验证。
三种新调度的全部二进制输出也相同。

## 数值与回归

沿用原阈值，CPU 5/5、NPU 数值 31/31、控制流 4/4、流水 11/11 通过。
流水用例的串行/重叠调度输出 B_rot/sigma/Vt 逐位一致，sweep 相同。
测试包含 block16/32、矩形与补齐、零/重谱/秩亏、条件数 1e4、极端尺度、
混合尺度 batch、2/4 AIV、固定 sweep、未收敛退出、重复调用及非法分组。

4096 最终默认配置的独立 FP64 验证：

| 指标 | 结果 |
|---|---:|
| 奇异值最大绝对误差 | 3.50152e-4 |
| B 非对角 Gram / 最大对角 | 9.45899e-7 |
| Vt 正交性最大绝对误差 | 1.00374e-6 |
| 行空间重构相对误差 | 2.01434e-5 |
| Frobenius norm 相对误差 | 8.46258e-7 |
| 归一化右特征残差 | 1.01124e-5 |

新的局部 pivot 顺序会改变输出向量和舍入路径，因此新旧实现之间比较独立
FP64 NumPy SVD 和分解不变量；不要求新旧 B/V 逐位相同。

## 单次启动与实际利用率

最终 40+24、4096/block32、repeat1 的 msprof 记录：

- Runtime `KernelLaunchWithHandle` Count=1。
- 算子 Count=1，Task Type=MIX_AIC，Block Dim=20，Mix Block Dim=40。
- Mixed task 时间 1727727.700 us；profiler 测量与上表的普通计时分开。
- profiler 输出 B_rot/sigma/Vt 与普通计时的默认配置 SHA256 一致。

| PipeUtilization 指标 | 修改前历史采集 | 本轮最终采集 |
|---|---:|---:|
| `aiv_vec_ratio` | 0.392 | 0.593 |
| `aiv_scalar_ratio` | 0.256 | 0.044 |
| `aic_mac_ratio` | 0.059 | 0.091 |

修改前 profiler 记录保存在 [overlap-results.md](overlap-results.md)。上述
ratio 是完整任务的流水忙碌比例，不能等同于峰值 FLOPS 达成率。仍有组间
全核屏障、阶段启动/排空、Gather 和向量依赖延迟；40+24 的参数阶段分别
有 40/24 个 AIV 任务。batch1 的 maxabs 和最终排序仍只由一个 AIV 执行。
本轮提高了向量计算占比并降低标量串行开销，尚未做到所有核心全程满载。
FP32 Gram 的高条件数/小奇异子空间精度限制仍存在。

## 复现与证据

```bash
cd /home/zhushengrui/project/paco-blocked-jacobi-svd
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake --build build-fused -j8
OPENBLAS_NUM_THREADS=8 python3 scripts/benchmark_large.py \
  --size 4096 --block 32 --repeat 4 \
  --artifacts artifacts/aiv-reproduce-4096
```

使用 `--no-overlap` 测串行调度，`--pipeline-jobs 32` 测 32+32，
`--exe artifacts/pre-aiv/jacobi_svd_main` 测保存的修改前版本。
各配置顺序运行，避免争用 NPU。所有 `.log` 存在项目 `logs/` 下。

两端均保存精简证据目录 `artifacts/evidence/aiv-vectorization/`：

- `verification.json`：通过数、四配置计时、输入/输出 SHA256、源码与程序
  SHA256、一次 launch 的 profiler 行。
- `benchmarks/`：四种配置的原始 timing/results JSON。
- `tests/`：最终三组 NPU 回归 results JSON。
- `msprof-final/`：最终 op_summary、op_statistic、api_statistic、task_time CSV。

完整输入和二进制输出保留在 server 的各 `artifacts/aiv-*-4096*` 目录；
完整 profiler 数据保留在 `artifacts/msprof-aiv-final/`。
