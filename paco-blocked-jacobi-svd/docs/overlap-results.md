# AIV/AIC 流水实现与 4096×4096 实测

本文件保留 AIV 批量向量化之前的历史测量。当前实现、同条件重测和利用率
见 [aiv-vectorization-results.md](aiv-vectorization-results.md)。

2026-11-16 在 server 的 Ascend 910B3 / CANN 8.2.RC1 上采集，
2026-11-17 整理。一次完整调用仍只有一个 mixed kernel launch，
启动 20 AIC + 40 AIV；全部数值计算、round/sweep 和收敛控制在 device。

## 旋转由 AIC 执行

选中两行块组成面板 X，AIV 通过小 Gram 的局部 Jacobi 求解累计旋转矩阵 R，
随后 **AIC 以 Cube GEMM 计算 X_new = R X**，并把结果写回对应的原行块。
block32、4096 列时，R 是 64×64，X 是 64×4096。AIC 按 32 行、128 列
的输出 tile 分配工作，覆盖完整面板。完整矩阵和面板常驻 GM，UB 只保留
小 Gram/R、临时向量或前后处理的行缓存。

代码路径：

- `blocked_jacobi_fused.cpp`：`cube.Apply(...)` 位于
  `__DAV_C220_CUBE__` 分支。
- `fused_cube.hpp`：`FusedCube::Apply` 用
  `mm_.SetMatrix(r, x, b, cp_, d_)` 设置 R、X 和输出 B，
  再调用 `mm_.Process(...)` 执行并写回。
- `paco_gemm.hpp`：`MatmulCustom::Process` 调用 `AscendC::Mmad`，
  通过 Cube 完成矩阵乘法。
- `fused_vector.hpp`：`Parameters` 只读取各任务的小 Gram、求解并写出 R
  和收敛标志。局部求解中的 Givens 向量更新作用于小 Gram 和累计 R；
  对选中原矩阵面板应用 R 的计算在 AIC 上。

## 两组流水

4096/block32/batch1 有 128 个行块，每个 sweep 有 127 个 round，
每轮 64 个互不重叠的块对任务。将它们分成 A/B 两组，执行：

| 阶段 | AIC | AIV |
|---|---|---|
| 0 | 等待 | Pack 全部面板 |
| 1 | Gram(A)：X_A X_A^T | 等待 |
| 2 | Gram(B)：X_B X_B^T | 求 A 组的 R_A |
| 3 | 旋转 A：R_A X_A | 求 B 组的 R_B |
| 4 | 旋转 B：R_B X_B | 等待 |

阶段 2/3 中 AIC 和 AIV 同时计算，阶段之间仍有全核 device barrier。
每组严格满足 Gram → 参数 → 旋转的依赖。面板/Gram/R 的 GM 存储按任务
隔离；不同组写回互不重叠的行块，因此无需额外矩阵缓冲区或 UB。
轮末仍同步所有核，再开始下一轮重新配对。这里实现的是组间流水，
没有跨 round 流水或逐任务 ready 队列。

默认按 AIV 参数任务的轮次分组：设 C 为 AIV 数，W=ceil(jobs/C)，
split=floor(W/2)×C。4096/block32 的 64 jobs/40 AIV 得到 **40+24**。
A 组参数阶段使用 40 AIV，B 组使用 24 AIV；两组的 Gram/旋转任务
都足以调度全部 20 AIC。jobs≤C 时默认保持串行，避免把一轮参数求解
拆成两轮。显式 `--pipeline-jobs N` 仍可用于比较任意合法分组。

## 4096×4096 分解耗时

FP32、batch1、标准正态输入，NumPy seed=20261116，block32、
local-sweeps1、tol=1e-6、最多 30 sweep。每配置独立运行 4 次，舍弃
首次，对余下 3 次取中位数。三种配置都在 **18 sweep 收敛**。

| 调度 | 设备时间中位数 | Host launch 至同步中位数 | 相对本轮串行耗时减少 |
|---|---:|---:|---:|
| 串行 `--no-overlap` | 2946.594 ms | 2946.859 ms | — |
| 流水 32+32 | 2679.671 ms | 2679.892 ms | 9.06% |
| 流水 40+24（当前默认） | **2648.205 ms** | **2648.469 ms** | **10.13%** |

40+24 相对串行加速约 1.113 倍。计时包括 device 归一化/pad、全部
sweep、收敛检查、norm/排序，以及 B_rot/sigma/Vt 输出；不含文件 I/O、
内存分配、初始 H2D、最终 D2H 和独立 CPU 校验。接口沿用 PACO 的行旋转
约定 B=Q A，不累计或输出 U，因此该时间对应现有 B_rot/sigma/Vt 输出范围。
历史串行数据见 [4096-results.md](4096-results.md)，性能改进比例只用
本轮 2946.594 ms 串行运行作对照。

32+32 的原始 server 目录名是 `artifacts/overlap-4096-auto`：当时自动
分组尚为均分。当前默认已经改为 40+24，与实测的显式 40+24 调度相同；
最终默认路径另通过完整回归和 4096 profiler 验证。

## 数值与回归

三种 4096 调度的 B_rot、sigma、Vt 逐位一致，独立 FP64 NumPy SVD
验证全部通过，沿用原阈值。最终默认 40+24 的 profiler 输出也与上述
40+24 输出 SHA256 一致。

| 指标 | 三种调度共同结果 |
|---|---:|
| 奇异值最大绝对误差 | 3.8286e-4 |
| B 非对角 Gram / 最大对角 | 9.5407e-7 |
| Vt 正交性最大绝对误差 | 1.0038e-6 |
| 行空间重构相对误差 | 2.1163e-5 |
| Frobenius norm 相对误差 | 1.0916e-6 |
| 归一化右特征残差 | 1.1091e-5 |

验证记录：CPU reference/validator 5/5、NPU 数值 31/31、控制流 4/4，
最终流水回归 11/11。流水回归包含 10 种调度/终止用例，对照新串行和
保存的修改前程序，三种输出逐位相同且 sweep 相同；另有一项覆盖
5 种非法参数，全部拒绝。覆盖自动/显式分组、奇数任务、单任务、
2/4 AIV 网格、固定 sweep、未收敛退出和重复调用。

4096 输出 SHA256：

```text
B_rot.bin  6b44ca0422e02df644c94f8173d833d7d47b491c3c4f002b8370b79eeb7d2f6b
sigma.bin  2b86fc99112ed55400ab1b22267cfd5793b1ede3c5f4158c0bb562cd63caf643
Vt.bin     4314b81866693dfaf1df5ff945a42bfcade5d461667c16437425cc924bfb84e2
```

## 单次启动与实际利用率

最终默认 40+24、4096/block32、repeat1 的 msprof 记录：

- Runtime `KernelLaunchWithHandle` Count=1。
- `blocked_jacobi_fused` Count=1，Task Type=MIX_AIC。
- Block Dim=20，Mix Block Dim=40。
- Mixed task 时间 2647388.38 us；本次 ACL event 时间 2648.052 ms。
- PipeUtilization：`aic_mac_ratio=0.059`、`aiv_vec_ratio=0.392`、
  `aiv_scalar_ratio=0.256`。

profiler 时间与上表非 profiler 中位数分开报告。全部物理核参与任务，
但 Cube 算术忙碌比例仍约 5.9%；当前组级流水仍有启动/排空阶段、
AIV 参数求解和同步等待。batch1 的 maxabs 归约和最终排序由一个 AIV
按矩阵执行。尚未达到全阶段核心满载，FP32 Gram 对高条件数输入的
小奇异子空间也仍有精度限制。

server 采集目录：

```text
artifacts/msprof-overlap-4096/PROF_000001_20261116142342768_DAJBDGBOHFRJAKKB/
```

本地原始 CSV 在 `artifacts/evidence/msprof-overlap-4096/`，
计时和回归 JSON 在 `artifacts/evidence/overlap/`，
日志在 `logs/artifacts/evidence/overlap/`。两端所有 `.log` 均归档至
项目的 `logs/`，包括 profiler 自动生成的日志，保留原相对路径。

## 复现

```bash
cd /home/zhushengrui/project/paco-blocked-jacobi-svd
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake --build build-fused -j8
OPENBLAS_NUM_THREADS=8 python3 scripts/benchmark_large.py \
  --size 4096 --block 32 --local-sweeps 1 --repeat 4 \
  --artifacts artifacts/overlap-reproduce-default
```

串行对照添加 `--no-overlap`，32+32 对照添加 `--pipeline-jobs 32`，
分别使用独立 artifacts 目录，按顺序运行以免争用 NPU。

profiler 可用以下绝对输入/输出路径复现，避免程序自身的工作目录处理
影响结果存放位置：

```bash
/usr/local/Ascend/ascend-toolkit/8.2.RC1/tools/profiler/bin/msprof \
  --output=artifacts/msprof-overlap-reproduce --task-time=on --runtime-api=on \
  --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
  --application="./build-fused/jacobi_svd_main 0 4096 4096 1 0 30 \
    --block 32 --local-sweeps 1 --repeat 1 \
    --input-dir /home/zhushengrui/project/paco-blocked-jacobi-svd/artifacts/overlap-4096-40/input \
    --output-dir /home/zhushengrui/project/paco-blocked-jacobi-svd/artifacts/overlap-profile-reproduce" \
  > logs/msprof-overlap-reproduce.log 2>&1
```
