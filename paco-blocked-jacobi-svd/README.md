# PACO blocked one-sided Jacobi SVD — single launch

针对 Ascend 910B3 / CANN 8.2 的独立实现，沿用 PACO 批量 SVD CLI 和二进制
接口。**每次完整 SVD 只启动一次 mixed AIC:AIV=1:2 kernel**；所有数值计算、
round 配对、sweep 循环和收敛控制都在 device 内完成。

本地目录：`C:/Users/dell/Desktop/svd/paco-blocked-jacobi-svd`；server 镜像：
`/home/zhushengrui/project/paco-blocked-jacobi-svd`。

日志统一放在项目的 `logs/` 中并保留原相对路径。测试与 benchmark 脚本
也写入此目录；项目外部的 artifacts 路径通过哈希区分。

全部源码、脚本、测试和文档的逐文件功能，以及本次无用函数清理范围，见
[file-guide.md](docs/file-guide.md)。

当前 AIV 批量更新和双缓冲实现见 [aiv-batching.md](docs/aiv-batching.md)：
block64 每次更新 32 对行，连续向量算术指令覆盖 4096 个 FP32 元素；
Pack 每次搬运 4 行，用两个缓冲区重叠 MTE2/MTE3。4096×4096、batch1、
FP32 epsilon 下，旧调度的完整分解设备时间中位数由当时同条件基线的
2025.754 ms 降至 **1692.535 ms**，减少 **16.45%**；均为 25 sweep，
输出逐位一致。上述数据使用旧 sweep 定义，不代表当前调度的性能。
每对一次的全局分阶段调度实测为 **966.515 ms**，25 sweep；
测量方式与精度结果见 [exact-sweep.md](docs/exact-sweep.md)。
2026-11-17 核组流水版本将每 sweep 的全局屏障从 254 次减至 2 次，4096 同输入、
同容差的 warm 中位数为 **904.740 ms**，减少 **6.39%**，仍为 25 sweep，
输出与原版逐位一致。协议、诊断和日志见 [core-local-pipeline.md](docs/core-local-pipeline.md)。
AIV UB 仍为 180 KiB，完整面板旋转仍由 AIC Cube 执行。
2026-11-18 进一步优化角度计算、无效任务和前后处理流水后，最终 warm 中位数为
**712.531 ms**；同次保存旧版重测为 900.693 ms，耗时减少 **20.89%**，相对
966.515 ms 历史基线减少 **26.28%**。均为 25 sweep，输出逐位一致，独立 FP64
校验通过。AIC Scalar 时间减少 23.48%，AIV Scalar 时间却增加 19.75%，仍需优化；
完整分项、流水限制和证据见 [scalar-pipeline.md](docs/scalar-pipeline.md)。
随后针对 AIC 等待减少 block64 单位局部轮的角度后半段及 G/R 更新，并省去
默认路径无人读取的单位 R 写回；最终同机 ABBA warm 中位数由 **713.453 ms**
降至 **620.730 ms**，减少 **13.00%**，相对 966.515 ms 历史基线累计减少
**35.78%**。AIC Scalar 绝对时间减少 **14.21%**；仍为25 sweep，输出逐位一致，
12个静态测试和105项设备/CLI回归通过，4096独立 FP64 校验通过。
同步协议保持不变，详细结果及原始证据见 [aic-wait.md](docs/aic-wait.md)。
随后将 UB 加零搬运改为 `Copy`，本轮同机 ABBA warm 中位数由 **621.655 ms**
降至 **615.016 ms**，减少 **1.07%**；均为25 sweep并收敛，输出逐位一致，
12个静态测试和105项设备/CLI回归、4096独立 FP64校验通过。
Copy 仍使用 Vector 管线，UB仍为180 KiB；实现、步长约束和证据见
[ub-copy.md](docs/ub-copy.md)。
CLI 默认 block16，使用本方案须指定 `--block 64`。
最初的 block64 设计与精度边界见 [block64-results.md](docs/block64-results.md)。

## 执行路径

唯一入口是 `blocked_jacobi_fused.cpp` 的 `blocked_jacobi_fused`。在 910B3 上
默认一次启动 20 AIC + 40 AIV；预先用 device 公式给每个 AIV 分配任务队列，
每组 AIC 轮询本组两个 AIV 的请求。整个调用的执行顺序为：

1. AIV：计算每个矩阵的 maxabs scale，归一化并补零。
2. Device 循环：公式生成 round-robin 块配对；AIV pack 两块面板 X；AIC 用
   Cube GEMM 计算 `X X^T`；AIV 对小 Gram 做 Jacobi 迭代并累积稠密旋转 R；
   AIC 用 Cube GEMM 计算完整面板 `R X` 并写回。
3. AIV：在完整 sweep 后归约收敛标志，决定继续或退出。
4. AIV：计算行 norm、排序奇异值、归一化生成 Vt，并恢复 B 和 sigma 的尺度。

Host 负责参数/文件检查、内存管理、原始输入上传、一次 launch、等待及结果
下载。`main.cpp` 没有 host sweep、数值缩放或排序。`--repeat N` 表示 N 次
独立完整调用，每次仍只启动一次 kernel；默认 N=1。

`DeviceBarrier` 使用 mode 0 同类核屏障和 mode 2 AIC/AIV 握手，确保阶段间
GM 数据可见。所有常驻核都参加，即使没有分配到计算任务；启动数量不得
超过物理核数。当前默认通过 GM 块版本交接跨组依赖，每个核组可处理已就绪的
后续轮次任务；每 sweep 仅在收敛汇总前后做两次全局屏障，初始化和最终输出
另各一次。本组 request/ack 也采用 64B GM mailbox；核内事件负责 DMA 完成，
不能代替跨组块版本。两槽缓冲在 Apply ack 前不可复用。

显式 `--pipeline-jobs N` 仍在每轮分成 A/B 两组，执行：

| 阶段 | AIC | AIV |
|---|---|---|
| 0 | 等待 | 打包全部面板 |
| 1 | Gram(A) | 等待 |
| 2 | Gram(B) | 参数求解(A) |
| 3 | 旋转(A) | 参数求解(B) |
| 4 | 旋转(B) | 等待 |

阶段 2/3 中两类核同时计算，阶段之间仍使用全核屏障。每轮最后的屏障
保证下一轮重新配对前所有行块均已写回。同一组的依赖严格保持
`Gram -> 参数 -> 旋转`，没有跨轮流水。GM 中的面板/Gram/R 按任务隔离，
无需增加矩阵缓冲区或 AIV UB。默认核组流水则允许不同轮次重叠，使用块版本
保证读取前驱的完整 AIC 写回。4096/block64 有 32 个参数任务，最多 32 个
AIV 有用，不代表 40 个 AIV 满载；12 个核组有两个活动槽，其余 8 个有一个。

AIV 的局部求解采用紧凑 Gram 和批量配对。在每个全局 sweep 的第一个块轮次，
每对块处理全部局部配对；其余块轮次仅处理两块之间的配对。这样每个原始行对
在一个 sweep 中恰好获得一次 pivot 机会；已满足阈值的 pivot 使用单位旋转。
block64 的完整局部轮次为 127 轮，跨块轮次为 64 轮，每轮同时求 64 对角度。
角度、两侧 Gram 更新、R 更新和行长度校正均用向量指令；行更新以 32 对行为一批，每条连续算术指令覆盖
4096 个有效 FP32 元素，列更新每条覆盖 2048 个元素。
Gather 映射在完整配对与跨块配对模式切换时重建，没有逐 pivot 的标量读取或
S/V 握手。block64 后续跨块局部轮用一次 V→S 归约检查当前64个pivot，全部不
活跃时省去角度后半段和 G/R 旋转，始终执行行列置换；首个完整局部轮不加此
逐轮检查。局部求解末仍归约整个任务的活跃状态。UB 为 180 KiB；矩阵仍在 GM。
block64 的后续跨块轮次增加一次全部 pivot 阈值预检查；全不活跃时省去局部
单位旋转循环。默认核组路径同时省去单位 R 的生成、写回及 AIC 的单位矩阵
Apply，仍发布块版本并等待 ack。预检查增加一次 V→S 归约，不改变配对机会或阈值。
Pack 复用求解器前 128 KiB 为两个四行缓冲，直接用 MTE2→MTE3 事件
传递 DMA 数据就绪状态，去掉原逐行搬运的 Vector 握手；缓冲复用及阶段依赖
仍由事件和 device barrier 保护。

`RotateRows`（block64 路径为 `RotateTile`）在 AIV 上更新小 Gram 和累计 R；
原矩阵面板的 `R X` 在 `FusedCube::ApplyJob`（全局路径为 `Apply`）中由 AIC 执行。
`Prepare` 已用 `CreateVecIndex`、广播、矢量算术和比较生成静态索引/单位阵掩码，
不再逐元素调用 `SetValue`；CANN 8.2 的序列接口内部仍保留 8 个标量种子写入。
实现与对照结果见 [prepare-vectorization.md](docs/prepare-vectorization.md)。

## 接口和构建

支持 `1 <= rows <= 4096`、`1 <= cols <= 4096` 和正 batch，block 为 16、32 或 64。
输入 `A_batch.bin` 为连续 row-major FP32，形状 `[batch, rows, cols]`。
输出仍遵循 PACO 的行旋转约定 `B = Q A`：

- `B_rot.bin`：`[batch, rows, cols]`，保留旋转后的行顺序。
- `sigma.bin`：`[batch, min(rows,cols)]`，降序奇异值。
- `Vt.bin`：`[batch, min(rows,cols), cols]`，按 sigma 排序的归一化 B 行。
- `metrics.json`：核心数量、实际 sweep/收敛状态、任务数及计时。

在 server 上构建和运行：

```bash
cd /home/zhushengrui/project/paco-blocked-jacobi-svd
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build-fused -DCMAKE_BUILD_TYPE=Release
cmake --build build-fused -j8
python3 scripts/gen_jacobi_svd_data.py --rows 128 --cols 128 --batch 8 --out-dir input
./build-fused/jacobi_svd_main 0 128 128 8 0 30 \
  --block 16 --local-sweeps 1 --input-dir input --output-dir output
```

位置参数依次为 device、rows、cols、batch、AIV 核数（0 表示全核）、最大 sweep。
限制核数时必须为正偶数，对应 AIC 数为其一半。`--local-sweeps` 只接受 1，
避免一个全局 sweep 内重复访问行对。`--fixed-sweeps` 固定执行
sweep 数，默认在完整 sweep 后检查收敛；`--tol` 默认 FP32 epsilon
`1.1920928955078125e-7`。当前采用逐对 Gram 相关性判据；这不构成与
GPU XGESVDJ 完整停止条件或输出精度等价的证明。数值验证使用独立 FP64
NumPy SVD，详见 [block64-results.md](docs/block64-results.md)。
`--no-overlap` 使用串行阶段；`--pipeline-jobs N` 指定每轮 A 组的块对
任务数，B 组处理剩余任务。N=0（默认）使用核组流水和块版本；正 N 必须小于
每轮总任务数，不能与 `--no-overlap` 同用。`metrics.json` 记录
`core_local_pipeline`、`synchronization`、`global_barriers_per_sweep`；
`phase_overlap` 专指旧显式 A/B 分组，默认核组流水时为 false。
`identity_apply_elision` 表示默认路径可省略单位 Apply；该路径的任务与
`padded_cube_flops` 估计是上界，由 `work_estimates_are_upper_bounds` 标明。

## 验证

本次 AIV 批量优化的分阶段计时、UB 布局与回归记录见
[aiv-batching.md](docs/aiv-batching.md)。

Gram 同步删减及无广播 Axpy 的对照结果见
[gram-synchronization.md](docs/gram-synchronization.md)。4096×4096 下，
只删除 Gram 每个 K 分段后的全流水屏障，block64/block32 的时间分别减少
8.13%/6.32%，输出逐位一致；两个旋转候选均因实测更慢而未设为默认。

```bash
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python3 tests/run_npu.py \
  --artifacts artifacts/npu-single-launch-final
OPENBLAS_NUM_THREADS=1 python3 tests/run_control.py \
  --artifacts artifacts/control-single-launch-final
OPENBLAS_NUM_THREADS=8 python3 tests/run_overlap.py
```

数值验证对照独立 NumPy SVD，检查奇异值、B 行正交性、能量、Vt 正交性、
行空间重构，以及 `A^T A v = sigma^2 v` 的归一化残差，确保 Vt 与 sigma 对应。
覆盖矩形/补齐边界、block16/32/64、零矩阵、重谱、秩亏、条件数 1e4、
`1e-38` 至 `1e30` 尺度和混合尺度 batch；控制流覆盖固定 sweep、重复调用
及仅使用 2/4 AIV 的常驻网格。
流水回归额外比较串行/流水的 B_rot、sigma、Vt 是否逐位一致，检查实际
sweep 相同，覆盖自动/显式分组、奇数任务、单任务、固定 sweep 和重复调用。
当前核组版本的静态、单 sweep FP64、完整数值、故障注入及 profiler 结果见
[core-local-pipeline.md](docs/core-local-pipeline.md)；计时复现脚本为
`tests/run_core_local_4096.py`，依赖服务器保留的 exact-sweep 输入和原版 executable。
2026-11-18 的 Scalar、Angles 数学、通信和 UB/MTE 流水优化见
[scalar-pipeline.md](docs/scalar-pipeline.md)，使用
`tests/benchmark_scalar_pipeline.py --label final --abba --validate` 复现同输入完整计时。
最新 AIC 等待优化见 [aic-wait.md](docs/aic-wait.md)，复现本轮完整对照时明确
指定起始 executable，并为新日志使用独立 label：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/benchmark_scalar_pipeline.py \
  --baseline artifacts/aic-wait-start/jacobi_svd_baseline \
  --label aic-wait-recheck --abba --validate
```

此前 tol1e-6 的 AIV 向量化实现与 4096×4096 实测见
[aiv-vectorization-results.md](docs/aiv-vectorization-results.md)（旧 sweep 定义）：40+24 默认
分组在 18 sweep 收敛，设备时间中位数 **1729.624 ms**，比同条件重跑的
修改前版本 2642.737 ms 减少 **34.55%**。完整面板旋转 `R X` 由 AIC Cube
执行；AIV 求小 Gram 的参数并生成 R。最终 profiler 确认一次 mixed launch、
20 AIC + 40 AIV，AIV Vector 忙碌比例 59.3%，Scalar 为 4.4%。

先前流水测量见 [overlap-results.md](docs/overlap-results.md)。早期串行
4096 数据见 [4096-results.md](docs/4096-results.md)；小矩阵记录见
[single-launch-results.md](docs/single-launch-results.md)。旧 `build-opt` 和
`benchmark-opt` 来自多 launch 版本，不能作为本版本的性能结果。

单次调用 profiler 命令（先准备相应 input）：

```bash
/usr/local/Ascend/ascend-toolkit/8.2.RC1/tools/profiler/bin/msprof \
  --output=logs/msprof-single-launch --task-time=on --runtime-api=on \
  --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
  --application="./build-fused/jacobi_svd_main 0 128 128 10 0 30 \
    --block 16 --local-sweeps 1 --repeat 1 --input-dir input --output-dir output"
```

## 当前限制

完整矩阵和打包面板常驻 GM。每个 AIV 的 UB 只存当前小 Gram/旋转矩阵，
或前后处理的两行缓存及 O(rows) 的 norm/排序数组；二者分阶段复用空间。
对齐路径以两个槽连接 MTE2→Vector→MTE3，补齐路径保留原实现；Pack 则使用
两个四行 DMA 槽。三个阶段不会同时占用同一 UB 区间。
4096×4096 的复现命令为：

```bash
OPENBLAS_NUM_THREADS=8 python3 scripts/benchmark_large.py \
  --size 4096 --block 64 --local-sweeps 1 --repeat 4 \
  --artifacts artifacts/large-4096-block64-tile16
```

全核启动不等于全程满载：参数阶段每个面板对应一个 AIV 任务；小 batch
会限制独立任务数量。`metrics.json` 给出各阶段有用核心数的上界。AIC 已
负责 Gram 和完整面板旋转 GEMM；小 Gram 的 AIV 求解、全核屏障及数据搬运
仍有开销，后续需按目标工作负载调优。
当前 maxabs 归约和最终排序按矩阵分配任务，batch1 时这两部分只使用一个
AIV；pack、参数求解、Gram 和面板更新按块分配。接口输出 B_rot/sigma/Vt，
不累计或输出 U。

FP32 Gram 会损失高条件数输入的小奇异子空间；低于最大行 norm 的 `1e-6`
阈值的 Vt 行置零，不补全零空间基。输入有限但奇异值超出 FP32 输出范围时
CLI 报错。达到最大 sweep 而未收敛时保留输出并在 metrics 报告 `converged=false`。
