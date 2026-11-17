# PACO blocked one-sided Jacobi SVD — single launch

针对 Ascend 910B3 / CANN 8.2 的独立实现，沿用 PACO 批量 SVD CLI 和二进制
接口。**每次完整 SVD 只启动一次 mixed AIC:AIV=1:2 kernel**；所有数值计算、
round 配对、sweep 循环和收敛控制都在 device 内完成。

本地目录：`C:/Users/dell/Desktop/svd/paco-blocked-jacobi-svd`；server 镜像：
`/home/zhushengrui/project/paco-blocked-jacobi-svd`。

日志统一放在项目的 `logs/` 中并保留原相对路径。测试与 benchmark 脚本
也写入此目录；项目外部的 artifacts 路径通过哈希区分。

当前 block64 实现与相同容差的 4096×4096 对比见
[block64-results.md](docs/block64-results.md)：每块 64 行，配对后的 Gram 为
128×128；AIV UB 总分配为 180 KiB，完整面板旋转仍由 AIC Cube 执行。
相同 FP32 epsilon 下，block64 设备时间中位数 2209.815 ms，block32 为
2496.125 ms，时间减少约 11.47%。CLI 默认 block16，使用本方案须指定
`--block 64`。剩余 Gather、精度限制和实际核心使用上界均在报告中列明。

## 执行路径

唯一入口是 `blocked_jacobi_fused.cpp` 的 `blocked_jacobi_fused`。在 910B3 上
默认一次启动 20 AIC + 40 AIV；每个核通过 grid-stride 分配工作，并参与
device barrier。整个调用的执行顺序为：

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
超过物理核数。默认在每轮内将独立块对分为 A/B 两组，执行：

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
无需增加矩阵缓冲区或 AIV UB。自动模式在任务数不超过 AIV 数量时使用
串行路径，避免将原本一轮可完成的参数求解拆成两轮。

AIV 的局部求解采用紧凑 Gram 和 circle 配对：block32 每轮批量处理
32 个不相交 pivot，63 轮覆盖全部 2016 对。角度、两侧 Gram 更新、R 更新
和行长度校正均用向量指令；连续行更新每条指令覆盖 2048 个有效 FP32 元素。
静态 Gather 映射每核只初始化一次，局部 sweep 末只进行一次 V→S 归约读取，
没有逐 pivot 的标量读取或 S/V 握手。UB 为 180 KiB；矩阵仍在 GM。

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
限制核数时必须为正偶数，对应 AIC 数为其一半。`--fixed-sweeps` 固定执行
sweep 数，默认在完整 sweep 后检查收敛；`--tol` 默认 FP32 epsilon
`1.1920928955078125e-7`。当前采用逐对 Gram 相关性判据；这不构成与
GPU XGESVDJ 完整停止条件或输出精度等价的证明。数值验证使用独立 FP64
NumPy SVD，详见 [block64-results.md](docs/block64-results.md)。
`--no-overlap` 使用串行阶段；`--pipeline-jobs N` 指定每轮 A 组的块对
任务数，B 组处理剩余任务。N=0（默认）按 AIV 任务轮次平衡两组，A 组
包含整数轮完整的 AIV 任务；任务数不超过 AIV 数时串行。正 N 必须小于每轮总
任务数，不能与 `--no-overlap` 同用。4096×4096/block32/batch1 默认
40+24，可用 `--pipeline-jobs 32` 对比 32+32。`metrics.json` 记录实际
`pipeline_groups` 和 `phase_overlap`；有用核心数上界按最大分组计算。

## 验证

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

此前 tol1e-6 的 AIV 向量化实现与 4096×4096 实测见
[aiv-vectorization-results.md](docs/aiv-vectorization-results.md)：40+24 默认
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
或前后处理的一行缓存及 O(rows) 的 norm/排序数组；二者分阶段复用空间。
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
