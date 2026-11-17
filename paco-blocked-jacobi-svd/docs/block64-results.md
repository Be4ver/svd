# Block64 AIV 优化与 4096×4096 实测

本地项目：C:/Users/dell/Desktop/svd/paco-blocked-jacobi-svd

服务器镜像：/home/zhushengrui/project/paco-blocked-jacobi-svd

## 计算分工

每块为 64 行；每次选择两个块，得到 128×cols_pad 的面板 X。
完整矩阵和面板保存在 GM。一次完整分解只有一次 mixed kernel launch，
缩放、补零、块配对、全部 sweep、收敛判断和输出整理均在 device 内完成。

| 步骤 | 执行部件 | 计算与存储 |
|---|---|---|
| 预处理及面板打包 | AIV | maxabs 缩放、补零；分行搬运到 GM 面板 |
| Gram | AIC | Cube GEMM 计算 G = X Xᵀ，B 操作数使用转置 |
| 参数与 R | AIV | 在 UB 中迭代 128×128 Gram，累积 128×128 旋转矩阵 R |
| 完整面板旋转 | AIC | Cube GEMM 计算 X_new = R X，分输出块和列 tile 写回 GM |
| 收敛及后处理 | AIV | sweep 标志归约、行 norm、sigma 排序、Vt 归一化 |

AIV 对小 Gram 的左右更新用于求旋转参数；对原矩阵完整面板的旋转始终由
AIC 执行。接口沿用行旋转约定 B = Q A，输出 B_rot、sigma、Vt，不累计 U。

## UB 与向量实现

64 行块对应的求解器为 [block_rotation128.hpp](../block_rotation128.hpp)。
不能只按 Gram 大小估算 UB：还要存储累积旋转 R、临时 tile 和索引。

| 存储 | 大小 |
|---|---:|
| G：128×128 FP32 | 64 KiB |
| R：128×128 FP32 | 64 KiB |
| 临时行 tile、cos/sin 广播、参数、索引和 mask | 52 KiB |
| 总分配 | 180 KiB |

前后处理复用该 UB，静态索引置于共享 scratch 区之外；没有把 4096×4096
输入放入 UB，也没有额外分配完整转置矩阵。

- 每轮同时求 64 个不相交 pivot 的角度；127 个局部 round 覆盖全部行对。
- 行更新采用 16 行 tile，每条连续向量算术指令处理 2048 个 FP32 元素。
- G 与 R 的对应行更新复用已经广播的 cos/sin。
- Gram 右乘直接使用向量 repeat stride，保留完整 J G Jᵀ 双边变换，
  包括不同 pivot 之间的交叉项。
- R 的行长度校正显式归约每行的两个 64 元素半段。
- 局部 sweep 末只执行一次 V→S 活跃标志读取；不同任务间保留必要的
  V→MTE2 顺序，避免新 Gram 载入覆盖尚在使用的 UB。

## Gather 与 GEMM 转置

已经去掉完整矩阵转置 Gather 和 R 行置换 Gather。Gram 右更新改用
repeat stride；circle 行置换改成有界 tile 的连续搬移。

目前仍保留三次 64 元素的 pivot 提取 Gather，以及 16 行 tile 的 Gram
列置换 Gather。列置换是 circle 配对的置换 P，数学作用为 P G Pᵀ；
它与矩阵转置不同。

[fused_cube.hpp](../fused_cube.hpp) 的 Gram 已使用 GEMM 转置操作数。
QDWH gemm.cpp 的转置标志作用于 AIC 的矩阵操作数，不能直接替代
AIV 中的小 Gram 右更新或 circle 置换。继续消除这些 Gather 需要改变
局部 Gram 布局或配对方案，并重新比较搬移和索引成本。

## 收敛与精度边界

默认 tol 改为 FP32 epsilon：1.1920928955078125e-7。
当前实际判据仍为 pairwise_relative_gram：pivot 的绝对相关项超过
tol × sqrt(abs(g_pp × g_qq)) + 1e-30 时执行旋转；一个完整外层 sweep
没有活跃旋转时停止。benchmark JSON 明确记录 tolerance 和 stopping_criterion。

**这个默认 epsilon 不等于已经证明与 GPU XGESVDJ 的完整收敛判据或输出
精度相同。没有运行 GPU 对照。** 当前验证依据是独立 FP64 NumPy SVD；
后续若要求与指定版本 XGESVDJ 严格对齐，需要确认该 API 的残差定义，
并在相同输入上比较残差、正交性及小奇异值误差。

FP32 Gram 对高条件数输入的小奇异子空间仍有固有限制。后处理沿用
低于最大行 norm 的 1e-6 阈值置零 Vt 行的行为，不补全零空间基。
tol 与该后处理阈值是不同参数，不能互相替代。

## 4096×4096 性能

910B3，batch1，seed20261116，local-sweeps1，最多 30 sweep；
两种块大小均使用上述 FP32 epsilon。每种 repeat4，去掉首次，
取其余三次的中位数；下面的性能不包含 NumPy 参考验证时间。

| 版本 | 收敛 sweep | 设备流时间中位数 ms | 调用壁钟中位数 ms |
|---|---:|---:|---:|
| block32 | 26 | 2496.12549 | 2496.37979 |
| block64，16 行 tile | 25 | 2209.81519 | 2210.05019 |

block64 相比相同容差的 block32 设备时间减少约 11.47%。
此前 block64 的 8 行 tile 为 2861.96582 ms；扩大为 16 行 tile 后减少约
22.79%。旧文档中 tol1e-6 的时间不能直接与本表比较。

block64 的独立 FP64 验证指标：

| 指标 | 结果 |
|---|---:|
| sigma 最大绝对误差 | 2.408091824e-4 |
| B 行正交性指标 | 1.096390086e-7 |
| Vt 正交性指标 | 5.492233603e-7 |
| 行空间重构指标 | 2.234854391e-6 |
| 能量相对误差 | 6.457894537e-7 |
| 右特征向量归一化残差 | 5.876066519e-6 |

右特征向量残差按参考谱范数的平方归一化。各指标的精确定义和断言
见 [tests/run_npu.py](../tests/run_npu.py) 的 validate。

原始结果已下载到本地：

- [block64 results.json](../artifacts/large-4096-block64-tile16/results.json)
- [block64 timing.json](../artifacts/large-4096-block64-tile16/timing.json)
- [相同容差 block32 results.json](../artifacts/large-4096-block32-eps/results.json)

## 核心使用与流水限制

4096 行按 64 行分为 64 块，每轮 32 对，每个 sweep 为 63 轮。
batch1 的参数任务只有 32 个，当前自动分组为 32+0：

| 阶段 | 有用核心数上界 |
|---|---:|
| pack | 40 AIV |
| 参数求解 | 32 AIV |
| Gram | 20 AIC |
| 完整面板旋转 | 20 AIC |

因此这个工作负载没有双组参数/旋转重叠，也不能声称所有 AIV 全程满载。
任务超过 AIV 数量时，保留 Gram(B)/参数(A)、旋转(A)/参数(B) 的双组流水。
batch80 的 block64 数值用例已覆盖 40+40 分组。

单次调用的真实 profiler 数据位于 logs/block64/msprof 下：

- op_statistic：blocked_jacobi_fused，MIX_AIC，Count=1。
- runtime API：KernelLaunchWithHandle，Count=1。
- op_summary：Block Dim=20，Mix Block Dim=40。
- 该次 profile 的 kernel 时间为 2202.27312 ms；AIV Vector 比例 0.583，
  Scalar 比例 0.060；AIC MAC 比例 0.086。

这些是整个 kernel 的聚合指标，不能当成参数阶段的逐核利用率。
op_summary 中 cube_utilization=100% 也不代表 MAC 管线全程繁忙。

## 验证记录与复现

最终 block64 tile16 数值套件完成 9 个用例，覆盖随机矩阵、矩形补齐、
batch80 流水、零矩阵、单位阵、重谱、秩亏和条件数 1e4。
block16/32 回归 31 项通过；该回归运行在最后一次仅涉及 block64 tile
大小的修改之前。最终控制流套件 6 项通过，包括 block64 的两 AIV、
多任务、local-sweeps2、repeat3 和固定 sweep 零矩阵。
CPU 单元测试 5 项通过。

    cd /home/zhushengrui/project/paco-blocked-jacobi-svd
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cmake --build build-fused -j8
    OPENBLAS_NUM_THREADS=1 python3 tests/run_block64.py --artifacts artifacts/block64-tile16-final
    OPENBLAS_NUM_THREADS=1 python3 tests/run_control.py --artifacts artifacts/block64-control
    OPENBLAS_NUM_THREADS=8 python3 scripts/benchmark_large.py --size 4096 --block 64 --local-sweeps 1 --repeat 4 --tol 1.1920928955078125e-7 --artifacts artifacts/large-4096-block64-tile16

所有 NPU 测试、benchmark 和 profile 串行运行。测试日志及 profiler 输出均在
logs/；本次服务器扫描没有发现 logs/ 之外的 .log 文件。
CLI 兼容保留默认 block16，使用本方案须显式指定 --block 64。
