# Prepare 矢量化与执行部件

`BlockRotationSmall::RotateRows` 在 AIV 上执行，使用矢量 `Mul/Sub/Add`
更新小 Gram（左右变换合为 `J G Jᵀ`）以及累计旋转矩阵（`J R`）。
block64 的对应路径是 `RotateTile/RotateBothRows/RotateColumns`，也在 AIV 上。
这些是求解旋转参数的局部计算；输入面板的完整旋转 `X_new = R X` 则由
`FusedCube::Apply` 中的 AIC Cube GEMM 执行。`__aicore__` 修饰符本身不表示
一定在 AIC 执行：此处调用链从 `__DAV_C220_VEC__` 的 `FusedVector` 进入。

## Prepare 实现

两条路径都改用 [rotation_prepare.hpp](../rotation_prepare.hpp) 的公共 helper：

1. `CreateVecIndex` 生成一次等差序列，复用于全部索引。
2. `Adds/CompareScalar/Select` 生成 circle 的下一轮行列顺序。
3. `Brcb` 广播行坐标；按倍增长度的矢量复制扩展列坐标，避免逐行标量循环。
4. `Muls/Adds/Add` 计算 transpose、circle 和 pivot 的字节偏移，`Cast` 转为索引。
5. 比较行列坐标是否相同，直接输出 packed 单位阵 predicate；不再逐字节写 mask。

所有偏移均为 0 到 65532 的精确整数，FP32 算术及转换无舍入误差。
block64 的列 Gather 只需要 16×128 tile map，倍增扩展时同时累加各行偏移。
初始化暂用尚未载入数据的 G/R 和现有临时区，UB 总分配保持 180 KiB；
不可变索引和 mask 仍在共享前后处理 scratch 之外。
首次初始化前保留 S→V 顺序，首次 Gram DMA 前保留 V→MTE2 顺序。

这里使用用户所说的 arange/等差序列功能，对应 CANN 基础 API 名为
`CreateVecIndex`。CANN 8.2.RC1 的 C220 实现内部仍用 **8 次 SetValue**
产生 32-byte seed，随后用矢量 Adds 扩展。因此本项目的 Prepare/helper
已经没有逐元素 SetValue，但不能声称底层实现完全没有标量写入。
每个参与参数求解的 AIV 每次完整调用只初始化一次，后续任务和 round 复用。

| 块宽 | Gram 维度 | 修改前 Prepare 的显式 SetValue 次数/核 | 修改后 API 内部 seed 写入 |
|---|---|---:|---:|
| 16 | 32×32 | 3248 | 8 |
| 32 | 64×64 | 12896 | 8 |
| 64 | 128×128 | 4288 | 8 |

## 910B3 验证

server，CANN 8.2.RC1，修改前保存二进制后进行对照：

- `tests/run_prepare.py`：12/12 通过。覆盖三种块宽、随机/单位/零矩阵、
  补齐边界、2 AIV 多任务复用和三次完整调用；对照独立 FP64 SVD，
  B_rot/sigma/Vt 的 SHA256 和实际 sweep 均与原版一致。
- 原有数值回归 31/31，block64 数值回归 9/9，控制流回归 6/6，CPU 测试 5/5。
- 数值与控制流回归继续检查单次 launch、device 前后处理及 device 收敛。

4096×4096，batch1，seed20261116，tol=FP32 epsilon，local-sweeps1，
自动分组和全部物理核。按修改前→修改后→修改后→修改前执行；每组 4 次，
每组舍弃首次，共汇总每个版本的 6 个 device 时间取中位数。
使用同一输入并复用输出目录；全部输出逐位一致，没有改变收敛判据。

| 块宽 | sweep | 修改前 device 中位数/ms | 修改后/ms | 时间减少 |
|---|---:|---:|---:|---:|
| 64 | 25 | 2207.453 | 2207.207 | 0.011% |
| 32 | 26 | 2495.024 | 2492.498 | 0.101% |

这次减少的是每核一次的初始化标量工作。block64 的整次时间没有可辨认的
加速，block32 差异也很小；不能据此声称迭代性能得到显著改善。
没有更改 AIC/AIV 工作划分、流水分组、精度阈值或一次完整分解的 launch 数。

日志及原版二进制统一保存于本地与服务器项目的 `logs/prepare-vector/`。
`benchmark.json` 含全部时间和输出哈希，`build.log`、`before-tests.log`、
`after-tests.log`、`numerical.log`、`block64.log`、`control.log`、`cpu.log`
保存构建和验证记录。数值 artifacts 在服务器 `artifacts/prepare-*` 下。
