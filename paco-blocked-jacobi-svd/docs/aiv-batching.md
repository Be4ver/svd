# AIV 多行批量更新与 Pack 双缓冲

> 本文的性能、sweep 及验证数据来自旧块调度：块内行对在每个全局 sweep
> 被重复访问。当前调度已改为每对一次，见 [exact-sweep.md](exact-sweep.md)；
> 历史耗时与 sweep 数不能直接作为当前结果。

2026-11-17，server，Ascend 910B3（20 AIC + 40 AIV），CANN 8.2.RC1。
本地与服务器项目目录分别为
`C:/Users/dell/Desktop/svd/paco-blocked-jacobi-svd` 和
`/home/zhushengrui/project/paco-blocked-jacobi-svd`。

## 计算分工与批量边界

每个 block 为 64 行，一对 block 对应面板 X（128×cols_pad）。
完整矩阵和面板在 GM；AIV 只载入小 Gram 和累计旋转 R。
一次完整 SVD 仍只启动一次 mixed kernel，预处理、配对、sweep、
收敛判断及后处理都在 device 内执行。

| 步骤 | 部件 | 本次实现 |
|---|---|---|
| 面板打包 | AIV 的 MTE2/MTE3 | 每个任务搬运 4 个连续行，两个 UB 缓冲交替使用 |
| G = X Xᵀ | AIC Cube | GEMM，保留此前删除 Gram 逐 K 分段全流水屏障的实现 |
| 求局部旋转参数 | AIV Vector | 同时计算 64 个不相交 pivot 的 cos/sin |
| 小 G 左更新及 R 更新 | AIV Vector | 每批 32 对行，每条连续算术指令处理 4096 个 FP32 元素 |
| 小 G 右更新 | AIV Vector | 每批 32 个物理行，repeat stride 复用 64 组参数，每条处理 2048 元素 |
| circle 置换及 R 行长校正 | AIV Vector | 行搬移最多 32 行；行长归约每批 32 行；列置换保留 16 行 Gather |
| 完整面板旋转 X_new = R X | AIC Cube | GEMM，分输出行块和列 tile 写回 GM |
| 收敛、norm、排序及 Vt | AIV | 沿用原 device 控制与后处理 |

参数求解在本次修改前已同时计算 64 个 pivot。本次主要扩大后续的小
G/R 更新与搬移粒度：行更新从 16 对行扩大到 32 对行，连续算术指令从
2048 元素扩大到 4096 元素；列更新从 16 个物理行扩大到 32 行。
这减少了循环、系数展开和指令发射次数。每轮的 pivot 两两不相交；
不同局部 round 之间仍依次执行，保留完整 G 的双边更新和交叉项。

## 紧凑系数与 UB

`block_rotation128.hpp` 不再把每个 cos/sin 展开到整行。
`Brcb` 加一次跨步 `Adds` 将每个标量填入两个相同的 8-float 块，
32 个系数只需 512 个 float。`Mul` 用 `src1BlkStride=0` 在一个
64-lane repeat 内复用系数，再在两个半行之间使用相同系数；64 个 repeat
覆盖 32×128 元素。G/R 共享这批紧凑系数，无需逐行读取标量参数。

| 存储 | float 偏移（末端不含） | 大小 |
|---|---|---:|
| G、R | [0, 32768) | 128 KiB |
| 两个行临时 tile | [32768, 40960) | 32 KiB |
| 紧凑 cos、sin | [40960, 41984) | 4 KiB |
| 参数及复用的 active/sign mask | [41984, 43200) | 4864 B |
| 16 行列置换索引 | [43200, 45248) | 8 KiB |
| 三组 pivot 索引 | [45248, 45440) | 768 B |
| 单位阵 mask | [45440, 45952) | 2 KiB |
| 分配末端 | 46080 | 总计 180 KiB |

C220 的物理 UB 为 192 KiB，但当前 CANN 为高级 API scratch 保留顶部
8 KiB，用户可用边界是 184 KiB。实际使用末端为 45952 个 float，
处于 180 KiB 分配内。列置换索引继续只存 16 行，避免扩大索引后越过
用户可用边界。没有增加完整矩阵缓存或全矩阵转置。

## Pack 的流水与等待

`fused_vector.hpp::Pack` 将逐行 Load/Store 改为四行任务。
所有 block 大小均为 4 的倍数，四行不会跨越配对的原始 block；
cols_pad 为 32 的倍数，DMA 块长度满足对齐要求。任务仍用 grid-stride
分给 AIV，`main.cpp` 同步记录 `pack_rows_per_task=4` 和实际任务数。

复用求解器前 128 KiB 为两个缓冲；最大 cols_pad=4096 时，每个缓冲
4×4096×4=64 KiB。每个缓冲的顺序为：

1. MTE3→MTE2 等待该缓冲上次写 GM 完成，再加载下一组四行。
2. MTE2→MTE3 直接发布输入 DMA 完成，随后写面板 GM。
3. 发布缓冲可复用事件，下一任务切换到另一个缓冲。
4. Pack 结束排空两组事件，再允许求解器复用这段 UB。

MTE2 可加载下一缓冲，同时 MTE3 写出上一缓冲。原搬运路径的
MTE2→Vector、Vector→MTE3 握手已去掉。入口保留一次 Vector→MTE2
依赖，防止覆盖前一阶段还在读取的 G/R；各缓冲保留必要的 DMA 复用依赖。
没有新增全核屏障，原 round 和 sweep 的屏障继续保护阶段间 GM 可见性。

局部求解仍只在 local sweep 末尾读取一次 Vector 归约结果，没有逐行
V→S 等待。此次没有更改 A/B 组调度；4096×4096、block64、batch1
每轮只有 32 个参数任务，自动模式为串行阶段。所有 40 个 AIV 都常驻并
参与屏障，参数求解阶段最多 32 个 AIV 有独立任务，不能据此声称全核满载。

## 4096×4096 分解实测

同一 seed20261116 FP32 输入，batch1，tol=`1.1920928955078125e-7`，
local-sweeps=1，max-sweeps=30，默认全核。block64 分组为 32+0，
block32 为 40+24。基线已包含此前 Prepare 矢量化和 Gram 屏障删减。

| 对照 | 版本 | 设备时间中位数 / ms | 实际 sweep |
|---|---|---:|---:|
| block64，批量更新前后 | 本次基线 | 2025.754 | 25 |
| block64，批量更新前后 | 32 对行更新、原 Pack | 1721.702 | 25 |
| block64，Pack 前后 | 32 对行更新、原 Pack（同组重跑） | 1722.093 | 25 |
| block64，Pack 前后 | **32 对行更新 + 四行 Pack 双缓冲** | **1692.535** | **25** |
| block32，Pack 前后 | 原 Pack | 2334.201 | 26 |
| block32，Pack 前后 | 四行 Pack 双缓冲 | 2270.943 | 26 |

32 对行更新使 block64 时间减少 **15.01%**；Pack 在此基础上再减少
**1.72%**。最终 block64 相对本次基线时间减少 **16.45%**，约 1.197 倍
加速。block32 的局部求解器未改，Pack 改动使其时间减少 **2.71%**。

两次对照分别按 ABBA 顺序串行运行。每次 CLI 完成四次独立完整分解，
舍弃首轮，再将每版本六个 warm device 时间汇总取中位数。设备计时
包含整个 kernel 的预处理至后处理，不含 host 文件读写和 H2D/D2H。
未用固定 sweep、放宽容差或未收敛时间报告加速。没有为本次实现新增
profiler 测量，因此不将时间改善解释为已测得的 Vector 满载比例。

原始脚本、两组 JSON、每次调用日志和候选二进制保存在两端的
`logs/aiv-batch/`，包括 `benchmark.py`、`row32-vs-before.json`、
`pack4-vs-row32.json`。复现对照需先 source CANN 环境，再执行：

```bash
python3 logs/aiv-batch/benchmark.py --label row32-vs-before \
  --versions before row32 --blocks 64 --repeat 4
python3 logs/aiv-batch/benchmark.py --label pack4-vs-row32 \
  --versions row32 pack4 --blocks 64 32 --repeat 4
```

输入复用 `artifacts/large-4096-block64-tile16/input/A_batch.bin`，
SHA256 为 `8485da69127c68fe91b3a84082c00b678f9a1842f83d4f6aff366c66df49867d`。
对照每个 block 内所有版本的 B_rot/sigma/Vt 哈希相同；block64 分别为：

```text
B_rot  6ed28b76e4b6235bb1a534b9108c438b01e9ca4632291ca3126fd382be7a3fa2
sigma  79d173d1c6c91490c23cfc9d5b4de6c43d44f5741bc8d7cabe627e4074500781
Vt     3e4e9d0927abd97964b737e2fb6a7f45d9b064d8e92748e232b0ce16efdc0c3a
```

这些哈希也与原 `artifacts/large-4096-block64-tile16/output` 一致，
可绑定该输入已有的独立 FP64 验证：B 行正交误差 1.0964e-7、
Vt 正交误差 5.4922e-7、重构相对误差 2.2349e-6、右特征对残差
5.8761e-6。本次通过哈希复用已有 4096 FP64 验证，没有重跑其完整 CPU SVD。
容差与逐对 Gram 停止条件保持原实现；这不构成与 GPU XGESVDJ 完整
停止条件和精度等价的证明。

## 最终验证

最终二进制与计时使用的 `pack4` 二进制逐位相同；所有 NPU 测试依次运行，
没有与计时或其他 NPU 作业并发。

| 验证 | 结果 | 覆盖 |
|---|---:|---|
| `tests/run_prepare.py` 对照修改前 manifest | 12/12 | block16/32/64；输出及 sweep 与基线逐位一致 |
| `tests/run_gram.py` | 12/12 | K=31/129/257/4096，全核/2 AIV、多任务、重复调用及 FP64 对照 |
| `tests/run_npu.py` | 31/31 | 矩形、补齐、零矩阵、重谱、秩亏、混合与极端尺度 |
| `tests/run_block64.py` | 9/9 | block64、奇数尺寸、80 个任务、条件数 1e4 |
| `tests/run_control.py` | 6/6 | 2/4 AIV、local-sweeps=2、固定 sweep、重复调用 |
| `tests/run_overlap.py` | 11/11 | 串行/流水逐位一致、奇数分组、未收敛及无效参数 |
| CPU `unittest discover -s tests -p 'test_*.py'` | 5/5 | 参考实现及输出校验器 |

设备测试同时检查单次 launch、device 前后处理和收敛控制。
静态复核与独立 CPU 索引模拟未发现批量系数、Pack 映射、事件平衡或 UB
生命周期问题。记录见 `logs/aiv-batch/pack4-prepare.log`、
`pack4-gram.log` 和 `final-*.log`；日志仍全部位于 `logs/`。
