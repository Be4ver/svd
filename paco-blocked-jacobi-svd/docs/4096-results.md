# 4096×4096 单次启动分块 Jacobi 实测

本文保留流水实现之前的串行实测。当前 AIV/AIC 流水、40+24 默认分组及
同轮串行对照见 [overlap-results.md](overlap-results.md)。下述时间和利用率
属于历史版本。

2026-11-16，server，Ascend 910B3（逻辑 device0），CANN 8.2.RC1。
单个 FP32、row-major、标准正态稠密矩阵，NumPy RNG seed=20261116。
每次完整计算一次 mixed kernel launch，20 AIC + 40 AIV；tol=1e-6，
local-sweeps=1，最多 30 sweep，由 device 在完整 sweep 后检查收敛。

## 矩阵分块和核分工

完整 4096×4096 矩阵占 64 MiB，常驻 GM，不装入 UB。沿用 PACO 的行旋转
约定 B=Q A，等价于对 A 的转置作列 Jacobi。block=b 表示 b 行一块，
每轮用 round-robin 配成不相交的两块面板 X，面板大小为 (2b)×4096。

| 工作 | block16 | block32 |
|---|---:|---:|
| 行块数 / 每个 sweep 的 round 数 | 256 / 255 | 128 / 127 |
| 每轮独立面板数 | 128 | 64 |
| 每个面板的小 Gram / R | 32×32 | 64×64 |
| AIV 参数任务数 / 参与核心数 | 128 / 40 | 64 / 40 |
| AIC Gram 任务数 / 参与核心数 | 128 / 20 | 64 / 20 |
| 每轮 AIC 面板更新 tile 数 | 8192 | 4096 |

每轮先由 AIV 将不连续的两块行打包到 GM。AIC 按 K=128 累积计算
G=X X^T；不同 AIV 各自读取一个小 G，在 UB 做局部 Jacobi 并累积 R，
再写回 GM。AIC 按 128 列分片计算整个 R X，写回对应原行块。
不同面板和不同列片用 grid-stride 分给全部核心。下一 round、下一 sweep
和收敛检查均在同一次 kernel 内执行。

AIV 的小 Gram 扩展布局和输入/旋转/临时缓冲区合计小于 168 KiB，
分阶段复用其中的 128 KiB 作为行缓存及 O(rows) 的 norm/排序数组。
4096 行列时前后处理所需空间小于 97 KiB。排序用 device min-heap，
复杂度 O(rows log rows)。这不随完整矩阵的元素数分配 UB。

各阶段通过全驻留核 barrier 同步，目前没有 AIC/AIV 阶段重叠。
maxabs 归约、最终排序按矩阵分配，batch1 时由一个 AIV 完成；迭代中的
pack、参数求解、Gram、完整面板更新按块并行。

## 收敛分解时间

每种配置独立执行 4 次，舍弃第 1 次，对余下 3 次取中位数。测量包含
device 归一化/pad、全部 sweep、收敛、norm/排序、B_rot/sigma/Vt 输出。
不含文件 I/O、内存分配、初始 H2D、最终 D2H、NumPy 验证。
接口不累计或输出 U，因此这里是现有 PACO 输出范围的计算时间。

| block | 实际 sweep | 收敛 | ACL event 设备时间 | host launch 到同步 |
|---|---:|---|---:|---:|
| 16 | 17 | 是 | 3652.514 ms | 3652.742 ms |
| 32 | 18 | 是 | **2906.392 ms** | **2906.663 ms** |

block32 在此输入上比 block16 少约 20.4% 时间；本次只比较这两种分块，
不能据此视作最佳调参结果。全核参与也不代表 Cube 全程满载。

## 数值校验

计时完成后，用 FP64 NumPy SVD 独立校验奇异值，并检查 FP64 计算的
B 行正交性、Vt 正交性、能量、行空间重构和 Vt/sigma 特征对应关系。
右特征残差为每行 `||v A^T A - sigma^2 v||_2 / sigma_ref_max^2`，
用独立参考谱范数归一化，避免阈值随矩阵秩增大而变宽松。

| 指标 | block16 | block32 |
|---|---:|---:|
| 奇异值最大绝对误差 | 4.9433e-4 | 3.8286e-4 |
| B 非对角 Gram / 最大对角 | 9.4303e-7 | 9.5407e-7 |
| Vt 正交性最大绝对误差 | 1.0038e-6 | 1.0038e-6 |
| 行空间重构相对误差 | 2.1640e-5 | 2.1163e-5 |
| Frobenius norm 相对误差 | 1.3695e-6 | 1.0916e-6 |
| 归一化右特征残差 | 1.1133e-5 | 1.1091e-5 |

两种配置均通过现有验证阈值。扩展到大矩阵时发现，重复 FP32 Givens
更新导致 R 行长度产生偏差，未经校正的 4096 输入能量误差约 7.73e-4。
现在每个 AIV 在输出小 R 前做一阶 Newton 行长度校正，消除一阶漂移；
512×512 也纳入常规数值回归，保持原有验证阈值。

本轮 CPU reference/validator 5/5、NPU 数值回归 31/31、控制流回归 4/4
通过。新增大矩阵错误 Vt 顺序的用例已验证旧 Frobenius 归一化会漏检、
谱范数归一化能拒绝。上述 4096 输出及整套 NPU 回归均已按新标准重新
校验，严格复核日志为 `logs/artifacts/evidence/large-4096/revalidate.log`。

FP32 Gram 对病态矩阵的小奇异子空间仍有精度限制。本次 4096 实测针对
上述标准正态输入，未覆盖所有谱分布。

## 单次启动和利用率

对 4096×4096、block32、repeat1 的实际执行采集 msprof：runtime API
`KernelLaunchWithHandle` 的 Count=1，算子 `blocked_jacobi_fused` 的
Count=1、Core Type=MIX_AIC，Block Dim=20、Mix Block Dim=40。
该次 mixed task 时间为 2900015.14 us。采集结果与上面的非 profiler
中位数分开保存。

完整算子 `aic_mac_ratio=0.054`，`aiv_vec_ratio=0.357`，
`aiv_scalar_ratio=0.230`。全部核心参与分块任务，但 Cube 算术忙碌比例
约 5.4%，当前仍有明显的 AIV 参数求解、数据搬运和阶段同步开销。
后续提高吞吐需要缩短局部 Jacobi 参数求解及等待时间，并考虑阶段流水。

server 完整采集目录：

```text
artifacts/msprof-4096/PROF_000001_20261116060848753_FHGAKOMEBRPCHRMC/
```

本地原始 CSV：`artifacts/evidence/msprof-4096/`。

## 复现与原始证据

```bash
cd /home/zhushengrui/project/paco-blocked-jacobi-svd
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build-fused -DCMAKE_BUILD_TYPE=Release
cmake --build build-fused -j8
OPENBLAS_NUM_THREADS=8 python3 scripts/benchmark_large.py \
  --size 4096 --block 32 --local-sweeps 1 --repeat 4 \
  --artifacts artifacts/large-4096-block32
```

block16 同理替换 `--block 16` 和 artifacts 目录。脚本要求 device 报告
converged=true，独立数值验证通过后才写出带 pass=true 的 results.json。

server 原始输入、输出、metrics 及验证结果分别保存在
`artifacts/large-4096-block16/` 和 `artifacts/large-4096-block32/`。
本地结果 JSON 位于 `artifacts/evidence/large-4096/`；日志统一迁移到
`logs/`，其中保留原相对路径。例如运行日志位于
`logs/artifacts/evidence/large-4096/`（本地）和
`logs/artifacts/large-4096-block32/`（server）。
