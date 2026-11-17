# 单次 mixed kernel 验证记录

本页是扩展 4096 支持和旋转行长度校正之前的小矩阵记录；当前目标尺寸的
实现及实测见 [4096-results.md](4096-results.md)。

2026-11-16，在 `server` 的 Ascend 910B3 device 0、CANN 8.2.RC1 上验证。
构建目录 `build-fused`，默认 20 AIC / 40 AIV，block16、local-sweeps1。

## 实际 launch 数

采集输入为 128×128、batch10，最多 30 sweep、repeat1；实际 device 执行
10 sweep。`msprof --runtime-api=on --task-time=on` 的结果：

| 证据 | 观测值 |
|---|---|
| runtime `KernelLaunchWithHandle` | Count = 1 |
| op `blocked_jacobi_fused` | Count = 1，Core Type = MIX_AIC |
| Block Dim / Mix Block Dim | 20 / 40 |
| mixed task 时间 | 10900.36 us |

`task_time` 中同一 stream46/task1 的两个 MIX_AIC 时间记录属于同一个 mixed
task；launch 数同时由 runtime API 和 op statistic 交叉确认。除该 task 外
还有 profiler 和 event 控制记录，没有独立 pack/Gram/参数/旋转/norm kernel。

本地原始 CSV 证据在 `artifacts/evidence/msprof-single-launch/`，文件时间戳
`20261116052311`；server 完整采集在：

```text
artifacts/msprof-single-launch/PROF_000001_20261116052305079_HFJFFPAQGMOMLORB/
```

Profiler 中 `aic_mac_ratio=0.003`，`aiv_vec_ratio=0.455`，
`aiv_scalar_ratio=0.383`，说明当前完整算子的 AIC 算术忙碌比例仍低，AIV
参数求解及阶段等待占用较多时间。单次 launch 解决了 host 调度问题，
并不代表已经充分利用 Cube 吞吐。

## 正确性与控制流

- 本地 CPU reference/validator：4/4 通过。
- 最终 NPU 数值回归：30/30 通过，包括一个预期 FP32 输出溢出报错用例。
- 最终 NPU 控制流：4/4 通过，覆盖重复调用、固定 sweep、2/4 AIV 网格。
- 五组性能样例同时通过数值与单次执行契约验证。

数值回归检查独立 NumPy 奇异值、B 行正交性/能量、Vt 正交性、行空间重构
及 Vt/sigma 的特征对应关系。所有可表示结果的归一化右特征残差最大
`3.32e-6`。validator 另有交换 Vt 行必须失败的回归用例。

NPU 完整输入/输出分别在 server 的
`artifacts/npu-single-launch-final/` 和
`artifacts/control-single-launch-final/`；日志在 `logs/` 下的对应相对路径；结果 JSON 已拷回本地
`artifacts/evidence/npu-results.json` 和 `control-results.json`。

## 当前单次启动版本的延迟

命令：`OPENBLAS_NUM_THREADS=1 python3 scripts/benchmark.py --blocks 16
--locals 1 --artifacts artifacts/benchmark-single-launch`。
每种形状连续调用完整算子 4 次，每次一次 launch；舍弃首次后取 3 次中位数。
stream 为 ACL event 时间；wall 包含 host launch 到同步，不含文件 I/O、
初始 H2D、最终 D2H 和分配。均使用 block16/local1，未进行完整参数搜索。

| shape / batch | sweeps | stream (ms) | launch-to-sync (ms) |
|---|---:|---:|---:|
| 20×20 / 512 | 9 | 8.377 | 8.500 |
| 32×32 / 512 | 10 | 17.174 | 17.381 |
| 64×64 / 256 | 9 | 46.351 | 46.600 |
| 128×128 / 4 | 9 | 10.125 | 10.287 |
| 128×128 / 40 | 10 | 38.942 | 39.160 |

原始性能结果：server `artifacts/benchmark-single-launch/results.json`，
本地 `artifacts/evidence/benchmark-results.json`。旧多 launch 版的调参和
profiler 数据保留在其他 artifacts 目录中，不计入这里的结果。
