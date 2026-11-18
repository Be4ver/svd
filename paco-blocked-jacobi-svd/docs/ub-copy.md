# UB 复制改用 Copy（2026-11-18）

## 实现范围

将 `block_rotation.hpp`、`block_rotation128.hpp`、`rotation_prepare.hpp` 中仅用于
UB 搬运的 `Adds(dst, src, 0.0f, ...)` 改为 `AscendC::Copy`。实际的索引偏移、
阈值常数和 norm 校正仍使用 Adds。Angles、GM 通信、任务映射、块版本、
request/ack 和 sweep 配对均保持原实现。

新增 `rotation_copy.hpp` 封装本项目的连续 FP32 复制。当前调用长度为 32，
或 64 的倍数，最大 8192 元素；mask 为 `min(count,64)`，repeat 为
`ceil(count/64)`，参数 `{1,1,8,8}`。此封装不支持任意尾长，不能直接扩展到
例如 96 元素的调用，否则最后一轮会多复制。最大 128 repeats，小于 255。

| 位置 | Copy 用途与实际步长 |
|---|---|
| `rotation_prepare::Columns` | 首行复制以及成倍扩展；每次复制区间相邻但不重叠。 |
| `rotation_prepare::Grids` | 保留原 mask/repeat，将 64 列广播结果压紧到 32 列，或填入 128 列行的两个半行。 |
| 小 solver `BroadcastRows` | 保留每行 d=32/64 的 mask 和原行步长，压紧广播结果。 |
| 128 阶 `BroadcastRows` | mask=8、32 repeats、repeatStride=2 个32B块；读偶数块，写奇数块，实际元素不重叠。 |
| `RotateTile` / `RotateColumns` | 复制临时运算结果回 G/R；列更新保留 dstRepeatSize=16、srcRepeatSize=8。 |
| `PermuteRows` | 保留 `matrix → tmp → matrix` 的分块行移位，以及 cos/sin 中保存的边界行；Copy 不承担重叠 memmove。 |
| `PermuteColumns` | Gather 的 16 行结果复制回 G。 |

Copy 沿用原 Vector 数据依赖，未增加缓冲；block64 AIV UB 仍为180 KiB。
原有 Pack 的 MTE2/MTE3 双缓冲保持不变。

## CANN / 910B3 接口依据

服务器 CANN 8.2.RC1 的 `aarch64-linux/ascendc/include/basic_api/` 中：

- `interface/kernel_struct_data_copy.h:234` 定义 `CopyRepeatParams`，顺序为
  dstStride、srcStride、dstRepeatSize、srcRepeatSize。
- `interface/kernel_operator_data_copy_intf.h:142/148` 的 Copy 接口要求 mask、
  uint8 repeat 和步长，没有只传总元素数的 overload；标注 `__inout_pipe__(V)`。
- `impl/dav_c220/kernel_operator_data_copy_impl.h:330–373` 的 FP32 Copy 将输入
  作为 uint32 位模式传给 `vcopy`，并设置 Vector mask。

因此 Copy 明确表达位复制，仍使用 Vector 管线，不会把这些 UB 搬运转移到
MTE2/MTE3，也不据此声称增加了 DMA 重叠。SDK 摘录与哈希位于
`logs/ub-copy/sdk-copy-implementation.txt` 和 `sdk-sha256.txt`。
通用 llvm-objdump 对本设备指令显示 `<not available>`，其记录仅为失败诊断，
不作为已反汇编验证 vcopy 的证据。

## 验证与计时

CANN 构建成功，12 个静态测试和105项设备/CLI回归通过：单 sweep 15、通用
数值31、block64 9、控制7、Prepare 12、Gram 12、不活跃任务8、调度11。
Prepare 的三种块宽、重复调用和 scratch 复用与保存的原版结果逐位一致。
真实 NPU 单 sweep 与独立 FP64 配对参考比较，覆盖 block16/32/64、
2/4/6/8/10 块、多 batch、补齐和部分核；其余对照中的单位轮、阈值边界、
默认核组/全局路径也通过逐位比较。

4096×4096、batch1、block64、FP32 epsilon，固定输入种子20261116，使用相同
输入和停止条件，串行 ABBA（旧、Copy、Copy、旧），每组4次完整分解，
排除各组首次，分别对6个 warm 样本取中位数：

| 版本 | 6个 warm 设备时间（ms） | 中位数（ms） |
|---|---|---|
| 本轮保存的 Adds 版 | 621.172363、621.862854、621.980286、621.259399、621.446533、622.155762 | 621.654694 |
| Copy 版 | 615.790527、615.403381、615.860962、614.628540、614.614929、614.168640 | 615.015961 |

本次耗时减少 **1.0679%**（6.638733 ms），相对966.515 ms历史基线累计减少
**36.37%**。这是一轮同机 ABBA 实测的小幅收益，不把 Copy 描述为大幅优化。
16次完整调用均为25 sweep、converged=true，每次一次 kernel launch；三份输出
`B_rot.bin`、`sigma.bin`、`Vt.bin` 在全部组之间逐位一致。
独立 FP64 校验：sigma 最大绝对误差2.17581e-4，B 正交误差1.11562e-7，
Vt 正交误差4.95332e-7，重构相对误差2.41502e-6，能量相对误差5.16556e-7，
右特征残差4.90179e-6（按参考谱范数平方归一化）。

起始 executable SHA256 为 `086f85ec…34bd4f`，Copy 版为
`2cf431b74a26217ddc87f9b1fa7ced87c13e66d6ec353a9943fb73af8fbedaf1`。
固定输入 SHA256 为
`8485da69127c68fe91b3a84082c00b678f9a1842f83d4f6aff366c66df49867d`。
完整 ABBA 数据在 `logs/scalar-pipeline/ub-copy/results.json`，
汇总在 `logs/ub-copy/verified-results.json`。

## Profiler 核查

修改前后各串行采集一次完整分解。均为一条 MIX_AIC op，20 AIC/40 AIV，
runtime API 中 KernelLaunch 总计1次；Cube MAC、Fixpipe 和 AIV Vector 活动非零。
profiler 的输出和 executable 哈希均绑定到 ABBA 结果。

| 管线 | Adds 版绝对时间（ms） | Copy 版绝对时间（ms） | 减少 |
|---|---:|---:|---:|
| AIV Vector | 278.118948 | 273.429303 | 1.69% |
| AIV Scalar | 186.599759 | 185.685921 | 0.49% |
| AIC Scalar | 541.844184 | 535.893345 | 1.10% |
| AIC MAC | 149.248082 | 149.247941 | 约0% |
| AIV MTE2 | 20.465025 | 20.455740 | 约0% |
| AIV MTE3 | 20.922862 | 20.896982 | 约0% |

上表是各一次 profiler 样本，不能当作多次统计；Scalar 含轮询，不能将其
全部解释为有效标量运算或纯等待。不同管线时间重叠，不可相加为总时长。
本轮未改变同步协议和通信量，没有新增 MTE2/MTE3 流水。
详见 `logs/ub-copy/profile-{baseline,final}-summary.json` 及其引用的 CSV。

## 文件与复现

源码差异保存在 `logs/ub-copy/implementation.diff`；修改前的三份头文件和
executable 保存在 `artifacts/ub-copy-start/`，起始 executable SHA256 为
`086f85ec657a1b504fe54a9eef1ece16d24acbfd3719bdbc338e0089df34bd4f`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake --build build-core-local -j8
bash artifacts/ub-copy-start/verify.sh
bash artifacts/ub-copy-start/profile.sh
python3 artifacts/ub-copy-start/collect_evidence.py
```

`verify.sh` 串行运行静态、Prepare 基线对照、单 sweep、完整数值、控制、Gram、
不活跃任务和调度回归，再运行同输入 ABBA。`profile.sh` 随后串行采集修改前后
profiler，`summarize_profile.py` 绑定 executable 与输出哈希并检查一次 mixed launch。
`collect_evidence.py` 核对结果、未修改的其他 kernel 源文件并打包紧凑证据；
`verify_sync.py` 在本地核对源码和下载证据 SHA256。大矩阵二进制留服务器。
