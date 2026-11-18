# AIC Scalar 与同步等待：单位轮更新省略

本轮从 `c79195aa84fd49e90a3d58889156d8c4e0faba7a71ba43aa87d5e4b3610ecf78`
开始，保留此前未提交工作。目标始终为 4096×4096、FP32、batch1、block64、
FP32 epsilon、`--local-sweeps 1`，一次 mixed launch 完成前处理、全部 sweep、
收敛和后处理。未放宽容差，未改变 CLI 或全局 sweep 的配对次序。

## 先定位等待来源

`artifacts/aic-wait-start/prepare_diagnostic.py` 在独立源码副本中插入 SYS_CNT
计时，正式 kernel 没有这些计时指令。固定 4096 输入 repeat2 为 713.345、
712.441 ms，均在 25 sweep 收敛。第二次调用的 32 个有效 AIV 累计区间为：

| 区间 | 占所计五个区间累计时间 |
|---|---:|
| 等两个输入块版本 | 16.107% |
| PackJob | 4.171% |
| Gram 请求发布及等待 ack | 10.909% |
| ParametersJob | 61.893% |
| Apply 请求发布及等待 ack | 6.919% |

这些是所有任务累计区间的比例，不能直接当作关键路径占比。区间包含各自的
event 等待，SYS_CNT 未换算成秒。AIC 的 GramJob+ApplyJob 计时约占 CubeSweep
的 29%（12 个双槽组）或 14%（8 个单槽组）；剩余时间包含轮询、发布和其他协议开销。
这支持优先缩短 AIV 求解，而不是把高 Scalar 比例直接解释成有效标量计算。
原始数据和解析为 `logs/aic-wait/diag-run.log`、`diagnostic-summary.json`。

4096、block64 每轮有 32 个 job，组 0..11 两个 AIV 有任务，组 12..19 只有
第一个 AIV 有任务。不能描述为参数阶段 40 个 AIV 全部满载。

## 实际采用的更新

`rotation_angles.hpp` 将原有计算按真实依赖分为 `RotationThreshold` 和
`RotationCoefficients`；block16/32 仍由 `RotationAngles` 依次调用。阈值保持：

`abs(gpq) > tol * sqrt(abs(gpp * gqq)) + 1e-30f`。

block64 的跨块局部轮先对当前 64 个 pivot 做原阈值判断并更新 active 累计，
用 WholeReduceMax 和 V_S event 判断是否全部不活跃。全不活跃时，原算法的
64 个旋转均为 c=1、s=0，因此省略后续 hypot、div/sqrt、G/R 行旋转及 G 列旋转。
**G 的行列置换和 R 的行置换始终执行**，否则下一轮会漏掉或重复 pivot。
有活跃 pivot 时仍执行原有系数计算，保留 sqrt/div 和 FP32 运算顺序。
每个 sweep 的首个外层轮次（127 个局部轮）不增加这个逐轮 V_S 判断。

另一个省略发生在整块 `CrossActive` 已确认全部 4096 个跨块 pivot 都低于阈值后：
默认核组路径直接返回 false，不再初始化 R、做行范数修正或写回 R/flag。
这利用已有负偶数 Apply token：AIC 不读取该任务的 R/flag，B 保持有效。
显式全局路径仍需要 R，`requireIdentity=true` 的默认参数保留单位 R 生成。
每个这样省略的任务少写 65,536B R 和 32B flag；没有采集 DDR 字节计数，
不报告整体带宽下降比例。原有 Pack 和 Gram 仍需执行，才能得到当前阈值判断。

上述两项作为一个候选共同测量，没有分别量化收益。UB 仍为 180KiB，阈值判断
复用已有 params/activeMask/归约位置，没有增加矩阵缓冲。V_S 是本 AIV 内的
Vector→Scalar 完成事件；它本身有代价，最终以完整分解计时判断取舍。

## 同步协议和流水边界

本轮没有改变 `core_local_pipeline.hpp`、`core_local_sync.hpp` 或 AIC 的 GEMM
顺序。AIV 等块版本，S_MTE2 acquire 后 Pack；MTE3_S release 后发布奇数 Gram
请求。AIC 计算 Gram，FIX_S 后 ack。AIV 求参数；有旋转时等待 R 写完后发布正
偶数 Apply 请求，无旋转时发布负偶数请求。AIC 完成整个 R×X 写回后，才发布
两个块的下一版本及 ack；单位请求也执行版本和 ack 发布。AIV 收到 ack 才能
覆盖本槽 X/G/R。不会用过期 R，也没有删除跨核组的数据交接。

信号各占独立 64B cache line，发布经过 DDR DSB、DCCI 和 DDR DSB，读取经过
DCCI、DDR DSB 和 volatile GM load。DMA 可见性仍由 MTE3_S、FIX_S、S_MTE2
排序保护。token 跨 sweep 单调递增，块 epoch 不取模；每次 launch 重新初始化。
已有 watchdog 和公平双槽轮询保留；空槽不参与服务，部分核沿用实际启动的组数。

每组 AIC 仍按 ready 槽执行 Gram 或 Apply；另一 AIV 可准备或求解自己的任务。
单 AIV 不同时拥有两个在途面板，AIC 的两次 GEMM 服务也不并行。已有四行 Pack
双槽重叠 MTE2/MTE3，前后处理双行槽重叠 MTE2/Vector/MTE3；本轮没有新增 DMA
流水或扩大 lookahead。收益来自少做单位更新后缩短任务响应时间，不能全部归因于
流水重叠。每 sweep 保留两次全局屏障用于 active 汇总和一致停止；初始化和最终
输出也保留同步。协议细节见 [core-local-pipeline.md](core-local-pipeline.md)。

## C220 API 审计与网络资料

已检查服务器实际 CANN 头文件，证据含文件 SHA256 和行号，保存在
`logs/aic-wait/sdk-sync-evidence.json`。`NotifyEventImpl` 调用
`ffts_cross_core_sync`；`WaitEventImpl` 忽略 modeId，调用阻塞
`wait_flag_dev(flagId)`。这些实现不足以证明独立任一 AIV 通知的 wait-any、
可中断等待或超时退出语义。固定阻塞某个槽可能挡住另一槽的 Apply，继而阻塞
跨组后继。因此继续公平轮询；硬件通知也不能替代 GM 块版本和 DMA 发布/获取。
同机 `impl/dav_c220/kfc/kfc_comm.h` 的 mailbox 继续作为 64B/DCCI 用法的参考。

网络查阅范围和取舍：

- [LAPACK DGESVJ 源码](https://www.netlib.org/lapack/double/dgesvj.f)：
  实际下载并检查阈值判断、未旋转 pivot 计数和 ROWSKIP/BLSKIP/SWBAND。
  借鉴避免无效旋转的方向；没有引入可能提前跳过 pivot 机会的扫描启发式。
  源码 SHA256 和摘录为 `logs/aic-wait/research-lapack.json`。
- [New Dynamic Orderings for the Parallel One–Sided Block-Jacobi SVD Algorithm](https://doi.org/10.1142/S0129626415500036)：
  读取 Crossref 元数据及摘要，未获取全文。它研究按当前块列正交性动态排序，
  各变种的通信量和局部/全局停止策略不同。尚未验证其算法满足当前 exact-once
  契约，因此本轮没有改变调度次序。证据为 `logs/aic-wait/research.json`。
- [New Fast and Accurate Jacobi SVD Algorithm I](https://doi.org/10.1137/050639193)、
  [II](https://doi.org/10.1137/05063920X) 及
  [DGEJSV 源码](https://www.netlib.org/lapack/double/dgejsv.f)：
  查阅摘要和驱动源码，QR 预处理方向会改变当前单 sweep 轨迹，未直接引入。

本实现仍使用逐对 Gram 相关性停止判据，没有证据证明与 GPU XGESVDJ 完整
停止条件等价。也没有通过降低轮询频率来单独美化 Scalar 指标。

## 验证与复现

`artifacts/aic-wait-start/verify_final.sh` 串行运行静态、独立单 sweep、完整数值、
阈值/稀疏、控制和调度回归，然后使用明确指定的本轮起始 executable 做 ABBA。
`profile_final.sh` 随后串行采集起始/最终版本。`collect_evidence.py` 检查实际套件
结果和 executable/输出绑定，`verify_sync.py` 检查本地与服务器源码和证据 SHA256。
所有新 .log 位于 `logs/aic-wait/`、`logs/artifacts/aic-wait/` 或本轮专用的
`logs/scalar-pipeline/aic-wait-*`。矩阵二进制保留服务器 `artifacts/aic-wait/`。

稀疏回归新增三例：不同强度的四个独立跨块对，其中三个活跃、一个低于阈值，
使活跃轮和单位轮交替；覆盖 256/255/383 行、补齐、batch1/5/3、40/2/6 AIV。
与独立 FP64 单 sweep 比较，并与本轮起始 executable 做三个输出的逐位对照。
静态图只验证调度，不证明设备内存可见性；本轮没有修改同步协议，因此沿用已有
故障注入证据，没有重复构建同一故障变体。

## 最终完整分解实测

固定输入 SHA256 为
`8485da69127c68fe91b3a84082c00b678f9a1842f83d4f6aff366c66df49867d`，
种子 20261116。A 为本轮开始时保留的 c79195aa executable，B 为最终
086f85ec executable；同机串行 ABBA，每组四次完整分解，分别去掉首次后合并
六次 warm 样本取中位数。计时覆盖设备前处理、全部 sweep、收敛和后处理，
不含文件读写、上传/下载及 CPU FP64 校验。

| 顺序 | 版本 | 完整设备时间 ms（首列为各组首次） |
|---|---|---|
| A1 | 本轮开始 | 714.807922, 713.518066, 713.547729, 713.287598 |
| B1 | 最终 | 622.077515, 620.851135, 621.094299, 621.005249 |
| B2 | 最终 | 620.979431, 620.590942, 620.607910, 619.990479 |
| A2 | 本轮开始 | 713.422241, 712.930969, 714.149597, 713.387695 |

warm 中位数由 **713.452881 ms 降至 620.729523 ms**，耗时减少 **12.9964%**。
相对最初 exact-sweep 的 966.515 ms 历史基线，累计减少 **35.7765%**，约
**1.5571×**；本次收益以同机 ABBA 的 13.00% 为准。此前候选初测的
621.145569 ms 不作为最终数字。全部 16 次均为 **25 sweep、converged=true**。
四组 B_rot、sigma、Vt 均逐位相同，最终输出 SHA256 为：

| 输出 | SHA256 |
|---|---|
| B_rot.bin | `3f914843fc7929ac18c5119ad833abb293cff479e37afe7bd1ae5772add1b403` |
| sigma.bin | `d71d2549ec6a2f49cc15f29ae6bf63ec193145c6117d278a562c7b397faab057` |
| Vt.bin | `8abba76d8c8735bc7feb638cccbabc8150a1edc4cd386fd3519a9e024afdba37` |

独立 FP64 校验：sigma 最大绝对误差 2.175810e-4，B 正交误差 1.115617e-7，
Vt 正交误差 4.953325e-7，重构误差 2.415023e-6，相对能量误差 5.165557e-7，
右特征残差 4.901788e-6（由参考谱范数平方归一化）。没有改变停止阈值。

静态测试 **12/12**，设备/CLI 回归 **105/105**：独立单 sweep 15、通用数值31、
block64专项9、控制7、Prepare12、Gram12、阈值/稀疏8、流水对照11。
其中控制套件含一项拒绝重复 local sweep，流水套件含一组五项非法参数拒绝；
这些检查不启动 NPU，不计作矩阵分解。
单 sweep 覆盖 block16/32/64、2/4/6/8/10 个块、多 batch、补齐和部分核，
最大相对 B 误差 **1.039619e-5**。实际命令及各案例输出保存在日志中。

## 最终 profiler 与解释

验证后分别串行 profile 本轮开始和最终 executable，每次完整分解一次。
两次均只有一条 MIX_AIC、Block Dim=20、Mix Block Dim=40，以及一次
KernelLaunchWithHandle；绑定 executable 和输出 SHA256，均25 sweep收敛。
MAC、Fixpipe、Vector 活动均非零，源码仍由 AIC 执行完整 R×X。

| profiler 项目 | 本轮开始 us | 最终 us | 时间减少 |
|---|---:|---:|---:|
| AIC Scalar | 630481.384 | 540895.903 | 14.21% |
| AIC MAC | 149286.786 | 149235.712 | 0.03% |
| AIC Fixpipe | 86212.517 | 85728.723 | 0.56% |
| AIV Scalar | 211280.328 | 185841.622 | 12.04% |
| AIV Vector | 325000.066 | 278118.948 | 14.42% |
| AIV MTE2 | 20505.235 | 20268.591 | 1.15% |
| AIV MTE3 | 20956.771 | 20781.950 | 0.83% |

profiler 单次任务时间为 713109.320→619612.760 us，独立于上述无 profiler 的
ABBA 计时。AIC Scalar 比例从 88.4% 降至 87.3%，仍然较高：它包含公平轮询
及缓存/发布协议，不能解释为纯运算，也不能把减少的 89.585481 ms 全部算作
纯等待下降。各管线重叠，时间不能相加。AIV Scalar 绝对时间下降，而比例因总
时间下降从29.6%变为30.0%；判断收益以完整设备耗时及绝对时间为依据。

本轮没有改 AIC 服务调度或跨组同步，也没有新增 MTE 流水；这是通过缩短 AIV
无效更新来减少 AIC 的轮询时间。AIC MAC 基本不变，与所有任务仍需 Gram、
活跃任务仍执行完整面板 Apply 一致。不报告未经采集的纯等待或 DDR 流量比例。

## 复现与证据位置

在服务器项目目录、已加载 CANN 环境下依次运行，不能并行占用 NPU：

```bash
bash artifacts/aic-wait-start/verify_final.sh
bash artifacts/aic-wait-start/profile_final.sh
python3 artifacts/aic-wait-start/collect_evidence.py
```

只复现完整 ABBA 和独立 FP64 校验：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/benchmark_scalar_pipeline.py \
  --baseline artifacts/aic-wait-start/jacobi_svd_baseline \
  --label aic-wait-final --abba --validate
```

该 label 是本次原始证据目录；重新采集时应换 label，并为 profiler 换新的输出
目录以保留本次数据。明确指定 baseline，避免脚本默认历史版本造成误比较。

- `logs/scalar-pipeline/aic-wait-final/results.json`：16次完整计时和 FP64 误差。
- `logs/aic-wait/profile-{baseline,final}-summary.json`：启动计数、管线和哈希绑定；
  原始 CSV 位于对应 `profile-*/*/mindstudio_profiler_output/`。
- `logs/aic-wait/verified-results.json`：测试、性能、profiler 汇总。
- `logs/aic-wait/implementation.diff`：相对本轮开始快照的四个正式源码差异。
- `logs/aic-wait/source-manifest.json`、`evidence-manifest.json`、
  `local-sync-results.json`：两端源码与原始证据的 SHA256 核对。
- `artifacts/aic-wait/evidence.tar.gz`：紧凑日志、JSON、CSV、diff；大矩阵保留服务器。

正式代码改动限于 `rotation_angles.hpp`、`block_rotation128.hpp`、
`block_rotation.hpp`、`fused_vector.hpp`，另扩展 `tests/run_inactive.py`。
同步、调度和 Cube 的五个文件与本轮开始快照一致；独立只读审查见
`logs/aic-wait/code-review.txt`。本地与服务器保留既有未提交修改，不提交或推送。
