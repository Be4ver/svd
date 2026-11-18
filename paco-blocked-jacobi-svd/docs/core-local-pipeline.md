# 核组流水与块依赖：实现和验证

实现面向 Ascend 910B3 / CANN 8.2.RC1。完整矩阵、面板 X、Gram 和 R 留在 GM；AIV 仅把搬运分片和小 Gram/R 放入 UB。一次分解只启动一个 MIX_AIC_1_2 kernel，初始化、全部 sweep、收敛和输出均在 device 内。

默认使用 core_local_pipeline.hpp；--no-overlap 保留全局分阶段路径，显式 --pipeline-jobs N 保留原来的两阶段分组路径。CLI 及 --local-sweeps 1 的限制未改变。

后续 AIV 单位更新省略将同输入完整时间降至620.730ms，AIC Scalar绝对时间减少
14.21%；本文件的调度、双槽、块版本及全局屏障协议均保持不变。最新同机对照、
精度和 profiler 证据见 [aic-wait.md](aic-wait.md)，下文数字保留各历史版本语境。

文件分工及清理范围见 [file-guide.md](file-guide.md)。调用审计后删除未用的 MatmulCustom::Process2，以及 Process 中恒为 -1 的 cubeidx 参数和不可达通知分支；四处有效矩阵乘调用已更新。两种显式全局路径仍有 CLI 和回归用途。

## 任务、核组与两个槽

令 G 为本次启动的 AIC 数，每组的逻辑 AIV 为 2*g、2*g+1。C220 的 AIV GetBlockIdx 是物理 block ID 乘 task ratio 再加 subblock ID，AIC 直接使用物理 block ID。job 的 owner 为 2*(job%G)+(job/G)%2，各 AIV 按 round、job 升序处理队列。

AIC 轮询两个槽的 Gram 或 Apply 请求，不能阻塞等待固定槽，否则可能挡住另一槽中能释放块依赖的 Apply。闲置 AIV 不发请求，闲置 AIC 的任务计数为零，直接到 sweep 汇总屏障；所有参与者进入相同的全局屏障。

4096、block64、batch1、G=20 时每轮 32 个 job：组 0..11 的两个槽都有任务，组 12..19 只有第一个槽有任务。准备/参数最多有 32 个有效 AIV，不能描述为 40 个 AIV 满载。AIC 执行本 job 的完整面板写回，Apply 不再沿用全局路径中按 tile 分散给所有 AIC 的方式。

## GM 协议和缓冲生命周期

core_local_sync.hpp 为每个 request、ack、矩阵块版本各分配独立的 64B cache line，另有一个幂等 abort line。request 只传一个 int64 token；job 和 round 由 AIV/AIC 按同一静态队列独立推导。AIC 仅在 Apply 完成后调用 AdvanceWorkerJob 推进游标，每个 sweep 重建游标，token 则跨 sweep 递增。正常运行中每个 request 只有所属 AIV 写，每个 ack 只有本组 AIC 写；同块版本的发布者通过前驱依赖串行交接。

1. 第 s 个 sweep、第 r 轮的输入 epoch 为 s*(nb-1)+r。AIV 等两个块版本都等于 epoch，再用 S_MTE2 把 acquire 排在面板 DMA 读取前。
2. AIV PackJob 用两个四行 UB 缓冲读取 B、写入 X；MTE3_S 等写完，发布奇数 token 的 Gram 请求。
3. AIC acquire 后用 S_MTE2 排序 DMA 读取，Cube 按原有 128 列部分和顺序计算 X X^T；FIX_S 等 Gram 写回，再发布同 token 的 ack。
4. AIV acquire ack 后，更新 UB 小 Gram、计算角度并累积 R。第 0 个外层轮次做完整局部配对，之后只做跨块配对。block64 的跨块任务先检查全部 4096 对原阈值；全不活跃时可跳过恒为单位旋转的局部循环。MTE3_S 等 R/flag 写完，有活跃 pivot 发布正偶数 token，否则发布负偶数 token。
5. AIC acquire 后，正偶数请求执行完整 R X，包括两个块的全部 128 列 tile，只有 FIX_S 完成全部写回才依次发布两个块的 epoch+1。负偶数请求的 B 未被修改，直接发布两个版本。两种情况都最后发布正值 Apply ack。AIC 用 abs(request) 匹配下一 token，负号不参与版本或 served 计数。
6. AIV 必须收到 Apply ack 才能覆盖本 job 的 X/G/R 或提交下一任务。一个槽最多有一个未完成请求，token 和 served 计数在 sweep 间递增；没有任务的第二槽不参与轮询，有任务的两槽仍公平轮询。

同块的下一任务可能属于另一核组。它等待的是 GM 块版本，不是本组 ack 或核内 event。两个版本不要求原子地同时发布：任一版本发布都已晚于整个 Apply 写回，各后继仍等待自己的两个输入版本。

标量发布为 SetValue → DDR DSB → DCCI(CACHELINE_OUT) → DDR DSB；读取为 DCCI → DDR DSB → volatile GM int64 load。DCCI 负责标量 D-cache clean/invalidate，不能代替 DMA 完成。MTE3_S/FIX_S 负责 release，S_MTE2 负责 acquire 后的 DMA 排序。矩阵由 DMA 访问，未用标量 D-cache 读面板，没有 DMA 写入信号 cache line。删除 job/round payload 减少标量访存，不改变 64B line 粒度，不能据此推算相同比例的 DDR 流量下降。

每次 launch 由 AIV 重置全部信号首字段，再做初始化全局屏障。一次 launch 中版本不清零或取模，最大 200 sweep 的 epoch 不溢出 int；request/ack 和 Cube 完成计数用 int64。连续调用重新初始化 scratch 和信号。

## 重叠和保留的同步

两个槽来自同组两个 AIV，不是单 AIV 同时拥有两个在途面板。AIC 执行槽 0 Apply 时，槽 1 AIV 可以做 Pack 或参数；跨组可提前处理依赖已满足的后续轮次。同一 AIV 仍等待自己的 Apply ack，没有额外 lookahead。AIC 的两次矩阵乘服务串行。

2026-11-18 增加前后处理的两行 MTE2→Vector→MTE3 流水：复用既有 row scratch 和 Pack 的 free event；每槽在 MTE3 完成后才能再次被 MTE2 覆盖。block64 的 CrossActive 临时复用尚未初始化的 R，不增加 180KiB UB。具体生命周期、数学简化、失败候选及本轮结果见 [scalar-pipeline.md](scalar-pipeline.md)。下列 2026-11-17 数字保留为本轮起始基线。

每 sweep 保留两次 DeviceBarrier：第一次保证 active 发布、全部 AIC 完成，AIV0 汇总；第二次统一停止决定。另有初始化和最终输出各一次。4096 旧自动路径为 4*63+2=254 次/sweep，候选为 2 次。这里统计 DeviceBarrier 调用，大量点对点 request、ack、版本轮询仍存在，不能宣称没有跨核同步。

同块前驱只来自更小轮次，队列边也向前，静态依赖图无环。AIC 轮询两槽且不持有块锁，避免额外服务顺序环。论证假设全部启动核组同时常驻（host 限制到设备核数）、DMA/event 正常完成和公平执行；静态图不证明设备可见性。

轮询每 1024 次检查 abort 和 SYS_CNT，2,000,000,000 tick 无进展则设 abort，未将 tick 换算为秒。软件 wait 退出后继续统一 sweep 屏障，跳过 Finalize，host 下载信号并报 core-local dependency watchdog expired。abort line 允许多个核幂等写 1。watchdog 无法中断卡住的硬件 event 或全局屏障，不是全部硬件故障的恢复机制。

## CANN 依据与轮询诊断

头文件根为 /usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/tikcfw，审计见 logs/core-local/sdk-evidence.json 和 sdk-audit.log。impl/dav_c220/kernel_operator_common_impl.h 给出编号和 SYS_CNT；impl/dav_c220/kfc/kfc_comm.h 使用成对 mailbox、64B line 和轮询前 DCCI。它的空汇编 Barrier 不是 DMA completion；GroupBarrier 的 GM polling 同样使用 DCCI，kernel_event.h 支持上述事件。

最早无 watchdog 的普通 GetValue 紧密 wait 在 32×49、block16、单 job 卡住。独立源码副本去掉 Wait watchdog 后复现 ACL507014；只改为 volatile 读取，同例 repeat3 完成。日志为 poll-diagnostic.log、poll-volatile.log，生成器为 artifacts/core-local-start/prepare_poll_diagnostic.py。证据支持普通读取被复用的编译优化问题；没有成功获得可读 AICore 反汇编，不能宣称指令级证明具体优化。

## 验证与性能

全部设备测试串行，最终数值、控制、对照和性能结果补充于此；保留原始失败记录。

- host 静态测试覆盖独立配对计数、真实 C++ 映射与 tournament 对照、随机队列进展、版本和 token 跨三次 sweep 复用。
- 实际 NPU 单 sweep FP64 对照覆盖 block16/32/64，每种 nb=2/4/6/8/10、batch=1/2/1/5/3，含补齐、2/6/8 AIV、部分核和空任务，repeat2。15/15 通过，最大相对 B 误差 1.04e-5。
- 最初扩展 nb8/10、block64 两例超过 2e-5（2.21e-5、4.05e-5），但与串行全局路径输出逐位相同。保留阈值，用更分离的行范数重测通过。该轨迹对照避开接近对角值，不代替通用随机矩阵完整收敛测试。失败记录为 sweep-watchdog.log、sweep-large-serial.log；清理后最终为 sweep-cleanup.log。

logs/core-local/ 存构建、套件、诊断日志，logs/artifacts/core-local/ 存每次调用日志；服务器 artifacts/core-local/ 存输入/输出及 JSON。4096 复用 artifacts/exact-sweep-4096/input。日志、结果清单同步本地，大型输入/输出二进制留服务器。

停止准则保持逐对相对 Gram 阈值，不证明与 GPU XGESVDJ 完整停止条件等价。4096 使用同 SHA256 输入、FP32 epsilon、完整收敛的 device event 时间，丢弃首次报告 warm 中位数，与历史 966.515ms 和保存的原始 executable 同机重测比较。

## 2026-11-17 清理后最终结果

使用相同输入 SHA256 `8485da69127c68fe91b3a84082c00b678f9a1842f83d4f6aff366c66df49867d`、block64、batch1、FP32 epsilon，maxSweeps=40；每次都是包含初始化、迭代、收敛汇总和输出的完整 device event 时间，所有调用均在 25 sweep 收敛。最终 executable SHA256 为 `a3f4bd59d5251849e5b1dc0b32f957347d7621504a2b902c416a494bfece67ad`。

| 版本 | 全部 device ms | 去首次 warm 中位数 |
|---|---|---:|
| 清理后核组流水 | 905.950745, 904.698059, 904.740234, 904.861389, 904.724792, 904.955322 | 904.740234 ms |
| 本次保存原版重测 | 966.725830, 965.702271, 965.768494, 965.777405 | 965.768494 ms |
| 历史基线 | 967.162659, 966.515320, 966.617371, 966.448303 | 966.515 ms |

相对历史基线设备时间减少 6.39%（1.06828×），相对同次原版重测减少 6.32%（1.06745×）。这是核组流水相对原版的整体收益，不能把它归因于删除死代码。清理前 payload 版本 warm 中位数 902.434082ms 的记录仍保留于 results-before-cleanup.json；本报告使用清理后重新测量的数值。

三个输出二进制与保存原版逐位相同。FP64 独立校验：sigma 最大绝对误差 2.17581e-04、B 正交 1.11562e-07、Vt 正交 4.95332e-07、重构 2.41502e-06、能量相对误差 5.16556e-07、右特征残差 4.90179e-06。

清理后重建并串行重跑：host 静态 11/11、单 sweep FP64 15/15、通用数值 31/31、block64 9/9、控制 7/7、默认/全局串行/显式分组及保存原版对照 11/11，全部通过。控制/路径测试包含重复调用和超过一个 job wave。host 静态中的 C++ 公式测试需要 g++；服务器完整通过，本地 Windows 未安装 g++，因此此前本地 discover 有一个 setup 错误（其余 Python 测试通过），不是设备测试失败。

`tests/prepare_core_local_fault.py` 创建独立源码副本，令一个任务等待不可能的块版本并缩短诊断 watchdog；`tests/run_core_local_fault.py` 确认显式拒绝、无 runtime 超时，再执行正式程序 repeat3。清理后故障在 4.292s host wall 时间后返回预期错误，健康调用设备时间为 0.907940、0.164580、0.177900ms。该诊断不修改正式 watchdog 参数，不能证明所有硬件异常可恢复。

最终 profiler 的单次完整调用只有一条 MIX_AIC op，Block Dim=20、Mix Block Dim=40，runtime `KernelLaunchWithHandle` count=1。AIC MAC 时间 185880.189us、FIXPIPE 109286.04us，AIV Vector 519207.221us；这些管线统计有重叠，不能相加作总时间。AIC Scalar 占比 90.8%，包含轮询等待，仍有优化空间。非零 MAC 同时包含 Gram 和 Apply；完整面板旋转位于实际执行路径的 ApplyJob，实际单 sweep 及完整输出对照共同验证它的结果。不能仅凭 MAC 总数区分两个阶段，也没有用 profiler 推断每个槽实际重叠的时长。msobjdump 确认单 MIX ELF；反汇编不可读的限制保留。

首次最终 profiler 尝试在 aclInit 前失败：磁盘只剩约 179MB，自动 storage_limit=160MB 低于 API 最小 200MB。释放本任务重复输出并指定 200MB 后完成采集；最终 profile 输出与已验证 candidate 逐位相同，冗余二进制删除，仅保留 metrics 和原始 CSV。失败日志 profile-payload-attempt1.log、profile-init-failure.log 保留；该失败未进入 kernel。

详细证据：`artifacts/core-local/4096/results.json`、各 `*-final/results.json`、`sweep-volatile/results.json`、`fault/results.json`；清理后套件 stdout 为 `logs/core-local/*-cleanup.log`，单次调用日志为 `logs/artifacts/core-local/`。最终 profiler 为 `logs/core-local/profile-payload/*/mindstudio_profiler_output/`，摘要 `profile-payload-summary.json` 绑定上述 executable 哈希，单 ELF 记录为 mixed-elf-cleanup.log。验证命令保存在 artifacts/core-local-start/verify_cleanup.sh。

所有新增 .log 均位于项目 logs 的专用目录。原始输入和完整输出保留服务器，结果 JSON、指标和日志同步本地。source-manifest.json 记录规范化换行后的源码哈希，evidence-manifest.json 记录下载证据哈希，verified-results.json 汇总通过情况。
