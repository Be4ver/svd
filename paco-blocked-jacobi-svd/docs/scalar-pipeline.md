# Scalar、Angles、通信和 AIV 流水优化（2026-11-18）

本轮沿用已验证的核组流水：一次 mixed launch，20 AIC/40 AIV，矩阵常驻 GM，
每 sweep 每个原始配对恰好一次，`--local-sweeps 1`，FP32 epsilon。保留开始前
本地/服务器修改；没有提交或推送。逐文件功能见 [file-guide.md](file-guide.md)，
跨组可见性依据见 [core-local-pipeline.md](core-local-pipeline.md)。

后续针对 AIC 等待的实现和最终620.730ms结果见 [aic-wait.md](aic-wait.md)。
下文保留本阶段712.531ms及当时逐项候选的测量，不能用作后续同机ABBA对照。

## Scalar 的实际含义和处理

上一版 profiler 的 AIC Scalar 为 820881.596us（90.8%），其中包含 mailbox
busy polling，不能把它全部视为有效标量运算。AIC 等 AIV 时仍执行 DCCI、DSB、
load 和循环，因此减少 AIV 无效计算也会减少 AIC 的 Scalar 等待时间。

本轮保留必要的 volatile 轮询及缓存维护；普通 GetValue 紧密等待曾在真实设备
卡住，不能为降低 Scalar 指标将其恢复。AIC 自己推导每槽的 job/round，删除
CoreSignals::Task/Job/Round 及其 payload 访存；只有一个任务槽的核组不再轮询
永远空闲的第二槽。两个有效槽仍公平轮询，避免固定槽等待造成依赖死锁。

最大收益来自 block64 的 CrossActive：后续外层轮次只允许跨块 pivot，先用原
`abs(c) > tol*sqrt(abs(a*b))+1e-30` 检查全部 4096 对。全不活跃意味着每次旋转
都是单位旋转，G 不变，64 次置换最终回到原位，因此可以省去这些局部循环；
任意一对活跃则执行完整原调度。首个外层轮次仍覆盖块内及跨块配对，不作此
提前退出。4096 每 sweep 仍有 8,386,560 次 pivot 机会，提前判定的机会没有遗漏。

## Angles 的数学与指令

`rotation_angles.hpp` 合并两种 solver 的重复计算；各自 Angles 只保留 Gather。
设 δ=(b-a)/2，h=max(|δ|,|c|,1e-30)，采用：

```text
hypot = h * sqrt(1 + (min(|δ|,|c|)/h)^2)
t     = sign(δ) * c / max(|δ|+hypot, 1e-30)
cs    = 1 / sqrt(1+t*t)
sn    = cs*t
```

不活跃 pivot 的 t 仍置零，阈值操作顺序不变。活跃时 |c|>threshold≥1e-30，
h 是实际最大值，因此式中比值有界。相对原来的两路缩放 hypot，减少一个
Div 和一个 Mul、增加一个 Min。FP32 运算次序变化不能仅靠代数推断等精度，
采用前运行了实际单 sweep 和完整输出对照。

C220 Level2 向量 API 每条操作都会切换/恢复 count mask；改用单 repeat 的
Level0 普通掩码，覆盖16/32/64个 pivot。Select 内部可能改变 mask，因此没有
无条件使用 isSetMask=false。此修改的独立性能收益很小，不能称主要加速来源。
服务器 CANN 头文件摘录与哈希保存在 `logs/scalar-pipeline/sdk-angles-evidence.json`，
可用 `artifacts/scalar-pipeline-start/audit_angles.py` 重新采集；这不是反汇编证据。

直接用 Rsqrt 替代 cs 的 sqrt/div 候选被拒绝：独立单 sweep FP64 对照产生约
0.01～0.2 的误差，日志 `math-sweep.log` 和 `math-sweep-results.json` 保留。
当前代码继续使用 sqrt/div。没有更改 Gram 的分段累加次序，也没有将旋转
改为会改变舍入次序的 FMA。

## 通信与单位 Apply

默认核组路径中，ParametersJob 没有任何活跃 pivot 时，AIV 提交负偶数 token；
AIC 省略单位矩阵 R×X 的面板计算/读写，仍发布两块 epoch+1 和正常正值 ack。
有活跃 pivot 使用正偶数请求，完整 ApplyJob 经 FIX_S 后才发布版本。奇数
Gram 请求不变，AIC 用 abs(request) 匹配 served+1。token 不取模，连续 launch
重置，跨 sweep 递增。

单位任务的 B 已经由前驱版本 acquire，且本任务未写 B，因此可以发布下一版；
后继仍须等待自己两个块的版本。AIV 收到 ack 前不能复用本 job 的 X/G/R。
这项优化没有跳过 Pack、Gram、R/flag 写回，也没有减少 request/ack 的数量。
它确实减少无需更新面板的 GEMM 和 GM 访问；没有采集 DDR 字节计数，故不报告
通信量下降百分比。删除 job/round 也不改变 request 的64B cacheline粒度。

保留 request/ack、每块 version、abort 各64B独立行；发布为 SetValue→DDR DSB→
DCCI→DDR DSB，读取为 DCCI→DDR DSB→volatile load。MTE3_S/FIX_S 等 DMA 完成，
S_MTE2 排序 acquire 后读取。每 sweep 两次全局收敛屏障，初始化/输出各一次，
核组间仍有真实 GM 依赖同步。没有用核内 event 代替跨组数据交接。

## UB 复用与部件流水

block64 仍分配180KiB UB。CrossActive 使用尚未初始化的 R 前16KiB保存全部
`abs(c)-threshold`，随后重新初始化 R。它不与活跃的旋转矩阵重叠。

FusedVector 的对齐前处理和最终 B 输出改用既有 row scratch 的两个行槽，最大
各16KiB；norm/sorted/perm 从至少3*cp处开始，不重叠。前处理保持 pre、inverse
两次 Muls，以保留极端尺度行为。补齐路径保留原实现。每槽事件生命周期：

```text
free(MTE3→MTE2) → GM→UB(MTE2)
→ ready(MTE2→V) → Muls(Vector)
→ output(V→MTE3) → UB→GM(MTE3) → free
```

两个槽交替发射，前一槽 MTE3 可以与后一槽 MTE2/Vector 重叠；重用前等待该槽
free，退出时 drain 两槽后再复用 UB。复用 Pack 已有的 free event，增加两组
ready/output event，不增加 UB。Pack 自身保留两个四行 DMA 槽，占用求解器前
128KiB；Pack、前后处理和参数求解分阶段使用 scratch。

同组两个 AIV 的不同槽仍可与 AIC 服务重叠：AIC 做一个槽的 Apply 时，另一个
AIV 可准备面板或求参数。每个 AIV 自己仍只有一个在途 job；没有把128阶 G/R
翻倍放入 UB。4096/batch1 只有32个有效参数任务，12组双槽、8组单槽，不能称
40 AIV 都满载。aggregate profiler 不能证明每槽具体重叠时长。

## 分项证据

每项均串行使用固定4096输入、相同精度和停止条件，ABBA各4次、每组丢首次，
统计6个warm样本中位数。每项都有25 sweep收敛且与起始输出逐位一致；下列数字
用于区分收益，不替代最终整体验证。对应JSON位于 `logs/scalar-pipeline/<名称>/results.json`。

| 名称 | 候选 warm ms | 同次对照 warm ms | 解释 |
|---|---:|---:|---|
| normal-mask | 901.770752 | 902.064118 | 0.033%，无明显独立收益 |
| minmax | 900.854431 | 903.224671 | 0.262%，数学简化收益较小 |
| preflight | 739.507843 | 901.751648 | 17.992%，省去无效局部循环 |
| row-pipeline | 737.889587 | 739.528534 | 0.222%，约1.64ms |
| static-mailbox | 737.795899 | 737.569397 | 慢0.031%，噪声级；只保留静态简化 |
| identity-apply | 713.867035 | 738.060883 | 3.278%，省去单位Apply |

分项试验的baseline参数会选择前一个候选；最终计时则与本轮开始保存的完整
executable作同机ABBA对照。原始源码/executable快照在服务器
`artifacts/scalar-pipeline-start/`，失败候选也保留，不能当作当前实现。

## 最终验证与复现

串行完整验证脚本为 `artifacts/scalar-pipeline-start/verify_final.sh`，单次
profiler脚本为同目录 `profile_final.sh`，`summarize_profile.py` 绑定最终
executable和输出哈希。所有新增.log位于 `logs/scalar-pipeline/` 或
`logs/artifacts/scalar-pipeline/`，大型矩阵二进制留服务器。

首次故障副本构建在 CANN 的 merge_mix_obj.sh 退出：测试脚本漏传
`-DCMAKE_BUILD_TYPE=Release`，生成的命令以空 `--build-type` 结尾，触发
脚本 `set -e` 下的 `shift 2` 失败。正式构建为 Release，未受影响。
补齐测试配置后构建成功；初次 fault-build.log 保留，修正后日志为
fault-configure-release.log / fault-build-release.log，resume_final.sh
从此处继续剩余验证，没有重复此前通过的回归。

### 最终完整分解计时

使用固定输入 SHA256
`8485da69127c68fe91b3a84082c00b678f9a1842f83d4f6aff366c66df49867d`，
4096×4096、FP32、batch1、block64、FP32 epsilon、maxSweeps=40。
按旧版→新版→新版→旧版顺序串行运行，每组4次，每组丢首次，分别汇总6个
warm样本。包含预处理、全部迭代、收敛汇总和后处理的设备事件时间为：

| 顺序 | executable | 4次完整 device ms |
|---|---|---|
| A1 | 本轮开始旧版 | 901.262268, 900.749390, 900.637085, 900.502136 |
| B1 | 最终新版 | 713.524414, 713.121582, 713.238464, 712.645386 |
| B2 | 最终新版 | 714.194458, 712.416992, 711.876526, 712.006958 |
| A2 | 本轮开始旧版 | 901.461182, 900.760132, 900.880066, 900.585144 |

新版 warm 中位数 **712.531189 ms**，同次旧版 **900.6932375 ms**，耗时减少
**20.89%**。相对历史 966.515 ms 减少 **26.28%**，加速 **1.35645×**。
所有16次分解均在 **25 sweep 收敛**，每次仅启动一次 mixed kernel。
计时区外的FP64校验不会计入设备时间。数据见 `logs/scalar-pipeline/final/results.json`。
最终 executable SHA256 为
`c79195aa84fd49e90a3d58889156d8c4e0faba7a71ba43aa87d5e4b3610ecf78`；
同次旧版为 `a3f4bd59d5251849e5b1dc0b32f957347d7621504a2b902c416a494bfece67ad`。

四组运行保存的三个输出文件逐位一致；CLI只下载各组最后一次repeat的矩阵，
没有逐一下载16次调用的输出。独立FP64校验结果：

| 指标 | 误差 |
|---|---:|
| sigma 最大绝对误差 | 2.1758097e-4 |
| B 行正交性 | 1.1156173e-7 |
| Vt 行正交性 | 4.9533248e-7 |
| 行空间重构相对误差 | 2.4150231e-6 |
| 能量相对误差 | 5.1655573e-7 |
| 右特征残差，按参考谱范数平方归一化 | 4.9017879e-6 |

停止准则始终为逐对Gram相关性阈值，没有证据表明等价于GPU XGESVDJ的完整停止条件。

### 最终回归和故障检查

正式最终 executable 完成 host静态12/12、独立FP64单sweep15/15、通用数值31/31、
block64 9/9、control 7/7、prepare 12/12、gram 12/12、阈值边界5/5、调度对照11/11。
共12项静态测试、102个实际NPU案例；调度对照包括默认、全局串行、显式分组和保存旧版。
单sweep覆盖block16/32/64，每种nb=2/4/6/8/10、不同batch、补齐及2/6/8 AIV部分核，
包括空任务和多wave。阈值测试覆盖零、低于/等于/高于阈值及补齐多batch，保留原阈值。
测试脚本还执行重复调用。最终逐套件日志和JSON为 `logs/scalar-pipeline/*-final*`。

独立故障副本等待不可能的块版本，3.465s host wall时间后返回预期watchdog错误，
没有ACL507014；随后正式程序repeat3的device时间为1.025600、0.188420、0.168140ms。
该副本不改变正式kernel，不能证明硬件event或全局屏障故障都能恢复。

### 最终 profiler 与 Scalar 限制

profiler单次完整调用为711690.140us，只出现一条MIX_AIC op，20 AIC/40 AIV，
runtime `KernelLaunchWithHandle` count=1，25 sweep收敛，输出与计时新版逐位一致。
`logs/scalar-pipeline/profile-final-summary.json` 绑定executable和输出SHA256。
下表将它与开始旧版的 `logs/core-local/profile-payload-summary.json` 比较：

| 管线 | 旧版 us | 新版 us | 变化 |
|---|---:|---:|---|
| AIC Scalar | 820881.596 | 628147.598 | 减少23.48% |
| AIC MAC | 185880.189 | 149267.278 | 减少 |
| AIC FIXPIPE | 109286.040 | 85938.034 | 减少 |
| AIV Vector | 519207.221 | 325000.066 | 减少 |
| AIV Scalar | 175606.076 | 210286.634 | **增加19.75%** |
| AIV MTE2 | 23249.436 | 20166.048 | 减少 |
| AIV MTE3 | 22056.729 | 20898.974 | 减少 |

AIC Scalar占比从90.8%变为88.3%，AIV Scalar从19.4%变为29.5%。这次优化降低
完整耗时和AIC Scalar，**没有降低AIV Scalar**。聚合统计包含等待/轮询，不能定位
某个函数或区分标量有效计算与等待；也不能确定AIV Scalar增加由哪项改动造成。
后续应先定位AIV依赖等待及标量指令热点，再决定是否增加任务lookahead或改变轮询。
不采用仅让Scalar指标看起来更低、却增加任务响应延迟的轮询改法。

MAC包含Gram与Apply；非零MAC本身不能区分二者，源码中实际执行的正Apply路径及
独立单sweep/完整输出校验共同确认完整面板旋转仍由AIC执行。管线时间有重叠，不能
相加，也不能用这些聚合计数证明每槽的具体重叠时长。两行流水的分项完整分解收益约
1.64ms，不能将全部20.89%收益归因于MTE流水。没有DDR字节计数，通信量只报告实际
省略的操作，不报告带宽节省百分比。

`collect_evidence.py` 汇总套件、计时和profiler到 `logs/scalar-pipeline/verified-results.json`，
生成规范化源码和原始证据SHA256清单；`verify_sync.py` 在本地核对。大型矩阵保留服务器
`artifacts/scalar-pipeline/`，紧凑日志/JSON/CSV同步本地。源码、文档和测试同步后保留未提交状态。
