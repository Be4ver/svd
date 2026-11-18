# 文件功能与调用关系

本说明覆盖项目维护的源码、脚本、测试和文档。`build*`、`logs`、`artifacts` 中的编译产物、逐次运行日志和矩阵数据按目录说明；它们不是额外的算子入口。历史文档中的调度、sweep 定义和性能数字只适用于各自版本，当前行为以 README、exact-sweep、core-local-pipeline、aic-wait 和最新 ub-copy 为准。

## 源码与构建

| 文件 | 功能 |
|---|---|
| `CMakeLists.txt` | 设置 C++17、910B3 和 CANN 路径，编译 mixed kernel 静态库及 ACL 主程序。 |
| `main.cpp` | 批量 SVD CLI；校验参数和输入，分配 GM，上传输入，每次完整调用启动一次 kernel，下载输出并记录设备时间、实际 sweep 和收敛状态。 |
| `blocked_jacobi_fused.cpp` | 唯一 device kernel 入口；组织初始化、全部 sweep、收敛和最终输出，选择默认核组流水或显式全局分阶段路径。 |
| `fused_common.hpp` | 共享完成标记和 `DeviceBarrier`；协调常驻 AIC/AIV 的必要全局阶段同步。 |
| `core_local_schedule.hpp` | 块 round-robin 配对、job/AIV/核组映射及 AdvanceWorkerJob 游标推进；AIC/AIV 独立推导同一静态队列，测试直接编译这些公式。 |
| `core_local_sync.hpp` | 64B GM request/ack、块 epoch 和 abort 信号；实现带缓存维护、内存排序和 watchdog 的发布/等待。 |
| `core_local_pipeline.hpp` | 默认调度；AIV 等块版本后提交 Gram/Apply，AIC 轮询本组有效槽；正 Apply 完整写回，负偶数 token 省略单位旋转，两者均发布块版本和 ack。 |
| `fused_vector.hpp` | AIV 工作：缩放与补齐、面板 Pack、小 Gram 参数求解、收敛汇总、norm/排序和 B/sigma/Vt 输出；含两种调度调用入口。 |
| `fused_cube.hpp` | AIC 工作：用 Cube GEMM 计算 Gram 和完整面板 `R X`；默认路径按 job 服务，显式全局路径按阶段分配任务。 |
| `block_rotation.hpp` | block16/32 的小 Gram Jacobi 向量实现，以及按面板尺寸分派到小矩阵或 128 阶求解器的封装。 |
| `block_rotation128.hpp` | block64 对应的 128×128 Gram/R 求解器；向量化计算角度、分批更新行列并累计旋转矩阵；全不活跃的跨块局部轮省略单位更新但保留置换，默认路径的单位面板可提前返回。 |
| `rotation_prepare.hpp` | 两种求解器共享的静态索引、配对置换、Gather 映射和单位阵掩码生成。 |
| `rotation_copy.hpp` | solver/Prepare 共享的连续 UB Copy 封装；只用于 32 或 64 倍数、最多 8192 个 FP32 元素，分段行移位仍保留临时区。 |
| `rotation_angles.hpp` | 两种求解器共享的阈值与系数计算；分开 RotationThreshold/RotationCoefficients 以便省略单位轮的角度后半段，使用普通掩码和 min/max hypot，保留原阈值及 sqrt/div 精度。 |
| `paco_gemm.hpp` | 底层 FP32 Cube 矩阵乘；管理 GM→L1→L0 搬运、Mmad、Fixpipe 及 Gram 部分和的原子累加。 |
| `.gitignore` | 排除构建目录、输入输出、实验产物、日志和 Python 缓存。 |
| `README.md` | 当前接口、构建运行方式、执行分工、验证入口及限制。 |

默认调用链：`main.cpp → blocked_jacobi_fused.cpp → core_local_pipeline.hpp → FusedVector / FusedCube`。AIV 求解器经 `block_rotation.hpp` 调用对应尺寸实现；AIC 的 Gram 和 Apply 都调用 `paco_gemm.hpp`。跨组的数据交接由 `core_local_sync.hpp` 的块版本保护。

`--no-overlap` 和显式 `--pipeline-jobs N` 仍调用 `Pack/Parameters/Gram/Apply`，这些函数属于有效接口路径。`OwnerWorker` 由静态测试验证队列映射；带 device 注解的构造函数用于 device 对象构造，也保留。

本次删除：`MatmulCustom::Process2` 无调用；`Process` 的 `cubeidx` 在全部四处调用中恒为 -1，因此删除该参数及其永不可达的旧跨核通知分支，并更新四处实参。保留现有矩阵乘和原子累加顺序。

2026-11-18 进一步删除不再需要的 `CoreSignals::Task/Job/Round`，request 只传 token；主程序 watchdog 诊断也不再读取旧 payload。两种 solver 的 Angles 数学实现合并到 `rotation_angles.hpp`，各自保留必要的 Gather 封装。显式全局路径仍受 CLI 和回归使用，不能删除。

## 使用与基准脚本

| 文件 | 功能 |
|---|---|
| `scripts/gen_jacobi_svd_data.py` | 按尺寸、batch、种子和目标条件数生成 FP32 输入及参考数据。 |
| `scripts/verify_jacobi_svd.py` | 用户可直接运行的输出检查器；对比奇异值、正交性及重构误差。 |
| `scripts/reference.py` | CPU 参考模型：tournament、单独 Gram Jacobi 求解、当前精确 sweep 配对和 blocked SVD；供静态/数值测试使用。 |
| `scripts/benchmark.py` | 小规模批量基准、block 参数扫描及可选原版对照。 |
| `scripts/benchmark_large.py` | 单个大矩阵的多次完整设备计时，计时区外进行独立数值校验。 |

## 测试

`test_*.py` 由 unittest 运行，主要检查 CPU 模型和静态调度；`run_*.py` 调用真实 NPU CLI。静态图测试不验证设备缓存可见性。所有真实设备套件串行执行。

| 文件 | 功能 |
|---|---|
| `tests/test_reference.py` | 检查配对覆盖、CPU Gram 旋转及 blocked SVD 数值契约。 |
| `tests/test_validation.py` | 检查验证器能识别 sigma 与右奇异向量错配。 |
| `tests/test_sweep_schedule.py` | 统计 block16/32/64 和 4096 规模的原始配对，检查完整/跨块置换恢复。 |
| `tests/test_async_dependencies.py` | 检查最初静态块对依赖图的覆盖、前驱关系及无环性。 |
| `tests/test_core_local.py` | 编译真实 C++ 映射公式；随机模拟两槽调度、空闲核、跨 sweep token/版本复用和进展。需要 g++。 |
| `tests/run_npu.py` | 通用真实 NPU 数值回归；提供共享日志定位、执行检查及独立 FP64 校验函数。 |
| `tests/run_block64.py` | block64 的补齐、秩亏、尺度等专项完整分解回归。 |
| `tests/run_control.py` | 固定 sweep、提前停止、重复调用、部分常驻核和错误参数回归。 |
| `tests/run_overlap.py` | 对比默认核组、全局串行、显式分组及可选保存原版，检查输出逐位相同及控制边界。 |
| `tests/run_sweep.py` | 独立 FP64 配对实现，对比真实 NPU 单 sweep；覆盖三种 block、多个块数、batch、补齐和部分核。 |
| `tests/run_inactive.py` | 验证无效面板、原阈值边界、稀疏活跃轮与单位轮交替、补齐和多 batch；独立 FP64 单 sweep，并与保存 executable 逐位对照。 |
| `tests/benchmark_scalar_pipeline.py` | 固定输入的串行 ABBA 完整分解计时；绑定 executable/输出哈希，统计 warm 中位数，可作独立 FP64 校验。 |
| `tests/run_prepare.py` | 检查 Prepare 索引/掩码生成、共享 scratch 复用，并可与保存结果逐位对比。 |
| `tests/run_gram.py` | 覆盖 Gram 的 K 尾部、多分段、4096 列以及单 AIC 多任务复用。 |
| `tests/run_core_local_4096.py` | 校验固定 4096 输入哈希，串行运行候选 6 次和保存原版 4 次，计算 warm 中位数并作 FP64/逐位对照。 |
| `tests/prepare_core_local_fault.py` | 生成独立故障源码副本，注入不可能块版本并缩短诊断 watchdog；不修改正式源码。 |
| `tests/run_core_local_fault.py` | 运行故障副本，确认预期错误后再次运行正式程序，检查错误退出及后续调用。 |

## 设计与结果文档

| 文件 | 功能 |
|---|---|
| `docs/file-guide.md` | 本文件，逐文件功能及清理范围。 |
| `docs/core-local-pipeline.md` | 当前核组映射、两槽协议、GM 可见性依据、watchdog、测试和最终性能。 |
| `docs/scalar-pipeline.md` | Scalar、Angles、通信和 UB 流水的分项试验、采用/拒绝理由及本轮最终性能。 |
| `docs/aic-wait.md` | 后续 AIC Scalar 等待定位、单位轮省略、C220 通知 API 审计、网络资料、当轮计时与 profiler。 |
| `docs/ub-copy.md` | 将 UB 加零搬运替换为 Copy 的 API 语义、步长/重叠约束、回归与计时；当前最新记录。 |
| `docs/exact-sweep.md` | 每对恰好一次的 sweep 定义、8,386,560 对计数及 966.515ms 基线。 |
| `docs/aiv-batching.md` | AIV 多行批量更新、Pack 双缓冲、UB 布局和当时的测量。 |
| `docs/gram-synchronization.md` | Gram 局部屏障删减及 Axpy 候选对照；记录采用与未采用的方案。 |
| `docs/prepare-vectorization.md` | Prepare 向量化、索引构造方式及验证。 |
| `docs/aiv-vectorization-results.md` | 早期 AIV 向量化结果，使用旧 sweep 定义。 |
| `docs/overlap-results.md` | 原有全局 A/B 分组重叠方案及其性能。 |
| `docs/block64-results.md` | 初始 block64 的实现、精度边界、核心利用率和测量。 |
| `docs/block64-review.md` | block64 独立静态审查记录。 |
| `docs/block64-plan.md` | block64 实现计划。 |
| `docs/4096-results.md` | 早期 4096 单次启动分解记录。 |
| `docs/single-launch-results.md` | 初次单 mixed launch 的启动计数、回归和耗时记录。 |
| `docs/single-launch-plan.md` | 将全部 sweep 和前后处理迁入一次 kernel 的计划。 |
| `docs/design.md` | 项目初始算法、行旋转约定、内存和同步设计。 |
| `docs/implementation-plan.md` | 初始 blocked Jacobi 实现步骤。 |
| `docs/superpowers/plans/2026-11-16-phase-overlap.md` | 全局分阶段 AIV/AIC 重叠实施计划。 |
| `docs/superpowers/plans/2026-11-17-aiv-vectorization.md` | AIV 批量向量化实施计划。 |
| `docs/superpowers/plans/2026-11-17-core-local.md` | 当前核组流水实施及验证清单。 |
| `docs/superpowers/plans/2026-11-18-scalar-pipeline.md` | 本轮逐项优化、独立验证、串行计时及同步计划。 |
| `docs/superpowers/plans/2026-11-18-aic-wait.md` | 后续 AIC Scalar 定位、候选实现、串行验证及源码同步计划。 |

## 产物和复现辅助文件

| 路径 | 功能 |
|---|---|
| `build*/` | 各版本 CMake/CANN 编译目录和 executable；默认源码没有第二个 kernel 入口。 |
| `logs/core-local/` | 本次构建、静态和 NPU 套件、故障诊断、profiler 原始记录与验证清单。清理后回归使用 `*-cleanup.log`。 |
| `logs/artifacts/core-local/` | 测试脚本为每个案例保存的实际 CLI stdout/stderr。 |
| `logs/exact-sweep/` | 精确 sweep 历史基线的测试及计时日志。其他日志子目录对应历史实验。 |
| `logs/scalar-pipeline/` | 本轮分项构建、失败候选、回归、ABBA 计时和 profiler 证据。逐次案例 stdout 位于 `logs/artifacts/scalar-pipeline/`。 |
| `artifacts/scalar-pipeline/` | 本轮 NPU 输入输出、故障注入副本和结果清单；矩阵二进制留服务器。 |
| `artifacts/scalar-pipeline-start/` | 开始状态及各候选源码/executable 快照；`verify_final.sh` 串行验证，`resume_final.sh` 从故障构建继续剩余验证，`profile_final.sh` 采集最终 profiler，`summarize_profile.py` 校验单次启动及输出哈希。 |
| `artifacts/scalar-pipeline-start/collect_evidence.py` | 检查最终套件数量、逐位对照和 profiler 绑定，生成规范化源码/证据 SHA256 清单，打包紧凑日志及 JSON/CSV。 |
| `artifacts/scalar-pipeline-start/verify_sync.py` | 本地检查服务器下载的源码清单和证据清单；源码规范化 CRLF/LF，日志按原始字节核对 SHA256。 |
| `artifacts/scalar-pipeline-start/audit_angles.py` | 保存服务器 C220 Div/Rsqrt/Select 和 event 头文件摘录及哈希，说明 mask 设置和事件排序的 SDK 依据。 |
| `logs/aic-wait/`、`logs/artifacts/aic-wait/` | AIC 等待优化的诊断、构建、105 项设备/CLI回归、最终 profiler 和源码/证据清单；ABBA 日志另在 `logs/scalar-pipeline/aic-wait-*`。 |
| `artifacts/aic-wait/`、`artifacts/aic-wait-start/` | 新一轮矩阵结果及开始状态快照；起始 executable 为 c79195aa，保留供串行对照。 |
| `artifacts/aic-wait-start/prepare_diagnostic.py` | 从当前源码生成独立 SYS_CNT 计时副本，分别记录 AIV 五个区间和 AIC Gram/Apply；正式源码不插桩。 |
| `artifacts/aic-wait-start/analyze_diagnostic.py` | 解析起始诊断的 warm 调用，统计有效 AIV 区间和每组 AIC 工作比例，不将 tick 擅自换算为秒。 |
| `artifacts/aic-wait-start/audit_sync.py` | 保存服务器 C220 notify/wait 和 KFC mailbox 实现摘录及 SHA256，限定硬件通知的已知语义。 |
| `artifacts/aic-wait-start/research.py` | 通过 Crossref 获取相关单边 Jacobi 动态排序及快速算法的元数据/摘要。 |
| `artifacts/aic-wait-start/research_lapack.py` | 下载 DGESVJ/DGEJSV 参考源码并保存阈值/跳过策略相关摘录和哈希。 |
| `artifacts/aic-wait-start/verify_final.sh` | 串行运行最终静态、数值、单 sweep、阈值/稀疏、控制、调度及固定输入 ABBA。 |
| `artifacts/aic-wait-start/profile_final.sh` | 在验证后串行采集本轮起始与最终 executable 的完整 4096 profiler。 |
| `artifacts/aic-wait-start/summarize_profile.py` | 检查单次 mixed launch、20/40 核、非零 Cube 活动和 executable/输出 SHA256。 |
| `artifacts/aic-wait-start/collect_evidence.py` | 校验本轮各套件、ABBA 和 profiler，汇总实际性能并打包日志/JSON/CSV及哈希清单。 |
| `artifacts/aic-wait-start/verify_sync.py` | 在本地核对本轮服务器源码清单和证据原始字节 SHA256。 |
| `logs/ub-copy/`、`logs/artifacts/ub-copy/` | Copy 替换的 SDK 依据、构建、静态/真实 NPU 回归及 profiler；ABBA 计时在 `logs/scalar-pipeline/ub-copy/`。 |
| `artifacts/ub-copy/`、`artifacts/ub-copy-start/` | Copy 回归输入输出和修改前三份头文件/executable 快照；矩阵二进制留服务器。 |
| `artifacts/ub-copy-start/verify.sh` | 串行设备回归和固定 4096 输入的 ABBA 完整分解计时。 |
| `artifacts/ub-copy-start/profile.sh` | 在验证结束后串行采集修改前后完整 4096 的 PipeUtilization。 |
| `artifacts/ub-copy-start/summarize_profile.py` | 校验一次 mixed launch、20/40 核、非零 Cube 活动及 executable/输出哈希。 |
| `artifacts/ub-copy-start/collect_evidence.py` | 核对回归、ABBA、profiler 与其他 kernel 源码未变，生成差异和源码/证据清单，打包紧凑日志。 |
| `artifacts/ub-copy-start/verify_sync.py` | 本地核对 Copy 版本的源码及下载证据 SHA256。 |
| `artifacts/core-local/` | 本次输入输出、各套件 `results.json`、故障副本及紧凑证据包。大矩阵二进制留服务器，日志和结果清单同步本地。 |
| `artifacts/exact-sweep-4096/` | 固定 4096 基线输入和输出；后续计时复用同一输入。 |
| `artifacts/core-local-start/local-source.tar.gz` | 开始前本地已有工作快照。 |
| `artifacts/core-local-start/server-source.tar.gz` | 开始前服务器已有工作快照。 |
| `artifacts/core-local-start/jacobi_svd_baseline` | 服务器保存的优化前 executable，用于同机数值和性能对照。 |
| `artifacts/core-local-start/jacobi_svd_before_cleanup` | 服务器保存的本次函数清理前 executable。 |
| `artifacts/core-local-start/audit_sdk.py` | 收集服务器 CANN 同步、编号和缓存 API 证据。 |
| `artifacts/core-local-start/prepare_poll_diagnostic.py` | 创建独立轮询诊断副本，比较普通和 volatile 标量读取。 |
| `artifacts/core-local-start/verify_payload.sh` | 清理前 volatile payload 版本的串行完整验证命令记录。 |
| `artifacts/core-local-start/verify_cleanup.sh` | 清理后版本的串行静态、NPU、故障与 4096 验证。 |
| `artifacts/core-local-start/profile_payload.sh` | 对当前 executable 的单次完整 4096 调用采集 profiler；目录名沿用 payload。 |
| `artifacts/core-local-start/summarize_profile.py` | 核验一条 mixed op、一次 launch 和非零 AIC/AIV 管线活动，绑定 executable 哈希。 |
| `artifacts/core-local-start/collect_evidence.py` | 检查套件结果，生成源码/证据 SHA256 清单并打包日志和 JSON。 |

工作区父目录中的其他项目、报告和已有产物不属于本次源码清理范围。
