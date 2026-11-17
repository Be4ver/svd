# Block64 独立静态审查记录

审查范围：block_rotation128.hpp、BlockRotation 包装、fused_vector.hpp、
fused_cube.hpp、paco_gemm.hpp、device 入口、CLI 及数值和控制流测试。
审查为只读，没有由审查者编译、执行 CPU/NPU 测试或修改文件。

未发现关键实现缺陷。静态检查确认 180 KiB UB 中各活跃区域分离，
最后一个 identity mask 字节位于偏移 183807，总分配为 184320 字节；
前后处理的复用区位于静态索引之前。检查了两个 64 元素半段的广播和
归约、16 行 tile 的 vector repeat stride、完整 J G Jᵀ 更新、circle
置换、R 累积，以及不同任务间 V→MTE2 和输出 V→MTE3/MTE3→V 顺序。
AIC 负责 Gram 与完整面板 R X；device 入口负责全部迭代及收敛控制。

重要限制：FP32 epsilon 的 pairwise_relative_gram 判据不能证明与 GPU
XGESVDJ 等价。后处理仍按最大行 norm 的 1e-6 阈值置零小 Vt 行。
测试通过支持当前有效子空间输出约定；不应据此宣称 GPU 级完整精度契约。

次要建议：独立 CLI metrics.json 尚未记录实际 tol；benchmark 的
results.json/timing.json 已记录 tolerance 与 stopping_criterion。
测试中的 kernel_launches_per_run 字段是 CLI 声明，独立证据来自
本次真实 profiler：KernelLaunchWithHandle Count=1，MIX_AIC Count=1，
Block Dim=20，Mix Block Dim=40。该 profiler 已由主任务解析核实。

审查者没有运行测试。主任务实际测试证据及性能结果见
[block64-results.md](block64-results.md)，原始日志位于 logs/block64/。

剩余 pivot 提取和 circle 列置换 Gather 不是矩阵转置；GEMM 转置标志
不能直接替代它们。完整转置及 R 行 Gather 在 block64 路径已移除。
