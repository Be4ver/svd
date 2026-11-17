# 单次启动改造

目标：一次 mixed AIC: AIV=1:2 kernel launch 内完成整个批量 SVD。host 只做参数检查、文件 I/O、分配、原始输入上传、启动、结果下载与计时。

采用原 PACO 的 CrossCore mode 0 全 AIV/全 AIC barrier 与 mode 2 簇内握手。每个阶段全体常驻核参加同步，即使没有分到任务也不能提前退出。AIC 数不得超过物理核数，避免驻留 barrier 死锁。

- [x] 在 NPU 验证中增加单次 launch/device 前后处理执行契约，确认旧实现失败。
- [x] 新增唯一入口 blocked_jacobi_fused，AIV device 归一化/pad、公式生成 tournament 配对、pack、Gram 参数、收敛、norm/sort/output；AIC 执行 Gram 和旋转 GEMM。
- [x] main.cpp 移除 host sweep、schedule、归一化与排序，仅调用一个 launch stub。repeat 是多次独立完整算子的基准测试，每次仍仅一次 launch。
- [x] NPU 小规模 smoke、30 组数值回归、固定 sweep/重复/缩减核数；msprof 检查一次执行只有一个 mixed kernel task。
- [x] 同步本地与 server 源码，重写文档并记录实测证据。

输出保留原 B_rot/sigma/Vt FP32 二进制。status 使用每核独立 32B 槽，所有收敛控制通过 GM->UB DMA 读取，避免 scalar cache 旧值。最后一个 sweep 状态和完成标志由 device 写出，host 验证，防止注册失败假成功。

实测结果见 [single-launch-results.md](single-launch-results.md)。
