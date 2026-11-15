# attic —— AIC/AIV 分工方案的实验记录

这里的文件**不参与编译**，记录"按 128 行分块 + AIV 算参数 / AIC 做旋转 + 簇内流水"
这一方案的验证过程与结论。

## 已确认可用的部分

1. **Cube 矩阵乘**（`gemm.hpp` 的 `MatmulCustom_01`）
   独立探针 `C = R * W`：256x128x256、128x128x128、256x512x256 三种形状误差为 0。

2. **簇内跨核同步协议**（`CrossCoreSetFlag/CrossCoreWaitFlag`）
   - 一个"核" = 1 AIC + 2 AIV；同步只在核内有效（AIV 侧 `blockIdx/2`，AIC 侧 `blockIdx` 即簇号）。
   - `CrossCoreSetFlag<0x2, PIPE_MTE3>(id)`（AIV 发）/ `CrossCoreWaitFlag(id)`（AIC 等），
     回程用另一个 id：`CrossCoreSetFlag<0x2, PIPE_FIX>(id)`（AIC 发）/ `CrossCoreWaitFlag(id)`（AIV 等）。
   - 探针：20 个簇各自独立跑 1/2/8/13 轮旋转，误差 9.5e-7，全部通过；
     用**两个不同 id** 分别做 ready/done 是必要的（共用一个 id 或全局同步都会挂死/出错）。

3. **完整簇式内核**（`jacobi_svd_cluster_impl.cpp`）
   已编译并在 910B3 上跑通（不挂死、每个核独立处理自己的矩阵、AIC 确实执行了矩阵乘），
   但**数值不收敛**，原因见下。

## 关键结论：每个块对只算一次 Gram 不收敛

原设计（也是原 `lu_custom.cpp` 的做法）是：对一个块对算一次 Gram `z`，
然后用这个 `z` 连续组装 b 轮（每轮 b/2 个 Givens）旋转。实测：

| 方式 | 1 轮 sweep 后 | 10 轮 sweep 后 |
| --- | --- | --- |
| 整个 sweep 复用同一份 Gram | 非对角 2.7e-1 / σ 误差 9.7 | 仍在 2.7e-1 / 9.5（不下降） |
| 每次旋转后解析更新 Gram (`G <- R G R^T`) | 1.4e-1 / 3.6 | 7.9e-7 / 7e-10（收敛） |

内核实测的非对角 2.796e-1、σ 误差 9.877 与 numpy 模拟（2.705e-1 / 9.672）一致，
说明内核实现是对的，**问题在算法**：旋转参数必须随已施加的旋转更新。

## 继续推进的三种做法

1. **块内旋转**：维护 b x b Gram（128x128 = 64KB，UB 放得下），每次旋转后解析更新
   `G <- R G R^T`（只动 i,j 行列）。这一半可以完全做对，且比纯 AIV 版本快约 10 倍。
2. **块对旋转**：参数需要 2b x 2b 的 Gram（256x256 = 256KB，UB 放不下）。
   可行路线：
   - 块对也按 64 行分块（128x128 Gram = 64KB，放得下）；
   - 或把 2b x 2b Gram 放 GM 逐轮读写（每轮 256KB，代价偏高）；
   - 或每个块对只做一轮（每轮重新算 Gram），用更多 sweep 换正确性（很慢）。
3. **回到纯 AIV 版本**（当前部署的 `jacobi_svd_custom.cpp`）：已验证，128x128 约
   29 ms/矩阵，批并行 40 核。
