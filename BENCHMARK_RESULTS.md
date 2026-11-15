# 算子下载与基准结果

本目录包含从 `server` 下载的两个算子：

- `paco-jacobi-svd/`
- `qdwh_sdc_fused_jacobi/`（服务器目录名为 `qdwh_sdc_fused_jacobi`）

## 基准条件

- 第一算子：512 个 32×32 矩阵，设备端批量 Jacobi SVD。
- 第二算子：单个 1024×1024 矩阵，融合 QDWH + SDC/Jacobi SVD。
- 时间均为 Ascend 设备上的算子执行时间；第一项是 launch + stream synchronize，第二项同时记录设备 event 和 host wait。
- 每次结果均通过对应的奇异值/正交性验证。

## 结果

| 算子 | 输入 | 时间 | 验证 |
|---|---:|---:|---|
| paco-jacobi-svd | 512 × (32×32) | **7.061 ms** | 512/512 PASS |
| qdwh-sdc-fused-jacobi | 1 × (1024×1024) | **51.599 ms kernel event**；51.800 ms host wait | PASS；23 个 leaves，最大叶块 63 |

第二项的 host wait 只比 kernel event 多约 0.201 ms；算子计算时间取设备 event 的 51.599 ms。

详细日志：

- [第一算子运行日志](paco-jacobi-svd/benchmark_32x32_batch512_run1/run.log)
- [第一算子验证日志](paco-jacobi-svd/benchmark_32x32_batch512_run1/verify.log)
- [第二算子 kernel 日志](qdwh_sdc_fused_jacobi/benchmark_1024x1024_run1/kernel.log)
- [第二算子奇异值验证](qdwh_sdc_fused_jacobi/benchmark_1024x1024_run1/leaf_sigma.log)
- [第二算子 QDWH/SDC 验证](qdwh_sdc_fused_jacobi/benchmark_1024x1024_run1/verify_qdwh_sdc.log)

复核结束后再次连接 `server` 时 SSH 暂时超时，但下载内容和已保存的测试日志均完整保留。

