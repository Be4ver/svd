# QDWH-SDC-Fused-Jacobi 方阵基准（2026-11-15）

测试在 `server` 的 Ascend 设备上完成，统一配置为 `blockDim=20`、`threshold=64`、`maxIter=4`、`cond=1e2`。时间取设备 `kernel_event`，同时记录 host wait。

| 方阵 | kernel 时间 | host wait | leaves | 验证 |
|---:|---:|---:|---:|:---:|
| 128×128 | **4.392 ms** | 4.621 ms | 3 | PASS |
| 512×512 | **20.540 ms** | 20.835 ms | 12 | PASS |
| 1024×1024 | **46.839 ms** | 47.185 ms | 23 | PASS |
| 2048×2048 | **120.302 ms** | 120.685 ms | 46 | PASS |

每个尺寸的运行日志和验证日志位于 [`qdwh_sdc_fused_jacobi/benchmark_square_sizes_20261115/`](C:/Users/dell/Desktop/svd/qdwh_sdc_fused_jacobi/benchmark_square_sizes_20261115/)。
