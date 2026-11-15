# SVD 算子下载与基准结果

生成时间：2026-11-15（本机时区 Asia/Shanghai）

## 下载目录

| 算子 | 服务器路径 | 本地目录 |
|---|---|---|
| paco-jacobi-svd | `server:/home/zhushengrui/project/paco-jacobi-svd` | `C:\Users\dell\Desktop\svd\paco-jacobi-svd` |
| qdwh-sdc-fused-jacobi | `server:/home/zhushengrui/project/svd/qdwh_sdc_fused_jacobi` | `C:\Users\dell\Desktop\svd\qdwh_sdc_fused_jacobi` |

本地目录包含顶层源码、CMake 文件和 Ascend 构建产物；本地 Windows 不能执行 Ascend NPU 二进制，计时来自服务器上的 Ascend 设备运行日志。

## 基准结果

### paco-jacobi-svd

- 配置：512 个 `32 x 32` 矩阵，AIV 核数 40，最大 sweep 30。
- 计时：**7.061 ms**。
- 计时口径：程序在 launch 前记录主机高精度时钟，launch 后同步 stream；因此覆盖算子执行及同步开销，不含输入上传和结果下载。
- 验证：512/512 个矩阵 PASS；最大奇异值绝对误差 `3.242e-05`，最差 B 行正交性 `9.345e-07`，最差 Vt 行正交性 `1.025e-06`。
- 日志：`paco-jacobi-svd/benchmark_32x32_batch512_run1/run.log`、`verify.log`。

### qdwh-sdc-fused-jacobi

- 配置：`1024 x 1024`，threshold=64，leaves=23，max_leaf=63，SDC groups=6。
- 计时：**51.599 ms kernel event**；host wait **51.800 ms**。
- 计时口径：Ascend `aclrtEventElapsedTime`，从 kernel launch 前 event 到完成 event，host wait 为同步后的主机等待时间。
- 验证：QDWH 与 SDC 均 PASS；`eig_err=2.367e-07`，`trace_err=5.047e-09`；端到端 sigma 最大绝对误差 `1.3838629e-06`，`pass=true`。
- 日志：`qdwh_sdc_fused_jacobi/benchmark_1024x1024_run1/kernel.log`、`verify_qdwh_sdc.log`、`end_to_end.json`。

## 备注

本轮后续重连时 `server` 的 `10.176.30.107:20016` 暂时无法建立 TCP 连接；上述结果是目录中已保存的服务器运行证据，未把 README 中的历史估计值替代实测日志值。
