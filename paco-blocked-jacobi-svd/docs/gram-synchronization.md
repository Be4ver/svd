# Gram 同步与 Axpy 旋转实测

2026-11-17，server，Ascend 910B3，CANN 8.2.RC1。
最终保留的计算改动只有 `FusedCube::Gram` 删除每个 128 列分段后的
`PipeBarrier<PIPE_ALL>()`。Prepare 的矢量化改动继续保留。

## 单核写入与流水依赖

每个 Gram 任务由一个 AIC 独占，不存在多个 AIC 对同一 Gram 的写入竞争。
原来逐段清空全部流水过于保守，不需要用这种屏障保护单核写入。
仍然保留以下数据依赖：

- TQue 的 EnQue/DeQue 将 MTE2→MTE1→M、M→FIX 串接起来。
- FreeTensor 的事件保护 L1、L0A/B、L0C 的复用，避免消费者读完前覆盖。
- 原 GEMM 对 `m*n < 2560` 的小 tile 保留 `PIPE_M` 同一累加器依赖。
- 阶段末尾 `DeviceBarrier` 经 `PIPE_FIX` 发布 Gram，AIV 等待后读取。

同一 AIC 的 Fixpipe 按发射顺序执行；首次覆盖写入、随后各段加到相同输出。
当前 `SetAtomicAdd` 用于完成**分段部分和的加法**，并非保护跨核写竞争。
直接删除它会把 Gram 变成最后一段的乘积。此次保留原有 128 列分段和
FP32 求和顺序，只移除逐段全流水屏障。

另试过整条 K 维在 L0C 连续累加、只 Fixpipe 一次：12 组小规模回归通过，
但目标 4096×4096/block64 在 30 sweep 后仍未收敛，原版为 25 sweep。
该方案同时改变 FP32 累加顺序，不能视为等价的屏障删减，因此未采用。
没有放宽 tol 或用未收敛时间报告加速。

## 两个 Muls、两个 Axpy

对一对行 p/q，数学上可以写成：

```cpp
Muls(newP, p, c, d);
Axpy(newP, q, -s, d);
Muls(newQ, q, c, d);
Axpy(newQ, p, s, d);
```

若原地更新，还需保存旧 p，最后复制 newP，防止 q 的计算读到新 p。
已实现并在 NPU 上试过这个版本。CANN 的 `Axpy` 接收标量系数，
而每个行对的 c/s 不同，因此无广播实现需要 V→S 等待、GetValue 读角度，
再逐行发射四条算术指令。block64 要遍历 64 对行，每条只处理 128 元素；
原版每个 tile 的指令处理 16 对行、2048 元素。四条算术指令是**每对行**
的计数，不能直接与原版**每批行对**的六条指令比较。

另试过两个 `Mul`、两个 `MulAddDst` 的批量版本：保留系数矢量化，
复用负 sin，block64 的列旋转通过 coefficient repeat stride=0 复用角度，
不广播列系数。这个版本也通过数值检查，但完整分解仍略慢。未对其具体
硬件瓶颈做 profiler 归因，不能仅凭指令条数判断吞吐。

因此默认仍使用原批量 `Mul/Sub/Add` 旋转；两个候选的源码和二进制保存在
本地、服务器的 `logs/gram-axpy/`，没有引入额外运行时调优开关。
AIV 只更新小 G/R，完整输入面板旋转 `R X` 仍由 AIC Cube GEMM 执行。

## 4096×4096 实测

同一 seed20261116 FP32 输入，batch1，tol=FP32 epsilon，local-sweeps1，
max-sweeps30，20 AIC+40 AIV，一次 mixed kernel launch 完成整个分解。
block64 为串行阶段，block32 默认分组 40+24；各版本分组保持一致。

| 实现 | block64 / ms | block32 / ms |
|---|---:|---:|
| 修改前 | 2208.357 | 2489.164 |
| **最终：只去 Gram 逐段全流水屏障** | **2028.787** | **2331.816** |
| 去屏障 + 逐行 Muls/Axpy，无角度广播 | 3307.306 | 4302.750 |
| 去屏障 + 批量 Mul/MulAddDst | 2117.847 | 2400.078 |

最终版本的时间分别减少 **8.13%** 和 **6.32%**。所有已收敛候选的实际
sweep 仍为 25/26，B_rot/sigma/Vt 在这个输入上均与修改前逐位一致。

修改前/只去屏障按 ABBA 顺序，每组 4 次、舍弃首轮，汇总每版本 6 个
device 时间取中位数。批量 MulAddDst 与只去屏障另做相同 ABBA 比较；
其中只去屏障参考值为 2027.912/2332.756 ms。逐行 Axpy 已显著变慢，
只执行两组、每组两次、舍弃首轮，汇总两个 device 时间取中位数。
各候选串行占用 NPU，输出目录复用，输入拷贝和文件读写不计入 device 时间。

`logs/gram-axpy/benchmark.py` 和各 `*benchmark.json` 保存命令、原始时间、
sweep、输出 SHA256；其余 `.log` 也全部位于 `logs/`。

## 回归范围

`tests/run_gram.py` 对 block16/32/64、K=31/129/257/4096、batch3、
全核及仅 2 AIV、三次完整调用进行独立 FP64 SVD 检查，覆盖尾块、
单个 AIC 多 Gram 任务和队列复用，同时检查单次 launch/device 前后处理与收敛。
数值、控制流、流水重叠回归与构建记录位于 `logs/gram-axpy/final-*.log`。

最终重新构建后：Gram 12/12、数值 31/31、block64 9/9、控制流 6/6、
流水/参数检查 11/11 全部通过。流水测试确认串行与重叠版本的输出逐位一致。
另有 CPU 测试 5/5 通过。静态复核未发现同步删减引入的依赖或生命周期问题。
