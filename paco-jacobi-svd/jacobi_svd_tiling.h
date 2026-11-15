#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace jacobi_svd {

/** 求解规模：批大小 + 单个矩阵的行列数（真实规模，未 pad） */
struct SvdProblem {
    int32_t rows = 0;        // M
    int32_t cols = 0;        // N
    int32_t batchSize = 1;   // 批大小
};

/** 主机侧为内核准备的 tiling 参数 */
struct SvdTiling {
    int32_t blockDim = 1;      // 参与计算的 AIV 核数
    int32_t rowBlock = 0;      // 主块行数，0 表示由内核按 UB 容量自动选择
    int32_t maxSweeps = 30;    // 最大扫描次数（收敛后内核会提前退出）
    int32_t rank = 0;          // min(rows, cols)
    int32_t rowsPad = 0;       // pad 后的行数（64 对齐）
    int32_t colsPad = 0;       // pad 后的列数（128 对齐）
    size_t aBytes = 0;         // 输入矩阵字节数（pad 后）
    size_t bBytes = 0;         // 旋转后矩阵字节数（pad 后）
    size_t sigmaBytes = 0;     // 行范数（未排序）字节数（pad 后）
};

/** 校验矩阵规模是否落在此实现的约束内；不合法时写入 err 并返回 false。 */
bool ValidateProblem(const SvdProblem &problem, std::string &err);

/** 查询设备的向量核（AIV）个数；查询失败返回 0。 */
int32_t QueryVectorCoreNum(int32_t deviceId);

/** 生成 tiling 参数；blockDim <= 0 时自动取设备向量核数。 */
SvdTiling MakeTiling(const SvdProblem &problem, int32_t blockDim, int32_t rowBlock, int32_t maxSweeps);

/** 行/列 pad 到 64 / 128 对齐（与内核的分块和搬运对齐要求一致）。 */
int32_t PadRows(int32_t rows);
int32_t PadCols(int32_t cols);

}  // namespace jacobi_svd
