#include "jacobi_svd_tiling.h"

#include <algorithm>
#include <iostream>

#include "acl/acl.h"

namespace jacobi_svd {

namespace {
constexpr int32_t kBlockRows = 64;    // 与内核中的 kB 一致
constexpr int32_t kColAlign = 32;     // 列按 32 对齐（cube 的 fractal 是 16，32 足够；AIV 只需 8）
constexpr int32_t kMaxCols = 256;     // 与内核中的 kMaxCols 一致
constexpr int32_t kMaxRows = 128;     // 单次处理的矩阵行数上限
}  // namespace

int32_t PadRows(int32_t rows)
{
    return ((rows + kBlockRows - 1) / kBlockRows) * kBlockRows;
}

int32_t PadCols(int32_t cols)
{
    return ((cols + kColAlign - 1) / kColAlign) * kColAlign;
}

bool ValidateProblem(const SvdProblem &problem, std::string &err)
{
    if (problem.rows <= 0 || problem.cols <= 0 || problem.batchSize <= 0) {
        err = "rows / cols / batchSize 必须为正整数";
        return false;
    }
    if (problem.rows > kMaxRows) {
        err = "rows 不能超过 " + std::to_string(kMaxRows);
        return false;
    }
    if (problem.cols > kMaxCols) {
        err = "cols 不能超过 " + std::to_string(kMaxCols);
        return false;
    }
    return true;
}

int32_t QueryVectorCoreNum(int32_t deviceId)
{
    int64_t value = 0;
    const aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId),
                                            ACL_DEV_ATTR_VECTOR_CORE_NUM, &value);
    if (ret != ACL_SUCCESS || value <= 0) {
        std::cerr << "[WARN] 查询向量核数失败(rc=" << ret << ")，默认使用 40" << std::endl;
        return 40;
    }
    return static_cast<int32_t>(value);
}

SvdTiling MakeTiling(const SvdProblem &problem, int32_t blockDim, int32_t rowBlock, int32_t maxSweeps)
{
    SvdTiling tiling;
    tiling.rank = std::min(problem.rows, problem.cols);
    tiling.rowsPad = PadRows(problem.rows);
    tiling.colsPad = PadCols(problem.cols);
    tiling.blockDim = (blockDim > 0) ? blockDim : QueryVectorCoreNum(0);
    tiling.rowBlock = std::max(rowBlock, 0);
    tiling.maxSweeps = (maxSweeps > 0) ? maxSweeps : 30;

    const size_t matElems = static_cast<size_t>(tiling.rowsPad) * static_cast<size_t>(tiling.colsPad);
    tiling.aBytes = matElems * static_cast<size_t>(problem.batchSize) * sizeof(float);
    tiling.bBytes = tiling.aBytes;
    tiling.sigmaBytes = static_cast<size_t>(tiling.rowsPad) * static_cast<size_t>(problem.batchSize) *
                        sizeof(float);
    return tiling;
}

}  // namespace jacobi_svd
