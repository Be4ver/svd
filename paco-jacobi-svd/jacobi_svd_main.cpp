#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_jacobi_svd_batched.h"

#include "data_utils.h"
#include "jacobi_svd_tiling.h"
#include "utils.h"

/**
 * 批量单边（块状）雅可比 SVD 主机侧驱动。
 *
 * 支持任意 rows <= 128、cols <= 256 的矩阵：主机侧把每个矩阵 pad 到
 * (PadRows(rows), PadCols(cols))（64 / 128 对齐，补 0）再送内核，
 * 内核按 pad 后的规模分块旋转；结果只取前 rows 行、前 cols 列。
 * 奇异值由 AIC 算出的 G = B·Bᵀ 的对角线开根得到。
 */

namespace {

struct RunConfig {
    int32_t deviceId = 0;
    jacobi_svd::SvdProblem problem;
    int32_t coreNum = 0;
    int32_t maxSweeps = 30;
    std::string inputFile = "../input/A_batch.bin";
    std::string outputDir = "../output";
};

void PrintUsage(const char *program)
{
    std::cout << "用法: " << program
              << " <deviceId> <rows> <cols> <batch> [coreNum] [maxSweeps]\n"
              << "  rows <= 128, cols <= 256；内部会 pad 到 64 / 128 对齐\n";
}

bool ParseArgs(int argc, char **argv, RunConfig &cfg)
{
    if (argc < 5) {
        PrintUsage(argv[0]);
        return false;
    }
    cfg.deviceId = std::atoi(argv[1]);
    cfg.problem.rows = std::atoi(argv[2]);
    cfg.problem.cols = std::atoi(argv[3]);
    cfg.problem.batchSize = std::atoi(argv[4]);
    if (argc > 5) cfg.coreNum = std::atoi(argv[5]);
    if (argc > 6) cfg.maxSweeps = std::atoi(argv[6]);
    // 负数 = 调试单测模式（-1 = AngleVector，-2 = 中间量），保持原值

    std::string err;
    if (!jacobi_svd::ValidateProblem(cfg.problem, err)) {
        std::cerr << "参数错误: " << err << std::endl;
        PrintUsage(argv[0]);
        return false;
    }
    return true;
}

/** 把 rows×cols 的矩阵按行拷进 rowsPad×colsPad 的缓冲（其余补 0） */
void PadBatch(const std::vector<float> &src, std::vector<float> &dst, int32_t rows, int32_t cols,
              int32_t rowsPad, int32_t colsPad, int32_t batch)
{
    std::fill(dst.begin(), dst.end(), 0.0f);
    for (int32_t m = 0; m < batch; ++m) {
        const float *s = src.data() + static_cast<size_t>(m) * rows * cols;
        float *d = dst.data() + static_cast<size_t>(m) * rowsPad * colsPad;
        for (int32_t r = 0; r < rows; ++r) {
            std::memcpy(d + static_cast<size_t>(r) * colsPad, s + static_cast<size_t>(r) * cols,
                        static_cast<size_t>(cols) * sizeof(float));
        }
    }
}

}  // namespace

int main(int argc, char **argv)
{
    RunConfig cfg;
    if (!ParseArgs(argc, argv, cfg)) return 1;

    const int32_t rows = cfg.problem.rows;
    const int32_t cols = cfg.problem.cols;
    const int32_t batch = cfg.problem.batchSize;
    const jacobi_svd::SvdTiling tiling =
        jacobi_svd::MakeTiling(cfg.problem, cfg.coreNum, 0, cfg.maxSweeps > 0 ? cfg.maxSweeps : 1);
    const int32_t rowsPad = tiling.rowsPad;
    const int32_t colsPad = tiling.colsPad;
    const int32_t clusterNum = tiling.blockDim / 2;

    std::cout << "==== 批量单边块状雅可比 SVD（AIV 旋转 + AIC 算 Gram）====" << std::endl;
    std::cout << "矩阵: " << rows << " x " << cols << " -> pad " << rowsPad << " x " << colsPad
              << ", 批大小: " << batch << ", AIV 核数: " << tiling.blockDim
              << ", 最大 sweep: " << tiling.maxSweeps << std::endl;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(cfg.deviceId));
    aclrtContext context = nullptr;
    CHECK_ACL(aclrtCreateContext(&context, cfg.deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const size_t trueBytes = static_cast<size_t>(rows) * cols * batch * sizeof(float);
    std::vector<float> aTrue(static_cast<size_t>(rows) * cols * batch);
    size_t readSize = 0;
    if (!ReadFile(cfg.inputFile, readSize, aTrue.data(), trueBytes) || readSize != trueBytes) {
        std::cerr << "读取输入失败或大小不符: " << cfg.inputFile << " (" << readSize
                  << " != " << trueBytes << ")" << std::endl;
        return 1;
    }
    std::cout << "输入: " << cfg.inputFile << " (" << readSize << " 字节)" << std::endl;

    std::vector<float> aHost(static_cast<size_t>(rowsPad) * colsPad * batch);
    PadBatch(aTrue, aHost, rows, cols, rowsPad, colsPad, batch);

    void *aRawDevice = nullptr;
    void *aDevice = nullptr;
    void *bDevice = nullptr;
    void *gDevice = nullptr;
    void *sigmaDevice = nullptr;
    void *flagDevice = nullptr;
    // flags 布局：[0, batch*8) = 旋转完成；[batch*8, 2*batch*8) = Gram 完成
    const size_t flagBytes = (static_cast<size_t>(std::max<int32_t>(batch * 16 + 64, 1024))) * sizeof(float);
    const size_t gBytes = static_cast<size_t>(rowsPad) * rowsPad * batch * sizeof(float);
    CHECK_ACL(aclrtMalloc(&aRawDevice, trueBytes + 256, ACL_MEM_MALLOC_HUGE_FIRST));  // +slack：pad 行可能多读 8 个 float
    CHECK_ACL(aclrtMalloc(&aDevice, tiling.aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&bDevice, tiling.bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&gDevice, gBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&sigmaDevice, tiling.sigmaBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&flagDevice, flagBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemsetAsync(flagDevice, flagBytes, 0, flagBytes, stream));
    // 只上传真实形状的原始数据；pad 缓冲在 device 上清零后由内核填充
    CHECK_ACL(aclrtMemcpy(aRawDevice, trueBytes, aTrue.data(), trueBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemsetAsync(aDevice, tiling.aBytes, 0, tiling.aBytes, stream));

    if (cfg.maxSweeps == -2) {
        // 单测输入：[a(8) | b(8) | c(8)]，其中 a/b/c 各前 4 个有效
        std::vector<float> t(24, 0.0f);
        const float av[4] = {4.0f, 1.0f, 3.0f, 2.0f};
        const float bv[4] = {1.0f, 9.0f, 3.0f, 2.0f};
        const float cv[4] = {0.5f, -2.0f, 0.0f, 1.5f};
        for (int32_t k = 0; k < 4; ++k) { t[k] = av[k]; t[8 + k] = bv[k]; t[16 + k] = cv[k]; }
        CHECK_ACL(aclrtMemcpy(bDevice, 24 * sizeof(float), t.data(), 24 * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    int32_t stages = 3;
    if (argc > 7) stages = std::atoi(argv[7]);

    const auto start = std::chrono::high_resolution_clock::now();
    const aclError launchRet = aclrtlaunch_jacobi_svd_batched(
        static_cast<uint32_t>(clusterNum), stream, rows, cols, batch, cfg.maxSweeps,
        tiling.blockDim, clusterNum, aRawDevice, aDevice, bDevice, gDevice, sigmaDevice, flagDevice, stages);
    CHECK_ACL(launchRet);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMs =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::vector<float> bHost(static_cast<size_t>(rowsPad) * colsPad * batch);
    std::vector<float> sigDev(static_cast<size_t>(rowsPad) * batch);
    CHECK_ACL(aclrtMemcpy(bHost.data(), tiling.bBytes, bDevice, tiling.bBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    // 奇异值已由 AIV 在设备侧从 G 的对角线开根写入 sigma，只回传 rows*batch 个 float
    CHECK_ACL(aclrtMemcpy(sigDev.data(), tiling.sigmaBytes, sigmaDevice, tiling.sigmaBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    // 调试：dump flags 缓冲前 24 个 float（ls / tau / s）
    {
        std::vector<float> dbg(24);
        CHECK_ACL(aclrtMemcpy(dbg.data(), 24 * sizeof(float), flagDevice, 24 * sizeof(float),
                              ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(cfg.outputDir + "/dbg.bin", dbg.data(), dbg.size() * sizeof(float));
    }
    std::vector<float> rowNorms(static_cast<size_t>(rows) * batch);
    for (int32_t m = 0; m < batch; ++m) {
        for (int32_t i = 0; i < rows; ++i) {
            rowNorms[static_cast<size_t>(m) * rows + i] = sigDev[static_cast<size_t>(m) * rowsPad + i];
        }
    }

    // 排序 + V^T（只取前 cols 列）
    std::vector<float> sigmaHost(static_cast<size_t>(tiling.rank) * batch);
    std::vector<float> vtHost(static_cast<size_t>(tiling.rank) * cols * batch);
    std::vector<int32_t> order(static_cast<size_t>(rows));
    for (int32_t m = 0; m < batch; ++m) {
        std::iota(order.begin(), order.end(), 0);
        const float *nrm = rowNorms.data() + static_cast<size_t>(m) * rows;
        std::sort(order.begin(), order.end(),
                  [nrm](int32_t lhs, int32_t rhs) { return nrm[lhs] > nrm[rhs]; });
        const float sigmaMax = nrm[order[0]];
        const float eps = 1e-6f * std::max(sigmaMax, 1e-30f);
        const float *bMat = bHost.data() + static_cast<size_t>(m) * rowsPad * colsPad;
        for (int32_t k = 0; k < tiling.rank; ++k) {
            const int32_t srcRow = order[static_cast<size_t>(k)];
            const float s = nrm[srcRow];
            sigmaHost[static_cast<size_t>(m) * tiling.rank + k] = s;
            float *dst = vtHost.data() + (static_cast<size_t>(m) * tiling.rank + k) * cols;
            const float *src = bMat + static_cast<size_t>(srcRow) * colsPad;
            if (s > eps) {
                const float inv = 1.0f / s;
                for (int32_t c = 0; c < cols; ++c) dst[c] = src[c] * inv;
            } else {
                std::memset(dst, 0, static_cast<size_t>(cols) * sizeof(float));
            }
        }
    }

    // 真实形状的 B（前 rows 行、前 cols 列），供验证脚本做正交性/重构检查
    std::vector<float> bTrue(static_cast<size_t>(rows) * cols * batch);
    for (int32_t m = 0; m < batch; ++m) {
        const float *bMat = bHost.data() + static_cast<size_t>(m) * rowsPad * colsPad;
        float *dst = bTrue.data() + static_cast<size_t>(m) * rows * cols;
        for (int32_t r = 0; r < rows; ++r) {
            std::memcpy(dst + static_cast<size_t>(r) * cols, bMat + static_cast<size_t>(r) * colsPad,
                        static_cast<size_t>(cols) * sizeof(float));
        }
    }
    WriteFile(cfg.outputDir + "/B_rot.bin", bTrue.data(), bTrue.size() * sizeof(float));
    WriteFile(cfg.outputDir + "/sigma.bin", sigmaHost.data(), sigmaHost.size() * sizeof(float));
    WriteFile(cfg.outputDir + "/Vt.bin", vtHost.data(), vtHost.size() * sizeof(float));

    std::cout << "耗时: " << elapsedMs << " ms (batch=" << batch << ", " << rows << "x" << cols
              << ")" << std::endl;
    std::cout << "[奇异值前 5 个] 矩阵 0: ";
    for (int32_t k = 0; k < std::min<int32_t>(5, tiling.rank); ++k) {
        std::cout << sigmaHost[static_cast<size_t>(k)] << " ";
    }
    std::cout << std::endl;

    CHECK_ACL(aclrtFree(aRawDevice));
    CHECK_ACL(aclrtFree(aDevice));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFree(gDevice));
    CHECK_ACL(aclrtFree(sigmaDevice));
    CHECK_ACL(aclrtFree(flagDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(cfg.deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
