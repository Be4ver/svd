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

/** 批量单边（块状）雅可比 SVD 主机侧驱动：AIV 完成旋转，AIC 用转置乘算 Gram */

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
              << "  rows, cols : 单个矩阵的行数、列数（rows 为 64 的倍数，cols 为 8 的倍数且 <= 256）\n"
              << "  batch      : 批大小\n"
              << "  coreNum    : 参与计算的 AIV 核数，缺省为设备向量核数\n"
              << "  maxSweeps  : 最大扫描次数，缺省 30\n";
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

    std::string err;
    if (!jacobi_svd::ValidateProblem(cfg.problem, err)) {
        std::cerr << "参数错误: " << err << std::endl;
        PrintUsage(argv[0]);
        return false;
    }
    return true;
}

void PostProcess(const std::vector<float> &rotated, const std::vector<float> &rowNorms,
                 int32_t rows, int32_t cols, int32_t rank,
                 std::vector<float> &sigmaSorted, std::vector<float> &vt)
{
    std::vector<int32_t> order(static_cast<size_t>(rows));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&rowNorms](int32_t lhs, int32_t rhs) {
        return rowNorms[static_cast<size_t>(lhs)] > rowNorms[static_cast<size_t>(rhs)];
    });
    for (int32_t k = 0; k < rank; ++k) {
        sigmaSorted[static_cast<size_t>(k)] = rowNorms[static_cast<size_t>(order[static_cast<size_t>(k)])];
    }
    const float sigmaMax = (rank > 0) ? sigmaSorted[0] : 0.0f;
    const float eps = 1e-6f * std::max(sigmaMax, 1e-30f);
    for (int32_t k = 0; k < rank; ++k) {
        const float s = sigmaSorted[static_cast<size_t>(k)];
        const int32_t srcRow = order[static_cast<size_t>(k)];
        float *dst = vt.data() + static_cast<size_t>(k) * cols;
        const float *src = rotated.data() + static_cast<size_t>(srcRow) * cols;
        if (s > eps) {
            const float inv = 1.0f / s;
            for (int32_t c = 0; c < cols; ++c) dst[c] = src[c] * inv;
        } else {
            std::memset(dst, 0, static_cast<size_t>(cols) * sizeof(float));
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
        jacobi_svd::MakeTiling(cfg.problem, cfg.coreNum, 0, cfg.maxSweeps);
    const int32_t clusterNum = tiling.blockDim / 2;

    std::cout << "==== 批量单边块状雅可比 SVD（AIV 全部旋转 + AIC 算 Gram）====" << std::endl;
    std::cout << "矩阵: " << rows << " x " << cols << ", 批大小: " << batch
              << ", AIV 核数: " << tiling.blockDim << ", 最大 sweep: " << tiling.maxSweeps << std::endl;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(cfg.deviceId));
    aclrtContext context = nullptr;
    CHECK_ACL(aclrtCreateContext(&context, cfg.deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    std::vector<float> aHost(static_cast<size_t>(rows) * cols * batch);
    size_t readSize = 0;
    if (!ReadFile(cfg.inputFile, readSize, aHost.data(), tiling.aBytes) ||
        readSize != tiling.aBytes) {
        std::cerr << "读取输入失败或大小不符: " << cfg.inputFile << std::endl;
        return 1;
    }
    std::cout << "输入: " << cfg.inputFile << " (" << readSize << " 字节)" << std::endl;

    void *aDevice = nullptr;
    void *bDevice = nullptr;
    void *gDevice = nullptr;
    void *flagDevice = nullptr;
    const size_t flagBytes = static_cast<size_t>(tiling.maxSweeps) * 64 * sizeof(float);
    const size_t gBytes = static_cast<size_t>(rows) * rows * batch * sizeof(float);
    CHECK_ACL(aclrtMalloc(&aDevice, tiling.aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&bDevice, tiling.bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&gDevice, gBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&flagDevice, flagBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(flagDevice, flagBytes, 0, flagBytes));
    CHECK_ACL(aclrtMemcpy(aDevice, tiling.aBytes, aHost.data(), tiling.aBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    const auto start = std::chrono::high_resolution_clock::now();
    const aclError launchRet = aclrtlaunch_jacobi_svd_batched(
        static_cast<uint32_t>(clusterNum), stream, rows, cols, batch, tiling.maxSweeps,
        tiling.blockDim, clusterNum, aDevice, bDevice, gDevice, flagDevice);
    CHECK_ACL(launchRet);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMs =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::vector<float> bHost(static_cast<size_t>(rows) * cols * batch);
    std::vector<float> rowNorms(static_cast<size_t>(rows) * batch);
    CHECK_ACL(aclrtMemcpy(bHost.data(), tiling.bBytes, bDevice, tiling.bBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> gHost(static_cast<size_t>(rows) * rows * batch);
    CHECK_ACL(aclrtMemcpy(gHost.data(), gBytes, gDevice, gBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int32_t m = 0; m < batch; ++m) {
        for (int32_t i = 0; i < rows; ++i) {
            const size_t idx = (static_cast<size_t>(m) * rows + i) * rows + i;
            rowNorms[static_cast<size_t>(m) * rows + i] = std::sqrt(std::max(gHost[idx], 0.0f));
        }
    }

    std::vector<float> sigmaHost(static_cast<size_t>(tiling.rank) * batch);
    std::vector<float> vtHost(static_cast<size_t>(tiling.rank) * cols * batch);
    for (int32_t m = 0; m < batch; ++m) {
        const size_t matOff = static_cast<size_t>(m) * rows * cols;
        const size_t normOff = static_cast<size_t>(m) * rows;
        const size_t outOff = static_cast<size_t>(m) * tiling.rank;
        std::vector<float> rotated(bHost.begin() + matOff, bHost.begin() + matOff +
                                   static_cast<size_t>(rows) * cols);
        std::vector<float> norms(rowNorms.begin() + normOff, rowNorms.begin() + normOff + rows);
        std::vector<float> sig(static_cast<size_t>(tiling.rank));
        std::vector<float> vt(static_cast<size_t>(tiling.rank) * cols);
        PostProcess(rotated, norms, rows, cols, tiling.rank, sig, vt);
        std::copy(sig.begin(), sig.end(), sigmaHost.begin() + outOff);
        std::copy(vt.begin(), vt.end(), vtHost.begin() + static_cast<size_t>(m) * tiling.rank * cols);
    }

    WriteFile(cfg.outputDir + "/sigma.bin", sigmaHost.data(), sigmaHost.size() * sizeof(float));
    WriteFile(cfg.outputDir + "/Vt.bin", vtHost.data(), vtHost.size() * sizeof(float));
    WriteFile(cfg.outputDir + "/B_rot.bin", bHost.data(), bHost.size() * sizeof(float));

    std::cout << "耗时: " << elapsedMs << " ms (batch=" << batch << ", " << rows << "x" << cols
              << ")" << std::endl;
    std::cout << "[奇异值前 5 个] 矩阵 0: ";
    for (int32_t k = 0; k < std::min<int32_t>(5, tiling.rank); ++k) {
        std::cout << sigmaHost[static_cast<size_t>(k)] << " ";
    }
    std::cout << std::endl;

    CHECK_ACL(aclrtFree(aDevice));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFree(gDevice));
    CHECK_ACL(aclrtFree(flagDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(cfg.deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
