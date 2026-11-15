#include "aclrtlaunch_qdwh_sdc_fused_custom.h"
#include "utils.h"
#include "fused_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK_ACL_FATAL(x) do { \
    aclError __ret = (x); \
    if (__ret != ACL_SUCCESS) { \
        std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        exit(1); \
    } \
} while (0)

// 与 kernel 侧一致（QdwhSdcFused::SdcCube 的容量检查）
static constexpr int kDevMaxNodes = 16384;
static constexpr int kDevMaxLeaves = 16384;

struct XferStats {
    int h2d = 0;
    int d2h = 0;
    size_t h2dBytes = 0;
    size_t d2hBytes = 0;
};

static std::string OutPath(const char *name)
{
    std::ifstream probe("../input/shapes.txt");
    const char *first = probe.good() ? "../output/" : "output/";
    return std::string(first) + name;
}

// 调试：下载 dbg 区（128 个 int32）并打印；kernel 异常时也能看到最后进度
static void DumpDbg(void *dbgDev)
{
    int buf[128];
    std::memset(buf, 0xcc, sizeof(buf));
    aclError e = aclrtMemcpy(buf, sizeof(buf), dbgDev, sizeof(buf), ACL_MEMCPY_DEVICE_TO_HOST);
    fprintf(stderr, "[fused] dbg dump (aclError=%d):\n", (int)e);
    for (int i = 0; i < 128; i += 8) {
        fprintf(stderr, "  [%3d] %10d %10d %10d %10d %10d %10d %10d %10d\n", i,
                buf[i], buf[i + 1], buf[i + 2], buf[i + 3],
                buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]);
    }
    if (e == ACL_SUCCESS) {
        float cProd = 0.0f;
        std::memcpy(&cProd, &buf[112], sizeof(cProd));
        fprintf(stderr, "[fused] phase1 cProd[0]=%.9f\n", cProd);
    }
}

static bool ReadShapes(const char *path, std::vector<int> &M, std::vector<int> &N)
{
    FILE *fp = fopen(path, "r");
    if (fp == nullptr) {
        return false;
    }
    char line[256];
    while (fgets(line, (int)sizeof(line), fp) != nullptr) {
        char *p = line;
        while (*p != '\0' && *p != '#') {
            if (*p == ',') *p = ' ';
            ++p;
        }
        *p = '\0';
        int m = 0, n = 0;
        if (sscanf(line, "%d %d", &m, &n) == 2 && m > 0 && n > 0) {
            M.push_back(m);
            N.push_back(n);
        }
    }
    fclose(fp);
    return !M.empty();
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr,
                "用法: %s <deviceId> <blockDim> <threshold> [maxIter=4] [alpha=1.0] [l0=1e-6] [--no-hu]\n"
                "说明: 融合 kernel = QDWH 极分解 + device 内 SDC 递归。\n"
                "      host 只在开始时上传 A 一次，launch 一次，结束时下载 leaves\n"
                "      （--no-hu 时不再下载 H/U，默认下载用于 verify.py）。\n",
                argv[0]);
        return 1;
    }
    const int deviceId = std::atoi(argv[1]);
    const int blockDim = std::atoi(argv[2]);
    const int threshold = std::atoi(argv[3]);
    // maxIter 默认 4：l0=1e-6 起 l 递推在第 4 轮就到 1，
    // 实测 4 轮比 15 轮还准一点（u_err 7.6e-7 vs 1.3e-6），耗时只有 ~40%。
    const int maxIter = (argc > 4) ? std::atoi(argv[4]) : 4;
    const double alpha = (argc > 5) ? std::atof(argv[5]) : 1.0;
    const double l0 = (argc > 6) ? std::atof(argv[6]) : 1e-6;
    bool dumpHU = true;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-hu") == 0) dumpHU = false;
    }

    if (blockDim < 16 || blockDim > 64 || (blockDim % 2) != 0) {
        fprintf(stderr, "[fused] 错误: 16 <= blockDim <= 64 且为偶数（当前 %d）\n", blockDim);
        return 1;
    }
    if (threshold < 16 || threshold > 64) {
        fprintf(stderr, "[fused] 错误: 融合 Jacobi 仅支持 16 <= threshold <= 64（当前 %d）\n", threshold);
        return 1;
    }
    if (maxIter <= 0 || maxIter > 64) {
        fprintf(stderr, "[fused] 错误: 1 <= max_iter <= 64（当前 %d）\n", maxIter);
        return 1;
    }
    if (!(alpha > 0.0) || !(l0 > 0.0) || l0 > 1.0) {
        fprintf(stderr, "[fused] 错误: alpha>0、0<l0<=1（当前 alpha=%g, l0=%g）\n", alpha, l0);
        return 1;
    }

    // ------------------------------------------------------------------
    // shapes（融合 kernel 目前按单矩阵设计：batch=1）
    // ------------------------------------------------------------------
    const char *shapeCands[3] = {"../input/shapes.txt", "input/shapes.txt", "./input/shapes.txt"};
    std::vector<int> Ms, Ns;
    const char *shapePath = nullptr;
    for (int i = 0; i < 3; ++i) {
        if (ReadShapes(shapeCands[i], Ms, Ns)) {
            shapePath = shapeCands[i];
            break;
        }
    }
    if (shapePath == nullptr) {
        fprintf(stderr, "[fused] 错误: 找不到/读不全 shapes.txt\n");
        return 1;
    }
    if (Ms.size() != 1) {
        fprintf(stderr, "[fused] 错误: shapes.txt 有 %zu 组 shape；融合 kernel 目前只支持单矩阵\n",
                Ms.size());
        return 1;
    }
    const int m = Ms[0];
    const int n = Ns[0];
    if (m <= 0 || n <= 0 || m > 8192 || n > 8192) {
        fprintf(stderr, "[fused] 错误: shape=(%d,%d) 非法（要求 1 <= M,N <= 8192）\n", m, n);
        return 1;
    }
    // 任意形状：行列分别对齐到 128，不一定相等；若 m<n 先转置，使 kernel 里
    // 行数 >= 列数。QDWH 的 H 是 nPad x nPad 的半正定矩阵，边长 = min 维的
    // 128 对齐值；SDC 只取原始 min 维 cOrig，叶子总和 = cOrig²。
    const int rOrig = (m > n) ? m : n;
    const int cOrig = (m > n) ? n : m;
    const int mPad = (rOrig + 127) / 128 * 128;
    const int nPad = (cOrig + 127) / 128 * 128;
    const bool transposed = (m < n);
    if (mPad > 8192 || nPad > 8192) {
        fprintf(stderr, "[fused] 错误: padded 尺寸 %d x %d 超过 8192\n", mPad, nPad);
        return 1;
    }
    if (mPad + nPad > kQrMaxQd) {
        fprintf(stderr, "[fused] 错误: padded m+n=%d 超过 QDWH/QR 点积区容量 %d"
                        "（当前 %d x %d）\n", mPad + nPad, kQrMaxQd, mPad, nPad);
        return 1;
    }
    fprintf(stderr, "[fused] shapes: %s -> %d x %d（%spad 到 %d x %d，H 边长 %d）\n",
            shapePath, m, n, transposed ? "转置后 " : "", mPad, nPad, cOrig);

    // ------------------------------------------------------------------
    // workspace 布局（host/device 共用 FusedLayout::Build）
    // ------------------------------------------------------------------
    const int leafCap = 2 * cOrig * cOrig;   // 叶子平方和恰为 cOrig²，2 倍留余量
    FusedLayout L;
    L.Build(mPad, nPad, kDevMaxNodes, kDevMaxLeaves, leafCap);
    const size_t totalBytes = (size_t)L.total * sizeof(float);
    fprintf(stderr,
            "[fused] workspace: %.1f MB (A=%.1f H=%.1f Q=%.1f arena=%.1f leaf=%.1f MB)\n",
            (double)totalBytes / 1048576.0,
            (double)((long)mPad * nPad * 4) / 1048576.0,
            (double)((long)cOrig * cOrig * 4) / 1048576.0,
            (double)(L.qbuf * 4) / 1048576.0,
            (double)(L.arenaFloats * 4) / 1048576.0,
            (double)((long)leafCap * 4) / 1048576.0);

    // ------------------------------------------------------------------
    // host 输入：A / Ω1 / Ω2 / eye
    // ------------------------------------------------------------------
    std::vector<float> aHost((size_t)m * n);
    {
        const char *matCands[3] = {"../input/matrix.bin", "input/matrix.bin", "./input/matrix.bin"};
        const char *matPath = nullptr;
        for (int i = 0; i < 3; ++i) {
            FILE *fp = fopen(matCands[i], "rb");
            if (fp != nullptr) { fclose(fp); matPath = matCands[i]; break; }
        }
        if (matPath == nullptr) {
            fprintf(stderr, "[fused] 错误: 找不到 matrix.bin\n");
            return 1;
        }
        try {
            ReadFile(matPath, aHost.data(), aHost.size() * sizeof(float));
        } catch (const std::exception &e) {
            fprintf(stderr, "[fused] 读取 matrix.bin 失败: %s\n", e.what());
            return 1;
        }
        fprintf(stderr, "[fused] matrix: %s (%zu float)\n", matPath, aHost.size());
    }
    // 补零到 mPad x nPad：m>=n 时左上 m x n 为 A；m<n 时放 A^T（左上 n x m）
    std::vector<float> aPad((size_t)mPad * nPad, 0.0f);
    if (!transposed) {
        for (int r = 0; r < m; ++r) {
            std::memcpy(aPad.data() + (size_t)r * nPad, aHost.data() + (size_t)r * n,
                        (size_t)n * sizeof(float));
        }
    } else {
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < m; ++c) {
                aPad[(size_t)r * nPad + c] = aHost[(size_t)c * n + r];
            }
        }
    }

    // Ω1/Ω2：与 sdc_dev_main 相同的 LCG + Box-Muller（保证可复现）
    std::vector<float> om1((size_t)nPad * nPad), om2((size_t)nPad * nPad);
    {
        unsigned int seed = 20261026u;
        auto frand = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return (float)((seed >> 8) * (1.0 / 16777216.0));
        };
        for (size_t i = 0; i < om1.size(); ++i) {
            float u1 = std::max(frand(), 1e-7f);
            float u2 = frand();
            float rr = std::sqrt(-2.0f * std::log(u1));
            om1[i] = rr * std::cos(6.2831853f * u2);
            om2[i] = rr * std::sin(6.2831853f * u2);
        }
    }
    std::vector<float> eyeHost(128 * 256, 0.0f);
    for (int i = 0; i < 128; ++i) eyeHost[i * 128 + i] = 1.0f;
    // 完整的 n x n 单位阵：device 侧用一次 GEMM（Q = I*I）生成 QR 的 Q 初值
    // n x n 单位阵：device 侧用一次 GEMM 生成 QR 的 Q 初值
    const long eyeSide = (long)nPad;
    std::vector<float> eyeNHost((size_t)eyeSide * eyeSide, 0.0f);
    for (long i = 0; i < eyeSide; ++i) eyeNHost[(size_t)i * eyeSide + i] = 1.0f;
    // shapes: pQdwh = (nPad, mPad+nPad)、pQr = (nPad, nPad)
    std::vector<int32_t> shapeHost(8, 0);
    shapeHost[0] = nPad;
    shapeHost[1] = mPad + nPad;
    shapeHost[2] = nPad;
    shapeHost[3] = nPad;

    // ------------------------------------------------------------------
    // device
    // ------------------------------------------------------------------
    CHECK_ACL_FATAL(aclInit(nullptr));
    aclrtContext context;
    CHECK_ACL_FATAL(aclrtSetDevice(deviceId));
    int64_t aicCount = 0, aivCount = 0;
    CHECK_ACL_FATAL(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicCount));
    CHECK_ACL_FATAL(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aivCount));
    // Global cross-core barriers require every participating MIX block to
    // remain resident: blockDim counts AICs, and each AIC has two AIVs.
    if (blockDim > aicCount || 2LL * blockDim > aivCount) {
        fprintf(stderr, "[fused] blockDim=%d exceeds resident MIX capacity: %lld AIC / %lld AIV\n",
                blockDim, (long long)aicCount, (long long)aivCount);
        CHECK_ACL_FATAL(aclrtResetDevice(deviceId));
        CHECK_ACL_FATAL(aclFinalize());
        return 1;
    }
    CHECK_ACL_FATAL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL_FATAL(aclrtCreateStream(&stream));

    XferStats xfer;
    void *wkDev = nullptr;
    CHECK_ACL_FATAL(aclrtMalloc(&wkDev, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    auto dev = [&wkDev](long off) {
        return static_cast<void *>(static_cast<char *>(wkDev) + (size_t)off * sizeof(float));
    };
    auto h2d = [&](long off, const void *src, size_t bytes) {
        CHECK_ACL_FATAL(aclrtMemcpy(dev(off), bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        xfer.h2d += 1;
        xfer.h2dBytes += bytes;
    };

    // --- 上传（计算前一次性完成；计算期间不再有 h2d/d2h）---
    h2d(L.a, aPad.data(), aPad.size() * sizeof(float));              // A_pad：唯一的一次 A 上传
    CHECK_ACL_FATAL(aclrtMemcpy(dev(L.a0), aPad.size() * sizeof(float), dev(L.a),
                                aPad.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE));
    h2d(L.om1, om1.data(), om1.size() * sizeof(float));
    h2d(L.om2, om2.data(), om2.size() * sizeof(float));
    h2d(L.eye, eyeHost.data(), eyeHost.size() * sizeof(float));
    h2d(L.eyeN, eyeNHost.data(), eyeNHost.size() * sizeof(float));
    h2d(L.shapes, shapeHost.data(), shapeHost.size() * sizeof(int32_t));
    {
        std::vector<int32_t> cntZero(8, 0);
        h2d(L.cnt, cntZero.data(), cntZero.size() * sizeof(int32_t));
    }
    {
        // 元数据数组清零，避免 device 读到未初始化的节点/叶表项
        std::vector<int32_t> z(kDevMaxNodes, 0);
        h2d(L.nodeH, z.data(), z.size() * sizeof(int32_t));
        h2d(L.nodeN, z.data(), z.size() * sizeof(int32_t));
        h2d(L.nodeLd, z.data(), z.size() * sizeof(int32_t));
        std::vector<int32_t> zl(kDevMaxLeaves, 0);
        h2d(L.leafN, zl.data(), zl.size() * sizeof(int32_t));
        h2d(L.leafLd, zl.data(), zl.size() * sizeof(int32_t));
        h2d(L.leafOff, zl.data(), zl.size() * sizeof(int32_t));
        h2d(L.sigmaOff, zl.data(), zl.size() * sizeof(int32_t));
        h2d(L.sigmaCount, zl.data(), zl.size() * sizeof(int32_t));
        const int32_t zero = 0;
        h2d(L.sigmaLeafCount, &zero, sizeof(zero));
        std::vector<float> sigmaZero((size_t)L.sigmaCap, 0.0f);
        h2d(L.sigma, sigmaZero.data(), sigmaZero.size() * sizeof(float));
        std::vector<int32_t> zd(128, 0);
        h2d(L.dbg, zd.data(), zd.size() * sizeof(int32_t));
    }

    // ------------------------------------------------------------------
    // 唯一一次 kernel launch；event 时间只覆盖 device queue 上的 kernel。
    // ------------------------------------------------------------------
    aclrtEvent startEvent = nullptr;
    aclrtEvent stopEvent = nullptr;
    CHECK_ACL_FATAL(aclrtCreateEvent(&startEvent));
    CHECK_ACL_FATAL(aclrtCreateEvent(&stopEvent));
    Timer timer;
    timer.start();
    CHECK_ACL_FATAL(aclrtRecordEvent(startEvent, stream));
    CHECK_ACL_FATAL(ACLRT_LAUNCH_KERNEL(qdwh_sdc_fused_custom)(
        blockDim, stream,
        static_cast<uint8_t *>(wkDev),
        mPad, nPad, cOrig, threshold, maxIter,
        kDevMaxNodes, kDevMaxLeaves, leafCap,
        (float)alpha, (float)l0));
    CHECK_ACL_FATAL(aclrtRecordEvent(stopEvent, stream));
    {
        aclError e = aclrtSynchronizeEvent(stopEvent);
        if (e != ACL_SUCCESS) {
            fprintf(stderr, "[fused] event sync error %d: %s\n",
                    (int)e, aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "(null)");
            DumpDbg(dev(L.dbg));
            return 1;
        }
    }
    float kernelEventMs = 0.0f;
    CHECK_ACL_FATAL(aclrtEventElapsedTime(&kernelEventMs, startEvent, stopEvent));
    timer.end();
    const long long kernelUs = (long long)timer.get_duration_us();
    CHECK_ACL_FATAL(aclrtDestroyEvent(startEvent));
    CHECK_ACL_FATAL(aclrtDestroyEvent(stopEvent));
    if (std::getenv("FUSED_DBG") != nullptr) {
        DumpDbg(dev(L.dbg));
    }

    // ------------------------------------------------------------------
    if (std::getenv("FUSED_AUDIT") != nullptr) {
        auto dump = [&](const char *name, long off, size_t count) {
            std::vector<float> data(count);
            CHECK_ACL_FATAL(aclrtMemcpy(data.data(), count * 4, dev(off), count * 4,
                                       ACL_MEMCPY_DEVICE_TO_HOST));
            FILE *f = fopen(OutPath(name).c_str(), "wb");
            if (!f) exit(1);
            fwrite(data.data(), 4, count, f); fclose(f);
        };
        dump("audit_arena.bin", L.arena, L.arenaFloats);
        dump("audit_node_h.bin", L.nodeH, kDevMaxNodes);
        dump("audit_node_n.bin", L.nodeN, kDevMaxNodes);
        dump("audit_node_ld.bin", L.nodeLd, kDevMaxNodes);
        dump("audit_leaf_raw.bin", L.leaf, leafCap);
        dump("audit_leaf_n.bin", L.leafN, kDevMaxLeaves);
        dump("audit_leaf_ld.bin", L.leafLd, kDevMaxLeaves);
        dump("audit_leaf_off.bin", L.leafOff, kDevMaxLeaves);
        dump("audit_sigma.bin", L.sigma, L.sigmaCap);
    }

    // 可选下载 H/U（kernel 已结束，属于“计算后”的传输）
    // ------------------------------------------------------------------
    if (dumpHU) {
        // 下载 H_pad (nPad²) / U_pad (mPad*nPad)，再取左上 cOrig 方阵 / m*n
        std::vector<float> hPad((size_t)nPad * nPad), uPad((size_t)mPad * nPad);
        CHECK_ACL_FATAL(aclrtMemcpy(hPad.data(), hPad.size() * 4, dev(L.arena),
                                    hPad.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL_FATAL(aclrtMemcpy(uPad.data(), uPad.size() * 4, dev(L.a),
                                    uPad.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        xfer.d2h += 2;
        xfer.d2hBytes += (hPad.size() + uPad.size()) * 4;
        std::vector<float> hOut((size_t)cOrig * cOrig), uOut((size_t)m * n);
        for (int r = 0; r < cOrig; ++r) {
            std::memcpy(hOut.data() + (size_t)r * cOrig, hPad.data() + (size_t)r * nPad,
                        (size_t)cOrig * sizeof(float));
        }
        if (!transposed) {
            for (int r = 0; r < m; ++r) {
                std::memcpy(uOut.data() + (size_t)r * n, uPad.data() + (size_t)r * nPad,
                            (size_t)n * sizeof(float));
            }
        } else {
            // kernel 算的是 A^T 的极分解：U_std = U_T^T
            for (int r = 0; r < m; ++r) {
                for (int c = 0; c < n; ++c) {
                    uOut[(size_t)r * n + c] = uPad[(size_t)c * nPad + r];
                }
            }
        }
        FILE *fh = fopen(OutPath("h.bin").c_str(), "wb");
        FILE *fu = fopen(OutPath("u.bin").c_str(), "wb");
        if (fh != nullptr) { fwrite(hOut.data(), 4, hOut.size(), fh); fclose(fh); }
        if (fu != nullptr) { fwrite(uOut.data(), 4, uOut.size(), fu); fclose(fu); }
        fprintf(stderr, "[fused] 写出 output/h.bin（%zu float）、output/u.bin（%zu float）\n",
                hOut.size(), uOut.size());
    }

    // ------------------------------------------------------------------
    // 下载 leaves（host 只在 kernel 结束后读一次）
    // ------------------------------------------------------------------
    auto cntDev = [&](int i) { return dev(L.cnt + i); };
    int lc = 0, ov = 0, grp = 0;
    CHECK_ACL_FATAL(aclrtMemcpy(&lc, 4, cntDev(3), 4, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_FATAL(aclrtMemcpy(&ov, 4, cntDev(5), 4, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_FATAL(aclrtMemcpy(&grp, 4, cntDev(7), 4, ACL_MEMCPY_DEVICE_TO_HOST));
    xfer.d2h += 3;
    xfer.d2hBytes += 12;
    if (lc < 0 || ov < 0) {
        fprintf(stderr, "[fused] kernel 没有跑起来 (leafCount=%d overflow=%d)\n", lc, ov);
        return 1;
    }
    if (ov != 0) {
        fprintf(stderr, "[fused] overflow=%d（arena/leaf 容量不够）\n", ov);
    }
    if (lc > kDevMaxLeaves) lc = kDevMaxLeaves;
    fprintf(stderr, "[fused] device 端完成: leaves=%d overflow=%d, kernel_event=%.3f ms, host_wait=%.3f ms, sdc_groups=%d (batch<=%d)\n",
            lc, ov, (double)kernelEventMs, (double)kernelUs / 1000.0, grp, kSdcBatchMax);

    int sigmaLc = 0;
    CHECK_ACL_FATAL(aclrtMemcpy(&sigmaLc, sizeof(sigmaLc), dev(L.sigmaLeafCount),
                                sizeof(sigmaLc), ACL_MEMCPY_DEVICE_TO_HOST));
    xfer.d2h += 1;
    xfer.d2hBytes += sizeof(sigmaLc);
    if (sigmaLc != lc) {
        fprintf(stderr, "[fused] sigma leaf count=%d 与 SDC leaf count=%d 不一致\n", sigmaLc, lc);
        return 1;
    }
    std::vector<int> ln(lc), lld(lc), lo(lc);
    std::vector<int> so(lc), sc(lc);
    if (lc > 0) {
        CHECK_ACL_FATAL(aclrtMemcpy(ln.data(), lc * 4, dev(L.leafN), lc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL_FATAL(aclrtMemcpy(lld.data(), lc * 4, dev(L.leafLd), lc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL_FATAL(aclrtMemcpy(lo.data(), lc * 4, dev(L.leafOff), lc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL_FATAL(aclrtMemcpy(so.data(), lc * 4, dev(L.sigmaOff), lc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL_FATAL(aclrtMemcpy(sc.data(), lc * 4, dev(L.sigmaCount), lc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        xfer.d2h += 5;
        xfer.d2hBytes += (size_t)lc * 20;
    }
    int sigmaValid = 0;
    for (int i = 0; i < lc; ++i) {
        if (sc[i] != ln[i] || so[i] < 0 || so[i] + sc[i] > L.sigmaCap) {
            fprintf(stderr, "[fused] 非法 sigma 元数据 leaf=%d off=%d count=%d n=%d cap=%d\n",
                    i, so[i], sc[i], ln[i], L.sigmaCap);
            return 1;
        }
        sigmaValid += sc[i];
    }
    if (sigmaValid > leafCap) {
        fprintf(stderr, "[fused] sigma 总数 %d 超出容量 %d\n", sigmaValid, leafCap);
        return 1;
    }
    std::vector<float> sigmaHost((size_t)sigmaValid);
    std::vector<int> sigmaCompactOff(lc, 0);
    int sigmaPrefix = 0;
    for (int i = 0; i < lc; ++i) {
        // Device scratch gives each leaf a 64-float aligned slot so parallel
        // AIV owners never write the same cache line. Compact on host for the
        // public output contract.
        const int scratchOff = i * 64;
        if (scratchOff + sc[i] > L.sigmaCap) {
            fprintf(stderr, "[fused] sigma scratch 越界 leaf=%d off=%d count=%d cap=%d\n",
                    i, scratchOff, sc[i], L.sigmaCap);
            return 1;
        }
        sigmaCompactOff[i] = sigmaPrefix;
        if (sc[i] > 0) {
            CHECK_ACL_FATAL(aclrtMemcpy(sigmaHost.data() + sigmaPrefix,
                                        (size_t)sc[i] * sizeof(float),
                                        dev(L.sigma + scratchOff),
                                        (size_t)sc[i] * sizeof(float),
                                        ACL_MEMCPY_DEVICE_TO_HOST));
            xfer.d2h += 1;
            xfer.d2hBytes += (size_t)sc[i] * sizeof(float);
        }
        sigmaPrefix += sc[i];
    }
    {
        FILE *fs = fopen(OutPath("sigma.bin").c_str(), "wb");
        FILE *fm = fopen(OutPath("sigma_meta.txt").c_str(), "w");
        if (fs == nullptr || fm == nullptr) {
            fprintf(stderr, "[fused] 无法写 sigma 输出\n");
            return 1;
        }
        if (!sigmaHost.empty()) fwrite(sigmaHost.data(), sizeof(float), sigmaHost.size(), fs);
        for (int i = 0; i < lc; ++i) {
            fprintf(fm, "%d %d %d %d\n", 0, ln[i], sigmaCompactOff[i], sc[i]);
        }
        fclose(fs);
        fclose(fm);
        fprintf(stderr, "[fused] 写出 sigma.bin (%zu float) + sigma_meta.txt\n", sigmaHost.size());
    }
    // 16MB 级别的 D2H：先用 aclrtMallocHost 申请锁页内存再拷到 vector
    std::vector<float> lh((size_t)leafCap, 0.0f);
    {
        void *lhHost = nullptr;
        const size_t leafBytes = lh.size() * sizeof(float);
        CHECK_ACL_FATAL(aclrtMallocHost(&lhHost, leafBytes));
        CHECK_ACL_FATAL(aclrtMemcpy(lhHost, leafBytes, dev(L.leaf), leafBytes,
                                    ACL_MEMCPY_DEVICE_TO_HOST));
        std::memcpy(lh.data(), lhHost, leafBytes);
        CHECK_ACL_FATAL(aclrtFreeHost(lhHost));
    }
    xfer.d2h += 1;
    xfer.d2hBytes += lh.size() * 4;

    // 叶块：按 ld 读出 -> 对称化 -> 紧凑
    std::vector<float> out;
    std::vector<int> outN, outOff;
    for (int i = 0; i < lc; ++i) {
        const int ln2 = ln[i], ld = lld[i];
        outOff.push_back((int)out.size());
        outN.push_back(ln2);
        out.resize(out.size() + (size_t)ln2 * ln2);
        float *Lp = out.data() + outOff.back();
        const float *P = lh.data() + lo[i];
        for (int r = 0; r < ln2; ++r) {
            for (int c = 0; c < ln2; ++c) {
                float a = P[(size_t)r * ld + c];
                float b = P[(size_t)c * ld + r];
                Lp[(size_t)r * ln2 + c] = 0.5f * (a + b);
            }
        }
    }

    {
        std::string hp = OutPath("leaf_h.bin");
        std::string mp = OutPath("leaf_meta.txt");
        FILE *fh = fopen(hp.c_str(), "wb");
        FILE *fm = fopen(mp.c_str(), "w");
        if (fh == nullptr || fm == nullptr) {
            fprintf(stderr, "[fused] 无法写 output\n");
            return 1;
        }
        if (!out.empty()) fwrite(out.data(), 4, out.size(), fh);
        for (int i = 0; i < lc; ++i) {
            fprintf(fm, "%d %d %d\n", 0, outN[i], outOff[i]);
        }
        fclose(fh);
        fclose(fm);
        fprintf(stderr, "[fused] 写出 %s (%zu float) + %s\n", hp.c_str(), out.size(), mp.c_str());
    }

    const int maxLeaf = lc ? *std::max_element(ln.begin(), ln.end()) : 0;
    fprintf(stderr,
            "[fused] 汇总: leaves=%d max_leaf=%d threshold=%d | host 传输: "
            "h2d %d 次/%.1f MB, d2h %d 次/%.1f MB（全部在计算前后）\n",
            lc, maxLeaf, threshold,
            xfer.h2d, (double)xfer.h2dBytes / 1048576.0,
            xfer.d2h, (double)xfer.d2hBytes / 1048576.0);

    CHECK_ACL_FATAL(aclrtFree(wkDev));
    CHECK_ACL_FATAL(aclrtDestroyStream(stream));
    CHECK_ACL_FATAL(aclrtDestroyContext(context));
    CHECK_ACL_FATAL(aclrtResetDevice(deviceId));
    CHECK_ACL_FATAL(aclFinalize());
    return 0;
}
