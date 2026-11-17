#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclrtlaunch_blocked_jacobi_fused.h"

static void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(what) + ": ACL " + std::to_string(status));
}
#define ACL(call) Check((call), #call)

struct Runtime {
    int device; aclrtContext context = nullptr; aclrtStream stream = nullptr;
    explicit Runtime(int d) : device(d) {
        ACL(aclInit(nullptr)); ACL(aclrtSetDevice(device));
        ACL(aclrtCreateContext(&context, device)); ACL(aclrtCreateStream(&stream));
    }
    ~Runtime() {
        if (stream) aclrtDestroyStream(stream);
        if (context) aclrtDestroyContext(context);
        aclrtResetDevice(device); aclFinalize();
    }
};

struct DeviceBuffer {
    void* ptr = nullptr; size_t bytes;
    explicit DeviceBuffer(size_t n) : bytes(n) { ACL(aclrtMalloc(&ptr, n, ACL_MEM_MALLOC_HUGE_FIRST)); }
    ~DeviceBuffer() { if (ptr) aclrtFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    void Upload(const void* src) { ACL(aclrtMemcpy(ptr, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE)); }
    void Download(void* dst) { ACL(aclrtMemcpy(dst, bytes, ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST)); }
};

static int Integer(const char* value) {
    size_t used = 0; int n = std::stoi(value, &used);
    if (value[used] != '\0') throw std::runtime_error("invalid integer");
    return n;
}
struct Config {
    int device, rows, cols, batch, cores = 0, sweeps = 30, block = 16, local = 1, repeat = 1, pipelineJobs = 0;
    float tol = std::numeric_limits<float>::epsilon();
    bool fixed = false, overlap = true;
    std::string input = "../input", output = "../output";
};

static Config Parse(int argc, char** argv) {
    if (argc < 5) throw std::runtime_error(
        "usage: jacobi_svd_main device rows cols batch [coreNum(AIV)] [maxSweeps] "
        "[--block 16|32|64] [--local-sweeps 1] [--repeat 1] [--tol 1.1920929e-7] "
        "[--fixed-sweeps] [--no-overlap] [--pipeline-jobs 0(auto)|N] "
        "[--input-dir ../input] [--output-dir ../output]");
    Config c{Integer(argv[1]), Integer(argv[2]), Integer(argv[3]), Integer(argv[4])};
    int i = 5;
    if (i < argc && argv[i][0] != '-') c.cores = Integer(argv[i++]);
    if (i < argc && argv[i][0] != '-') c.sweeps = Integer(argv[i++]);
    while (i < argc) {
        std::string flag = argv[i++];
        if (flag == "--fixed-sweeps") { c.fixed = true; continue; }
        if (flag == "--no-overlap") { c.overlap = false; continue; }
        if (i == argc) throw std::runtime_error("missing value for " + flag);
        const char* value = argv[i++];
        if (flag == "--block") c.block = Integer(value);
        else if (flag == "--local-sweeps") c.local = Integer(value);
        else if (flag == "--repeat") c.repeat = Integer(value);
        else if (flag == "--pipeline-jobs") c.pipelineJobs = Integer(value);
        else if (flag == "--tol") { size_t used; c.tol = std::stof(value, &used); if (value[used]) throw std::runtime_error("invalid tol"); }
        else if (flag == "--input-dir") c.input = value;
        else if (flag == "--output-dir") c.output = value;
        else throw std::runtime_error("unknown option " + flag);
    }
    if (c.rows < 1 || c.rows > 4096 || c.cols < 1 || c.cols > 4096 || c.batch < 1 || c.batch > 1000000 ||
        (c.block != 16 && c.block != 32 && c.block != 64) || c.sweeps < 1 || c.sweeps > 200 || c.local < 1 || c.local > 50 ||
        c.repeat < 1 || c.repeat > 100 || c.cores < 0 || c.pipelineJobs < 0 ||
        (!c.overlap && c.pipelineJobs != 0) || !std::isfinite(c.tol) || c.tol < 1e-8f || c.tol > 1e-2f)
        throw std::runtime_error("unsupported shape or invalid tuning argument");
    return c;
}

template<class T> static void Write(const std::string& path, const std::vector<T>& data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    if (!f) throw std::runtime_error("cannot write " + path);
}
static int Cores(int device, aclrtDevAttr attr) {
    int64_t count = 0;
    ACL(aclrtGetDeviceInfo(device, attr, &count));
    if (count < 1 || count > 256) throw std::runtime_error("invalid device core count");
    return static_cast<int>(count);
}

static int Run(const Config& c) {
    Runtime runtime(c.device);
    const int allAiv = Cores(c.device, ACL_DEV_ATTR_VECTOR_CORE_NUM);
    const int allAic = Cores(c.device, ACL_DEV_ATTR_AICORE_CORE_NUM);
    const int aiv = c.cores == 0 ? allAiv : c.cores;
    if (aiv > allAiv || aiv % 2 || aiv / 2 > allAic)
        throw std::runtime_error("coreNum must be even and fit resident AIV/AIC cores");
    const int aic = aiv / 2;
    const int nb = std::max(2, ((c.rows + c.block - 1) / c.block + 1) / 2 * 2);
    const int rp = nb * c.block, cp = (c.cols + 31) / 32 * 32;
    if (static_cast<int64_t>(c.batch) * rp > std::numeric_limits<int32_t>::max() ||
        static_cast<int64_t>(c.batch) * nb * ((cp + 127) / 128) > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("batch/shape exceeds device task index range");
    const int d = 2 * c.block, np = nb / 2, jobs = c.batch * np, rounds = nb - 1;
    if (c.pipelineJobs >= jobs && c.pipelineJobs != 0)
        throw std::runtime_error("pipeline-jobs must be smaller than block-pair tasks per round");
    // Splitting <= AIV-count jobs doubles the number of parameter waves.
    // Auto mode avoids that regression; explicit splits remain available.
    // Balance parameter waves, placing complete AIV waves in group A. Unlike
    // an element-wise half split, this never adds an extra parameter wave.
    const int waves = (jobs + aiv - 1) / aiv;
    const int autoSplit = (waves / 2) * aiv;
    const int split = !c.overlap ? 0 : (c.pipelineJobs ? c.pipelineJobs : autoSplit);
    const int groupA = split ? split : jobs, groupB = split ? jobs - split : 0;
    const int phaseJobs = std::max(groupA, groupB);
    const int rank = std::min(c.rows, c.cols);
    const size_t elements = static_cast<size_t>(c.batch) * rp * cp;
    std::vector<float> original(static_cast<size_t>(c.batch) * c.rows * c.cols);
    std::ifstream in(c.input + "/A_batch.bin", std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(original.size() * sizeof(float)))
        throw std::runtime_error("input file missing or size mismatch");
    in.seekg(0); in.read(reinterpret_cast<char*>(original.data()), original.size() * sizeof(float));
    if (!in || !std::all_of(original.begin(), original.end(), [](float x){ return std::isfinite(x); }))
        throw std::runtime_error("input read failed or nonfinite input");
    DeviceBuffer input(original.size() * sizeof(float)), b(elements * sizeof(float)), panels(b.bytes),
        gram(static_cast<size_t>(jobs) * d * d * sizeof(float)), rotations(gram.bytes),
        flags(static_cast<size_t>(jobs) * 8 * sizeof(float)), scales(static_cast<size_t>(c.batch) * 8 * sizeof(float)),
        status((aiv + 1) * 8 * sizeof(int32_t)), bout(input.bytes),
        sout(static_cast<size_t>(c.batch) * rank * sizeof(float)),
        vtout(static_cast<size_t>(c.batch) * rank * c.cols * sizeof(float));
    input.Upload(original.data());
    std::vector<double> deviceTimes, hostTimes;
    std::vector<int32_t> state((aiv + 1) * 8);
    int finished = 0; bool converged = false;
    std::cout << "single-launch mixed kernel AIV=" << aiv << "/" << allAiv << " AIC=" << aic << "/" << allAic
              << " shape=" << c.rows << "x" << c.cols << " batch=" << c.batch << " block=" << c.block
              << " overlap=" << (split > 0) << " groups=" << groupA << "+" << groupB << '\n';
    aclrtEvent begin, end;
    ACL(aclrtCreateEvent(&begin)); ACL(aclrtCreateEvent(&end));
    for (int rep = 0; rep < c.repeat; ++rep) {
        // Completion sentinel detects missing registration/device execution.
        ACL(aclrtMemset(status.ptr, status.bytes, 0xff, status.bytes));
        const auto t0 = std::chrono::steady_clock::now();
        ACL(aclrtRecordEvent(begin, runtime.stream));
        ACL(aclrtlaunch_blocked_jacobi_fused(aic, runtime.stream,
            input.ptr, b.ptr, panels.ptr, gram.ptr, rotations.ptr, flags.ptr, scales.ptr, status.ptr,
            bout.ptr, sout.ptr, vtout.ptr, c.rows, c.cols, c.batch, c.block, c.sweeps, c.local, c.tol, c.fixed ? 1 : 0, split));
        ACL(aclrtRecordEvent(end, runtime.stream));
        ACL(aclrtSynchronizeStream(runtime.stream));
        const double hostMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        float deviceMs = 0; ACL(aclrtEventElapsedTime(&deviceMs, begin, end));
        status.Download(state.data());
        const auto* result = state.data() + aiv * 8;
        if (result[2] != 0x424a5356 || result[0] < 1 || result[0] > c.sweeps || (result[1] != 0 && result[1] != 1))
            throw std::runtime_error("invalid/unwritten device completion status; check kernel registration");
        finished = result[0]; converged = result[1] != 0;
        deviceTimes.push_back(deviceMs); hostTimes.push_back(hostMs);
        std::cout << "run=" << rep << " sweeps=" << finished << " converged=" << converged
                  << " stream_ms=" << deviceMs << " operator_wall_ms=" << hostMs << '\n';
    }
    ACL(aclrtDestroyEvent(begin)); ACL(aclrtDestroyEvent(end));
    std::vector<float> bt(original.size()), sigma(sout.bytes / sizeof(float)), vt(vtout.bytes / sizeof(float));
    bout.Download(bt.data()); sout.Download(sigma.data()); vtout.Download(vt.data());
    if (!std::all_of(sigma.begin(), sigma.end(), [](float x){ return std::isfinite(x); }) ||
        !std::all_of(bt.begin(), bt.end(), [](float x){ return std::isfinite(x); }) ||
        !std::all_of(vt.begin(), vt.end(), [](float x){ return std::isfinite(x); }))
        throw std::runtime_error("SVD result exceeds FP32 output range or nonfinite device output");
    std::filesystem::create_directories(c.output);
    Write(c.output + "/B_rot.bin", bt); Write(c.output + "/sigma.bin", sigma); Write(c.output + "/Vt.bin", vt);
    std::ofstream metrics(c.output + "/metrics.json");
    metrics << std::setprecision(9) << "{\n  \"rows\": " << c.rows << ", \"cols\": " << c.cols << ", \"batch\": " << c.batch
            << ", \"aiv\": " << aiv << ", \"aic\": " << aic << ", \"block\": " << c.block
            << ", \"local_sweeps\": " << c.local << ", \"sweeps\": " << finished << ", \"converged\": " << (converged ? "true" : "false")
            << ",\n  \"kernel_launches_per_run\": 1, \"device_preprocess\": true, \"device_postprocess\": true, \"device_convergence\": true"
            << ",\n  \"rows_pad\": " << rp << ", \"cols_pad\": " << cp << ", \"rounds_per_sweep\": " << rounds
            << ",\n  \"phase_overlap\": " << (split ? "true" : "false")
            << ", \"pipeline_groups\": [" << groupA << ", " << groupB << "]"
            << ", \"fixed_sweeps\": " << (c.fixed ? "true" : "false")
            << ",\n  \"tasks_per_round\": {\"pack\": " << c.batch * rp << ", \"gram\": " << jobs
            << ", \"parameters\": " << jobs << ", \"apply\": " << jobs * 2 * ((cp + 127) / 128) << "}"
            << ",\n  \"useful_core_upper_bound\": {\"pack_aiv\": " << std::min(aiv, c.batch * rp)
            << ", \"parameters_aiv\": " << std::min(aiv, phaseJobs)
            << ", \"gram_aic\": " << std::min(aic, phaseJobs)
            << ", \"apply_aic\": " << std::min(aic, phaseJobs * 2 * ((cp + 127) / 128)) << "}"
            << ",\n  \"padded_cube_flops\": " << 4LL * jobs * d * d * cp * rounds * finished
            << ",\n  \"stream_ms\": [";
    for (size_t i = 0; i < deviceTimes.size(); ++i) metrics << (i ? ", " : "") << deviceTimes[i];
    metrics << "],\n  \"operator_wall_ms\": [";
    for (size_t i = 0; i < hostTimes.size(); ++i) metrics << (i ? ", " : "") << hostTimes[i];
    metrics << "]\n}\n";
    if (!metrics) throw std::runtime_error("metrics write failed");
    return 0;
}

int main(int argc, char** argv) {
    try { return Run(Parse(argc, argv)); }
    catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
