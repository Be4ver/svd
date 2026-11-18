#pragma once
#include "fused_common.hpp"

namespace blocked_jacobi {
// One 64B cache line per request, ack and matrix block. No DMA writes to
// these lines and no false sharing between independent scalar writers.
// This follows CANN C220 KFC mailboxes: invalidate before polling and clean
// after publication. DMA release/acquire events are the caller's obligation.
class CoreSignals {
public:
    __aicore__ inline void Init(GM_ADDR workspace, int workers, int blocks)
    {
        workers_ = workers;
        abort_ = 2 * workers + blocks;
        data_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(workspace));
    }
    __aicore__ inline int Request(int worker) { return worker; }
    __aicore__ inline int Ack(int worker) { return workers_ + worker; }
    __aicore__ inline int Version(int block) { return 2 * workers_ + block; }
    __aicore__ inline int64_t Read(int line)
    {
        auto address = data_[static_cast<int64_t>(line) * 8];
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(address);
        DataSyncBarrier<MemDsbT::DDR>();
        // GetValue is not volatile: a tight wait can reuse a scalar load even
        // across DCCI. Force a fresh GM observation on every polling iteration.
        return *reinterpret_cast<volatile __gm__ int64_t*>(address.GetPhyAddr(0));
    }
    __aicore__ inline bool Aborted() { return Read(abort_) != 0; }
    __aicore__ inline void Abort() { Publish(abort_, 1); }
    __aicore__ inline bool Wait(int line, int64_t value)
    {
        const int64_t start = GetSystemCycle();
        int polls = 0;
        while (Read(line) != value) {
            if ((++polls & 1023) == 0) {
                if (Aborted()) return false;
                // SYS_CNT watchdog, deliberately far above one task's latency.
                if (GetSystemCycle() - start > 2000000000LL) { Abort(); return false; }
            }
        }
        return !Aborted();
    }
    __aicore__ inline void Publish(int line, int64_t value)
    {
        auto address = data_[static_cast<int64_t>(line) * 8];
        address.SetValue(0, value);
        DataSyncBarrier<MemDsbT::DDR>();
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(address);
        DataSyncBarrier<MemDsbT::DDR>();
    }
private:
    GlobalTensor<int64_t> data_;
    int workers_, abort_;
};
}
