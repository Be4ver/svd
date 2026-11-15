from pathlib import Path
import shutil
base=Path('/tmp/jacobi64-study-20261112-iOuF3K')
dst=base/'src-tournament'
shutil.copytree(base/'src',dst,dirs_exist_ok=True)
s=(dst/'jacobi_svd_custom.cpp').read_text()
v=(base/'src-vector/jacobi_svd_custom.cpp').read_text()
start=v.index('                Mul(scls_, red_[0], red_[64], 64);')
end=v.index('            }'+chr(10)+'        }'+chr(10)+'        WholeReduceMax', start)
angle=v[start:end]
angle=angle.replace('Brcb(rowA_[0], csV_, 1, {1, 8});','Brcb(rowA_[0], csV_, 4, {1, 8});')
angle=angle.replace('Brcb(rowA_[64], snV_, 1, {1, 8});','Brcb(rowA_[256], snV_, 4, {1, 8});')
angle=angle.replace('Brcb(rowA_[128], tmpV_, 1, {1, 8});','Brcb(rowA_[512], tmpV_, 4, {1, 8});')
angle=angle.replace('Brcb(rowA_[192], tV_, 1, {1, 8});','Brcb(rowA_[768], tV_, 4, {1, 8});')
angle=angle.replace('rowA_[64], 64, K','rowA_[256], 64, 32').replace('rowA_[128], 64, K','rowA_[512], 64, 32').replace('rowA_[192], 64, K','rowA_[768], 64, 32').replace('rowA_[0], 64, K','rowA_[0], 64, 32')
new=r'''    // Research-only 64x64 resident, one AIV per matrix, tournament ordering.
    __aicore__ inline void RunResidentTournament(__gm__ float *input, __gm__ float *output,
        int32_t m, int64_t matElems, int32_t maxSweeps)
    {
        GlobalTensor<float> in, out;
        in.SetGlobalBuffer(input + static_cast<int64_t>(m) * matElems);
        out.SetGlobalBuffer(output + static_cast<int64_t>(m) * matElems);
        LoadRows(tile_, in, 0, 4096);
        LocalTensor<float> cur = tile_;
        LocalTensor<float> next = tile_[4096];
        Duplicate(red_, 0.0f, 192);
        const BinaryRepeatParams br{1, 1, 0, 8, 8, 1};
        for (int32_t sweep = 0; sweep < maxSweeps; ++sweep) {
            Duplicate(ls_, 0.0f, 64);
            for (int32_t round = 0; round < 63; ++round) {
                LocalTensor<float> u = cur;
                LocalTensor<float> v = cur[2048];
                Mul(prod3_, u, u, 2048);
                WholeReduceSum(red_[0], prod3_, 64, 32, 1, 1, 8);
                Mul(prod3_, v, v, 2048);
                WholeReduceSum(red_[64], prod3_, 64, 32, 1, 1, 8);
                Mul(prod3_, u, v, 2048);
                WholeReduceSum(red_[128], prod3_, 64, 32, 1, 1, 8);
''' + angle + r'''
                // Circle schedule: A'=[A0,B0,A1..A30], B'=[B1..B31,A31].
                Adds(next[0], cur[0], 0.0f, 64);
                Adds(next[64], cur[2048], 0.0f, 64);
                Adds(next[128], cur[64], 0.0f, 30 * 64);
                Adds(next[2048], cur[2112], 0.0f, 31 * 64);
                Adds(next[4032], cur[1984], 0.0f, 64);
                LocalTensor<float> swap = cur; cur = next; next = swap;
            }
            WholeReduceMax(red_[200], ls_, 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            if (red_.GetValue(200) == 0.0f) break;
        }
        StoreRows(out, 0, cur, 4096);
    }

'''
needle='    /** 从 G 的对角线取 sqrt 写 sigma（AIC 算完 Gram 之后由 AIV 调用） */'
assert s.count(needle)==1
s=s.replace(needle,new+needle)
needle='    if (rows <= kB && colsWork <= 64 && perCore > 1) {'
assert s.count(needle)==1
s=s.replace(needle,'''    if (rows == 64 && colsWork == 64 && perCore == 1) {
        for (int32_t m = mStart; m < mEnd; ++m) {
            op.RunResidentTournament(reinterpret_cast<__gm__ float *>(APad),
                reinterpret_cast<__gm__ float *>(B), m, matElems, maxSweeps);
        }
    } else if (rows <= kB && colsWork <= 64 && perCore > 1) {''')
(dst/'jacobi_svd_custom.cpp').write_text(s)
