from pathlib import Path
import shutil
base=Path('/tmp/jacobi64-study-20261112-iOuF3K')
dst=base/'src-vector'
shutil.copytree(base/'src',dst,dirs_exist_ok=True)
s=(dst/'jacobi_svd_custom.cpp').read_text()
needle='    __aicore__ inline bool RunMulti(int32_t nRows, int32_t colsW, int32_t K)'
assert s.count(needle)==1
new=r'''    // Research prototype only: fixed 64-lane angle chain, UB coefficient broadcast.
    __aicore__ inline bool RunMulti(int32_t nRows, int32_t colsW, int32_t K)
    {
        if (colsW != 64 || K > 8) return RunMultiOriginal(nRows, colsW, K);
        Duplicate(red_, 0.0f, 192);
        Duplicate(ls_, 0.0f, 64);
        const BinaryRepeatParams br{1, 1, 0, 8, 8, 1};
        for (int32_t i = 0; i < nRows; ++i) {
            LocalTensor<float> u = tile_[(i * K) * colsW];
            for (int32_t j = i + 1; j < nRows; ++j) {
                LocalTensor<float> v = tile_[(j * K) * colsW];
                Mul(prod3_, u, u, K * 64);
                WholeReduceSum(red_[0], prod3_, 64, K, 1, 1, 8);
                Mul(prod3_, v, v, K * 64);
                WholeReduceSum(red_[64], prod3_, 64, K, 1, 1, 8);
                Mul(prod3_, u, v, K * 64);
                WholeReduceSum(red_[128], prod3_, 64, K, 1, 1, 8);
                Mul(scls_, red_[0], red_[64], 64);
                Sqrt(scls_, scls_, 64);
                Muls(thrV_, scls_, 1e-6f, 64);
                Abs(absV_, red_[128], 64);
                Compare(mask_, absV_, thrV_, CMPMODE::GT, 64);
                Select(actV_, mask_, onesV_, zerosV_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Max(ls_, ls_, actV_, 64);
                Sub(diffV_, red_[64], red_[0], 64);
                Select(diffV_, mask_, diffV_, zerosV_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Muls(denV_, red_[128], 2.0f, 64);
                Select(denV_, mask_, denV_, onesV_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Div(tauV_, diffV_, denV_, 64);
                Abs(diffV_, tauV_, 64);
                Mul(svV_, tauV_, tauV_, 64);
                Adds(svV_, svV_, 1.0f, 64);
                Sqrt(svV_, svV_, 64);
                Add(denV_, diffV_, svV_, 64);
                Div(tV_, onesV_, denV_, 64);
                Compare(mask2_, tauV_, zerosV_, CMPMODE::GE, 64);
                Muls(tmpV_, tV_, -1.0f, 64);
                Select(tV_, mask2_, tV_, tmpV_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Select(tV_, mask_, tV_, zerosV_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                Mul(tmpV_, tV_, tV_, 64);
                Adds(tmpV_, tmpV_, 1.0f, 64);
                Sqrt(tmpV_, tmpV_, 64); // invCs
                Div(csV_, onesV_, tmpV_, 64);
                Mul(snV_, csV_, tV_, 64);
                Muls(snV_, snV_, -1.0f, 64);
                Brcb(rowA_[0], csV_, 1, {1, 8});
                Brcb(rowA_[64], snV_, 1, {1, 8});
                Brcb(rowA_[128], tmpV_, 1, {1, 8});
                Brcb(rowA_[192], tV_, 1, {1, 8});
                Mul(u, u, rowA_[0], 64, K, br);
                MulAddDst(u, v, rowA_[64], 64, K, br);
                Mul(v, v, rowA_[128], 64, K, br);
                MulAddDst(v, u, rowA_[192], 64, K, br);
            }
        }
        WholeReduceMax(red_[200], ls_, 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        return red_.GetValue(200) > 0.0f;
    }

    __aicore__ inline bool RunMultiOriginal(int32_t nRows, int32_t colsW, int32_t K)'''
s=s.replace(needle,new)
(dst/'jacobi_svd_custom.cpp').write_text(s)
