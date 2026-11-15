from pathlib import Path
p=Path('/home/zhushengrui/project/svd/qdwh_sdc_fused_jacobi/fused_layout.hpp')
s=p.read_text()
s=s.replace('    long leafOff;      // 叶块在 leaf 缓冲中的偏移\n    long kTmp;', '    long leafOff;      // 叶块在 leaf 缓冲中的偏移\n    long sigma;        // 每个叶块贡献的奇异值，连续 float32 输出\n    long sigmaOff;     // 每叶 sigma 在 sigma 缓冲中的偏移（int32）\n    long sigmaCount;   // 每叶 sigma 数量（int32）\n    long sigmaLeafCount; // 最终叶块数量（int32）\n    long kTmp;')
s=s.replace('    leafOff = o;    o += maxLeaves;\n    kTmp', '    leafOff = o;    o += maxLeaves;\n    sigma = o;      o += leafCap;\n    sigmaOff = o;   o += maxLeaves;\n    sigmaCount = o; o += maxLeaves;\n    sigmaLeafCount = o; o += 1;\n    kTmp')
p.write_text(s)

p=Path('/home/zhushengrui/project/svd/qdwh_sdc_fused_jacobi/fused_main.cpp')
s=p.read_text()
s=s.replace('        h2d(L.leafOff, zl.data(), zl.size() * sizeof(int32_t));\n        std::vector<int32_t> zd', '        h2d(L.leafOff, zl.data(), zl.size() * sizeof(int32_t));\n        h2d(L.sigmaOff, zl.data(), zl.size() * sizeof(int32_t));\n        h2d(L.sigmaCount, zl.data(), zl.size() * sizeof(int32_t));\n        const int32_t zero = 0;\n        h2d(L.sigmaLeafCount, &zero, sizeof(zero));\n        std::vector<float> sigmaZero((size_t)leafCap, 0.0f);\n        h2d(L.sigma, sigmaZero.data(), sigmaZero.size() * sizeof(float));\n        std::vector<int32_t> zd')
old='''    std::vector<int> ln(lc), lld(lc), lo(lc);
    if (lc > 0) {
        CHECK_ACL_FATAL(aclrtMemcpy(ln.data(), lc * 4, dev(L.leafN), lc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL_FATAL(aclrtMemcpy(lld.data(), lc * 4, dev(L.leafLd), lc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL_FATAL(aclrtMemcpy(lo.data(), lc * 4, dev(L.leafOff), lc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        xfer.d2h += 3;
        xfer.d2hBytes += (size_t)lc * 12;
    }
'''
new='''    int sigmaLc = 0;
    CHECK_ACL_FATAL(aclrtMemcpy(&sigmaLc, sizeof(sigmaLc), dev(L.sigmaLeafCount),
                                sizeof(sigmaLc), ACL_MEMCPY_DEVICE_TO_HOST));
    xfer.d2h += 1;
    xfer.d2hBytes += sizeof(sigmaLc);
    if (sigmaLc != lc) {
        fprintf(stderr, "[fused] sigma leaf count=%d 与 SDC leaf count=%d 不一致\\n", sigmaLc, lc);
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
        if (sc[i] != ln[i] || so[i] < 0 || so[i] + sc[i] > leafCap) {
            fprintf(stderr, "[fused] 非法 sigma 元数据 leaf=%d off=%d count=%d n=%d cap=%d\\n",
                    i, so[i], sc[i], ln[i], leafCap);
            return 1;
        }
        sigmaValid += sc[i];
    }
    if (sigmaValid > leafCap) {
        fprintf(stderr, "[fused] sigma 总数 %d 超出容量 %d\\n", sigmaValid, leafCap);
        return 1;
    }
    std::vector<float> sigmaHost((size_t)sigmaValid);
    if (sigmaValid > 0) {
        CHECK_ACL_FATAL(aclrtMemcpy(sigmaHost.data(), sigmaHost.size() * sizeof(float),
                                    dev(L.sigma), sigmaHost.size() * sizeof(float),
                                    ACL_MEMCPY_DEVICE_TO_HOST));
        xfer.d2h += 1;
        xfer.d2hBytes += sigmaHost.size() * sizeof(float);
    }
    {
        FILE *fs = fopen(OutPath("sigma.bin").c_str(), "wb");
        FILE *fm = fopen(OutPath("sigma_meta.txt").c_str(), "w");
        if (fs == nullptr || fm == nullptr) {
            fprintf(stderr, "[fused] 无法写 sigma 输出\\n");
            return 1;
        }
        if (!sigmaHost.empty()) fwrite(sigmaHost.data(), sizeof(float), sigmaHost.size(), fs);
        for (int i = 0; i < lc; ++i) {
            fprintf(fm, "%d %d %d %d\\n", 0, ln[i], so[i], sc[i]);
        }
        fclose(fs);
        fclose(fm);
        fprintf(stderr, "[fused] 写出 sigma.bin (%zu float) + sigma_meta.txt\\n", sigmaHost.size());
    }
'''
if old not in s:
    raise SystemExit('expected leaf metadata download block missing')
s=s.replace(old,new)
p.write_text(s)
