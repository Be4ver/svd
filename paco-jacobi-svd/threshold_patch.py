from pathlib import Path
p=Path('/home/zhushengrui/project/svd/qdwh_sdc_fused_jacobi/fused_main.cpp')
s=p.read_text()
s=s.replace('    if (threshold < 16 || threshold > 8192) {\n        fprintf(stderr, "[fused] 错误: threshold 非法（当前 %d）\\n", threshold);', '    if (threshold < 16 || threshold > 64) {\n        fprintf(stderr, "[fused] 错误: 融合 Jacobi 仅支持 16 <= threshold <= 64（当前 %d）\\n", threshold);')
p.write_text(s)
