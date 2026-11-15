#!/usr/bin/env bash
# =============================================================================
# 一键：生成数据 -> 编译融合 kernel -> 一次 launch 跑完 QDWH+SDC -> verify
#
#   ./run.sh                     # 默认 1024x1024 / cond=1e2 / threshold=64
#   SHAPES=1280x1024 ./run.sh
#   REGEN=0 ./run.sh             # 复用已有 input/
#
# 环境变量：SHAPES COND THRESHOLD DEVICE BLOCKDIM MAXITER J REGEN
# =============================================================================
set -e
cd "$(dirname "$0")"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

SHAPES=${SHAPES:-1024x1024}
COND=${COND:-1e2}
THRESHOLD=${THRESHOLD:-64}   # 叶子 n <= 64
DEVICE=${DEVICE:-0}
BLOCKDIM=${BLOCKDIM:-20}    # 20 个 AIC + 40 个 AIV（QR 的 vec 侧仍只用 32 个，见 README）
MAXITER=${MAXITER:-4}      # l 递推实测第 4 轮就到 1，4 轮足够且更准
J=${J:-16}
REGEN=${REGEN:-1}

mkdir -p input output
if [ "${REGEN}" = "1" ]; then
    python3 gen_data.py --shapes "${SHAPES}" --cond "${COND}" --out-dir input
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fused_main -j "${J}"

./build/fused_main "${DEVICE}" "${BLOCKDIM}" "${THRESHOLD}" "${MAXITER}" 1.0 1e-6 2>&1 \
    | tee output/fused_run.txt

python3 verify.py --threshold "${THRESHOLD}" | tee output/verify.txt
