#!/usr/bin/env bash
set -eo pipefail
repo=/home/zhushengrui/project/svd/qdwh_sdc_fused_jacobi
root=/tmp/qdwh-square-bench-20261115
rm -rf "$root"
mkdir -p "$root"
cd "$repo"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
timeout 600s cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >"$root/build.log" 2>&1
timeout 600s cmake --build build --target fused_main -j16 >>"$root/build.log" 2>&1
for shape in 128x128 512x512 1024x1024 2048x2048; do
  size="${shape%%x*}"
  dir="$root/$size"
  mkdir -p "$dir"
  rm -rf input output
  mkdir -p input output
  python3 gen_data.py --shapes "$shape" --cond 1e2 --out-dir input >"$dir/generate.log" 2>&1
  timeout 240s ./build/fused_main 0 20 64 4 1.0 1e-6 >"$dir/run.log" 2>&1
  python3 verify.py --threshold 64 >"$dir/verify.log" 2>&1
  printf 'SIZE=%s\n' "$size"
  grep -E 'shape|kernel_event|host_wait|leaves=|RESULT:|PASS|FAIL' "$dir/run.log" "$dir/verify.log" || true
done
printf 'ROOT=%s\n' "$root"

