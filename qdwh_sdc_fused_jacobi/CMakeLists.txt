cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(qdwh_sdc_fused)

# Ascend 910B3 / AscendC。先 source /usr/local/Ascend/ascend-toolkit/set_env.sh，
# 使 ASCEND_HOME_PATH 指向 CANN。
set(SOC_VERSION "ascend910b3")
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH} CACHE PATH "ASCEND CANN package installation directory")

include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

# ---------------------------------------------------------------------------
# device kernel：融合 kernel（QDWH + device 内 SDC 递归，一次 launch）
#
#   qdwh_sdc_fused_custom.cpp  (kernel 入口，只做 Init + RunAiv/RunCube 分支)
#     -> fused_qdwh_sdc.hpp    (class QdwhSdcFused：Phase1/Phase2 的阶段与同步)
#          -> sdc_node.hpp     (SDC 单节点 split：build_plan / tail_aiv / tail_cube)
#               -> qr_custom.cpp  (compute_weight / qdwh_cube / MatmulCustom)
#                    -> vec.hpp    (qdwh_vec：AIV 侧全部向量实现)
#                    -> gemm.cpp   (MatmulCustom：分块 GEMM)
#          -> fused_layout.hpp   (host/device 共享的工作区布局)
# ---------------------------------------------------------------------------

ascendc_library(fused_kernels STATIC qdwh_sdc_fused_custom.cpp)
ascendc_compile_definitions(fused_kernels PRIVATE HAVE_TILING)
ascendc_compile_options(fused_kernels PRIVATE "-Wno-deprecated-declarations")

# ---------------------------------------------------------------------------
# host program：fused_main（上传 A、launch 一次、下载 leaves）
# ---------------------------------------------------------------------------
add_executable(fused_main fused_main.cpp)
target_link_libraries(fused_main PRIVATE fused_kernels)
add_dependencies(fused_main fused_kernels)
