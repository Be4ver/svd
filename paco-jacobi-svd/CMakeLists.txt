cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(jacobi_svd)

# 目标芯片: 昇腾 910B3
set(SOC_VERSION "ascend910b3")
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH} CACHE PATH "ASCEND CANN package installation directory")

include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

# ---------------------------------------------------------------- 设备侧 kernel
# 编译 jacobi_svd_custom.cpp，产出 libjacobi_svd_kernels.a 以及
# include/jacobi_svd_kernels/aclrtlaunch_jacobi_svd_batched.h
ascendc_library(jacobi_svd_kernels STATIC jacobi_svd_custom.cpp)
ascendc_compile_options(jacobi_svd_kernels PRIVATE "-Wno-deprecated-declarations")
ascendc_include_directories(jacobi_svd_kernels PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)

# ---------------------------------------------------------------- 主机侧
add_library(jacobi_svd_tiling STATIC jacobi_svd_tiling.cpp)
add_executable(jacobi_svd_main jacobi_svd_main.cpp)

target_include_directories(jacobi_svd_tiling PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
target_include_directories(jacobi_svd_main PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_directories(jacobi_svd_main PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(jacobi_svd_main PRIVATE jacobi_svd_kernels jacobi_svd_tiling ascendcl)

# ---------------------------------------------------------------- 数据生成 / 结果验证
# 可用 cmake -DSVD_DEMO_ARGS="--rows 256 --cols 128 --batch 16" 覆盖默认参数
set(SVD_DEMO_ARGS "--rows 128 --cols 128 --batch 8" CACHE STRING "gen/verify 使用的数据规模")
separate_arguments(SVD_DEMO_ARG_LIST UNIX_COMMAND "${SVD_DEMO_ARGS}")

add_custom_target(gen_data
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_jacobi_svd_data.py ${SVD_DEMO_ARG_LIST}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "生成批量输入矩阵与参考奇异值")

add_custom_target(verify_svd
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_jacobi_svd.py ${SVD_DEMO_ARG_LIST}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "验证 NPU 输出的奇异值 / 正交性 / 重构误差")
