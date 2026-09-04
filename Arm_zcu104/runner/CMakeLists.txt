cmake_minimum_required(VERSION 3.19)
project(hyperstarcop_arm_executorch LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EXECUTORCH_INSTALL_DIR)
  message(FATAL_ERROR "Informe -DEXECUTORCH_INSTALL_DIR=/caminho/install")
endif()

include(
  "${EXECUTORCH_INSTALL_DIR}/lib/cmake/ExecuTorch/executorch-config.cmake"
)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgcodecs)

add_executable(
  hyperstarcop_arm_executorch
  hyperstarcop_arm_executorch.cpp
)

target_include_directories(
  hyperstarcop_arm_executorch PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  hyperstarcop_arm_executorch
  PRIVATE
    executorch
    extension_module_static
    extension_tensor
    extension_threadpool
    xnnpack_backend
    portable_ops_lib
    ${OpenCV_LIBS}
)

target_link_options(
  hyperstarcop_arm_executorch PRIVATE -Wl,--gc-sections -s
)
