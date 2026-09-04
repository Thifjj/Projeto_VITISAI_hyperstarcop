set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-11)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-11)

get_filename_component(
  ZCU104_SYSROOT
  "${CMAKE_CURRENT_LIST_DIR}/../sysroot-zcu104"
  ABSOLUTE
)

set(CMAKE_SYSROOT "${ZCU104_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${ZCU104_SYSROOT}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# GCC's Ubuntu cross package searches /usr/aarch64-linux-gnu/include before the
# requested sysroot. Disable that implicit search so no glibc 2.39 headers leak
# into binaries intended for the PetaLinux 2.34 target.
set(ZCU104_GCC_INCLUDE "/usr/lib/gcc-cross/aarch64-linux-gnu/11/include")
set(ZCU104_CXX_INCLUDE "${ZCU104_SYSROOT}/usr/include/c++/11.2.0")
set(CMAKE_C_FLAGS_INIT
  "-nostdinc -isystem ${ZCU104_GCC_INCLUDE} -isystem ${ZCU104_SYSROOT}/usr/include"
)
set(CMAKE_CXX_FLAGS_INIT
  "-nostdinc -nostdinc++ -isystem ${ZCU104_GCC_INCLUDE} -isystem ${ZCU104_CXX_INCLUDE} -isystem ${ZCU104_CXX_INCLUDE}/aarch64-xilinx-linux -isystem ${ZCU104_SYSROOT}/usr/include"
)

set(CMAKE_EXE_LINKER_FLAGS_INIT
  "-Wl,-rpath-link,${ZCU104_SYSROOT}/lib -Wl,-rpath-link,${ZCU104_SYSROOT}/usr/lib"
)
