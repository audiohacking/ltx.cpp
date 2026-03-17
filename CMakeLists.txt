cmake_minimum_required(VERSION 3.16)
project(ltx.cpp VERSION 0.1.0 LANGUAGES CXX C)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

option(LTX_CUDA    "Enable CUDA backend"   OFF)
option(LTX_VULKAN  "Enable Vulkan backend" OFF)
option(LTX_METAL   "Enable Metal backend"  OFF)
option(LTX_HIP     "Enable ROCm/HIP backend" OFF)

# ── GGML submodule ──────────────────────────────────────────────────────────
if(LTX_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(LTX_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(LTX_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(LTX_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# ── Common interface library ─────────────────────────────────────────────────
add_library(ltx_common INTERFACE)
target_include_directories(ltx_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(ltx_common INTERFACE ggml)

# ── ltx-generate binary ──────────────────────────────────────────────────────
add_executable(ltx-generate src/ltx-generate.cpp)
target_link_libraries(ltx-generate PRIVATE ltx_common)

if(MSVC)
    target_compile_options(ltx-generate PRIVATE /W3)
else()
    # -Wno-unused-but-set-variable suppresses warnings from vendored stb_image.h.
    # The flag is supported by GCC and Apple Clang 13+; guard it so older
    # toolchains don't fail with an "unrecognized option" error.
    check_cxx_compiler_flag(-Wno-unused-but-set-variable HAS_WNO_UNUSED_BUT_SET_VARIABLE)
    set(_ltx_generate_warn_flags -Wall -Wextra -Wno-unused-parameter)
    if(HAS_WNO_UNUSED_BUT_SET_VARIABLE)
        list(APPEND _ltx_generate_warn_flags -Wno-unused-but-set-variable)
    endif()
    target_compile_options(ltx-generate PRIVATE ${_ltx_generate_warn_flags})
endif()

# ── quantize utility ─────────────────────────────────────────────────────────
add_executable(ltx-quantize src/ltx-quantize.cpp)
target_link_libraries(ltx-quantize PRIVATE ltx_common)

install(TARGETS ltx-generate ltx-quantize RUNTIME DESTINATION bin)
