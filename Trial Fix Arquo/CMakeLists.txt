cmake_minimum_required(VERSION 3.14)
project(MediapipeHolisticCPP)

set(CMAKE_CXX_STANDARD 17)

# ── OpenCV ────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ── Mediapipe (harus sudah di-build dengan Bazel) ─────────────
# Ganti path ini sesuai lokasi build Mediapipe kamu
set(MEDIAPIPE_DIR "C:/mediapipe" CACHE PATH "Path ke folder Mediapipe")

include_directories(
    ${MEDIAPIPE_DIR}
    ${MEDIAPIPE_DIR}/bazel-mediapipe
)

# ── Target ────────────────────────────────────────────────────
add_executable(mediapipe_holistic_cpp Mediapipe_Holistic_CPP.cpp)

target_link_libraries(mediapipe_holistic_cpp
    ${OpenCV_LIBS}
    # Tambahkan library Mediapipe sesuai hasil build Bazel
    # Contoh: ${MEDIAPIPE_DIR}/bazel-bin/mediapipe/framework/libcalculator_framework.a
)
