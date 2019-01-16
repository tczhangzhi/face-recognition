# CMake 版本号
cmake_minimum_required(VERSION 3.5)

# 项目名称
PROJECT(face-recognition)

# 二进制文件
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./dist)

# 引入 OpenCV 包
find_package(OpenCV REQUIRED)

# 头文件
include_directories(
  ${OpenCV_INCLUDE_DIRS},
  ${CMAKE_CURRENT_LIST_DIR}/include
)

# 源文件
aux_source_directory(./src src)
add_executable(main ${src})

# 库文件
link_directories(${OpenCV_LIBRARY_DIRS})
target_link_libraries(main ${OpenCV_LIBS})