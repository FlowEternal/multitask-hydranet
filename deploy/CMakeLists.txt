# cmake最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
set(PROJECT_NAME 	ONNX_Hydranet_API)
set(API_NAME		ONNX_Hydranet_API)
set(DEMO_NAME		ONNX_Hydranet_DEMO)
project(PROJECT_NAME)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin_arm)
	set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin_arm)
else()
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
endif()

# 是否使用c++11标准
option(USE_CPP11 "Enable project use c++11" ON)
if(USE_CPP11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

# 是否用OpenMP库
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 加入opencv和MNN头文件和库文件
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
	message(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/opencv/include)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/opencv/include)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/common)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/framework)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/graph)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/optimizer)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/providers)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/session)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/providers/cpu)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/providers/cuda)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/include/onnxruntime/core/providers/tensorrt)
else()
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv/include)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnx/include)
endif()


# 将源文件提取出来
FILE(GLOB FILES_MODEL "src/model/*.cpp")
FILE(GLOB FILES_INTERFACE "src/interface/*.cpp")

# 将头文件提取出来
FILE(GLOB FILES_MODEL_H "./src/model/*.h")
FILE(GLOB FILES_INTERFACE_H "./src/interface/*.h")

# 将头文件目标包含进include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/interface)

# 分组
source_group(Headers FILES ${FILES_MODEL_H})
source_group(InterFace FILES ${FILES_INTERFACE_H} ${FILES_INTERFACE})

# 生成动态库
add_library(${API_NAME} SHARED 
			${FILES_MODEL} 
			${FILES_INTERFACE} 
			${FILES_MODEL_H}
			${FILES_INTERFACE_H}
)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
	message(STATUS "current platform: Linux ")
	target_link_libraries(${API_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/opencv/lib/libopencv_world.so)
	target_link_libraries(${API_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/onnx/lib/libonnxruntime.so)  
else()
	message(STATUS "current platform: Windows")
	target_link_libraries(${API_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv/lib/opencv_world451.lib)
	target_link_libraries(${API_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnx/lib/onnxruntime.lib)  
endif()

# DEMO编译
FILE(GLOB DEMO_FILE "src/demo/*.cpp")
add_executable(${DEMO_NAME} ${DEMO_FILE})
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
	target_link_libraries(${DEMO_NAME} ${API_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_arm/opencv/lib/libopencv_world.so)
else()
	target_link_libraries(${DEMO_NAME} ${API_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv/lib/opencv_world451.lib)
endif()
