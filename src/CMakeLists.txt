file(GLOB COMMON_SOURCES CONFIGURE_DEPENDS
    common/*.cpp
)

add_library(common STATIC ${COMMON_SOURCES})
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

file(GLOB INFERENCE_SOURCES CONFIGURE_DEPENDS
    inference/*.cpp
)
option(USE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(USE_OPENCV "Enable OpenCV image processing" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime (contains include/lib/bin)")
set(ONNXRUNTIME_INCLUDE "" CACHE PATH "Override path to onnxruntime_cxx_api.h directory")
set(ONNXRUNTIME_LIB "" CACHE FILEPATH "Override path to onnxruntime.lib")
set(ONNXRUNTIME_BIN_DIR "" CACHE PATH "Directory containing ONNX Runtime DLLs to copy next to executables")

add_library(inference STATIC ${INFERENCE_SOURCES})
target_link_libraries(inference PUBLIC common)
target_include_directories(inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(USE_ONNXRUNTIME)
    if(NOT ONNXRUNTIME_INCLUDE)
        find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
            HINTS ${ONNXRUNTIME_ROOT}
            PATH_SUFFIXES include)
    endif()
    if(NOT ONNXRUNTIME_LIB)
        find_library(ONNXRUNTIME_LIB NAMES onnxruntime
            HINTS ${ONNXRUNTIME_ROOT}
            PATH_SUFFIXES lib)
    endif()
    if(ONNXRUNTIME_INCLUDE AND ONNXRUNTIME_LIB)
        target_sources(inference PRIVATE inference/onnx_backend.cpp)
        target_include_directories(inference PRIVATE ${ONNXRUNTIME_INCLUDE})
        target_link_libraries(inference PRIVATE ${ONNXRUNTIME_LIB})
        target_compile_definitions(inference PUBLIC DIP_HAS_ONNX)
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Provide -DONNXRUNTIME_ROOT or -DONNXRUNTIME_INCLUDE and -DONNXRUNTIME_LIB. Example: -DONNXRUNTIME_ROOT=C:/libs/onnxruntime")
    endif()
endif()

if(USE_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
    target_link_libraries(inference PRIVATE ${OpenCV_LIBS})
    target_include_directories(inference PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(inference PUBLIC DIP_HAS_OPENCV)
endif()

option(USE_NETWORKING "Enable networking for multi-device" ON)
if(USE_NETWORKING)
    file(GLOB NETWORK_SOURCES CONFIGURE_DEPENDS
        networking/*.cpp
        master/net_master.cpp
    )
    add_library(networking STATIC ${NETWORK_SOURCES})
    target_include_directories(networking PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_compile_definitions(networking PUBLIC DIP_HAS_NETWORKING)
    if(MSVC)
        target_link_libraries(networking PUBLIC ws2_32)
    endif()
endif()

if(BUILD_MASTER)
    add_executable(master master/main.cpp)
    target_link_libraries(master PRIVATE common inference networking)
    target_include_directories(master PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    if(ONNXRUNTIME_BIN_DIR)
        add_custom_command(TARGET master POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll" $<TARGET_FILE_DIR:master>
        )
    endif()
endif()

if(BUILD_WORKER)
    add_executable(worker worker/main.cpp)
    target_link_libraries(worker PRIVATE common inference networking)
    target_include_directories(worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    if(ONNXRUNTIME_BIN_DIR)
        add_custom_command(TARGET worker POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll" $<TARGET_FILE_DIR:worker>
        )
    endif()
endif()
