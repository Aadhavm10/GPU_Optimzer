# NVML monitoring library
add_library(nvml_monitoring
    nvml_monitor.cpp
    metrics_collector.cpp
    data_processor.cpp
    gpu_metrics.cpp
)

target_include_directories(nvml_monitoring PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nvml_monitoring
    ${NVML_LIBRARY}
    Threads::Threads
)
