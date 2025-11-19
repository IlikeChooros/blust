project(BlustBench)

add_executable(
    BlustBench
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/make_csv.cpp
)

target_link_libraries(BlustBench PUBLIC blust)