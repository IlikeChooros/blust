add_executable(${PROJECT_NAME} main.cpp)

add_subdirectory(blust)

target_link_libraries(${PROJECT_NAME} blust)