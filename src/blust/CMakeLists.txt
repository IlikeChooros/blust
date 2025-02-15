add_library(blust)

target_sources(blust PRIVATE
    src/BaseLayer.cpp
)

target_include_directories(blust PUBLIC include)