cmake_minimum_required(VERSION 3.10)
project(AiEngineTests)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

add_library(AiEngine SHARED IMPORTED)
set_target_properties(AiEngine PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libAiEngine.so)

add_executable(run_tests test_bridge.cpp)

target_link_libraries(run_tests gtest_main AiEngine log)
