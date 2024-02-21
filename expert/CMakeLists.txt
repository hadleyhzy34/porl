cmake_minimum_required(VERSION 3.0.0)
project(
  AStar
  VERSION 0.1.0
  LANGUAGES C CXX)

include(GNUInstallDirs)
set(PACKAGE_NAME matplotlib_cpp)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}/cmake)
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_ROOT_DIR "/home/hadley/anaconda3/envs/new")

# Library target
add_library(matplotlib_cpp INTERFACE)
target_include_directories(
  matplotlib_cpp INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/examples>
                           $<INSTALL_INTERFACE:include>)
target_compile_features(matplotlib_cpp INTERFACE cxx_std_11)
# TODO: Use `Development.Embed` component when requiring cmake >= 3.18
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
target_link_libraries(matplotlib_cpp INTERFACE Python3::Python Python3::Module)
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  target_link_libraries(matplotlib_cpp INTERFACE Python3::NumPy)
else()
  target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
endif()
install(TARGETS matplotlib_cpp EXPORT install_targets)

include(CTest)
enable_testing()

add_library(ast src/AStar.h src/AStar.cpp)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(AStar ast matplotlib_cpp)
# target_link_libraries(AStar PRIVATE matplotlib_cpp)
set_target_properties(AStar PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                       "${CMAKE_BINARY_DIR}/bin")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
