find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable(Display /home/lanpokn/Documents/2022/pbrt/pbrt-v4/pbrt-v4/IEBCS-main/cpp/utils/display.cpp)
target_link_libraries(Display ${OpenCV_LIBS})