cmake_minimum_required(VERSION 3.16)
project(hermes_client)

set(manus_sdk_root "C:/path/to/Manus_Core_C++_SDK_v1.6.0_bbde")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

add_executable(${PROJECT_NAME}_node src/test.cpp)

target_compile_options(${PROJECT_NAME}_node PRIVATE /wd4005 /wd4099 /wd4244 /wd4251 /wd4275)
target_link_options(${PROJECT_NAME}_node PRIVATE -IGNORE:4006 -IGNORE:4099 -IGNORE:4217 -IGNORE:4286)

set_property(TARGET ${PROJECT_NAME}_node PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME}_node PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(${PROJECT_NAME}_node SYSTEM PRIVATE "${manus_sdk_root}/cpp/Externals/protobuf-3.11.2/src")
target_include_directories(${PROJECT_NAME}_node SYSTEM PRIVATE "${manus_sdk_root}/cpp/Externals/spdlog/include")
target_include_directories(${PROJECT_NAME}_node SYSTEM PRIVATE "${manus_sdk_root}/core")
target_include_directories(${PROJECT_NAME}_node SYSTEM PRIVATE ${catkin_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}_node PRIVATE debug "${manus_sdk_root}/cpp/Generated/Output/Debug/x64/HermesSDK.lib")
target_link_libraries(${PROJECT_NAME}_node PRIVATE optimized "${manus_sdk_root}/cpp/Generated/Output/Release/x64/HermesSDK.lib")
target_link_libraries(${PROJECT_NAME}_node PRIVATE ${catkin_LIBRARIES})