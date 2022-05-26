cmake_minimum_required(VERSION 2.8.3)
project(estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# 添加cmake模块以使用g2o
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/cmake_modules)

find_package(OpenCV 3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3)




include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
  ${catkin_INCLUDE_DIRS} 
)


# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# catkin_package()
# pose_estimation_BA_main
add_executable(pose_estimation_BA_main src/pose_estimation_BA_main.cpp)
# # 添加依赖库
target_link_libraries(pose_estimation_BA_main ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} 
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
) 
# motion_pose_polar_constraints_main
add_executable(motion_pose_polar_constraints_main src/motion_pose_polar_constraints_main.cpp src/features_pose.cpp)
# 添加依赖库
target_link_libraries(motion_pose_polar_constraints_main ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} 
) 
add_executable(fetures_extrac_macth_main src/fetures_extrac_macth_main.cpp)
# 添加依赖库
target_link_libraries(fetures_extrac_macth_main ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} 
) 