cmake_minimum_required(VERSION 3.0.2)
project(wpb_task1_owner_search)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  geometry_msgs
  move_base_msgs
  nav_msgs
  perception_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
)

catkin_package(
  CATKIN_DEPENDS actionlib actionlib_msgs cv_bridge geometry_msgs move_base_msgs nav_msgs perception_msgs rospy sensor_msgs std_msgs std_srvs tf
)

catkin_install_python(PROGRAMS
  scripts/owner_voice_reid_test.py
  scripts/owner_voice_reid_optimized.py
  scripts/person_reid_owner_test.py
  scripts/robot_mic_test.py
  scripts/task1_find_owner_real.py
  scripts/yoloworld_async.py
  scripts/yoloworld_debug_viewer_stable.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config data docs launch maps models third_party tools
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
