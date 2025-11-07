# 程式碼備份

## 檔案位置

### Master container
> 包含 realsense-ros 與 offboard_py 兩個 packages
>
> | 檔案 | 位置 |
> |:------|:-------|
> | /master_container/offboard_py/offb_node.py | catkin_ws/src/offboard_py/scripts/offb_node.py |
> | /master_container/realsense-ros/orb_marvros.py | catkin_ws/src/realsense-ros/realsense2_camera/scripts/orb_mavros.py |
> | /master_container/realsense-ros/rs_camera.launch | catkin_ws/src/realsense-ros/realsense2_camera/launch/rs_camera.launch |

### ORB-SLAM3 container
> | 檔案 | 位置 |
> |:------|:-------|
> | /orb_slam3_container/orb_slam_ros/rs_d435i_rgbd.launch | catkin_ws/src/orb_slam3_ros/launch/rs_d435i_rgbd.launch |
> | /orb_slam3_container/orb_slam_ros/realsense_D415.launch | catkin_ws/src/orb_slam3_ros/launch/realsense_D415.launch |
> | /orb_slam3_container/orb_slam_ros/orb_slam3_D415.rviz | catkin_ws/src/orb_slam3_ros/config/orb_slam3_D415.rviz |
> | /orb_slam3_container/orb_slam_ros/Realsense_D415.yaml | catkin_ws/src/orb_slam3_ros/config/Stereo/RealSense_D415.yaml |

### YOLO container
> | 檔案 | 位置 |
> |:------|:-------|
> | /yolo_container/yolo_detect_v2.py | catkin_ws/src/yolo_ros/scripts/yolo_detect.py |
> | /yolo_container/yolo_detect.launch | catkin_ws/src/yolo_ros/launch/yolo_detect.launch |
> | /yolo_container/yolo.rviz | catkin_ws/src/yolo_ros/config/yolo.rviz |

---
## 執行指令
