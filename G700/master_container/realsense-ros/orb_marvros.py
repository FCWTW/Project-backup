#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import time

class SlamToMavrosBridge:
    def __init__(self):
        """
        初始化 SLAM 到 Mavros 的橋接節點。
        """
        rospy.init_node('slam_to_mavros_bridge', anonymous=True)
        
        # 建立一個發布者，將 PoseStamped 訊息發布給 Mavros
        # Mavros 會監聽 /mavros/vision_pose/pose 這個 Topic 來獲取外部定位資料
        self.vision_pose_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
        
        # 建立一個訂閱者，訂閱 ORB-SLAM3 發布的定位資訊
        # 根據你的 ORB-SLAM3 設置，這個 Topic 名稱可能需要調整
        # rospy.Subscriber('/orb_slam3_stereo/camera_pose', PoseStamped, self.vision_pose_callback)
        rospy.Subscriber('/orb_slam3/camera_pose', PoseStamped, self.vision_pose_callback)
        
        # 等待 Mavros 服務啟動
        rospy.loginfo("Waiting for Mavros to be ready...")
        time.sleep(5) # 暫停 5 秒，確保 Mavros 已啟動

        rospy.loginfo("Bridge node started. Publishing vision poses to Mavros.")

    def vision_pose_callback(self, data):
        """
        將 ORB-SLAM3 的 PoseStamped 訊息直接轉發給 Mavros。
        此處假設 ORB-SLAM3 的輸出座標系與 Mavros 的期望座標系（例如 ENU）一致。
        """
        try:
            # 創建一個新的 PoseStamped 訊息，以確保時間戳是最新的
            px4_pose = PoseStamped()
            px4_pose.header = data.header
            px4_pose.header.stamp = rospy.Time.now()

            # 直接複製位置和姿態
            px4_pose.pose.position = data.pose.position
            px4_pose.pose.orientation = data.pose.orientation

            # 發布訊息
            self.vision_pose_pub.publish(px4_pose)
            
            # 加入這行來顯示發布訊息
            # rospy.loginfo("Successfully published vision pose data.")

        except Exception as e:
            rospy.logerr("Error in vision pose callback: %s", str(e))

if __name__ == '__main__':
    try:
        # 創建並運行橋接器
        bridge = SlamToMavrosBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass