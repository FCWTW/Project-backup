#!/usr/bin/env python3
import ros_numpy
import rospy
import time
import cv_bridge
import cv2
import torch
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from ultralytics import YOLO
import message_filters
import numpy as np
import image_geometry
import struct

# 檢查CUDA是否可用
def check_cuda():
    if torch.cuda.is_available():
        device = 'cuda'
        rospy.loginfo(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        rospy.logwarn("找不到可用的GPU，將使用CPU進行推論")
    return device

device = check_cuda()

# 初始化ROS節點和YOLO模型
rospy.init_node("yolo_detector_pointcloud")
detection_model = YOLO("yolo11n.pt").to(device)
detection_model.fuse()

# 創建影像和點雲發布者
det_image_pub = rospy.Publisher("/yolo/detection/image", Image, queue_size=5)
# 發布整個場景的點雲
scene_pointcloud_pub = rospy.Publisher("/yolo/scene/pointcloud", PointCloud2, queue_size=5)

# CV Bridge instance
bridge = cv_bridge.CvBridge()
# Camera Model instance (for CameraInfo)
cam_model = image_geometry.PinholeCameraModel()

# 全局變數
processing = False

def callback(rgb_msg, depth_msg, depth_info_msg):
    global processing
    if processing:
        return
    processing = True
    try:
        start_time = time.time()

        # 1. 解析相機參數
        cam_model.fromCameraInfo(depth_info_msg)
        fx = cam_model.fx()
        fy = cam_model.fy()
        cx = cam_model.cx()
        cy = cam_model.cy()

        # 2. 轉換ROS影像為OpenCV格式
        try:
            rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            depth_image_meters = depth_image_raw.astype(np.float32) / 1000.0
            h, w = depth_image_meters.shape
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            processing = False
            return

        # 3. YOLO推論
        with torch.cuda.amp.autocast(enabled=(device=='cuda')):
            det_results = detection_model(rgb_image, verbose=False, conf=0.5)
        result = det_results[0] if isinstance(det_results, list) else det_results
        boxes = result.boxes
        names = result.names

        # 4. 使用numpy優化的方式生成點雲
        # 先生成整個場景的點雲 (使用適當的採樣間隔)
        step = 3  # 適當的採樣間隔，平衡點雲密度和處理速度
        v_indices, u_indices = np.mgrid[0:h:step, 0:w:step]
        v_indices = v_indices.flatten()
        u_indices = u_indices.flatten()
        
        depths = depth_image_meters[v_indices, u_indices]
        valid_mask = (depths > 0.01) & (depths < 20.0) & (~np.isnan(depths))
        
        v_valid = v_indices[valid_mask]
        u_valid = u_indices[valid_mask]
        z_valid = depths[valid_mask]
        
        # 計算3D座標
        x_valid = (u_valid - cx) * z_valid / fx
        y_valid = (v_valid - cy) * z_valid / fy
        
        # 獲取顏色
        colors = rgb_image[v_valid, u_valid]
        
        # 轉換為列表格式
        scene_points = np.vstack((x_valid, y_valid, z_valid)).T
        scene_colors = colors[:, [2,1,0]]  # BGR轉RGB
        
        # 創建一個索引陣列，用於標記點屬於哪個物體
        point_labels = np.zeros(len(scene_points), dtype=np.int32) - 1  # -1表示不屬於任何物體
        
        # 5. 為每個物體框單獨處理
        for i, box in enumerate(boxes):
            cls_id = int(box.cls)
            cls_name = names[cls_id]
            conf = float(box.conf.cpu().numpy())
            xyxy = box.xyxy.cpu().numpy()[0]
            
            # 使用優化的方式標記物體框內的點
            u_min, v_min, u_max, v_max = map(int, [max(0, xyxy[0]), max(0, xyxy[1]), 
                                              min(w-1, xyxy[2]), min(h-1, xyxy[3])])
            
            # 找出屬於該物體框的點
            in_box_mask = (u_valid >= u_min) & (u_valid <= u_max) & (v_valid >= v_min) & (v_valid <= v_max)
            if np.any(in_box_mask):
                # 找出該物體框內的最近點
                box_depths = z_valid[in_box_mask]
                min_depth = np.min(box_depths)
                
                # 標記屬於該物體的點
                point_labels[in_box_mask] = i
                
                rospy.loginfo(f"物體 {cls_name} (置信度: {conf:.2f}): 最近距離 {min_depth:.2f} 米")
        
        # 為物體點著色 (用於點雲)
        scene_colors_copy = scene_colors.copy()
        for i, box in enumerate(boxes):
            # 找出屬於該物體的點
            obj_mask = (point_labels == i)
            if np.any(obj_mask):
                # 獲取該物體的深度範圍
                obj_depths = z_valid[obj_mask]
                min_depth = np.min(obj_depths)
                max_depth = np.max(obj_depths)
                depth_range = max(max_depth - min_depth, 0.1)  # 避免除以零
                
                # 根據深度為物體點著色
                normalized_depths = (obj_depths - min_depth) / depth_range
                r = (255 * (1 - normalized_depths)).astype(np.uint8)
                g = np.ones_like(r) * 128
                b = (255 * normalized_depths).astype(np.uint8)
                
                # 更新顏色
                scene_colors_copy[obj_mask, 0] = r
                scene_colors_copy[obj_mask, 1] = g
                scene_colors_copy[obj_mask, 2] = b
        
        # 6. 發布整個場景的點雲
        if len(scene_points) > 0:
            # 定義PointCloud2字段
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1),
            ]
            
            # 創建header
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_depth_optical_frame"
            
            # 打包點和顏色
            packed_points = []
            for i in range(len(scene_points)):
                pt = scene_points[i]
                color = scene_colors_copy[i]
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                rgb_packed = (r << 16) | (g << 8) | b
                packed_points.append([pt[0], pt[1], pt[2], rgb_packed])
            
            # 創建PointCloud2消息
            scene_cloud_msg = pc2.create_cloud(header, fields, packed_points)
            scene_pointcloud_pub.publish(scene_cloud_msg)
            rospy.loginfo(f"發布了包含 {len(scene_points)} 個點的場景點雲。")

        # 7. 使用YOLO的原始偵測結果
        det_annotated = result.plot()
        
        # 8. 在原始偵測結果上添加深度點和深度信息
        for i, box in enumerate(boxes):
            # 找出屬於該物體的點
            obj_mask = (point_labels == i)
            if np.any(obj_mask):
                # 獲取該物體的深度範圍
                obj_depths = z_valid[obj_mask]
                min_depth = np.min(obj_depths)
                max_depth = np.max(obj_depths)
                depth_range = max(max_depth - min_depth, 0.1)
                
                # 獲取點的座標和深度
                obj_u = u_valid[obj_mask]
                obj_v = v_valid[obj_mask]
                normalized_depths = (obj_depths - min_depth) / depth_range
                
                # 每隔幾個點繪製一次，減少密度但保持視覺效果
                skip = 3  # 跳過的點數，可以根據需要調整
                for j in range(0, len(obj_u), skip):
                    u, v = int(obj_u[j]), int(obj_v[j])
                    depth_val = normalized_depths[j]
                    
                    # 根據深度生成顏色 (紅色表示近，藍色表示遠)
                    r = int(255 * (1 - depth_val))
                    g = 0  # 移除綠色成分，增強紅藍對比
                    b = int(255 * depth_val)
                    
                    # 直接在偵測結果上繪製點
                    cv2.circle(det_annotated, (u, v), 3, (b, g, r), -1)  # OpenCV使用BGR
                
                # 添加深度信息文字
                xyxy = box.xyxy.cpu().numpy()[0]
                u_min, v_min = map(int, [xyxy[0], xyxy[1]])
                cls_name = names[int(box.cls)]
                depth_text = f"depth: {min_depth:.2f}m"
                
                right_top_x = u_min  
                right_top_y = v_min - 25  
                cv2.putText(det_annotated, depth_text, (right_top_x, right_top_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # 使用黃色
        
        # 9. 發布帶有深度信息的偵測結果
        det_image_pub.publish(bridge.cv2_to_imgmsg(det_annotated, encoding="bgr8"))

        process_time = time.time() - start_time
        rospy.loginfo(f"檢測與點雲處理完成。處理時間: {process_time:.3f}秒。使用設備: {device}")

    except Exception as e:
        rospy.logerr(f"處理圖像時出錯: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        processing = False

# 使用message_filters同步RGB、深度影像和相機參數
# 這邊要改成要接收的rostopic
rgb_topic = "/camera/color/image_raw"
depth_topic = "/camera/depth/image_rect_raw"
depth_info_topic = "/camera/depth/camera_info"

rgb_sub = message_filters.Subscriber(rgb_topic, Image)
depth_sub = message_filters.Subscriber(depth_topic, Image)
depth_info_sub = message_filters.Subscriber(depth_info_topic, CameraInfo)

# 使用ApproximateTimeSynchronizer實現時間同步
ts = message_filters.ApproximateTimeSynchronizer(
    [rgb_sub, depth_sub, depth_info_sub],
    queue_size=10,
    slop=0.1
)
ts.registerCallback(callback)

rospy.loginfo(f"YOLO檢測器已啟動，使用{device}進行推論，並結合深度資訊生成點雲")
rospy.loginfo(f"訂閱: RGB: {rgb_topic}, 深度: {depth_topic}, 相機參數: {depth_info_topic}")
rospy.loginfo(f"發布標註影像到: /yolo/detection/image")
rospy.loginfo(f"發布場景點雲到: /yolo/scene/pointcloud")

rospy.spin()


