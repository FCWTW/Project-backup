#!/usr/bin/env python3

"""
ROS YOLO Detection Node with Depth Integration and Point Cloud Generation

This node performs YOLO object detection on RGB-D camera streams,
generates 3D point clouds, and publishes detection results with depth information.

Author: [Your Name]
Date: [Date]
"""

import ros_numpy
import rospy
import time
import cv2
import cv_bridge
import message_filters
import numpy as np
import image_geometry
import struct
import socket
import pickle
import sys

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

HOST = '172.18.0.1'
PORT = 5001

# COCO Dataset Class Names
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Global variables
processing = False
sock = None
connection_established = False
last_process_time = 0.0
frame_count = 0
skip_count = 0

def initialize_node():
    """Initialize ROS node and parameters"""
    rospy.loginfo(f"Python version: {sys.version}")
    rospy.loginfo(f"NumPy version: {np.__version__}")
    
    rospy.init_node("yolo_detector")
    
    target_fps = rospy.get_param('~target_fps', 1.0)
    process_interval = 1.0 / target_fps
    rospy.loginfo(f"Target processing FPS: {target_fps}, interval: {process_interval:.3f}s")
    
    return target_fps, process_interval

def setup_publishers():
    """Setup ROS publishers"""
    det_image_pub = rospy.Publisher("/yolo/detection/image", Image, queue_size=5)
    scene_pointcloud_pub = rospy.Publisher("/yolo/scene/pointcloud", PointCloud2, queue_size=5)
    return det_image_pub, scene_pointcloud_pub

def connect_to_host():
    """Establish connection to host with retry mechanism"""
    global sock, connection_established
    max_retries = 5
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            if sock:
                sock.close()
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((HOST, PORT))
            connection_established = True
            rospy.loginfo("Successfully connected to server at %s:%d", HOST, PORT)
            return True
            
        except socket.error as e:
            rospy.logwarn(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                rospy.loginfo(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                rospy.logerr("Failed to connect after %d attempts", max_retries)
                connection_established = False
                return False
    
    return False

def safe_recv(sock, size):
    """Safely receive data from socket ensuring complete data reception"""
    data = b''
    while len(data) < size:
        try:
            packet = sock.recv(size - len(data))
            if not packet:
                raise socket.error("Connection closed by peer")
            data += packet
        except socket.timeout:
            raise socket.error("Receive timeout")
        except socket.error as e:
            raise e
    return data

def send_image_and_receive_result(resized_image, transform_info):
    """Send resized image and transform info, receive YOLO inference results"""
    global sock, connection_established
    
    try:
        # Compress image to JPEG to reduce transmission size
        _, img_encoded = cv2.imencode('.jpg', resized_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_bytes = img_encoded.tobytes()

        # Prepare payload
        payload = {
            "image": img_bytes,
            "transform_info": transform_info
        }
        payload_bytes = pickle.dumps(payload)
        print(f"payload size: {len(payload_bytes)}")

        # Send packet size and data
        sock.sendall(struct.pack('>I', len(payload_bytes)))
        sock.sendall(payload_bytes)
        rospy.logdebug("Resized image + transform_info sent to host (%d bytes)", len(payload_bytes))

        # Receive result size and data
        size_data = safe_recv(sock, 4)
        data_len = struct.unpack('>I', size_data)[0]
        result_data = safe_recv(sock, data_len)
        result = pickle.loads(result_data)

        rospy.logdebug("Received result from host (%d bytes)", data_len)
        return result

    except socket.error as e:
        rospy.logerr("Socket error during communication: %s", e)
        connection_established = False
        raise e
    except Exception as e:
        rospy.logerr("Error during image processing: %s", e)
        raise e

class LetterBox:
    """Image preprocessing with letterboxing for YOLOv8n"""
    
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Calculate scaled ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Return transformation information
        transform_info = {
            'ratio': ratio,
            'pad': (left, top),
            'original_shape': shape,
            'new_shape': new_shape
        }
        
        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            labels["transform_info"] = transform_info
            return labels
        else:
            return img, transform_info

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels for letterboxed image"""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

def generate_colors(num_classes):
    """Generate fixed color palette for different classes"""
    colors = []
    golden_angle = 137.508
    
    for i in range(num_classes):
        hue = (i * golden_angle) % 360
        
        # Alternate saturation and value for better variation
        if i % 4 == 0:
            saturation, value = 255, 255
        elif i % 4 == 1:
            saturation, value = 200, 255
        elif i % 4 == 2:
            saturation, value = 255, 200
        else:
            saturation, value = 180, 255
        
        # Convert HSV to BGR
        hsv = np.uint8([[[int(hue//2), saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors[:num_classes]

def visualizer(image, boxes, labels):
    """Draw detection results on image with different colors for each class"""
    if not boxes:
        return image
    h, w = image.shape[:2]
    colors = generate_colors(len(labels))

    for i in range(len(boxes)):
        x1 = max(0, int(boxes[i]["xyxy"][0]))
        y1 = max(0, int(boxes[i]["xyxy"][1]))
        x2 = min(w, int(boxes[i]["xyxy"][2]))
        y2 = min(h, int(boxes[i]["xyxy"][3]))
        cls_id = int(boxes[i]["cls"])
        conf = boxes[i]["conf"]

        if x1 >= x2 or y1 >= y2:
            continue

        color = colors[cls_id % len(colors)]
        cls_name = labels[cls_id] if cls_id < len(labels) else f"class_{cls_id}"
        print(f'Detected: {cls_name}, Confidence: {conf:.2f}, Coordinates: ({x1}, {y1}, {x2}, {y2}), Color: {color}')

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
        label_text = f'{cls_name} {conf:.2f}'
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
        brightness = sum(color) / 3
        text_color = (255, 255, 255) if brightness < 127 else (0, 0, 0)
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return image

def callback(rgb_msg, depth_msg, depth_info_msg):
    """Main callback function for processing synchronized RGB-D data"""
    global processing, connection_established, last_process_time, frame_count, skip_count
    
    frame_count += 1
    
    if processing:
        skip_count += 1
        return
    
    # Frame rate control
    current_time = time.time()
    if current_time - last_process_time < process_interval:
        skip_count += 1
        return
    
    processing = True
    last_process_time = current_time
    
    # Report statistics every 100 frames
    if frame_count % 100 == 0:
        rospy.loginfo(f"Frame statistics: processed={frame_count-skip_count}, skipped={skip_count}, total={frame_count}")
    
    try:
        start_time = time.time()

        # Check connection status
        if not connection_established:
            rospy.logwarn("No connection to host, attempting to reconnect...")
            if not connect_to_host():
                rospy.logwarn("Cannot connect to host, skipping frame")
                return

        # Parse camera parameters
        cam_model.fromCameraInfo(depth_info_msg)
        fx, fy, cx, cy = cam_model.fx(), cam_model.fy(), cam_model.cx(), cam_model.cy()

        # Convert ROS images to OpenCV format
        try:
            rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            depth_image_meters = depth_image_raw.astype(np.float32) / 1000.0
            h, w = depth_image_meters.shape
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # Image preprocessing
        if rgb_image is None:
            rospy.logerr("rgb_image is None!")
            return
        
        rospy.loginfo(f"RGB image shape: {rgb_image.shape}")
        letterbox = LetterBox(new_shape=(640, 640))
        
        try:
            resized_image, transform_info = letterbox(image=rgb_image)
            rospy.loginfo(f"Prepared image size for transmission: {resized_image.shape}")
        except Exception as e:
            rospy.logerr(f"LetterBox failed: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return

        # Send image and receive YOLO results
        img_bbox = rgb_image.copy()
        try:
            result_data = send_image_and_receive_result(resized_image, transform_info)
            boxes = result_data["boxes"]
            img_bbox = visualizer(rgb_image, boxes, COCO_CLASSES)
        except Exception as e:
            rospy.logerr("Failed to process image with host: %s", e)
            if not connect_to_host():
                rospy.logwarn("Cannot reconnect, skipping frame")
            return
    
        # Generate scene point cloud with optimized sampling
        step = 4  # Sampling interval to reduce computation
        v_indices, u_indices = np.mgrid[0:h:step, 0:w:step]
        v_indices = v_indices.flatten()
        u_indices = u_indices.flatten()
        
        depths = depth_image_meters[v_indices, u_indices]
        valid_mask = (depths > 0.01) & (depths < 20.0) & (~np.isnan(depths))
        
        v_valid = v_indices[valid_mask]
        u_valid = u_indices[valid_mask]
        z_valid = depths[valid_mask]
        
        # Calculate 3D coordinates
        x_valid = (u_valid - cx) * z_valid / fx
        y_valid = (v_valid - cy) * z_valid / fy
        
        # Get colors
        colors = rgb_image[v_valid, u_valid]
        
        # Create point arrays
        scene_points = np.vstack((x_valid, y_valid, z_valid)).T
        scene_colors = colors[:, [2,1,0]]  # BGR to RGB
        
        # Create point labels array (-1 means not belonging to any object)
        point_labels = np.zeros(len(scene_points), dtype=np.int32) - 1
        
        # Process each detected object
        for i, box in enumerate(boxes):
            cls_id = int(box['cls'])
            cls_name = COCO_CLASSES[cls_id]
            conf = float(box['conf'])
            xyxy = box['xyxy']
            
            # Mark points within object bounding box
            u_min, v_min, u_max, v_max = map(int, [max(0, xyxy[0]), max(0, xyxy[1]), 
                                              min(w-1, xyxy[2]), min(h-1, xyxy[3])])
            
            # Find points belonging to this object
            in_box_mask = (u_valid >= u_min) & (u_valid <= u_max) & (v_valid >= v_min) & (v_valid <= v_max)
            if np.any(in_box_mask):
                box_depths = z_valid[in_box_mask]
                min_depth = np.min(box_depths)
                point_labels[in_box_mask] = i
                rospy.loginfo(f"Object {cls_name} (confidence: {conf:.2f}): closest distance {min_depth:.2f} m")
        
        # Color object points in point cloud
        scene_colors_copy = scene_colors.copy()
        for i, box in enumerate(boxes):
            obj_mask = (point_labels == i)
            if np.any(obj_mask):
                # Get depth range for this object
                obj_depths = z_valid[obj_mask]
                min_depth = np.min(obj_depths)
                max_depth = np.max(obj_depths)
                depth_range = max(max_depth - min_depth, 0.1)
                
                # Color based on depth
                normalized_depths = (obj_depths - min_depth) / depth_range
                r = (255 * (1 - normalized_depths)).astype(np.uint8)
                g = np.ones_like(r) * 128
                b = (255 * normalized_depths).astype(np.uint8)
                
                scene_colors_copy[obj_mask, 0] = r
                scene_colors_copy[obj_mask, 1] = g
                scene_colors_copy[obj_mask, 2] = b
        
        # Publish scene point cloud
        if len(scene_points) > 0:
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1),
            ]
            
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_depth_optical_frame"
            
            # Pack points and colors
            packed_points = []
            for i in range(len(scene_points)):
                pt = scene_points[i]
                color = scene_colors_copy[i]
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                rgb_packed = (r << 16) | (g << 8) | b
                packed_points.append([pt[0], pt[1], pt[2], rgb_packed])
            
            scene_cloud_msg = pc2.create_cloud(header, fields, packed_points)
            scene_pointcloud_pub.publish(scene_cloud_msg)
            rospy.loginfo(f"Published scene point cloud with {len(scene_points)} points.")
        
        # Add depth points and information to detection results
        for i, box in enumerate(boxes):
            obj_mask = (point_labels == i)
            if np.any(obj_mask):
                obj_depths = z_valid[obj_mask]
                min_depth = np.min(obj_depths)
                max_depth = np.max(obj_depths)
                depth_range = max(max_depth - min_depth, 0.1)
                
                obj_u = u_valid[obj_mask]
                obj_v = v_valid[obj_mask]
                normalized_depths = (obj_depths - min_depth) / depth_range
                
                # Draw depth points with reduced density
                skip = 4
                for j in range(0, len(obj_u), skip):
                    u, v = int(obj_u[j]), int(obj_v[j])
                    depth_val = normalized_depths[j]
                    
                    # Color based on depth (red=near, blue=far)
                    r = int(255 * (1 - depth_val))
                    g = 0
                    b = int(255 * depth_val)
                    
                    cv2.circle(img_bbox, (u, v), 2, (b, g, r), -1)
                
                # Add depth text
                xyxy = box['xyxy']
                u_min, v_min = map(int, [xyxy[0], xyxy[1]])
                depth_text = f"depth: {min_depth:.2f}m"
                cv2.putText(img_bbox, depth_text, (u_min, v_min - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Publish detection results with depth information
        det_image_pub.publish(bridge.cv2_to_imgmsg(img_bbox, encoding="bgr8"))

        process_time = time.time() - start_time
        rospy.loginfo(f"Detection and point cloud processing completed. Process time: {process_time:.3f}s")

    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
        connection_established = False
    finally:
        processing = False

def cleanup():
    """Clean up resources"""
    global sock
    if sock:
        try:
            sock.close()
        except:
            pass
    rospy.loginfo("Resources cleaned up")

if __name__ == "__main__":
    try:
        # Initialize node and get parameters
        target_fps, process_interval = initialize_node()
        
        # Setup publishers and other components
        det_image_pub, scene_pointcloud_pub = setup_publishers()
        bridge = cv_bridge.CvBridge()
        cam_model = image_geometry.PinholeCameraModel()
        
        # Establish initial connection
        if not connect_to_host():
            rospy.logfatal("Cannot establish initial connection to host. Exiting.")
            exit(1)
        
        # Register cleanup function
        rospy.on_shutdown(cleanup)
        
        # Setup message synchronization
        rgb_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth/image_rect_raw"
        depth_info_topic = "/camera/depth/camera_info"
        
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        depth_info_sub = message_filters.Subscriber(depth_info_topic, CameraInfo)
        
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, depth_info_sub],
            queue_size=10,
            slop=0.1
        )
        ts.registerCallback(callback)
        
        # Log startup information
        rospy.loginfo(f"YOLO detector started with processing frequency: {target_fps} FPS")
        rospy.loginfo(f"Subscribed topics - RGB: {rgb_topic}, Depth: {depth_topic}, Camera Info: {depth_info_topic}")
        rospy.loginfo(f"Publishing annotated images to: /yolo/detection/image")
        rospy.loginfo(f"Publishing scene point cloud to: /yolo/scene/pointcloud")
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by user")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        cleanup()