import socket, pickle
import struct
import cv2
import numpy as np
import os
from utils.neuronpilot import neuronrt
import time, argparse
import logging
from typing import Optional, Tuple, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = '0.0.0.0'
PORT = 5001
SOCKET_TIMEOUT = 30.0

def receive_image(conn: socket.socket) -> Optional[Tuple[np.ndarray, dict]]:
    """
    接收圖片數據，加入錯誤處理和超時機制
    """
    try:
        # 設置超時
        conn.settimeout(SOCKET_TIMEOUT)
        
        # 先接收 4 bytes，表示圖片大小
        size_data = conn.recv(4)
        if len(size_data) != 4:
            logger.warning("Failed to receive complete size data")
            return None
            
        data_len = struct.unpack('>I', size_data)[0]
        
        # 檢查數據大小是否合理 (最大50MB)
        if data_len > 50 * 1024 * 1024:
            logger.warning(f"Image size too large: {data_len} bytes")
            return None
            
        logger.debug(f"Expecting {data_len} bytes of image data")
        
        data = b''
        while len(data) < data_len:
            remaining = data_len - len(data)
            chunk_size = min(4096, remaining)
            packet = conn.recv(chunk_size)
            if not packet:
                logger.warning("Connection closed while receiving image data")
                break
            data += packet
            
        if len(data) != data_len:
            logger.warning(f"Incomplete image data: received {len(data)}, expected {data_len}")
            return None

        logger.info(f"Received raw data size: {len(data)} bytes")
        
        # === 解 pickle ===
        payload = pickle.loads(data)
        if not isinstance(payload, dict):
            logger.error("Received payload is not a dict")
            return None
        
        # 取出影像 bytes 與 transform_info
        img_bytes = payload.get("image", None)
        transform_info = payload.get("transform_info", None)

        if img_bytes is None or transform_info is None:
            logger.error("Payload missing required keys (image or transform_info)")
            return None
    
        # JPEG 解碼
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image from payload")
            return None
        
        # logger.info(f"Decoded image shape: {img.shape}")
        # logger.info(f"Decoded image dtype: {img.dtype}")
        # logger.info(f"Decoded image min/max values: {img.min()}/{img.max()}")
        
        # 檢查圖片是否有異常值
        if img.max() <= 1.0:
            logger.warning("Image values seem to be normalized (0-1), might need scaling")
        elif img.max() > 255:
            logger.warning("Image values exceed 255, might be in wrong format")
    
        logger.debug(f"Decoded resized image shape: {img.shape}, dtype: {img.dtype}")
        return img, transform_info
        
    except socket.timeout:
        logger.warning("Socket timeout while receiving image")
        return None
    except Exception as e:
        logger.error(f"Error receiving image: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def send_result(conn: socket.socket, result_data: Dict[str, Any]) -> bool:
    """
    發送結果數據，加入錯誤處理
    """
    try:
        payload = pickle.dumps(result_data)
            
        # 發送大小和數據
        conn.sendall(struct.pack('>I', len(payload)) + payload)
        logger.debug(f"Sent result: {len(payload)} bytes")
        return True
        
    except Exception as e:
        logger.error(f"Error sending result: {e}")
        return False

def preprocess_image(image):
    """
    Preparing input images for NeuroPilot YOLO models
    """
    input_data = image.astype(np.float32)
    
    # Convert BGR to RGB
    input_data = input_data[..., ::-1]
    
    # Normalization
    input_data /= 255.0
    
    # Ensure continuous data storage
    input_data = np.ascontiguousarray(input_data)
    
    # Adding batch dimensions
    input_data = np.expand_dims(input_data, axis=0)
    
    print(f"Preprocessed input shape: {input_data.shape}, type: {input_data.dtype}")
    return input_data

def postprocess(preds, transform_info, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300, nc=80):
    """
    YOLO output post-processing functions for the NeuroPilot SDK
    
    preds: model output, shape (1, 8400, 84)
    transform_info: pre-processed transform information
    """
    results = []
    
    for i, pred in enumerate(preds):
        # Calculate the maximum class confidence for each box
        class_scores = np.max(pred[:, 5:5+nc], axis=1)
        class_ids = np.argmax(pred[:, 5:5+nc], axis=1)
        
        # Filter out boxes with confidence levels greater than the threshold
        conf_mask = class_scores > conf_thres
        filtered_pred = pred[conf_mask]
        filtered_scores = class_scores[conf_mask]
        filtered_class_ids = class_ids[conf_mask]
        
        print(f"Confidence threshold: {conf_thres}")
        print(f"The range of class confidence: {np.min(class_scores)} - {np.max(class_scores)}")
        print(f"Number of remaining boxes after filtering: {filtered_pred.shape[0]}")
        
        if filtered_pred.shape[0] == 0:  # Nothing detected
            results.append(None)
            continue
            
        # Convert coordinates to xyxy format (still normalized coordinates)
        boxes = filtered_pred[:, :4].copy()
        boxes = xywh2xyxy(boxes)
        
        # Converts normalized coordinates to absolute coordinates relative to the model input image
        input_h, input_w = transform_info['new_shape']
        boxes[:, [0, 2]] *= input_w  # x
        boxes[:, [1, 3]] *= input_h  # y
        
        # Remove letterbox padding and convert back to the original zoomed image coordinates
        pad_left, pad_top = transform_info['pad']
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        
        # Zoom back to original image size
        ratio_w, ratio_h = transform_info['ratio']
        boxes[:, [0, 2]] /= ratio_w  # x
        boxes[:, [1, 3]] /= ratio_h  # y
        
        # Ensure the coordinates are within the original image
        orig_h, orig_w = transform_info['original_shape']
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # Combination result: [x1, y1, x2, y2, conf, class_id]
        det = np.column_stack((boxes, filtered_scores, filtered_class_ids))
        
        # Non Max Suppression
        if det.shape[0] > 1:
            boxes_for_nms, scores = det[:, :4], det[:, 4]
            nms_indices = non_max_suppression(boxes_for_nms, scores, iou_thres)
            if isinstance(nms_indices, list):
                nms_indices = np.array(nms_indices)
            det = det[nms_indices[:max_det]]
        
        print(f"Number of remaining boxes after NMS: {det.shape[0]}")
        results.append(det)
    return results

def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y

def non_max_suppression(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--tflite_model", type=str, required=True, help="Path to .tflite")
    parser.add_argument("-d", "--device", type=str, default='mdla3.0', choices = ['mdla3.0', 'mdla2.0', 'vpu'], help="Device name for acceleration")
    args = parser.parse_args()

    if not os.path.exists(args.tflite_model):
        raise FileNotFoundError(f"Model file doesn't exist: {args.tflite_model}")
    
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./bin', exist_ok=True)
    logging.getLogger().setLevel(logging.DEBUG)

    # 初始化 neuronrt.Interpreter
    logger.info(f"Loading model: {args.tflite_model}")
    interpreter = neuronrt.Interpreter(model_path=args.tflite_model, device=args.device)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 獲取輸入形狀
    if len(input_details) > 0 and len(input_details[0]['shape']) >= 3:
        input_shape = tuple(input_details[0]['shape'][1:3])  # [batch, height, width, channels]
        if input_shape[0] == 0 or input_shape[1] == 0:
            input_shape = (640, 640)
    else:
        input_shape = (640, 640)
    
    logger.info(f"Input shape: {input_shape}")

    # 啟動伺服器
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允許重用地址
        s.bind((HOST, PORT))
        s.listen(1)
        logger.info(f"Server listening on {HOST}:{PORT}")
        
        while True:  # 外層循環處理多個連接
            try:
                conn, addr = s.accept()
                logger.info(f"Connected by {addr}")
                
                with conn:
                    while True:  # 內層循環處理單個連接的多個請求
                        try:
                            start_time = time.time()
                            
                            # 接收圖片
                            img, transform_info = receive_image(conn)
                            if img is None:
                                logger.info("No image received. Closing connection.")
                                continue

                            # 預處理
                            input_data = preprocess_image(img)

                            # 確保dtype正確
                            input_dtype = input_details[0]['dtype']
                            if input_data.dtype != input_dtype:
                                input_data = input_data.astype(input_dtype)

                            # 推論
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            output_data = interpreter.get_tensor(output_details[0]['index'])

                            # 後處理
                            output_data = output_data.transpose(0, 2, 1)
                            results = postprocess(output_data, transform_info, 
                                                conf_thres=0.25, 
                                                iou_thres=0.45)

                            # 取得 bounding box
                            for det in results:
                                if det is None:
                                    continue
                                
                                boxes = []
                                # Draw each bounding box
                                for i in range(det.shape[0]):
                                    x1, y1, x2, y2, conf, cls_id = det[i]
                                    cls_id = int(cls_id) + 1

                                    boxes.append({
                                        "cls": int(cls_id),
                                        "conf": float(conf),
                                        "xyxy": [float(x1), float(y1), float(x2), float(y2)]
                                    })

                            # 準備結果
                            result_data = {
                                "boxes": boxes,
                                "processing_time": time.time() - start_time
                            }
                            
                            # 發送結果
                            if not send_result(conn, result_data):
                                break
                                
                            logger.info(f"Processing time: {result_data['processing_time']:.3f}s, "
                                      f"Detected objects: {len(boxes)}")

                        except Exception as e:
                            logger.error(f"Error processing request: {e}")
                            break
                            
            except KeyboardInterrupt:
                logger.info("Server stopped by user")
                break
            except Exception as e:
                logger.error(f"Server error: {e}")
                continue