import numpy as np
import cv2
from utils.neuronpilot import neuronrt

def preprocess_image(image, input_shape=(640, 640)):
    """
    為 NeuroPilot YOLO 模型準備輸入圖像
    
    Args:
        image (numpy.ndarray): 原始 BGR 圖像
        input_shape (tuple): 模型期望的輸入圖像尺寸
    
    Returns:
        numpy.ndarray: 處理後的模型輸入
    """
    # 調整圖像大小
    resized_image = cv2.resize(image, input_shape, interpolation=cv2.INTER_LINEAR)
    
    # 將圖像轉換為 float32
    input_data = resized_image.astype(np.float32)
    
    # 歸一化: 像素值範圍從 [0, 255] 縮放到 [0, 1]
    input_data /= 255.0
    
    # 添加 batch 維度
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data

def postprocess_detection(interpreter, image_shape, conf_threshold=0.5, iou_threshold=0.45):
    """
    處理 NeuroPilot YOLO 模型的原始輸出
    
    Args:
        interpreter (neuronrt.Interpreter): NeuroPilot 解釋器
        image_shape (tuple): 原始圖像尺寸 (height, width)
        conf_threshold (float): 置信度閾值
        iou_threshold (float): IoU 閾值
    
    Returns:
        dict: 處理後的檢測結果
    """
    # 獲取輸出詳情
    output_details = interpreter.get_output_details()
    
    # 讀取所有輸出
    outputs = [interpreter.get_tensor(detail['index']) for detail in output_details]
    
    # 注意：以下邏輯需要根據實際模型輸出調整
    # 假設輸出格式為 [num_detections, [x1, y1, x2, y2, conf, class]]
    detections = outputs[0]  # 可能需要調整
    
    # 篩選高於置信度閾值的檢測
    valid_detections = detections[detections[:, 4] > conf_threshold]
    
    # 調整框的座標到原始圖像尺寸
    h, w = image_shape
    input_shape = (640, 640)  # 與預處理保持一致
    scale_x = w / input_shape[0]
    scale_y = h / input_shape[1]
    
    valid_detections[:, [0, 2]] *= scale_x
    valid_detections[:, [1, 3]] *= scale_y
    
    # 非極大值抑制
    def nms(boxes, scores, iou_threshold):
        # 根據分數排序
        sorted_indices = np.argsort(scores)[::-1]
        
        keep = []
        while sorted_indices.size > 0:
            current = sorted_indices[0]
            keep.append(current)
            
            if sorted_indices.size == 1:
                break
            
            # 計算 IoU
            ious = compute_iou(boxes[current], boxes[sorted_indices[1:]])
            
            mask = ious <= iou_threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return keep

    def compute_iou(box, boxes):
        # 計算交並比
        # 實現取決於框的具體格式
        # 這是一個基本實現，可能需要根據實際情況調整
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box_area + boxes_area - intersection
        
        return intersection / union

    # 應用 NMS
    if len(valid_detections) > 0:
        keep_indices = nms(valid_detections[:, :4], valid_detections[:, 4], iou_threshold)
        final_detections = valid_detections[keep_indices]
    else:
        final_detections = []
    
    # 組織返回結果
    results = {
        'boxes': final_detections[:, :4],  # [x1, y1, x2, y2]
        'scores': final_detections[:, 4],  # 置信度
        'classes': final_detections[:, 5].astype(int)  # 類別 ID
    }
    
    return results

# 使用範例
def detect_with_neuronpilot(image):
    # 載入模型
    interpreter = neuronrt.Interpreter(model_path='yolo8n.tflite', device='mdla3.0')
    
    # 預處理
    input_data = preprocess_image(image)
    
    # 設置輸入
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    
    # 推理
    interpreter.invoke()
    
    # 後處理
    results = postprocess_detection(interpreter, image.shape[:2])
    
    return results