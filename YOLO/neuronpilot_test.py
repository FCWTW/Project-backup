import os
import numpy as np
import cv2
from utils.neuronpilot import neuronrt
import time

'''
Source:
https://github.com/R300-AI/ITRI-AI-Hub
https://github.com/R300-AI/MTK-genio-demo

Tool:
https://netron.app/
'''

class LetterBox:
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
        shape = img.shape[:2]  # 原始圖像高度, 寬度
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 計算縮放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
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
        
        # 返回變換資訊
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
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

def preprocess_image(image, input_shape=(640, 640)):
    """
    為 NeuroPilot YOLO 模型準備輸入圖像
    """
    # 使用 LetterBox 進行預處理，保持縱橫比
    letterbox = LetterBox(new_shape=input_shape)
    resized_image, transform_info = letterbox(image=image)
    
    # 確認輸入圖像形狀
    print(f"調整後的圖像形狀: {resized_image.shape}")
    
    # 將圖像轉換為 float32
    input_data = resized_image.astype(np.float32)
    
    # 將 BGR 轉換為 RGB
    input_data = input_data[..., ::-1]
    
    # 歸一化: 像素值範圍從 [0, 255] 縮放到 [0, 1]
    input_data /= 255.0
    
    # 確保數據連續存儲
    input_data = np.ascontiguousarray(input_data)
    
    # 添加 batch 維度
    input_data = np.expand_dims(input_data, axis=0)
    
    print(f"預處理後的輸入形狀: {input_data.shape}, 類型: {input_data.dtype}")
    return input_data, transform_info

def postprocess(preds, transform_info, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300, nc=80):
    """
    適用於NeuroPilot SDK的YOLO輸出後處理函數
    
    preds: 模型輸出，形狀為(1, 8400, 84)
    transform_info: 預處理時的變換資訊
    """
    results = []
    
    for i, pred in enumerate(preds):  # 遍歷每個batch
        # 計算每個框的最大類別置信度
        class_scores = np.max(pred[:, 5:5+nc], axis=1)
        class_ids = np.argmax(pred[:, 5:5+nc], axis=1)
        
        # 篩選出置信度大於閾值的框
        conf_mask = class_scores > conf_thres
        filtered_pred = pred[conf_mask]
        filtered_scores = class_scores[conf_mask]
        filtered_class_ids = class_ids[conf_mask]
        
        print(f"置信度閾值: {conf_thres}")
        print(f"類別置信度範圍: {np.min(class_scores)} - {np.max(class_scores)}")
        print(f"篩選後剩餘框數: {filtered_pred.shape[0]}")
        
        if filtered_pred.shape[0] == 0:  # 沒有檢測到物體
            results.append(None)
            continue
            
        # 處理坐標 - YOLO輸出的是歸一化座標 (相對於模型輸入尺寸)
        boxes = filtered_pred[:, :4].copy()
        
        # 轉換為xyxy格式 (仍然是歸一化座標)
        boxes = xywh2xyxy(boxes)
        
        # 將歸一化座標轉為相對於模型輸入圖像的絕對座標
        input_h, input_w = transform_info['new_shape']
        boxes[:, [0, 2]] *= input_w  # x座標
        boxes[:, [1, 3]] *= input_h  # y座標
        
        # 移除 letterbox padding，轉換回縮放後的原始圖像座標
        pad_left, pad_top = transform_info['pad']
        boxes[:, [0, 2]] -= pad_left  # 移除左側padding
        boxes[:, [1, 3]] -= pad_top   # 移除頂部padding
        
        # 縮放回原始圖像尺寸
        ratio_w, ratio_h = transform_info['ratio']
        boxes[:, [0, 2]] /= ratio_w  # x座標還原
        boxes[:, [1, 3]] /= ratio_h  # y座標還原
        
        # 確保座標在原始圖像範圍內
        orig_h, orig_w = transform_info['original_shape']
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # 組合結果: [x1, y1, x2, y2, conf, class_id]
        det = np.column_stack((boxes, filtered_scores, filtered_class_ids))
        
        # 執行 NMS
        if det.shape[0] > 1:
            boxes_for_nms, scores = det[:, :4], det[:, 4]
            nms_indices = non_max_suppression(boxes_for_nms, scores, iou_thres)
            if isinstance(nms_indices, list):
                nms_indices = np.array(nms_indices)
            det = det[nms_indices[:max_det]]
        
        print(f"NMS後剩餘框數: {det.shape[0]}")
        print(f"檢測到的類別: {np.unique(det[:, 5])}")
        
        results.append(det)
    
    return results

def generate_colors(num_classes):
    """
    生成固定的顏色調色板
    """
    colors = []
    golden_angle = 137.508
    
    for i in range(num_classes):
        hue = (i * golden_angle) % 360
        
        # 交替使用不同的飽和度和亮度組合，增加變化
        if i % 4 == 0:
            saturation, value = 255, 255
        elif i % 4 == 1:
            saturation, value = 200, 255
        elif i % 4 == 2:
            saturation, value = 255, 200
        else:
            saturation, value = 180, 255
        
        # 轉換HSV到BGR
        hsv = np.uint8([[[int(hue//2), saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    
    return colors[:num_classes]

def visualizer(image, results, labels, input_shape=(640, 640)):
    """
    將檢測結果繪製到圖像上，每個類別使用不同顏色
    """
    if results is None or len(results) == 0 or results[0] is None:
        return image

    h, w = image.shape[:2]
    print(f"原始圖像尺寸: {image.shape}")
    
    # 生成顏色調色板
    colors = generate_colors(len(labels))
    
    # 處理檢測結果
    for det in results:
        if det is None:
            continue
        
        # 繪製每個檢測框
        for i in range(det.shape[0]):
            x1, y1, x2, y2, conf, cls_id = det[i]
            cls_id = int(cls_id) + 1
            
            # 確保坐標為整數並在圖像範圍內
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            # 檢查坐標是否有效
            if x1 >= x2 or y1 >= y2:
                continue
                
            # 獲取類別顏色
            color = colors[cls_id % len(colors)]
            
            # 輸出檢測結果
            cls_name = labels[cls_id] if cls_id < len(labels) else f"class_{cls_id}"
            print(f'檢測到: {cls_name}, 置信度: {conf:.2f}, 坐標: ({x1}, {y1}, {x2}, {y2}), 顏色: {color}')
            
            # 繪製矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
            
            # 繪製類別標籤背景和文字
            label_text = f'{cls_name} {conf:.2f}'
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            
            # 根據背景顏色選擇文字顏色（確保可讀性）
            brightness = sum(color) / 3
            text_color = (255, 255, 255) if brightness < 127 else (0, 0, 0)
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    return image

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

# COCO 數據集的類別標籤
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

def analyze_yolo_output(output_data):
    """分析YOLO輸出的結構"""
    print(f"輸出形狀: {output_data.shape}")
    
    # 檢查不同部分的值範圍
    for i in range(0, min(84, output_data.shape[2]), 5):
        segment = output_data[0, :, i:i+5]
        print(f"輸出索引 {i}~{i+4} 的值範圍: {np.min(segment)} ~ {np.max(segment)}")
        print(f"輸出索引 {i}~{i+4} 的非零值百分比: {np.mean(segment != 0) * 100:.2f}%")
    
    # 檢查前幾個檢測框的詳細內容
    for i in range(min(5, output_data.shape[1])):
        box_data = output_data[0, i]
        max_class_idx = np.argmax(box_data[5:])
        max_class_conf = box_data[5 + max_class_idx]
        print(f"框 {i}: 坐標[{box_data[0]:.4f}, {box_data[1]:.4f}, {box_data[2]:.4f}, {box_data[3]:.4f}], " 
              f"置信度: {box_data[4]:.4f}, 最高類別: {max_class_idx}, 類別置信度: {max_class_conf:.4f}")

def detect_with_neuronpilot(image_path):
    """
    使用 NeuroPilot 進行物體檢測並儲存結果
    Args:
    image_path (str): 輸入圖像的路徑
    Returns:
    dict: 檢測結果
    """
    # 載入圖像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Can't load image from: {image_path}")
    
    # 載入模型
    model_path = '/home/ubuntu/MTK-genio-demo/models/yolov8n_float32.tflite'
    print(f"嘗試載入模型: {model_path}")
    
    # 檢查模型檔案是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型檔案不存在: {model_path}")

    # 建立 neuronrt.Interpreter 需要的資料夾
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./bin', exist_ok=True)

    # 初始化 neuronrt.Interpreter
    interpreter = neuronrt.Interpreter(model_path=model_path, device='mdla3.0')
    interpreter.allocate_tensors()
    
    # 獲取輸入和輸出詳情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 預處理圖像
    if len(input_details) > 0 and len(input_details[0]['shape']) >= 3:
        input_shape = tuple(input_details[0]['shape'][1:3])  # [batch, height, width, channels]
        if input_shape[0] == 0 or input_shape[1] == 0:
            input_shape = (640, 640)
    else:
        input_shape = (640, 640)
    
    print("前處理中...")
    input_data, transform_info = preprocess_image(image, input_shape)
    
    # 確保輸入數據類型與期望類型匹配
    input_dtype = input_details[0]['dtype']
    if input_data.dtype != input_dtype:
        print(f"轉換輸入數據類型從 {input_data.dtype} 到 {input_dtype}")
        input_data = input_data.astype(input_dtype)
    
    # 設置輸入張量
    print("設置輸入張量...")
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 執行推理
    print("執行推理...")
    start = time.time()
    interpreter.invoke()
    inference_time = time.time() - start
    print(f"推理完成，耗時: {inference_time:.4f} 秒")

    # 後處理
    print("後處理中...")
    imgsz = (640, 640)  # resize 後輸入網路的大小
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 84, 8400)

    output_data = output_data.transpose(0, 2, 1)
    # analyze_yolo_output(output_data)
    results = postprocess(output_data, transform_info, conf_thres=0.25, iou_thres=0.45, nc=80)
    
    # 繪製檢測結果
    print("繪製結果中...")
    output_image = visualizer(image, results, COCO_CLASSES)
    
    # 儲存結果圖像
    output_path = image_path.rsplit('.', 1)[0] + "_detection.jpg"
    cv2.imwrite(output_path, output_image)
    
    # print(f'偵測到 {len(results[0])} 個物體')
    print(f"結果已儲存至: {output_path}")

if __name__ == "__main__":
    # 設定圖像路徑
    image_path = "data/dog.png"
    
    # 檢查圖像是否存在
    if not os.path.exists(image_path):
        print(f"警告: 圖像檔案不存在: {image_path}")
        # 嘗試在當前目錄尋找圖像
        current_dir = os.getcwd()
        print(f"正在當前目錄尋找圖像: {current_dir}")
        image_found = False
        for file in os.listdir(current_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(current_dir, file)
                print(f"找到替代圖像: {image_path}")
                image_found = True
                break
        
        if not image_found:
            raise FileNotFoundError(f"找不到任何可用的圖像檔案")
    
    print(f"使用圖像: {image_path}")

    # 執行物體檢測
    detect_with_neuronpilot(image_path)