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
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(image, input_shape=(640, 640)):
    """
    為 NeuroPilot YOLO 模型準備輸入圖像
    Args:
    image (numpy.ndarray): 原始 BGR 圖像
    input_shape (tuple): 模型期望的輸入圖像尺寸
    Returns:
    numpy.ndarray: 處理後的模型輸入
    """
    # 使用 LetterBox 進行預處理，保持縱橫比
    letterbox = LetterBox(new_shape=input_shape)
    resized_image = letterbox(image=image)

    show_image("preprocess", resized_image)
    
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
    return input_data

def postprocess(preds, imgsz, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300, nc=80):
    """
    適用於NeuroPilot SDK的YOLO輸出後處理函數
    
    preds: 模型輸出，形狀為(1, 8400, 84)
    imgsz: 輸入圖像尺寸(寬, 高)
    conf_thres: 置信度閾值
    iou_thres: IoU閾值用於非極大值抑制
    nc: 類別數量
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
            
        # 處理坐標
        boxes = filtered_pred[:, :4].copy()
        
        # 轉換為xyxy格式
        boxes = xywh2xyxy(boxes)
        
        # 將相對坐標轉為絕對坐標
        boxes[:, [0, 2]] *= imgsz[0]  # x座標乘以寬度
        boxes[:, [1, 3]] *= imgsz[1]  # y座標乘以高度
        
        # 組合結果: [x1, y1, x2, y2, conf, class_id]
        det = np.column_stack((boxes, filtered_scores, filtered_class_ids))
        
        # 執行非極大值抑制
        if det.shape[0] > 1:
            boxes, scores = det[:, :4], det[:, 4]
            nms_indices = non_max_suppression(boxes, scores, iou_thres)
            if isinstance(nms_indices, list):
                nms_indices = np.array(nms_indices)
            det = det[nms_indices[:max_det]]
        
        print(f"NMS後剩餘框數: {det.shape[0]}")
        print(f"檢測到的類別: {np.unique(det[:, 5])}")
        
        results.append(det)
    
    return results

def visualizer(image, results, labels, input_shape=(640, 640)):
    """
    將檢測結果繪製到圖像上
    
    image: 原始圖像
    results: 後處理後的檢測結果
    labels: 類別標籤列表
    """
    if results is None or len(results) == 0 or results[0] is None:
        return image

    h, w = image.shape[:2]
    
    # 處理檢測結果
    for det in results:
        if det is None:
            continue
        
        # 繪製每個檢測框
        for i in range(det.shape[0]):
            x1, y1, x2, y2, conf, cls_id = det[i]
            cls_id = int(cls_id)
            
            # 確保坐標為整數並在圖像範圍內
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            # 檢查坐標是否有效
            if x1 >= x2 or y1 >= y2:
                continue
                
            # 輸出檢測結果
            cls_name = labels[cls_id] if cls_id < len(labels) else f"class_{cls_id}"
            print(f'檢測到: {cls_name}, 置信度: {conf:.2f}, 坐標: ({x1}, {y1}, {x2}, {y2})')
            
            # 繪製矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            
            # 繪製類別標籤
            label_text = f'{cls_name} {conf:.2f}'
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

'''
def postprocess(preds, imgsz, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300, nc=0, max_time_img=0.05, max_nms=30000, max_wh=7680, in_place=True, rotated=False):
    preds[:, [0, 2]] *= imgsz[0] ; preds[:, [1, 3]] *= imgsz[1]
    xc = np.max(preds[:, 4: nc + 4], axis = 1) > conf_thres
    preds = np.transpose(preds, (0, 2, 1)) 
    preds[..., :4] = xywh2xyxy(preds[..., :4])
    x = preds[0][xc[0]]

    if not x.shape[0]:
        return None
    box, cls, keypoints = x[:, :4], x[:, 4:5], x[:, 5:]
    j = np.argmax(cls, axis=1)
    print(f'j:{j}')
    conf = cls[[i for i in range(len(j))], j]
    concatenated = np.concatenate((box, conf.reshape(-1, 1), j.reshape(-1, 1).astype(float), keypoints), axis=1)
    x = concatenated[conf.flatten() > conf_thres]

    if x.shape[0] > max_nms:  # excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
    cls = x[:, 5:6] * (0 if agnostic else max_wh)
    scores, boxes = x[:, 4], x[:, :4] + cls
    i = non_max_suppression(boxes, scores, iou_thres)
    if isinstance(i, list):
        i = np.array(i)

    return [x[i[:max_det]]]

def visualizer(image, results, labels, input_shape=(640, 640)):
    if results is None or len(results[0]) == 0:
        return image

    ih, iw = input_shape
    h, w = image.shape[:2]
    print(f'input shape: {ih}, {iw}')

    for bboxes in results[0]:
        x1 = max(0, int(bboxes[0]))
        y1 = max(0, int(bboxes[1]))
        x2 = min(w, int(bboxes[2]))
        y2 = min(h, int(bboxes[3]))
        print(f'x and y in visualizer: {x1}, {x2}, {y1}, {y2}')
        conf, cls = bboxes[4], bboxes[5]
        print(labels[int(cls)])
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
        cv2.putText(image, f'{labels[int(cls)]} {conf:.2f}', (x1, y1 - 2), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
    return image
'''

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
        raise ValueError(f"無法讀取圖像: {image_path}")
    
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
    print("初始化 neuronrt.Interpreter...")
    interpreter = neuronrt.Interpreter(model_path=model_path, device='mdla3.0')
    interpreter.allocate_tensors()
    
    # 獲取輸入和輸出詳情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"輸入詳情: {input_details}")
    print(f"輸出詳情: {output_details}")

    # 預處理圖像
    if len(input_details) > 0 and len(input_details[0]['shape']) >= 3:
        input_shape = tuple(input_details[0]['shape'][1:3])  # [batch, height, width, channels]
        if input_shape[0] == 0 or input_shape[1] == 0:
            input_shape = (640, 640)  # 使用默認值
    else:
        input_shape = (640, 640)  # 使用默認值
    
    print(f"使用模型輸入形狀: {input_shape}")
    print("前處理中...")
    input_data = preprocess_image(image, input_shape)
    
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
    analyze_yolo_output(output_data)
    results = postprocess(output_data, imgsz, conf_thres=0.25, iou_thres=0.45, nc=80)
    
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
    image_path = "data/bus.jpg"
    
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