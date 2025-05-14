import numpy as np
import cv2
from utils.neuronpilot import neuronrt
import os
import time

os.environ["NEUROPILOT_LOG_LEVEL"] = "3"

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
    # print(f"調整圖像大小至 {input_shape}")
    resized_image = cv2.resize(image, input_shape, interpolation=cv2.INTER_LINEAR)
    
    # 確認輸入圖像形狀
    # print(f"調整後的圖像形狀: {resized_image.shape}")
    
    # 將圖像轉換為 float32
    input_data = resized_image.astype(np.float32)
    
    # 歸一化: 像素值範圍從 [0, 255] 縮放到 [0, 1]
    input_data /= 255.0
    
    # 添加 batch 維度
    input_data = np.expand_dims(input_data, axis=0)
    
    print(f"預處理後的輸入形狀: {input_data.shape}, 類型: {input_data.dtype}")
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
    print(f"輸出詳情: {output_details}")
    
    # 讀取所有輸出
    outputs = []
    for detail in output_details:
        output = interpreter.get_tensor(detail['index'])
        outputs.append(output)
        print(f"輸出形狀: {output.shape}, 類型: {output.dtype}")
    
    # 注意：以下邏輯需要根據實際模型輸出調整
    # 假設我們的模型是 YOLOv8n，預期輸出形狀為 [1, 84, 8400]
    # 其中 84 = 4 (bbox) + 1 (objectness) + 80 (COCO classes)
    
    # 處理輸出格式
    if len(outputs) == 1:
        if len(outputs[0].shape) == 3 and outputs[0].shape[1] > 5:  # YOLOv8 格式
            # YOLOv8 格式: [1, 84, 8400]
            # 轉置為 [1, 8400, 84]
            output = outputs[0]
            if output.shape[1] > output.shape[2]:
                output = np.transpose(output, (0, 2, 1))
            
            # 移除 batch 維度
            output = output[0]
            
            # 分離 bounding box, objectness, class predictions
            boxes = output[:, :4]  # 前 4 個是 bounding box
            scores = output[:, 4:5]  # 第 5 個是 objectness
            class_probs = output[:, 5:]  # 剩餘的是類別概率
            
            # 獲取類別 ID 和最終的分數
            class_ids = np.argmax(class_probs, axis=1)
            class_scores = np.max(class_probs, axis=1)
            
            # 計算最終分數
            final_scores = scores[:, 0] * class_scores
            
            # 創建 [x1, y1, x2, y2, score, class_id] 格式的 detections
            detections = np.column_stack((boxes, final_scores, class_ids))
        else:
            # 其他格式：假設輸出已經是 [num_detections, 6] 格式
            # 其中 6 = [x1, y1, x2, y2, score, class_id]
            detections = outputs[0]
    else:
        # 多個輸出：假設不同的輸出表示不同的 yolo 層
        # 簡單起見，我們將所有輸出合併
        detections = np.concatenate(outputs, axis=1)
    
    # 篩選高於置信度閾值的檢測
    valid_indices = detections[:, 4] > conf_threshold
    valid_detections = detections[valid_indices]
    
    print(f"篩選後的有效檢測數量: {len(valid_detections)}")
    
    # 調整框的座標到原始圖像尺寸
    h, w = image_shape
    input_shape = (640, 640)  # 與預處理保持一致
    
    # 檢查框的座標是否已經是歸一化的（介於 0 和 1 之間）
    if np.all(valid_detections[:, :4] <= 1) and np.all(valid_detections[:, :4] >= 0):
        # 如果是歸一化的，則轉換為絕對座標
        valid_detections[:, 0] *= w  # x1
        valid_detections[:, 1] *= h  # y1
        valid_detections[:, 2] *= w  # x2
        valid_detections[:, 3] *= h  # y2
    else:
        # 如果不是歸一化的，則根據輸入形狀進行縮放
        scale_x = w / input_shape[0]
        scale_y = h / input_shape[1]
        valid_detections[:, 0] *= scale_x  # x1
        valid_detections[:, 1] *= scale_y  # y1
        valid_detections[:, 2] *= scale_x  # x2
        valid_detections[:, 3] *= scale_y  # y2
    
    # 確保框座標在圖像範圍內
    valid_detections[:, 0] = np.clip(valid_detections[:, 0], 0, w)
    valid_detections[:, 1] = np.clip(valid_detections[:, 1], 0, h)
    valid_detections[:, 2] = np.clip(valid_detections[:, 2], 0, w)
    valid_detections[:, 3] = np.clip(valid_detections[:, 3], 0, h)
    
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
        print(f"NMS 後的最終檢測數量: {len(final_detections)}")
    else:
        final_detections = []
    
    # 組織返回結果
    results = {
        'boxes': final_detections[:, :4],  # [x1, y1, x2, y2]
        'scores': final_detections[:, 4],  # 置信度
        'classes': final_detections[:, 5].astype(int)  # 類別 ID
    }
    return results

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

def draw_detections(image, results):
    """
    在圖像上繪製檢測結果
    Args:
    image (numpy.ndarray): 原始圖像
    results (dict): 檢測結果，包含 'boxes', 'scores', 'classes'
    Returns:
    numpy.ndarray: 繪製了檢測結果的圖像
    """
    # 複製圖像，避免修改原始圖像
    output_image = image.copy()
    
    # 確保有檢測結果
    if len(results['boxes']) == 0:
        return output_image
    
    # 為每個類別分配不同的顏色
    np.random.seed(42)  # 確保顏色的一致性
    colors = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)
    
    # 繪製每個檢測框
    for i in range(len(results['boxes'])):
        box = results['boxes'][i].astype(int)
        score = results['scores'][i]
        class_id = results['classes'][i]
        
        # 獲取類別名稱和顏色
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
        color = tuple(map(int, colors[class_id % len(colors)]))
        
        # 繪製邊界框
        cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # 準備標籤文本
        label = f"{class_name}: {score:.2f}"
        
        # 獲取文本大小
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 繪製文本背景
        cv2.rectangle(output_image, (box[0], box[1] - text_height - 4), (box[0] + text_width, box[1]), color, -1)
        
        # 繪製文本
        cv2.putText(output_image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return output_image

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
    model_path = '/root/catkin_ws/src/yolo_ros/yolov8n_float32.tflite'
    print(f"嘗試載入模型: {model_path}")
    
    # 檢查模型檔案是否存在
    import os
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
    input_shape = tuple(input_details[0]['shape'][1:3])  # 轉換為元組
    if input_shape[0] == 0 or input_shape[1] == 0:
        input_shape = (640, 640)  # 使用默認值
        print(f"輸入形狀不完整，使用默認值: {input_shape}")
    
    # print(f"模型輸入形狀: {input_shape}")
    print("前處理中...")
    input_data = preprocess_image(image, input_shape)
    
    # 確保輸入數據類型與期望類型匹配
    input_dtype = input_details[0]['dtype']
    if input_data.dtype != input_dtype:
        print(f"轉換輸入數據類型從 {input_data.dtype} 到 {input_dtype}")
        input_data = input_data.astype(input_dtype)
    
    # 設置輸入張量
    print("設置輸入張量...")
    print("output_handlers_with_shape:", interpreter.output_handlers_with_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # 執行推理
    print("執行推理...")
    start = time.time()
    interpreter.invoke()
    print("推理完成，耗時:", time.time() - start)

    print("成功推論！")
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']
    output_tensor = interpreter.get_tensor(output_index)
    print("推論輸出 shape:", output_tensor.shape)

    # 後處理
    print("後處理中...")
    results = postprocess_detection(interpreter, image.shape[:2])
    
    # 繪製檢測結果
    print("繪製結果中...")
    output_image = draw_detections(image, results)
    
    # 儲存結果圖像
    output_path = image_path.rsplit('.', 1)[0] + "_detection.jpg"
    cv2.imwrite(output_path, output_image)
    
    print(f"檢測結果已儲存至: {output_path}")
    print(f"檢測到 {len(results['boxes'])} 個物體")
    
    return results

if __name__ == "__main__":
        # 設定圖像路徑
    image_path = "/root/catkin_ws/src/yolo_ros/bus.jpg"
    
    # 檢查圖像是否存在
    import os
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
    results = detect_with_neuronpilot(image_path)
    
    # 打印檢測結果
    if results and len(results['boxes']) > 0:
        print("\n檢測結果詳情:")
        for i in range(len(results['boxes'])):
            box = results['boxes'][i]
            score = results['scores'][i]
            class_id = results['classes'][i]
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
            print(f"物體 {i+1}: {class_name}, 置信度: {score:.4f}, 位置: {box.astype(int)}")
    else:
        print("未檢測到任何物體")
    '''
    try:
        # 設定圖像路徑
        image_path = "/root/catkin_ws/src/yolo_ros/bus.jpg"
        
        # 檢查圖像是否存在
        import os
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
        results = detect_with_neuronpilot(image_path)
        
        # 打印檢測結果
        if results and len(results['boxes']) > 0:
            print("\n檢測結果詳情:")
            for i in range(len(results['boxes'])):
                box = results['boxes'][i]
                score = results['scores'][i]
                class_id = results['classes'][i]
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
                print(f"物體 {i+1}: {class_name}, 置信度: {score:.4f}, 位置: {box.astype(int)}")
        else:
            print("未檢測到任何物體")
    
    except FileNotFoundError as e:
        print(f"檔案錯誤: {e}")
    except Exception as e:
        import traceback
        print(f"執行過程中發生錯誤: {e}")
        print("錯誤詳情:")
        traceback.print_exc()
        # 檢查最終的圖像是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到任何可用的圖像檔案")
        
        print(f"使用圖像: {image_path}")
        results = detect_with_neuronpilot(image_path)
        
        # 打印檢測結果
        if len(results['boxes']) > 0:
            print("\n檢測結果詳情:")
            for i in range(len(results['boxes'])):
                box = results['boxes'][i]
                score = results['scores'][i]
                class_id = results['classes'][i]
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
                print(f"物體 {i+1}: {class_name}, 置信度: {score:.4f}, 位置: {box.astype(int)}")
    
    except Exception as e:
        import traceback
        print(f"執行過程中發生錯誤: {e}")
        print("錯誤詳情:")
        traceback.print_exc()
        
        # 如果是 neuronrt 相關錯誤，提供一些可能的解決方案
        if "neuronrt" in str(e).lower():
            print("\n可能的解決方案:")
            print("1. 確認 neuronrt 模組已正確安裝")
            print("2. 檢查 neuronrt.Interpreter 的使用方法是否正確")
            print("3. 查閱 neuronrt 的文檔以獲取正確的 API 用法")
            print("4. 確認模型檔案(.tflite)存在並且格式正確")
        
        # 如果是圖像相關的錯誤
        if "imread" in str(e).lower() or "image" in str(e).lower():
            print("\n可能的圖像相關解決方案:")
            print("1. 確認圖像檔案路徑正確")
            print("2. 檢查圖像檔案是否損壞")
            print("3. 嘗試使用絕對路徑而不是相對路徑")
    '''