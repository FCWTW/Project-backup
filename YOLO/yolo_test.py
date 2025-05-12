import cv2
import numpy as np
import time
from utils.neuronpilot import neuronrt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression"""
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def compute_iou(box1, boxes):
    """Compute IoU between one box and many"""
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box1_area + boxes_area - inter_area

    return inter_area / (union_area + 1e-6)

def postprocess(output, conf_threshold=0.3, iou_threshold=0.5, input_shape=(640, 640), original_shape=(480, 640)):
    """Parse YOLOv8 output"""
    predictions = output[0]  # shape: (1, N, 85)
    if predictions.ndim == 2:
        predictions = np.expand_dims(predictions, axis=0)
    predictions = predictions[0]

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        obj_conf = sigmoid(pred[4])
        class_scores = sigmoid(pred[5:])
        conf = obj_conf * np.max(class_scores)
        class_id = np.argmax(class_scores)
        if conf > conf_threshold:
            # YOLOv8 predicts: [cx, cy, w, h]
            cx, cy, w, h = pred[:4]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # scale to original image size
            scale_x = original_shape[1] / input_shape[1]
            scale_y = original_shape[0] / input_shape[0]
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y

            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(class_id)

    boxes = np.array(boxes)
    confidences = np.array(confidences)
    class_ids = np.array(class_ids)

    keep = nms(boxes, confidences, iou_threshold)

    return boxes[keep], confidences[keep], class_ids[keep]

# --- 主流程 ---

# 載入模型
model_path = "./models/yolov8n_float32.tflite"
device = "mdla3.0"
interpreter = neuronrt.Interpreter(model_path=model_path, device=device)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 載入圖像
img = cv2.imread("./data/bus.jpg")
original_shape = img.shape[:2]  # (h, w)
input_shape = input_details[0]['shape'][1:3]  # (640, 640)

# 前處理
img_resized = cv2.resize(img, (input_shape[1], input_shape[0]))
img_input = img_resized.astype(np.float32) / 255.0
img_input = np.expand_dims(img_input, axis=0)

# 推論
interpreter.set_tensor(input_details[0]['index'], img_input)
for _ in range(10):
    interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# 後處理
boxes, scores, class_ids = postprocess(output, input_shape=input_shape, original_shape=original_shape)

# 畫框
for box, score, class_id in zip(boxes, scores, class_ids):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{class_id}:{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 顯示結果
cv2.imshow("YOLOv8 Neuron SDK", img)
cv2.waitKey(0)
cv2.destroyAllWindows()