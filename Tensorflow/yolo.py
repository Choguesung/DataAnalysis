import cv2

# YOLO 설정 파일과 가중치 파일 경로 설정
yolo_cfg = "yolov3.cfg"
yolo_weights = "yolov3.weights"
yolo_classes = "coco.names"

# 클래스 이름 파일 로드
with open(yolo_classes, "r") as f:
    classes = f.read().strip().split("\n")

# YOLO 모델 로드
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# 입력 이미지 로드
image = cv2.imread("image.jpg")

# 이미지에서 객체 감지를 위한 전처리
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# 객체 감지 수행
outs = net.forward(net.getUnconnectedOutLayersNames())

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 결과 이미지 표시
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
