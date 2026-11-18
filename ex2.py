import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict(r"C:\Users\guita\iCloudDrive\3年秋\領域実習B\7_YOLO\RJ米田_公開レポジトリ\ex2-26.png", conf=0.1)

# 入力画像
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes

for box in boxes:
    # 物体領域のxy座標を取得する．
    if results[0].names[int(box.cls[0])] == "person":
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            thickness=3,
        )
person_count = int((results[0].boxes.cls == 0).sum().item())
print(f"検出した人数: {person_count}人")

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()