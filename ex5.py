import cv2
from ultralytics import YOLO

model = YOLO("best260408.pt")

# 入力動画
cap = cv2.VideoCapture(r"ex5-26.mp4")

# フレーム番号を0にする
cnt = 0

while cap.isOpened():
    success, frame = cap.read()
    # フレームを読み出す
    if not success:
        break

    # 読み出しに成功すれば以下を実行する
    results = model.predict(frame, conf=0.5)
    # 認識した物体領域を取得する．
    boxes = results[0].boxes

    for box in boxes:
        # 物体領域のxy座標を取得する．
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            thickness=3,
        )

    cv2.imshow("", frame)

    if(cnt % 30 == 0):
        helmet_count = int((results[0].boxes.cls == 0).sum().item())
        print(f"フレーム{cnt}で検出したヘルメット数: {helmet_count}個")

    # フレーム番号をインクリメントする
    cnt += 1

    # ESCが押されれば終了
    if cv2.waitKey(20) == 27:
        break  

cap.release()
cv2.destroyAllWindows()