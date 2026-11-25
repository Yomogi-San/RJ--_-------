import cv2
from ultralytics import YOLO

model = YOLO("best260408.pt")

results = model.predict(r"ex5-26.mp4", conf=0.5)

# フレーム番号を0にする
cnt = 0

while cap.isOpened():
    cap = results[cnt].orig_img
    success, frame = cap.read()
    # フレームを読み出す
    if not success:
        break

    # 読み出しに成功すれば以下を実行する
    # 認識した物体領域を取得する．
    boxes = results[cnt].boxes

    for box in boxes:
        # 物体領域のxy座標を取得する．
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(
            cap,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            thickness=3,
        )

    if(cnt % 30 == 0):
        helmet_count = int((results[cnt].boxes.cls == 0).sum().item())
        print(f"フレーム{cnt}で検出したヘルメット数: {helmet_count}個")

    # フレーム番号をインクリメントする
    cnt += 1
    
    if cv2.waitKey(20) == 27:
        break
    # ESCが押されれば終了

cap.release()
cv2.destroyAllWindows()