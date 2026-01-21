import cv2
from deepface import DeepFace

# เปิดกล้อง
cap = cv2.VideoCapture(0)

print("กด Q เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # ค้นหาใบหน้าใน dataset
        result = DeepFace.find(
            img_path=frame,
            db_path="dataset",
            model_name="Facenet512",
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            identity = result[0].iloc[0]["identity"]
            name = identity.split("\\")[-2]  # ดึงชื่อโฟลเดอร์
            text = f"FOUND: {name}"
            color = (0, 255, 0)
        else:
            text = "UNKNOWN"
            color = (0, 0, 255)

    except Exception as e:
        text = "NO FACE"
        color = (0, 0, 255)

    # แสดงผลบนจอ
    cv2.putText(
        frame, text, (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, color, 2
    )

    cv2.imshow("DeepFace Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
