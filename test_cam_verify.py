import cv2
from verify import verify_face

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("เปิดกล้องไม่สำเร็จ (ลองเปลี่ยนเป็น VideoCapture(1))")
        return

    print("กด ESC เพื่อออก")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name, dist = verify_face(frame, enforce_detection=True)

        text = f"{name}"
        if dist is not None:
            text += f"  ระยะ={dist:.3f}"

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("ทดสอบระบุตัวตน", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
