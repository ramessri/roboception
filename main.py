import cv2
import time

url = "http://[IP_ADDRESS]/video"

cap = cv2.VideoCapture(url)

fps_time = time.time()
frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("frame failed")
        break

    frames += 1

    if time.time() - fps_time >= 1:
        print("FPS:", frames)
        frames = 0
        fps_time = time.time()

    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == 27:
        break