import cv2
import numpy as np
import mediapipe as mp
import math

# Load Haar cascades (backup)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Initialize cameras
driver_cam = cv2.VideoCapture(0)
road_cam = cv2.VideoCapture("video.mp4")

# Drowsiness variables
score = 0
THRESHOLD = 15
EAR_THRESHOLD = 0.25

# Set video display size
target_width = 640
target_height = 480

# --------------------------- Eye Aspect Ratio Functions --------------------------- #
def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def eye_aspect_ratio(landmarks, eye_indices, shape):
    points = [(int(landmarks.landmark[i].x * shape[1]), 
               int(landmarks.landmark[i].y * shape[0])) for i in eye_indices]
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    v1 = euclidean_dist(points[1], points[5])
    v2 = euclidean_dist(points[2], points[4])
    h = euclidean_dist(points[0], points[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0

# --------------------------- Lane Detection --------------------------- #
def detect_lanes(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    region = np.array([[(0, h), (w, h), (w, int(h*0.6)), (0, int(h*0.6))]])
    cv2.fillPoly(mask, region, 255)
    cropped = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 50, minLineLength=80, maxLineGap=150)

    left_line_pos = []
    right_line_pos = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            if x1 < w // 2 and x2 < w // 2:
                left_line_pos.append((x1 + x2) // 2)
            elif x1 > w // 2 and x2 > w // 2:
                right_line_pos.append((x1 + x2) // 2)

    lane_center = None
    if left_line_pos and right_line_pos:
        left_avg = np.mean(left_line_pos)
        right_avg = np.mean(right_line_pos)
        lane_center = (left_avg + right_avg) / 2
    elif left_line_pos:
        lane_center = np.mean(left_line_pos) + w * 0.25
    elif right_line_pos:
        lane_center = np.mean(right_line_pos) - w * 0.25

    return frame, lane_center

# --------------------------- Main Loop --------------------------- #
while True:
    ret1, df = driver_cam.read()
    if not ret1:
        print("Webcam not found! Try changing camera index from 0 → 1 → 2")
        break

    df = cv2.resize(df, (target_width, target_height))
    
    # Primary: MediaPipe Face Mesh with EAR (works with closed eyes)
    rgb = cv2.cvtColor(df, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    eyes_open = False
    ear_value = 0
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Right eye landmarks
            right_eye = [33, 159, 158, 133, 153, 145]
            # Left eye landmarks
            left_eye = [362, 380, 374, 263, 386, 385]
            
            ear_right = eye_aspect_ratio(face_landmarks, right_eye, df.shape)
            ear_left = eye_aspect_ratio(face_landmarks, left_eye, df.shape)
            ear_value = (ear_right + ear_left) / 2
            
            # Draw ONLY eye landmarks (removed full face mesh)
            for idx in right_eye + left_eye:
                point = face_landmarks.landmark[idx]
                px, py = int(point.x * df.shape[1]), int(point.y * df.shape[0])
                color = (0, 255, 0) if ear_value > EAR_THRESHOLD else (0, 0, 255)
                cv2.circle(df, (px, py), 2, color, -1)
            
            eyes_open = ear_value > EAR_THRESHOLD
    
    # Fallback: Haar cascades (only if MediaPipe fails)
    if not results.multi_face_landmarks:
        gray = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = df[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                eyes_open = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Drowsiness logic using EAR
    if eyes_open:
        score = max(0, score - 1)
        state = "Awake"
        color = (0, 255, 0)
    else:
        score += 1
        state = "Sleepy" if score > THRESHOLD else "Awake"
        color = (0, 0, 255) if state == "Sleepy" else (0, 255, 0)

    # Display status and EAR
    cv2.putText(df, f"Driver Status: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(df, f"EAR: {ear_value:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(df, f"Score: {score}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Driver Camera (EAR Drowsiness Detection)", df)
    cv2.moveWindow("Driver Camera (EAR Drowsiness Detection)", 50, 50)

    # Road camera processing
    ret2, rf = road_cam.read()
    if not ret2:
        road_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    rf = cv2.resize(rf, (target_width, target_height))
    rf, lane_center = detect_lanes(rf)

    # Lane-based instruction when driver is sleepy
    instruction = ""
    frame_center = target_width / 2
    if state == "Sleepy" and lane_center is not None:
        if lane_center < frame_center - 15:
            instruction = "Turn Left"
        elif lane_center > frame_center + 15:
            instruction = "Turn Right"
        else:
            instruction = "Drive Straight"
        cv2.putText(rf, f"DRIVER SLEEPY - {instruction}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Lane Detection", rf)
    cv2.moveWindow("Lane Detection", 700, 50)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
driver_cam.release()
road_cam.release()
face_mesh.close()
cv2.destroyAllWindows()
