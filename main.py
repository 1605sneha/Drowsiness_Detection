import math
import os
import threading
import time

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# --------------------------- Config --------------------------- #
THRESHOLD = 15          # frames of closed eyes before "Sleepy"
EAR_THRESHOLD = 0.25
TARGET_W, TARGET_H = 640, 480
ROAD_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")

RIGHT_EYE = [33, 159, 158, 133, 153, 145]
LEFT_EYE = [362, 380, 374, 263, 386, 385]

# Shared between the webcam thread and the Streamlit script
shared = {"state": "Awake", "ear": 0.0, "score": 0}
lock = threading.Lock()

# Models (created once)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# --------------------------- Eye Aspect Ratio --------------------------- #
def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(landmarks, eye_indices, shape):
    pts = [
        (int(landmarks.landmark[i].x * shape[1]), int(landmarks.landmark[i].y * shape[0]))
        for i in eye_indices
    ]
    v1 = euclidean_dist(pts[1], pts[5])
    v2 = euclidean_dist(pts[2], pts[4])
    h = euclidean_dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0


# --------------------------- Driver webcam callback --------------------------- #
def process(frame: av.VideoFrame) -> av.VideoFrame:
    df = frame.to_ndarray(format="bgr24")
    df = cv2.resize(df, (TARGET_W, TARGET_H))

    rgb = cv2.cvtColor(df, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    eyes_open = False
    ear_value = 0.0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ear_right = eye_aspect_ratio(face_landmarks, RIGHT_EYE, df.shape)
            ear_left = eye_aspect_ratio(face_landmarks, LEFT_EYE, df.shape)
            ear_value = (ear_right + ear_left) / 2
            eyes_open = ear_value > EAR_THRESHOLD

            dot_color = (0, 255, 0) if eyes_open else (0, 0, 255)
            for idx in RIGHT_EYE + LEFT_EYE:
                p = face_landmarks.landmark[idx]
                cv2.circle(df, (int(p.x * TARGET_W), int(p.y * TARGET_H)), 2, dot_color, -1)
    else:
        # Fallback: Haar cascades
        gray = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.3, 5):
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = df[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                eyes_open = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    with lock:
        if eyes_open:
            shared["score"] = max(0, shared["score"] - 1)
        else:
            shared["score"] += 1
        sleepy = shared["score"] > THRESHOLD
        shared["state"] = "Sleepy" if sleepy else "Awake"
        shared["ear"] = ear_value
        state, score = shared["state"], shared["score"]

    color = (0, 0, 255) if state == "Sleepy" else (0, 255, 0)
    cv2.putText(df, f"Driver Status: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(df, f"EAR: {ear_value:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(df, f"Score: {score}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return av.VideoFrame.from_ndarray(df, format="bgr24")


# --------------------------- Lane Detection --------------------------- #
def detect_lanes(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    region = np.array([[(0, h), (w, h), (w, int(h * 0.6)), (0, int(h * 0.6))]])
    cv2.fillPoly(mask, region, 255)
    cropped = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=150)

    left_pos, right_pos = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            if x1 < w // 2 and x2 < w // 2:
                left_pos.append((x1 + x2) // 2)
            elif x1 > w // 2 and x2 > w // 2:
                right_pos.append((x1 + x2) // 2)

    lane_center = None
    if left_pos and right_pos:
        lane_center = (np.mean(left_pos) + np.mean(right_pos)) / 2
    elif left_pos:
        lane_center = np.mean(left_pos) + w * 0.25
    elif right_pos:
        lane_center = np.mean(right_pos) - w * 0.25

    return frame, lane_center


# --------------------------- UI --------------------------- #
st.title("Drowsiness Detection")
st.caption("Click START and allow camera access. Keep your face visible and well lit.")

ctx = webrtc_streamer(
    key="drowsiness",
    video_frame_callback=process,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.subheader("Road view")
road_placeholder = st.empty()
status_placeholder = st.empty()

if not os.path.exists(ROAD_VIDEO):
    road_placeholder.error(
        f"Road video not found at: {ROAD_VIDEO}. "
        f"Files next to main.py: {sorted(os.listdir(os.path.dirname(ROAD_VIDEO)))}"
    )
elif not ctx.state.playing:
    road_placeholder.info("Click START above. The road view appears once the webcam is running.")
else:
    road_cam = cv2.VideoCapture(ROAD_VIDEO)
    if not road_cam.isOpened():
        road_placeholder.error(
            "video.mp4 was found but OpenCV could not decode it (often an H.265/HEVC "
            "or AV1 file). Re-encode it as H.264 with: "
            "ffmpeg -i video.mp4 -vcodec libx264 -an -pix_fmt yuv420p video_h264.mp4"
        )
    else:
        failures = 0
        try:
            while ctx.state.playing:
                ok, rf = road_cam.read()
                if not ok:
                    failures += 1
                    if failures > 5:
                        road_placeholder.error("Could not read frames from video.mp4.")
                        break
                    road_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                failures = 0

                rf = cv2.resize(rf, (TARGET_W, TARGET_H))
                rf, lane_center = detect_lanes(rf)

                with lock:
                    state = shared["state"]

                if state == "Sleepy" and lane_center is not None:
                    frame_center = TARGET_W / 2
                    if lane_center < frame_center - 15:
                        instruction = "Turn Left"
                    elif lane_center > frame_center + 15:
                        instruction = "Turn Right"
                    else:
                        instruction = "Drive Straight"
                    cv2.putText(rf, f"DRIVER SLEEPY - {instruction}", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
                    status_placeholder.error("Driver is sleepy!")
                else:
                    status_placeholder.success("Driver is awake")

                road_placeholder.image(rf, channels="BGR")
                time.sleep(0.03)
        finally:
            road_cam.release()
