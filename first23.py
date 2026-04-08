import os
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import cv2
import numpy as np
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist
import dlib

# ============= Audio Alarm Setup =============
try:
    from pygame import mixer
    mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    print(" Warning: pygame.mixer not available. Alarm sound disabled.")
    PYGAME_AVAILABLE = False

# ============= Config Paths (Update to your system) =============
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ALARM_MP3 = "alrm.mp3"
BACKGROUND_IMAGE = "bkg.jpg"
# ============= Thresholds =============
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20  # pixel distance (works if face is close)

# ============= Globals =============
alarm_status = False
alarm_status2 = False
COUNTER = 0
vs = None
detect_thread = None
stop_event = threading.Event()

# ============= Load Classifiers =============
detector = cv2.CascadeClassifier(HAAR_PATH)
if not os.path.exists(DLIB_PREDICTOR_PATH):
    raise FileNotFoundError(f"Dlib predictor not found at {DLIB_PREDICTOR_PATH}")
predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

if PYGAME_AVAILABLE and os.path.exists(ALARM_MP3):
    mixer.music.load(ALARM_MP3)


# ============= Utility Functions =============
def play_alarm():
    if PYGAME_AVAILABLE:
        if not mixer.music.get_busy():
            mixer.music.play()


def eye_aspect_ratio(eye):
    """Compute EAR."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance


# ============= Detection Thread =============
def detect_drowsiness():
    global alarm_status, alarm_status2, COUNTER, vs
    stop_event.clear()
    while not stop_event.is_set():
        frame = vs.read()
        if frame is None:
            continue
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # EAR
            ear, leftEye, rightEye = final_ear(shape)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Lips
            distance = lip_distance(shape)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            # Drowsiness detection
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        play_alarm()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_status = False

            # Yawn detection
            if distance > YAWN_THRESH:
                cv2.putText(frame, "YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2:
                    alarm_status2 = True
                    play_alarm()
            else:
                alarm_status2 = False

            # Show metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (350, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (350, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    if vs is not None:
        vs.stop()


# ============= GUI Functions =============
def start_detection():
    global detect_thread, vs
    if detect_thread and detect_thread.is_alive():
        messagebox.showinfo("Info", "Detection already running.")
        return
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    detect_thread = threading.Thread(target=detect_drowsiness, daemon=True)
    detect_thread.start()


def stop_detection():
    stop_event.set()


def exit_application():
    stop_event.set()
    time.sleep(0.5)
    try:
        if vs:
            vs.stop()
    except Exception:
        pass
    root.destroy()


def show_about_info():
    about_window = tk.Toplevel(root)
    about_window.title("About Project")
    about_window.geometry("400x200")
    about_label = tk.Label(
        about_window,
        text=(
            "🚗 Drive Safe Project\n\n"
            "This project detects drowsiness and yawning in real-time\n"
            "using computer vision techniques.\n\n"
            "Developed by BGS Students (2023/24)"
        ),
        justify="center"
    )
    about_label.pack(fill="both", expand=True, padx=10, pady=10)


# ============= GUI Setup =============
root = tk.Tk()
root.title("Drowsiness Detection")
root.geometry("1050x600")

# Background image
if os.path.exists(BACKGROUND_IMAGE):
    bg_img = Image.open(BACKGROUND_IMAGE)
    bg_photo = ImageTk.PhotoImage(bg_img)
    background_label = tk.Label(root, image=bg_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Control Buttons
frame_controls = tk.Frame(root, bg="white", padx=10, pady=10)
frame_controls.place(relx=0.02, rely=0.05)

start_button = tk.Button(frame_controls, text="▶ Start Detection",
                         command=start_detection, bg="green", fg="white", font=("Arial", 12))
start_button.pack(fill="x", pady=5)

stop_button = tk.Button(frame_controls, text="⏸ Stop Detection",
                        command=stop_detection, bg="red", fg="white", font=("Arial", 12))
stop_button.pack(fill="x", pady=5)

about_button = tk.Button(frame_controls, text="ℹ About",
                         command=show_about_info, bg="#007ACC", fg="white", font=("Arial", 12))
about_button.pack(fill="x", pady=5)

exit_button = tk.Button(frame_controls, text="❌ Exit",
                        command=exit_application, bg="black", fg="white", font=("Arial", 12))
exit_button.pack(fill="x", pady=5)

root.protocol("WM_DELETE_WINDOW", exit_application)
root.mainloop()

