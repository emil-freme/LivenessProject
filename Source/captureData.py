import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import time

def distance3(a, b):
    import math
    return math.sqrt( (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2 )

def origin_size_to_points(x, y, w, h):
    return (x, x+w, y, y+h)

def save_frame(frame, username):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    name_bytes = bytes(name, "utf-8")
    png_info = PngInfo()
    png_info.add_text("Name", name_bytes)
    pil_image.save(f"captures/{int(time.time())}.png", format="PNG", pnginfo=png_info)

capture = cv2.VideoCapture(0)
framecount = 1

base_options_face = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
options_face = vision.FaceDetectorOptions(
        base_options=base_options_face,
        min_detection_confidence=0.6,
        min_suppression_threshold=0.5)

face_detector = vision.FaceDetector.create_from_options(options_face)

name = input("Name: ")
last_time = time.time()
look_text = ["Forward", "Left", "Right", "Up", "Down"]
look_step = 0

while True:
    success, frame = capture.read()
    if(not success):
        continue

    cv2.putText(frame, f"{100-framecount} Look {look_text[look_step]}", (10, 20), cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1, thickness=2, color=(0, 0, 255))
    cv2.imshow("Capture", frame)

    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    face_result = face_detector.detect(mp_frame)

    if(not len(face_result.detections) > 0):
        continue
    
    face_detection = face_result.detections[0]
    bbox = face_detection.bounding_box
    (x1, x2, y1, y2) = origin_size_to_points(
            bbox.origin_x,
            bbox.origin_y,
            bbox.width,
            bbox.height)

    cropped_frame = frame[y1:y2, x1:x2].astype("uint8")

    if time.time() - last_time > 1:
        save_frame(cropped_frame, name)
        if framecount%20 == 0:
            look_step += 1
        framecount += 1
        last_time = int(time.time())

    if(framecount == 100):
        break;
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    

    
