import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
from typing import Tuple, Union
import math
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # redA

def distance3(a, b):
    import math
    return math.sqrt( (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2 )

class LivenessDetection:
    def __init__(self):
        self.last_right_EAR = 0
        self.last_left_EAR  = 0
        self.treshold       = 2
        self.isLive         = False


    def eye_aspect_ratio(self, eye_landmakrs):
        TOP = 0
        BOTTOM = 1
        OUTER = 2
        INNER = 3
        vertical_distance = distance3(eye_landmakrs[TOP], eye_landmakrs[BOTTOM])
        horizontal_distance = distance3(eye_landmakrs[OUTER], eye_landmakrs[INNER])
        return horizontal_distance / vertical_distance 

    def update_ears(self, landmarks_result):
        # Details: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
        #L TOP 159, BOTTOM 145, OUTER 33, INNER 132
        #R TOP 386, BOTTOM 374, OUTER 263, INNER 362

        if(len(landmarks_result.face_landmarks)<1):
            return

        l_eyes_ids = [159, 145, 33, 133]
        r_eyes_ids = [386, 374, 263, 362]
        face_landmarks = landmarks_result.face_landmarks[0]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        left_eye_landmarks = [face_landmarks_proto.landmark[i] for i in l_eyes_ids]
        right_eye_landmarks = [face_landmarks_proto.landmark[i] for i in r_eyes_ids]

        new_left_EAR = self.eye_aspect_ratio(left_eye_landmarks)
        new_right_EAR = self.eye_aspect_ratio(right_eye_landmarks)

        if self.last_left_EAR == 0 or self.last_right_EAR == 0:
            self.last_left_EAR = new_left_EAR
            self.last_right_EAR = new_right_EAR
            return;

        if(abs(new_left_EAR - self.last_left_EAR) > self.treshold or
           abs(new_right_EAR - self.last_right_EAR) > self.treshold):

            self.isLive = True







def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    l_eyes_ids = [159, 145, 33, 133]
    l_eyes_landmarks = [face_landmarks_proto.landmark[i] for i in l_eyes_ids]
    for i, el in enumerate(l_eyes_landmarks):

        height, width, _ = rgb_image.shape
        keypoint_px = _normalized_to_pixel_coordinates(el.x, el.y, width,
                                                       height)
        color, thickness, radius = (128, 255, 128), 2, 2
        cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
        cv2.putText(annotated_image, f"{l_eyes_ids[i]}", keypoint_px, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)


    
    keypoint_px = _normalized_to_pixel_coordinates(0.1, 0.9, width,
                                                       height)
#    cv2.putText(annotated_image, f"{eye_aspect_ratio(l_eyes_landmarks)}", keypoint_px, cv2.FONT_HERSHEY_PLAIN,
#                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    r_eyes_ids = [386, 374, 263, 362]
    r_eyes_landmarks = [face_landmarks_proto.landmark[i] for i in r_eyes_ids]
    for i, el in enumerate(r_eyes_landmarks):

        height, width, _ = rgb_image.shape
        keypoint_px = _normalized_to_pixel_coordinates(el.x, el.y, width,
                                                       height)
        color, thickness, radius = (128, 255, 128), 2, 2
        cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
        cv2.putText(annotated_image, f"{l_eyes_ids[i]}", keypoint_px, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                    math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
        is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
        image,
        detection_result, 
        isLive
        ) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
  Args:
      image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
      Image with bounding boxes.
  """
    annotated_image = image.copy()
    height, width, _ = image.shape

    try:
        for detection in detection_result.detections:
          # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for i, keypoint in enumerate(detection.keypoints):
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
            cv2.putText(annotated_image, f"{i}", keypoint_px, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)


            # Draw label and score
            text_location = (MARGIN + bbox.origin_x,
                             MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, f"IS LIVE: {isLive}", text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    except:
        ##print("No detection")
        pass

    return annotated_image



def origin_size_to_points(x, y, w, h):
    return (x, x+w, y, y+h)
##############################

base_options_face = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
options_face = vision.FaceDetectorOptions(
        base_options=base_options_face,
        min_detection_confidence=0.6,
        min_suppression_threshold=0.5)

face_detector = vision.FaceDetector.create_from_options(options_face)

base_options_landmarks = python.BaseOptions(model_asset_path="face_landmarker.task")
options_landmarks = vision.FaceLandmarkerOptions(
        base_options=base_options_landmarks,
        min_tracking_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_face_detection_confidence=0.6,
        num_faces=1)

landmark_detector = vision.FaceLandmarker.create_from_options(options_landmarks)


capture = cv2.VideoCapture(0)

fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = 0
liveness = LivenessDetection()
while True: 
    success, frame = capture.read()

    timestamp_ms = int((frame_count/fps) * 1000)

    frame_count += 1

    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    result_face = face_detector.detect(mp_frame)
    for i, face_detection in enumerate(result_face.detections):
        bbox = face_detection.bounding_box
        (x1, x2, y1, y2) = origin_size_to_points(
                bbox.origin_x,
                bbox.origin_y,
                bbox.width,
                bbox.height)

        cropped_frame = frame[y1:y2, x1:x2].astype("uint8")
        mp_frame_l = mp.Image(image_format=mp.ImageFormat.SRGB, 
                              data=cropped_frame)
        result_landmark = landmark_detector.detect(
                mp_frame_l)


        liveness.update_ears(result_landmark)
        print(f"{liveness.isLive}")

        annotated_l = draw_landmarks_on_image(mp_frame_l.numpy_view(), result_landmark)
        cv2.imshow(f"croppped {i}", annotated_l)


    annotated = visualize(mp_frame.numpy_view(), result_face, liveness.isLive)




    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("Liveness Project", annotated)

capture.release()
cv2.destroyAllWindows()
