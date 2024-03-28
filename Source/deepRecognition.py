import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from PIL import Image, PngImagePlugin
import numpy as np
import os

debug = True

images_directory = "captures/"
batch_size = 32

def process_image(image_path):
    image = Image.open(image_path)
    image_resized = image.resize((256, 256))
    tag = image.info.get("Name", "notag")
    image_arr = np.array(image_resized)

    base_options_landmarks = python.BaseOptions(model_asset_path="face_landmarker.task")
    options_landmarks = vision.FaceLandmarkerOptions(
            base_options=base_options_landmarks,
            min_tracking_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_face_detection_confidence=0.6,
            num_faces=1)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
    landmark_detector = vision.FaceLandmarker.create_from_options(options_landmarks)

    result_landmark = landmark_detector.detect(mp_image)

    face_landmarks = result_landmark.face_landmarks[0]

    return image_arr, face_landmarks, tag

def image_generator(images_dir,batch_size):
    images_path = os.listdir(images_dir)
    images_path.remove(".gitkeep")
    print(images_dir)
    print(images_path)

    for i in range(0, len(images_path), batch_size):
        batch_paths = images_path[i:i+batch_size]
        process_batch = [process_image(f"{images_dir}/{path}") for path in batch_paths]
        images, landmarks, tag = zip(*process_batch)
        yield np.array(images), np.array(landmarks), np.array(tag)

        


dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(images_directory, batch_size),
        output_types=(tf.uint8, tf.float32, tf.string),
        output_shapes=(
            (None, 256, 256, 3),
            (None, 468, 3),
            (None, )
            )
        )
#Multi-model input
# Define the image input path
image_input = Input(shape=(224, 224, 3))  # Adjust the shape based on your image preprocessing
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
image_model = models.Model(image_input, x)

# Define the landmark input path
landmark_input = Input(shape=(468*3,))  # Adjust based on your landmark vector size
y = layers.Dense(128, activation='relu')(landmark_input)
y = layers.Dropout(0.5)(y)
landmark_model = models.Model(landmark_input, y)

# Combine the outputs of the two models
combined = layers.concatenate([image_model.output, landmark_model.output])

# Add a fully connected layer
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dropout(0.5)(z)

# Output layer
output = layers.Dense(1, activation='softmax')(z)

# Build the final model
model = models.Model(inputs=[image_input, landmark_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(dataset, epochs=10)
