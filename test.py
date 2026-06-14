import cv2
import mediapipe as mp
import urllib.request
import os
import time
from datetime import datetime

ModelPath = 'blaze_face_short_range.tflite'
UrlModel = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_detector/blaze_face_short_range/float16/1/'
    'blaze_face_short_range.tflite'
)

if not os.path.exists(ModelPath):
    urllib.request.urlretrieve(UrlModel, ModelPath)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_detections(image, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for keypoint in detection.keypoints:
            kp_x = int(keypoint.x * image.shape[1])
            kp_y = int(keypoint.y * image.shape[0])
            cv2.circle(image, (kp_x, kp_y), 3, (0, 0, 255), -1)


ImageFiles = []

imageoptions = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=ModelPath),
    running_mode=VisionRunningMode.IMAGE,
)

with FaceDetector.create_from_options(imageoptions) as detector:
    for idx, file in enumerate(ImageFiles):
        image = cv2.imread(file)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detection_result = detector.detect(mp_image)

        if not detection_result.detections:
            continue

        annotated_image = image.copy()
        draw_detections(annotated_image, detection_result)
        cv2.imwrite(f'annotated_image_{idx}.png', annotated_image)


VideoOptions = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=ModelPath),
    running_mode=VisionRunningMode.VIDEO,
)


def find_real_camera(max_index=5):
    for idx in range(max_index):
        test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not test.isOpened():
            test.release()
            continue
        frame = None
        for _ in range(10):
            ok, frame = test.read()
        if ok and frame is not None and frame.mean() > 5:
            return test
        test.release()
    return None


os.makedirs('Capturas', exist_ok=True)

cap = find_real_camera()

if cap is None:
    exit(1)

frametimestampms = 0
failcount = 0
prevcount = 0
lastprinttime = time.time()

with FaceDetector.create_from_options(VideoOptions) as detector:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            failcount += 1
            if failcount > 10:
                break
            continue

        failcount = 0
        frametimestampms += 33

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detection_result = detector.detect_for_video(mp_image, frametimestampms)
        FaceCount = len(detection_result.detections)

        draw_detections(image, detection_result)
        flipped = cv2.flip(image, 1)

        if FaceCount > prevcount:
            print("SE DETECTO UNA PERSONA")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(f'Capturas/captura_{timestamp}.png', flipped)
        prevcount = FaceCount

        now = time.time()
        if now - lastprinttime >= 5:
            print(f"Personas detectadas: {FaceCount}")
            lastprinttime = now

        cv2.putText(flipped, f"Personas: {FaceCount}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Face Detection', flipped)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
