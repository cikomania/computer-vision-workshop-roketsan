import os
import glob
import cv2
from ultralytics import YOLO

IMAGE_DIR = "video_sequences"
OUTPUT_VIDEO = "vehicle_tracking.mp4"
MODEL_NAME = "yolo11n.pt"
CONFIDENCE = 0.2
FPS = 20
VEHICLE_CLASSES = [2, 3, 5, 7]
model = YOLO(MODEL_NAME)
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

first_frame = cv2.imread(image_paths[0])
height, width = first_frame.shape[:2]
video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO, 
    cv2.VideoWriter_fourcc(*"mp4v"), 
    FPS, 
    (width, height)
)

for i, image_path in enumerate(image_paths):
    frame = cv2.imread(image_path)
    # YOLO + ByteTrack
    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        conf=CONFIDENCE,
        verbose=False
    )[0]
    boxes = result.boxes

    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for box, class_id, track_id, confidence in zip(xyxy, class_ids, track_ids, confidences):

            x1, y1, x2, y2 = map(int, box)

            class_name = model.names[class_id]

            # bounding Box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # araç sınıfı + takip ID
            label = f"{class_name} | ID:{track_id} | {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # frame numarasını göster
    cv2.putText(
        frame,
        f"Frame: {i + 1}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # videoya yaz
    video_writer.write(frame)

    print(f"{i + 1}/{len(image_paths)} işlendi.")

video_writer.release()

print("\nAraç takibi tamamlandı.")
print("Video:", OUTPUT_VIDEO)