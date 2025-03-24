import cv2
import glob
import os
import numpy as np
from datetime import datetime
from rich.progress import track


input_folder = "timelapse_images/Picture/"
image_pattern = os.path.join(input_folder, "*.bmp")
all_files = sorted(glob.glob(image_pattern))

if not all_files:
    raise FileNotFoundError("No BMP images found in folder")

brightness_threshold = 10

# filter out dark frames
cache_file = "valid_frames_cache.txt"

if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        valid_files = [line.strip() for line in f if line.strip()]
else:
    valid_files = []
    for image_path in track(all_files, description="Filtering frames... "):
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        if frame.mean() < brightness_threshold:
            print(
                f"Skipping dark frame: {image_path} (mean brightness: {frame.mean():.2f})"
            )
            continue

        valid_files.append(image_path)

    with open(cache_file, "w") as f:
        for file_path in valid_files:
            f.write(file_path + "\n")

if not valid_files:
    raise FileExistsError(
        "No valid (bright enough) frames found after filtering")

num_frames = len(valid_files)
fps = 30

# determine video dimensions
first_frame = cv2.imread(valid_files[0])
if first_frame is None:
    raise ValueError("Unable to read first frame")
height, width, _ = first_frame.shape

# set up video writer
output_video = "timelapse.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for i, image_path in enumerate(track(valid_files, description="Creating timelapse... ")):
    if i % 4 != 0:
        continue

    frame = cv2.imread(image_path)
    if frame is None:
        continue

    filename = os.path.splitext(os.path.basename(image_path))[0]
    parts = filename.split(" - ")

    if len(parts) == 2:
        timestamp_str = parts[1]
        try:
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            overlay_text = dt.strftime("%d/%m/%Y, %H:%M:%S")
        except Exception as e:
            print(f"Unable to parse date from {filename}: {e}")
            overlay_text = timestamp_str
    else:
        overlay_text = filename

    position = (15, 45)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.5
    color = (255, 255, 255)  # White text.
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        overlay_text, font, font_scale, thickness
    )

    cv2.rectangle(
        frame,
        (position[0] - 5, position[1] - text_height - 5),
        (position[0] + text_width + 5, position[1] + baseline + 5),
        (0, 0, 0),
        cv2.FILLED,
    )

    cv2.putText(
        frame,
        overlay_text,
        position,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    video_writer.write(frame)

video_writer.release()
print(f"Timelapse video saved as {output_video}")
