import cv2
import os

INPUT_IMAGE = "image.png"

OUTPUT_FOLDER = "ground_truth_parts"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

print("Loading image")

img = cv2.imread(INPUT_IMAGE)

height, width, _ = img.shape

print("Image size:", height, width)

# Define grid layout
rows = 3
cols = 3

h_step = height // rows
w_step = width // cols

count = 1

for r in range(rows):

    for c in range(cols):

        y1 = r * h_step
        y2 = (r + 1) * h_step

        x1 = c * w_step
        x2 = (c + 1) * w_step

        crop = img[y1:y2, x1:x2]

        filename = f"img_{count}.png"

        path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        cv2.imwrite(path, crop)

        print("Saved:", filename)

        count += 1