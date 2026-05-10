import os
import zipfile
import requests
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# DOWNLOAD KITTI DATASET (IMAGES + LABELS)
def download_kitti(output_dir="kitti"):
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "data_object_image_2.zip":
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "data_object_label_2.zip":
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    }

    for filename, url in files.items():
        out_path = Path(output_dir) / filename

        if out_path.exists():
            print(f"{filename} already downloaded.")
            continue

        print(f"Downloading {filename}...")
        r = requests.get(url, stream=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print(f"Extracting {filename}...")
        with zipfile.ZipFile(out_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

    print("KITTI dataset ready!")


# VISUALIZE KITTI GROUND TRUTH LABELS
def visualize_kitti_sample(image_path, label_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x1, y1, x2, y2 = parse_kitti_label(line)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, cls, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    plt.imshow(img)
    plt.axis("off")
    plt.show()


# PARSE KITTI LABEL FILE
def parse_kitti_label(line):
    parts = line.strip().split()
    cls = parts[0]
    x1, y1, x2, y2 = map(int, map(float, parts[4:8]))
    return cls, x1, y1, x2, y2

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    download_kitti()

    # sample_img = "kitti/training/image_2/000000.png"
    # sample_label = "kitti/training/label_2/000000.txt"

    # print("Visualizing KITTI ground truth...")
    # visualize_kitti_sample(sample_img, sample_label)