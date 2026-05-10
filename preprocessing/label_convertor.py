import os
from pathlib import Path
import cv2
import shutil
import random
import argparse

# KITTI classes for training
CLASSES = ["Car", "Pedestrian", "Cyclist"]
class_map = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

# CONVERT KITTI -> YOLO FORMAT
def convert_kitti2yolo(kitti_label_path, yolo_label_path, img_width, img_height):
    """Convert kitti label file to yolo format label file """
    
    # read label file
    with open(kitti_label_path, "r") as f:
        lines = f.readlines()
        
    yolo_lines = []

    # iterate label file
    for line in lines:
        parts = line.strip().split()
        cls = parts[0]
        
        # consider only required classes
        if cls not in CLASSES:
            continue  # skip other classes
        cls_id = CLASSES.index(cls)
        
        # Bounding Box in pixel coordinates
        x1, y1, x2, y2 = map(float, parts[4:8])

        # Convert to YOLO format
		# divide x_center and width by image width, and y_center and height by image height
        xc = (x1 + x2) / 2.0 / img_width
        yc = (y1 + y2) / 2.0 / img_height
        # normalized width =  box_width_pixel / image_width  
        w = (x2 - x1) / img_width
        # normalized height =  box_height_pixel / image_height 
        h = (y2 - y1) / img_height

        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # write the data in converted format(yolo)
    with open(yolo_label_path, "w") as f:
        f.write("\n".join(yolo_lines))


# SPLIT DATASET INTO TRAINING & VALIDATION
def prepare_yolo_dataset(kitti_root="kitti/training", out_root="kitti_yolo", train_ratio=0.8):
    """Convert full KITTI dataset into YOLO format with train/val split."""
    
    img_dir = Path(kitti_root) / "image_2"
    label_dir = Path(kitti_root) / "label_2"

    img_files = sorted(list(img_dir.glob("*.png")))
    random.seed(42)
    random.shuffle(img_files)

    # Create YOLO folder structure
    for split in ["train", "val"]:
        (Path(out_root) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = int(len(img_files) * train_ratio)

    for i, img_path in enumerate(img_files):
        split = "train" if i < n_train else "val"

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        label_path = label_dir / (img_path.stem + ".txt")

        out_img_path = Path(out_root) / "images" / split / img_path.name
        out_label_path = Path(out_root) / "labels" / split / (img_path.stem + ".txt")

        # Copy image
        shutil.copy(str(img_path), str(out_img_path))
    
        # Convert label
        convert_kitti2yolo(label_path, out_label_path, w, h)

    print("KITTI -->  YOLO conversion complete!")
    print(f"Train images: {n_train}")
    print(f"Val images:   {len(img_files) - n_train}")

# CONVERT YOLO -> KITTI FORMAT
def convert_yolo2kitti(pred_file, img_file, out_file, class_map):
    img = cv2.imread(img_file)
    h, w = img.shape[:2]

    with open(pred_file) as f:
        lines = f.readlines()

    with open(out_file, "w") as out:
        for line in lines:
            cls, cx, cy, bw, bh, conf = map(float, line.split())
            cls = int(cls)

            # Convert YOLO → absolute xyxy
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h

            obj_type = class_map[cls]

            # KITTI format (dummy values for 3D fields)
            out.write(
                f"{obj_type} 0 0 0 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"0 0 0 0 0 0 0\n"
            )

def batch_convert_yolo2kitti(pred_dir="runs/detect/predict/labels", img_dir="data/training/image_2", out_dir="runs/detect/predict/kitti_labels"):
    """
    """
    # create out_dir folder if doesnt exists
    os.makedirs(out_dir, exist_ok=True)

    # batch trigger
    for fname in os.listdir(pred_dir):
        fid = fname.replace(".txt", "")
        yolo_file = os.path.join(pred_dir, fname)
        img_file = os.path.join(img_dir, f"{fid}.png")
        out_file = os.path.join(out_dir, f"{fid}.txt")

        # convert the label file
        convert_yolo2kitti(yolo_file, img_file, out_file, class_map)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='label file convertor')
    parser.add_argument('convert_typ', type=str,
                        help='yolo2kitti or kitti2yolo')
    args = parser.parse_args()
    current_dir = os.path.dirname(__file__)
    parent_dir  = os.path.dirname(current_dir)

    kitti_root = os.path.join(parent_dir, "data/training")
    out_root  = os.path.join(parent_dir, "kitti_yolo")
    if args.convert_typ == "kitti2yolo":
        prepare_yolo_dataset(kitti_root, out_root)
    
    if args.convert_typ == "yolo2kitti":
        batch_convert_yolo2kitti()