from ultralytics import YOLO
import glob
import cv2
import matplotlib.pyplot as plt

# helper functions to draw YOLO boxes
def draw_yolo_boxes(img, labels, class_names, color=(0,255,0)):
    h, w = img.shape[:2]
    img_copy = img.copy()

    for lbl in labels:
        cls, xc, yc, bw, bh = lbl
        xc *= w
        yc *= h
        bw *= w
        bh *= h

        x1 = int(xc - bw/2)
        y1 = int(yc - bh/2)
        x2 = int(xc + bw/2)
        y2 = int(yc + bh/2)

        cv2.rectangle(img_copy, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img_copy, class_names[int(cls)], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_copy


# Load ground‑truth labels
def load_yolo_label(label_path):
    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            labels.append((cls, xc, yc, w, h))
    return labels


# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# Class names (KITTI subset)
CLASSES = ["Car", "Pedestrian", "Cyclist"]

# Pick a validation image
img_path = sorted(glob.glob("kitti_yolo/images/val/*.png"))[0]
label_path = img_path.replace("images", "labels").replace(".png", ".txt")

# Load image + ground truth
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gt_labels = load_yolo_label(label_path)

# Draw ground truth
img_gt = draw_yolo_boxes(img, gt_labels, CLASSES, color=(0,255,0))

# Run prediction
results = model.predict(img_path, conf=0.25)
pred_labels = []

for box in results[0].boxes:
    cls = int(box.cls[0])
    xc, yc, w, h = box.xywhn[0].tolist()
    pred_labels.append((cls, xc, yc, w, h))

# Draw predictions
img_pred = draw_yolo_boxes(img, pred_labels, CLASSES, color=(255,0,0))

# Show side-by-side
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.title("Ground Truth")
plt.imshow(img_gt)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Prediction")
plt.imshow(img_pred)
plt.axis("off")

plt.show()
