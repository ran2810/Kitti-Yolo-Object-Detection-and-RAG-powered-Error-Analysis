from ultralytics import YOLO

def evaluate(model_path="../runs/detect/train/weights/best.pt", data="../data/kitti.yaml"):
    model = YOLO(model_path)
    metrics = model.val(data=data)

    print("\n--- KITTI Evaluation ---")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print("\nPer-class AP50:")
    for cls, ap in enumerate(metrics.box.maps):
        print(f"  Class {cls}: {ap:.4f}")

if __name__ == "__main__":
    evaluate()