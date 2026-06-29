import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
from tqdm import tqdm

# ------------------------------------------------------------
# KITTI PARSER 
# ------------------------------------------------------------
def parse_kitti_label_file(path):
    """
    Parse KITTI label file into structured objects.
    :param path: label files relative path
    """
    objects = []
    if not os.path.exists(path):
        return objects

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            obj = {
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": list(map(float, parts[4:8])),
                "dimensions": list(map(float, parts[8:11])),
                "location": list(map(float, parts[11:14])),
                "rotation_y": float(parts[14])
            }
            objects.append(obj)
    return objects


# ------------------------------------------------------------
# SCENE DOCUMENT GENERATION
# ------------------------------------------------------------
def build_doc(frame_id, label_dir, image_dir):
    """
    Build a scene-level document for FAISS indexing.
    :param frame_id: image frame id
    :param label_dir: label files relative path
    :param image_dir: image files relative path
    """
    label_path = os.path.join(label_dir, f"{frame_id}.txt")
    image_path = os.path.join(image_dir, f"{frame_id}.png")

    # parse label file
    objects = parse_kitti_label_file(label_path)

    num_cars = sum(1 for o in objects if o["type"] == "Car")
    num_peds = sum(1 for o in objects if o["type"] == "Pedestrian")
    num_cyc = sum(1 for o in objects if o["type"] == "Cyclist")

    max_occ = max((o["occluded"] for o in objects), default=0)
    max_trunc = max((o["truncated"] for o in objects), default=0.0)

    # create summary
    summary = (
        f"Frame {frame_id} contains {num_cars} cars, {num_peds} pedestrians, "
        f"and {num_cyc} cyclists. Max occlusion level is {max_occ}, "
        f"max truncation is {max_trunc}."
    )

    # return label file data as doc
    return {
        "id": frame_id,
        "image_path": image_path,
        "label_path": label_path,
        "objects": objects,
        "summary_text": summary,
        "num_cars": num_cars,
        "num_pedestrians": num_peds,
        "num_cyclists": num_cyc,
        "max_occlusion": max_occ,
        "max_truncation": max_trunc
    }


# ------------------------------------------------------------
# ERROR ANALYSIS HELPERS
# ------------------------------------------------------------
def compute_iou(boxA, boxB):
    """
    Compute IoU between two KITTI bbox arrays [x1,y1,x2,y2].
    IoU = IntersectionArea/UnionArea
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    # top-left
    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    # bottom-right
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)
    
    # if no intersection -> -ve value set to zero. 
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    # get areas of A, B and intersection
    inter_area = inter_w * inter_h
    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)

    # union = total area - intersection
    union = areaA + areaB - inter_area

    # check if union > 0
    return inter_area / union if union > 0 else 0.0

# ------------------------------------------------------------
# ERROR DOCUMENT GENERATION
# ------------------------------------------------------------
def generate_error_docs(frame_ids, label_dir, pred_dir, image_dir):
    """
    Generate FP/FN/IoU error documents for RAG error analysis.
    """
    error_docs = []

    for fid in tqdm(frame_ids, desc="Generating error documents"):
        gt_path = os.path.join(label_dir, f"{fid}.txt")
        pred_path = os.path.join(pred_dir, f"{fid}.txt")
        img_path = os.path.join(image_dir, f"{fid}.png")

        # parse ground truth label file
        gt_objs = parse_kitti_label_file(gt_path)

        # prediction file missing -> no detections by yolo model
        if not os.path.exists(pred_path):
            for obj in gt_objs:
                error_docs.append({
                    "id": fid,
                    "error_type": "FN",
                    "class": obj["type"],
                    "iou": 0.0,
                    "occlusion_level": obj["occluded"],
                    "truncation_value": obj["truncated"],
                    "summary_text": (
                        f"False negative: missed {obj['type']} in frame {fid} "
                        f"(no predictions). Occlusion {obj['occluded']}, truncation {obj['truncated']}."
                    ),
                    "image_path": img_path
                })
            continue
 
        # parse prediction label file
        pred_objs = parse_kitti_label_file(pred_path)

        gt_boxes = [(o["type"], o["bbox"], o["occluded"], o["truncated"]) for o in gt_objs]
        pred_boxes = [(o["type"], o["bbox"]) for o in pred_objs]

        matched_gt = set()
        matched_pred = set()

        # Match predictions to GT
        for p_idx, (p_cls, p_box) in enumerate(pred_boxes):
            best_iou = 0
            best_gi = None

            for g_idx, (g_cls, g_box, g_occ, g_trunc) in enumerate(gt_boxes):
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = g_idx

            if best_iou >= 0.5:
                matched_gt.add(best_gi)
                matched_pred.add(p_idx)
            else:
                # False Positive
                error_docs.append({
                    "id": fid,
                    "error_type": "FP",
                    "class": p_cls,
                    "iou": float(best_iou),
                    "bbox": p_box, 
                    "confidence": pred_objs[p_idx].get("score", None), 
                    "occlusion_level": None,
                    "truncation_value": None,
                    "summary_text": f"False positive: predicted {p_cls} with IoU {best_iou:.2f} in frame {fid}.",
                    "image_path": img_path
                })

        # False Negatives
        for g_idx, (g_cls, g_box, g_occ, g_trunc) in enumerate(gt_boxes):
            if g_idx not in matched_gt:
                error_docs.append({
                    "id": fid,
                    "error_type": "FN",
                    "class": g_cls,
                    "iou": 0.0,
                    "bbox": g_box,
                    "occlusion_level": g_occ,
                    "truncation_value": g_trunc,
                    "summary_text": ( f"False negative: missed {g_cls} in frame {fid} " 
                                     f"with occlusion {g_occ} and truncation {g_trunc}." ),
                    "image_path": img_path
                })

    return error_docs


# ------------------------------------------------------------
# MAIN INDEX BUILDER
# ------------------------------------------------------------
def build_kitti_index(label_dir, image_dir, pred_dir):
    """
    Docstring for build_kitti_index for scene and error -> docs & index both modes(scene and error) and model
    """

    frame_ids = [f.split(".")[0] for f in os.listdir(label_dir)]

    # -------------------------
    # Scene documents
    # -------------------------
    scene_docs = []
    for fid in frame_ids:
        doc = build_doc(fid, label_dir, image_dir)
        scene_docs.append(doc)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    summary_texts  = []
    for d in scene_docs:
        summary_texts .append(d["summary_text"])
    scene_embeddings = model.encode(summary_texts , convert_to_numpy=True)

    # dimension (384 based on model used above) --> fast and 100% recall
    scene_index = faiss.IndexFlatL2(scene_embeddings.shape[1])
    scene_index.add(scene_embeddings)

    # -------------------------
    # Error documents
    # -------------------------
    error_docs = generate_error_docs(frame_ids, label_dir, pred_dir, image_dir)

    if len(error_docs) > 0:
        error_texts  = []
        for d in error_docs:
            error_texts.append(d["summary_text"])
        error_embeddings = model.encode(error_texts, convert_to_numpy=True)

        error_index = faiss.IndexFlatL2(error_embeddings.shape[1])
        error_index.add(error_embeddings)
    else:
        error_index = None

    return scene_docs, scene_index, error_docs, error_index, model


# ------------------------------------------------------------
# CLIP IMAGE INDEX
# ------------------------------------------------------------
def build_clip_index(image_dir, model_name="clip-ViT-B-32"):
    """
    Build a CLIP image embedding index over all KITTI frames using provided model (B-32/L-14) -> Clip index and frames
    """
    print(f"Loading CLIP model: {model_name}")
    clip_model = SentenceTransformer(model_name)

    image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".png"))

    frame_ids  = []
    embeddings = []

    # iterate over kitti training images
    for fname in tqdm(image_files, desc=f"Building CLIP index ({model_name})"):
        frame_id = fname.rsplit(".", 1)[0]
        img_path = os.path.join(image_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            # clip encode returns float64 but FAISS needs float32 -> cast
            emb = clip_model.encode(img, convert_to_numpy=True).astype("float32")
            # get L2 dist
            norm = np.linalg.norm(emb)
            # normalize
            if norm > 0:
                emb /= norm
            frame_ids.append(frame_id)
            embeddings.append(emb)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    embeddings = np.stack(embeddings).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"CLIP index built: {len(frame_ids)} frames, dim={dim}")
    return frame_ids, index


# Model name -> short tag used in filenames
_MODEL_TAG_MAP = {
    "clip-ViT-B-32": "B32",
    "clip-ViT-L-14": "14", 
}

# get tag to add as suffix for file naming
def _model_tag(model_name):
    return _MODEL_TAG_MAP.get(model_name, model_name.split("-")[-1])


# ------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build KITTI FAISS indexes")
    parser.add_argument(
        "--clip-model",
        default="clip-ViT-B-32",
        choices=["clip-ViT-B-32", "clip-ViT-L-14"],
        help="CLIP model to use for image index (default: clip-ViT-B-32)"
    )
    parser.add_argument(
        "--skip-text", action="store_true",
        help="Skip rebuilding text (scene/error) indexes — only rebuild CLIP index"
    )
    args = parser.parse_args()

    label_dir = "../data/training/label_2"
    image_dir = "../data/training/image_2"
    pred_dir  = "../runs/detect/predict/kitti_labels"

    # Text indexes (scene + error) 
    if not args.skip_text:
        scene_docs, scene_index, error_docs, error_index, model = build_kitti_index(
            label_dir, image_dir, pred_dir
        )
        # write scenes files -> .json doc and index(.faiss) 
        with open("../data/kitti_docs.json", "w") as f:
            json.dump(scene_docs, f, indent=2)
        faiss.write_index(scene_index, "../data/kitti_index.faiss")

        # write error files -> .json doc and index(.faiss) 
        with open("../data/error_docs.json", "w") as f:
            json.dump(error_docs, f, indent=2)
        if error_index:
            faiss.write_index(error_index, "../data/error_index.faiss")

        # write model name
        with open("../data/embedding_model.txt", "w") as f:
            f.write("all-MiniLM-L6-v2")
        print("Text indexes saved.")
    else:
        print("Skipping text indexes (--skip-text).")

    # CLIP image index 
    tag = _model_tag(args.clip_model)
    index_path = f"../data/clip_index_{tag}.faiss"
    ids_path   = f"../data/clip_frame_ids_{tag}.json"

    # get index and frame_ids files
    clip_frame_ids, clip_index = build_clip_index(image_dir, args.clip_model)

    # write clip files -> .json doc and index(.faiss) 
    with open(ids_path, "w") as f:
        json.dump(clip_frame_ids, f, indent=2)
    faiss.write_index(clip_index, index_path)

    print(f"CLIP index saved: {index_path}  ({clip_index.d}-dim)")
    print(f"Frame IDs saved:  {ids_path}")