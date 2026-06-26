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
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

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

        # prediction file missing → no detections
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
        for pi, (p_cls, p_box) in enumerate(pred_boxes):
            best_iou = 0
            best_gi = None

            for gi, (g_cls, g_box, g_occ, g_trunc) in enumerate(gt_boxes):
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= 0.5:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                # False Positive
                error_docs.append({
                    "id": fid,
                    "error_type": "FP",
                    "class": p_cls,
                    "iou": float(best_iou),
                    "bbox": p_box, 
                    "confidence": pred_objs[pi].get("score", None), 
                    "occlusion_level": None,
                    "truncation_value": None,
                    "summary_text": f"False positive: predicted {p_cls} with IoU {best_iou:.2f} in frame {fid}.",
                    "image_path": img_path
                })

        # False Negatives
        for gi, (g_cls, g_box, g_occ, g_trunc) in enumerate(gt_boxes):
            if gi not in matched_gt:
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
    Docstring for build_kitti_index
    
    :param label_dir: label files relative path
    :param image_dir: image files relative path
    :param pred_dir: prediction label files relative path
    """

    frame_ids = [f.split(".")[0] for f in os.listdir(label_dir)]

    # -------------------------
    # Scene documents
    # -------------------------
    scene_docs = [build_doc(fid, label_dir, image_dir) for fid in frame_ids]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    scene_embeddings = model.encode(
        [d["summary_text"] for d in scene_docs],
        convert_to_numpy=True
    )

    # dimension (384 based on model used above). good for the N < 100k. fast and 100% recall
    scene_index = faiss.IndexFlatL2(scene_embeddings.shape[1])
    scene_index.add(scene_embeddings)

    # -------------------------
    # Error documents
    # -------------------------
    error_docs = generate_error_docs(frame_ids, label_dir, pred_dir, image_dir)

    if len(error_docs) > 0:
        error_embeddings = model.encode(
            [d["summary_text"] for d in error_docs],
            convert_to_numpy=True
        )
        error_index = faiss.IndexFlatL2(error_embeddings.shape[1])
        error_index.add(error_embeddings)
    else:
        error_index = None

    return scene_docs, scene_index, error_docs, error_index, model


# ------------------------------------------------------------
# CLIP IMAGE INDEX
# ------------------------------------------------------------
def build_clip_index(image_dir):
    """
    Build a CLIP image embedding index over all KITTI frames.

    Uses cosine similarity (IndexFlatIP on L2-normalised vectors).
    Dimension: 512 (clip-ViT-B-32).
    """
    clip_model = SentenceTransformer("clip-ViT-B-32")

    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(".png")
    )

    frame_ids = []
    embeddings = []

    # iterate over images. show progress
    for fname in tqdm(image_files, desc="Building CLIP index"):
        frame_id = fname.rsplit(".", 1)[0]
        img_path = os.path.join(image_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            # default tensors. enable numpy true
            emb = clip_model.encode(img, convert_to_numpy=True).astype("float32")
            # L2-normalise so IndexFlatIP == cosine similarity
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            frame_ids.append(frame_id)
            embeddings.append(emb)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    embeddings = np.stack(embeddings).astype("float32")   # (N, 512)
    dim = embeddings.shape[1]

    # IndexFlatIP: inner product on normalised vectors = cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"CLIP index built: {len(frame_ids)} frames, dim={dim}")
    return frame_ids, index


# ------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    label_dir = "../data/training/label_2"
    image_dir = "../data/training/image_2"
    pred_dir = "../runs/detect/predict/kitti_labels"

    # Text indexes 
    scene_docs, scene_index, error_docs, error_index, model = build_kitti_index(
        label_dir, image_dir, pred_dir
    )

    # scene index
    with open("../data/kitti_docs.json", "w") as f:
        json.dump(scene_docs, f, indent=2)
    faiss.write_index(scene_index, "../data/kitti_index.faiss")

    # error idnex
    with open("../data/error_docs.json", "w") as f:
        
        json.dump(error_docs, f, indent=2)
    if error_index:
        faiss.write_index(error_index, "../data/error_index.faiss")

    # store mode name
    with open("../data/embedding_model.txt", "w") as f:
        f.write("all-MiniLM-L6-v2")

    #  CLIP image index 
    clip_frame_ids, clip_index = build_clip_index(image_dir)

    faiss.write_index(clip_index, "../data/clip_index.faiss")
    with open("../data/clip_frame_ids.json", "w") as f:
        json.dump(clip_frame_ids, f, indent=2)

    print("All indexes saved to ../data/")