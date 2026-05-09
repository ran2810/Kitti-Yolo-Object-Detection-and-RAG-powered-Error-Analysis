import streamlit as st
import json, os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import re
import cv2

# enable to print statements
print_debug = True

def load_fuzzy_rules():
    """
    # load json for fuzzy rules
    """
    with open("../data/fuzzy_rules.json", "r") as f:
        return json.load(f)

FUZZY_RULES = load_fuzzy_rules()


def expand_fuzzy_terms(query):
    """
    check for synonyms or letter casing
    
    :param query: user query from streamlit
    """
    query_lower = query.lower()

    detected = []

    for key, rule in FUZZY_RULES.items():
        # Check canonical term - lower case
        if key in query_lower:
            detected.append(key)
            continue

        # Check synonyms
        for syn in rule["synonyms"]:
            if syn in query_lower:
                detected.append(key)
                break

    return detected

# ------------------------------------------------------------
# KITTI PARSER (for GT + predictions)
# ------------------------------------------------------------
def parse_kitti_label_file(path):
    if not os.path.exists(path):
        return []
    objs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            objs.append({
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": list(map(float, parts[4:8])),
                "dimensions": list(map(float, parts[8:11])),
                "location": list(map(float, parts[11:14])),
                "rotation_y": float(parts[14])
            })
    return objs

# ---------------------------------------------------------
# JSON SANITIZATION
# ---------------------------------------------------------
def sanitize_json(text):
    """
    Fix malformed operator keys like "" >= or " >=".
    
    :param text: Description
    """
    text = re.sub(r'"\s*>=\s*"', '">="', text)
    text = re.sub(r'"\s*>\s*"', '">"', text)
    text = re.sub(r'"\s*<=\s*"', '"<="', text)
    text = re.sub(r'"\s*<\s*"', '"<"', text)
    text = text.replace('""', '"')
    return text


def extract_json_block(text):
    """
    Extract the first JSON block from LLM output.
    
    :param text: Description
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    block = sanitize_json(match.group(0))
    try:
        return json.loads(block)
    except:
        return None


# ---------------------------------------------------------
# LLM INTERPRETER (Ollama Llama3 - Scene + Error Mode)
# ---------------------------------------------------------
def interpret_query_with_llm(query, mode):
    """
    convert the query into filter (json format - dict) by asking prompt to LLM model
    
    :param query: user query from streamlit
    """
    fuzzy_hits = expand_fuzzy_terms(query)

    fuzzy_instructions = ""
    for term in fuzzy_hits:
        fuzzy_instructions += f'"{term}" → {json.dumps(FUZZY_RULES[term]["filters"])}\n'

    prompt = f"""
You are a query interpreter for a KITTI dataset explorer.

Your ONLY job is to output a JSON object with:
- "filters": dict of conditions
- "semantic_query": text for semantic search

Two modes:

1. Scene Search:
   Valid fields:
     - num_cars
     - num_pedestrians
     - num_cyclists
     - max_occlusion
     - max_truncation

2. Error Analysis:
   Valid fields:
     - error_type (FP or FN)
     - class (Car, Pedestrian, Cyclist)
     - iou
     - occlusion_level
     - truncation_value

Valid operators: ">", ">=", "<", "<=", "=="

Fuzzy → Numeric rules:
{fuzzy_instructions}

User query: "{query}"
Mode: "{mode}"

Return ONLY JSON. No explanations.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        # get the raw response
        raw = response.json().get("response", "").strip()
        if print_debug:
            print("\nRAW LLM OUTPUT:\n", raw)

        # Extract JSON block
        parsed = extract_json_block(raw)
        if parsed is None:
            raise ValueError("No JSON found")
        
        if print_debug:
            print("\nPARSED JSON:\n", parsed)
        
        # return query as dict (query -> filters)
        return parsed
    
        # # Replace Mongo-style operators
        # json_text = json_text.replace("$gt", ">")
        # json_text = json_text.replace("$lt", "<")
        # json_text = json_text.replace("$gte", ">=")
        # json_text = json_text.replace("$lte", "<=")

    except Exception as e:
        print("LLM error:", e)
        return {"filters": {}, "semantic_query": query}


# ---------------------------------------------------------
# LOAD RAG COMPONENTS
# ---------------------------------------------------------
@st.cache_resource
def load_rag():
    """
    load RAG components (labels file doc, faiss index, transformer embedded model)
    """
    # load for the input dataset
    with open("../data/kitti_docs.json", "r") as f:
        scene_docs = json.load(f)
    scene_index = faiss.read_index("../data/kitti_index.faiss")

    # load the prediction errors
    with open("../data/error_docs.json", "r") as f:
        error_docs = json.load(f)
    error_index = faiss.read_index("../data/error_index.faiss")

    # load model name
    with open("../data/embedding_model.txt", "r") as f:
        model_name = f.read().strip()

    model = SentenceTransformer(model_name)

    return scene_docs, scene_index, error_docs, error_index, model


scene_docs, scene_index, error_docs, error_index, emb_model = load_rag()

# ---------------------------------------------------------
# FILTER ENGINE (supports dict, list, scalar)
# ---------------------------------------------------------
def apply_filters(docs, filters):
    results = []

    for d in docs:
        ok = True

        for key, cond in filters.items():

            # 1) List → equality set
            if isinstance(cond, list):
                if d.get(key) not in cond:
                    ok = False
                    break
                continue

            # 2) Scalar → equality
            if not isinstance(cond, dict):
                if d.get(key) != cond:
                    ok = False
                    break
                continue

            # 3) Operator dict
            for op, val in cond.items():
                dv = d.get(key)

                if dv is None:
                    ok = False
                    break

                if op == ">=" and not (dv >= val): ok = False
                if op == "<=" and not (dv <= val): ok = False
                if op == ">"  and not (dv >  val): ok = False
                if op == "<"  and not (dv <  val): ok = False
                if op == "==" and not (dv == val): ok = False

            if not ok:
                break

        if ok:
            results.append(d)

    return results


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------
def semantic_search(query, docs, index, embed_model, top_k=10):
    emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(emb, top_k)
    return [docs[i] for i in I[0]]

# ------------------------------------------------------------
# VISUALIZATION HELPERS
# ------------------------------------------------------------
def draw_boxes(img, boxes, color, label_prefix):
    """
    Docstring for draw_boxes
    
    :param img: Description
    :param boxes: Description
    :param color: Description
    :param label_prefix: Description
    """
    VALID_CLASSES = {"Car", "Pedestrian", "Cyclist"} 
    for b in boxes: 
        cls = b["class"] 
        # Only draw GT boxes for Car / Pedestrian / Cyclist 
        if cls not in VALID_CLASSES: 
            continue
        x1, y1, x2, y2 = map(int, b["bbox"])
        label = f"{label_prefix} {cls}"
        # if "confidence" in b and b["confidence"] is not None:
        #     label += f" conf {b['confidence']:.2f}"
        # if "occlusion_level" in b and b["occlusion_level"] is not None:
        #     label += f" occ {b['occlusion_level']}"
        # if "truncation_value" in b and b["truncation_value"] is not None:
        #     label += f" trunc {b['truncation_value']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def render_side_by_side(frame_id, image_path, frame_errors):
    """
    Docstring for render_side_by_side
    
    :param frame_id: Description
    :param image_path: Description
    :param frame_errors: Description
    """

    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_path = os.path.join("../data/training/label_2", f"{frame_id}.txt")
    pred_path = os.path.join("../runs/detect/predict/kitti_labels", f"{frame_id}.txt")

    gt_objs = parse_kitti_label_file(gt_path)
    pred_objs = parse_kitti_label_file(pred_path)

    gt_img = img.copy()
    pred_img = img.copy()

    # draw GT boxes on left in green color
    gt_boxes = [{
        "bbox": o["bbox"],
        "class": o["type"],
        "occlusion_level": o["occluded"],
        "truncation_value": o["truncated"]
    } for o in gt_objs]
    gt_img = draw_boxes(gt_img, gt_boxes, (0, 255, 0), "GT")

    # draw GT boxes on left in yellow color
    pred_boxes = [{
        "bbox": o["bbox"],
        "class": o["type"]
    } for o in pred_objs]
    pred_img = draw_boxes(pred_img, pred_boxes, (255, 255, 0), "Pred")

    for e in frame_errors:
        if "bbox" not in e:
            continue
        # FP as red color
        if e["error_type"] == "FP":
            pred_img = draw_boxes(pred_img, [e], (255, 0, 0), "FP")
        else:
            # FN as blue color
            gt_img = draw_boxes(gt_img, [e], (0, 128, 255), "FN")

    combined = np.hstack([gt_img, pred_img])
    return combined


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("KITTI RAG Explorer (Scene Search + Error Analysis)")
st.write("Ask natural language questions about KITTI scenes, objects, occlusion, truncation, or counts.")
st.write("color Code info: FP as red, FN as blue")

# USER UI
query_mode = st.sidebar.selectbox("Query Mode", ["Scene Search", "Error Analysis"])
query = st.text_input("Enter your query")
top_k = st.slider("Number of results", 1, 10, 5)


if query:

    # ---------------------------------------------------------
    # STEP 1 — LLM INTERPRETATION (query - filters)
    # ---------------------------------------------------------
    parsed = interpret_query_with_llm(query, query_mode)
    filters = parsed.get("filters", {})
    semantic_query = parsed.get("semantic_query", query)

    st.markdown("### LLM Interpretation")
    st.json(parsed)

    # Select dataset
    if query_mode == "Scene Search":
        docs = scene_docs
        index = scene_index
    else:
        docs = error_docs
        index = error_index

    # ---------------------------------------------------------
    # STEP 2 — APPLY NUMERIC FILTERS
    # ---------------------------------------------------------
    filtered_docs = apply_filters(docs, filters)

    if len(filtered_docs) > 0:
        st.success(f"Found {len(filtered_docs)} frames matching filters")

        # Sort by the most relevant field 
        sort_field = None
        for field, cond in filters.items():
            if isinstance(cond, list) or not isinstance(cond, dict):
                sort_field = field
                break
            for op in cond.keys():
                if op in [">", ">=", "<", "<="]:
                    sort_field = field
                    break
            if sort_field:
                break

        if sort_field:
            filtered_docs = sorted(
                filtered_docs,
                key=lambda d: d.get(sort_field, 0),
                reverse=True
            )


        # display filtered result images
        for d in filtered_docs[:top_k]:
            st.subheader(f"Frame {d['id']}")
            st.write(d["summary_text"])
            current_dir = os.path.dirname(__file__)
            parent_dir  = os.path.dirname(current_dir)
            img_path = os.path.normpath(os.path.join(parent_dir, d["image_path"]))
            if query_mode == "Error Analysis":
                st.write(f"**Error Type:** {d['error_type']}")
                st.write(f"**Class:** {d['class']}")
                st.write(f"**IoU:** {d['iou']}")

                frame_errors = [e for e in error_docs if e["id"] == d["id"]]

                combined = render_side_by_side(d["id"], img_path, frame_errors)
                if combined is not None:
                    st.image(combined, caption="GT (left) vs Predictions (right)")
                else:
                    st.image(img_path)
            else:
                st.image(img_path)
        st.stop()

    if filters:
        st.warning("No matches for filters. Falling back to semantic search.")

    # ---------------------------------------------------------
    # STEP 3 — SEMANTIC SEARCH
    # ---------------------------------------------------------
    results = semantic_search(semantic_query, docs, index, emb_model, top_k)

    st.markdown("### Semantic Search Results")

    for d in results:
        st.subheader(f"Frame {d['id']}")
        st.write(d["summary_text"])
        current_dir = os.path.dirname(__file__)
        parent_dir  = os.path.dirname(current_dir)
        img_path = os.path.normpath(os.path.join(parent_dir, d["image_path"]))
        
        if query_mode == "Error Analysis":
            st.write(f"**Error Type:** {d['error_type']}")
            st.write(f"**Class:** {d['class']}")
            st.write(f"**IoU:** {d['iou']}")
            frame_errors = [e for e in error_docs if e["id"] == d["id"]]

            combined = render_side_by_side(d["id"], img_path, frame_errors)
            if combined is not None:
                st.image(combined, caption="GT (left) vs Predictions (right)")
            else:
                st.image(img_path)

        st.image(img_path)

