import streamlit as st
import json, os, time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import re
import cv2
from pathlib import Path

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False  

# enable to print statements
print_debug = True

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Minimum cosine similarity GAP between the two modes to trust the embedding result.
# Gap = best_mode_score - other_mode_score (not the absolute score).
# Typical ranges based on examples: confident classification gap ~0.10-0.25, ambiguous query gap ~0.01-0.05.
# If gap is below this, the classifier is uncertain and falls back to keyword check.
CONFIDENCE_THRESHOLD = 0.08

# ---------------------------------------------------------
# AUTO MODE DETECTION — embedding-based intent classification
# ---------------------------------------------------------
# Example queries that represent each mode -> Encode at startup -> Find similiarity via cosine
MODE_EXAMPLES = {
    "Error Analysis": [
        "find missed pedestrians",
        "wrongly detected objects with high occlusion",
        "false positives for cyclists",
        "frames where cars were not detected",
        "detection errors with high truncation",
        "objects the model failed to detect",
        "incorrectly predicted bounding boxes",
        "undetected cyclists in the scene",
        "high IoU mismatches",
        "false negative cars with occlusion",
    ],
    "Scene Search": [
        "scenes with many cars",
        "frames with high occlusion",
        "find busy intersections with pedestrians",
        "scenes with cyclists and pedestrians together",
        "frames with low truncation and many objects",
        "crowded urban scenes",
        "find frames with more than three pedestrians",
        "scenes where cyclists are present",
        "frames with high object density",
        "urban driving with multiple road users",
    ]
}

# Keyword fallback — used when embedding similarity is below CONFIDENCE_THRESHOLD
# so that unambiguous error terms still work even if the model is not yet loaded
ERROR_KEYWORDS = {
    "missed", "false positive", "false negative", "fp", "fn",
    "error", "iou", "wrong detection", "missed detection",
    "undetected", "incorrectly detected", "wrongly detected"
}


def build_mode_embeddings(model):
    """
    Encode MODE_EXAMPLES once at startup using the provided sentence-transformer model.
    Returns a dict: { mode_name -> np.ndarray of shape (n_examples, dim) }

    :param model: loaded SentenceTransformer instance
    """
    mode_embeddings = {}
    for mode, examples in MODE_EXAMPLES.items():
        # encode examples. Normalize so that cosine similarity = dot product
        embs = model.encode(examples, convert_to_numpy=True, normalize_embeddings=True)
        mode_embeddings[mode] = embs  # shape: (n_examples, 384)
    return mode_embeddings


def auto_detect_mode(query, user_mode, mode_embeddings, model):
    """
    Classify query intent using cosine similarity against MODE_EXAMPLES embeddings.

    :param query:           user query string
    :param user_mode:       mode hint from sidebar
    :param mode_embeddings: dict returned by build_mode_embeddings()
    :param model:           loaded SentenceTransformer instance
    :return: (effective_mode, was_overridden, scores)
             scores = {'Error Analysis': float, 'Scene Search': float}
    """
    # encode query — normalised. cosine similarity = dot product
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    scores = {}
    for mode, embs in mode_embeddings.items():
        # mean cosine similarity across all examples for this mode
        scores[mode] = float(np.mean(embs @ q_emb))

    # iterate via scores.get for values
    best_mode  = max(scores, key=scores.get)
    score_gap  = scores[best_mode] - scores[min(scores, key=scores.get)]

    if print_debug:
        print(f"\nAUTO-DETECT scores: {scores}  gap: {score_gap:.4f}")

    if score_gap >= CONFIDENCE_THRESHOLD:
        # Embedding classifier is confident
        effective_mode = best_mode
    else:
        # Ambiguous — fall back to keyword check
        query_lower = query.lower()
        if any(kw in query_lower for kw in ERROR_KEYWORDS):
            effective_mode = "Error Analysis"
        # keep sidebar hint
        else:
            effective_mode = user_mode   

    was_overridden = (effective_mode != user_mode)
    return effective_mode, was_overridden, scores


# ---------------------------------------------------------
# LLM CALL ROUTER — Ollama (local) or Groq (cloud)
# ---------------------------------------------------------
def call_llm(prompt, llm_config):
    """
    Route LLM call to Ollama (local) or Groq (cloud) based on llm_config.
    Returns raw response string from the model.

    :param prompt: prompt string to send to the model
    :param llm_config: dict with keys 'provider', 'model', 'api_key' (Groq only)
    """
    provider = llm_config.get("provider", "Ollama")
    model    = llm_config.get("model", "llama3")

    if provider == "Groq":
        api_key = llm_config.get("api_key", "")
        if not api_key:
            raise ValueError("Groq API key is required.")
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    else:  # Ollama (local)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "").strip()

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

    :param text: LLM raw output string
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    block = sanitize_json(match.group(0))
    try:
        return json.loads(block)
    except:
        return None


# Operator string patterns the LLM sometimes outputs instead of {"op": value}
# e.g. ">0", "> 0", ">=2", "<= 1.0"
_OP_STRING_RE = re.compile(r'^(>=|<=|>|<|==)\s*(-?\d+(?:\.\d+)?)$')

# MongoDB-style operator aliases some models use
_MONGO_OPS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<=", "$eq": "=="}

def _coerce_numeric(val):
    """Convert string number to int or float."""
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return val

def normalize_filters(filters):
    """
    Normalise LLM filter output to the format apply_filters() expects:
      - operator dict:  {">=": 2}
      - scalar:         "FN"  (equality)
      - list:           ["FP", "FN"]  (membership)

    Handles common LLM output variations:
      ">0"          -> {">":  0}     (op+value fused into string)
      "> 0"         -> {">":  0}     (op+value with space)
      {"$gt": 0}    -> {">":  0}     (MongoDB-style operators)
      {">": "2"}    -> {">":  2}     (numeric value as string)
      0  (scalar)   -> {">" : 0}     (bare 0 for numeric range fields means > 0)

    :param filters: raw filters dict from LLM
    :return: normalised filters dict
    """
    # Numeric range fields — bare scalar 0 almost always means "> 0" in context
    RANGE_FIELDS = {
        "num_cars", "num_pedestrians", "num_cyclists",
        "max_occlusion", "max_truncation",
        "occlusion_level", "truncation_value", "iou"
    }

    normalised = {}

    for key, cond in filters.items():

        #  list: keep as-is 
        if isinstance(cond, list):
            normalised[key] = cond
            continue

        #  string: may be fused op+value e.g. ">0" or plain scalar 
        if isinstance(cond, str):
            m = _OP_STRING_RE.match(cond.strip())
            if m:
                op, val = m.group(1), _coerce_numeric(m.group(2))
                normalised[key] = {op: val}
            else:
                normalised[key] = cond   # plain string equality (e.g. "FN")
            continue

        #  numeric scalar 
        if isinstance(cond, (int, float)):
            if key in RANGE_FIELDS and cond == 0:
                # "occlusion_level: 0" from LLM almost always means > 0
                normalised[key] = {">": 0}
            else:
                normalised[key] = cond   # genuine equality (e.g. num_cars: 3)
            continue

        # dict: operator dict
        if isinstance(cond, dict):
            clean = {}
            for op, val in cond.items():
                # remap MongoDB operators
                op = _MONGO_OPS.get(op, op)
                # coerce string numeric values
                val = _coerce_numeric(val) if isinstance(val, str) else val
                clean[op] = val
            normalised[key] = clean
            continue

        # fallback: keep unchanged
        normalised[key] = cond

    return normalised

# ---------------------------------------------------------
# LLM INTERPRETER (Ollama Llama3 - Scene + Error Mode)
# ---------------------------------------------------------
# improvise prompt to handle false negative(s) cars(for cars) with occlusion ( 0)
def interpret_query_with_llm(query, mode, llm_config):
    """
    Convert the user query into a filter dict + semantic query string
    by asking the LLM. Routes to Ollama or Groq based on llm_config.

    :param query: user query string from streamlit
    :param mode: effective mode ('Scene Search' or 'Error Analysis')
    :param llm_config: dict with keys 'provider', 'model', 'api_key'
    """
    fuzzy_hits = expand_fuzzy_terms(query)

    fuzzy_instructions = ""
    for term in fuzzy_hits:
        fuzzy_instructions += f'"{term}" -> {json.dumps(FUZZY_RULES[term]["filters"])}\n'

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

Valid operators for numeric fields: ">", ">=", "<", "<=", "=="
ALWAYS use a dict for operator conditions: {{">=": 2}} not ">=2" or ">= 2".
NEVER use MongoDB-style operators ($gt, $gte, etc.).
NEVER use bare 0 for a range condition — write {{">": 0}} to mean "greater than zero".

Examples (Error Analysis):
  "false negatives for cars with any occlusion"
  -> {{"filters": {{"error_type": "FN", "class": "Car", "occlusion_level": {{">": 0}}}}, "semantic_query": "false negatives for cars with occlusion"}}

  "false positives for pedestrians with occlusion level 2 or more"
  -> {{"filters": {{"error_type": "FP", "class": "Pedestrian", "occlusion_level": {{">=": 2}}}}, "semantic_query": "false positives for occluded pedestrians"}}

Examples (Scene Search):
  "scenes with more than 3 cars"
  -> {{"filters": {{"num_cars": {{">": 3}}}}, "semantic_query": "scenes with many cars"}}

Fuzzy -> Numeric rules:
{fuzzy_instructions}

User query: "{query}"
Mode: "{mode}"

Return ONLY the JSON object. No explanation, no markdown, no code fences.
"""

    try:
        t0 = time.perf_counter()
        raw = call_llm(prompt, llm_config)
        llm_latency_ms = (time.perf_counter() - t0) * 1000

        if print_debug:
            print("\nRAW LLM OUTPUT:\n", raw)
            print(f"\nLLM latency: {llm_latency_ms:.0f} ms")

        parsed = extract_json_block(raw)
        if parsed is None:
            raise ValueError("No JSON found")

        # Normalise operator formats before returning (handles LLM drift)
        if "filters" in parsed and isinstance(parsed["filters"], dict):
            parsed["filters"] = normalize_filters(parsed["filters"])

        if print_debug:
            print("\nPARSED JSON:\n", parsed)

        return parsed, llm_latency_ms

    except Exception as e:
        print("LLM error:", e)
        return {"filters": {}, "semantic_query": query}, None


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

    # CLIP image index (optional — skip if not yet built)
    clip_index, clip_frame_ids, clip_model = None, [], None
    if os.path.exists("../data/clip_index.faiss"):
        clip_index = faiss.read_index("../data/clip_index.faiss")
        with open("../data/clip_frame_ids.json", "r") as f:
            clip_frame_ids = json.load(f)
        clip_model = SentenceTransformer("clip-ViT-B-32")

    return scene_docs, scene_index, error_docs, error_index, model, \
           clip_index, clip_frame_ids, clip_model


scene_docs, scene_index, error_docs, error_index, emb_model, \
    clip_index, clip_frame_ids, clip_model = load_rag()

# Build mode example embeddings once at startup (reuses already-loaded emb_model)
mode_embeddings = build_mode_embeddings(emb_model)

# ---------------------------------------------------------
# FILTER ENGINE (supports dict, list, scalar)
# ---------------------------------------------------------
def apply_filters(docs, filters):
    results = []

    for d in docs:
        ok = True

        for key, cond in filters.items():

            # 1) List -> equality set
            if isinstance(cond, list):
                if d.get(key) not in cond:
                    ok = False
                    break
                continue

            # 2) Scalar -> equality
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


# ---------------------------------------------------------
# CLIP VISUAL SEARCH
# ---------------------------------------------------------
def clip_encode_images(pil_images, model):
    """
    Encode a list of PIL images with CLIP and return the mean
    L2-normalised embedding (support-set average).
    """
    embs = []
    for img in pil_images:
        e = model.encode(img, convert_to_numpy=True).astype("float32")
        norm = np.linalg.norm(e)
        if norm > 0:
            e /= norm
        embs.append(e)
    mean_emb = np.mean(embs, axis=0).astype("float32")
    mean_emb /= np.linalg.norm(mean_emb)          # renormalise after mean
    return mean_emb


def clip_encode_text(text, model):
    """
    Encode a text string with CLIP text encoder, L2-normalised.
    """
    e = model.encode(text, convert_to_numpy=True).astype("float32")
    e /= np.linalg.norm(e)
    return e


def clip_search(query_emb, clip_index, clip_frame_ids, scene_docs, top_k=10):
    """
    Search the CLIP image index and return matching scene docs with scores.

    :param query_emb: normalised 512-dim query vector (text or image)
    :param clip_index: IndexFlatIP
    :param clip_frame_ids: ordered list matching index positions
    :param scene_docs: list of scene doc dicts (for image_path lookup)
    :param top_k: number of results
    :return: list of (score, frame_id, image_path)
    """
    q = query_emb.reshape(1, -1)
    scores, indices = clip_index.search(q, top_k)

    # Build a quick lookup from frame_id -> image_path
    id_to_doc = {d["id"]: d for d in scene_docs}

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        fid = clip_frame_ids[idx]
        doc = id_to_doc.get(fid, {})
        results.append({
            "frame_id": fid,
            "score": float(score),
            "image_path": doc.get("image_path", ""),
            "summary_text": doc.get("summary_text", ""),
        })
    return results

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
st.write("Color code: FP = red, FN = blue")

# ---------------------------------------------------------
# SIDEBAR — LLM Provider
# ---------------------------------------------------------
st.sidebar.markdown("## LLM Provider")
llm_provider = st.sidebar.radio("Backend", ["Ollama (local)", "Groq (cloud)"])

if llm_provider == "Ollama (local)":
    ollama_model = st.sidebar.selectbox("Ollama Model", ["llama3", "llama3.1", "mistral"])
    st.sidebar.caption(f"Running locally via Ollama — model: {ollama_model}")
    llm_config = {"provider": "Ollama", "model": ollama_model, "api_key": ""}
else:
    if not GROQ_AVAILABLE:
        st.sidebar.error("groq package not installed -> pip install groq")
    groq_model = st.sidebar.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    )
    groq_key = st.sidebar.text_input("Groq API Key", type="password")
    st.sidebar.caption(f"Groq cloud — model: {groq_model}")
    llm_config = {"provider": "Groq", "model": groq_model, "api_key": groq_key}

st.sidebar.divider()

# ---------------------------------------------------------
# SIDEBAR — Mode Hint (can be overridden by auto-detection)
# ---------------------------------------------------------
st.sidebar.markdown("### Mode Hint")
st.sidebar.caption(
    "Auto-detection overrides this if error-related keywords "
    "(missed, FP, FN, IoU ...) are found in your query."
)
query_mode = st.sidebar.selectbox("Mode Hint", ["Scene Search", "Error Analysis"])

# ---------------------------------------------------------
# MAIN — Query input
# ---------------------------------------------------------
query = st.text_input("Enter your query")
top_k = st.slider("Number of results", 1, 10, 5)


if query:

    # ---------------------------------------------------------
    # AUTO MODE DETECTION (embedding-based)
    # ---------------------------------------------------------
    effective_mode, was_overridden, scores = auto_detect_mode(
        query, query_mode, mode_embeddings, emb_model
    )

    # print the mode 
    if was_overridden:
        st.info(
            f"Auto-detected mode: **{effective_mode}** "
            f"(overrode sidebar hint: '{query_mode}')  "
            f"— scores: Error Analysis {scores['Error Analysis']:.3f} "
            f"| Scene Search {scores['Scene Search']:.3f}"
        )
    else:
        st.info(
            f"Mode: **{effective_mode}**  "
            f"— scores: Error Analysis {scores['Error Analysis']:.3f} "
            f"| Scene Search {scores['Scene Search']:.3f}"
        )

    # ---------------------------------------------------------
    # LLM INTERPRETATION (query -> filters)
    # ---------------------------------------------------------
    parsed, llm_latency_ms = interpret_query_with_llm(query, effective_mode, llm_config)
    filters = parsed.get("filters", {})
    semantic_query = parsed.get("semantic_query", query)

    st.markdown("### LLM Interpretation")
    if llm_latency_ms is not None:
        provider_label = llm_config.get("provider", "LLM")
        model_label    = llm_config.get("model", "")
        color = "green" if llm_latency_ms < 1000 else ("orange" if llm_latency_ms < 3000 else "red")
        st.caption(
            f"**{provider_label}** ({model_label}) — "
            f":{color}[{llm_latency_ms:.0f} ms]"
        )
    st.json(parsed)

    # Select dataset based on effective mode
    if effective_mode == "Scene Search":
        docs = scene_docs
        index = scene_index
    else:
        docs = error_docs
        index = error_index

    # ---------------------------------------------------------
    # APPLY NUMERIC FILTERS
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
            image_path = Path(*Path(d["image_path"]).parts[1:]) # Drop the leading ".."
            # if print_debug:
            #     print("\n image path:", image_path , " parent_dir", parent_dir)
            img_path = parent_dir / image_path
            if effective_mode == "Error Analysis":
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
    # SEMANTIC SEARCH
    # ---------------------------------------------------------
    results = semantic_search(semantic_query, docs, index, emb_model, top_k)

    st.markdown("### Semantic Search Results")

    for d in results:
        st.subheader(f"Frame {d['id']}")
        st.write(d["summary_text"])
        image_path = Path(*Path(d["image_path"]).parts[1:]) # Drop the leading ".."
        img_path = parent_dir / image_path
        
        if effective_mode == "Error Analysis":
            st.write(f"**Error Type:** {d['error_type']}")
            st.write(f"**Class:** {d['class']}")
            st.write(f"**IoU:** {d['iou']}")
            frame_errors = [e for e in error_docs if e["id"] == d["id"]]

            # show GT and predicted images
            combined = render_side_by_side(d["id"], img_path, frame_errors)
            if combined is not None:
                st.image(combined, caption="GT (left) vs Predictions (right)")
            else:
                st.image(img_path)
        # for scene search - show the image directly
        st.image(img_path)


# ---------------------------------------------------------
# VISUAL SEARCH (CLIP) — separate section below text query
# ---------------------------------------------------------
st.divider()
st.markdown("## Visual Search (CLIP)")

if clip_index is None:
    st.warning(
        "CLIP index not found. Run `generate_faiss_doc.py` to build it first.  \n"
        "`python queries/generate_faiss_doc.py`"
    )
else:
    clip_search_mode = st.radio(
        "Query type",
        ["Text -> Images", "Image -> Images"],
        horizontal=True
    )
    clip_top_k = st.slider("Visual results", 1, 10, 5, key="clip_k")

    # Text -> Images 
    # works with full frame level context only but not minor object like - pedestrian crossing sign or . 
    if clip_search_mode == "Text -> Images":
        clip_text_query = st.text_input(
            "Describe what you want to find visually -- full frame level context only",
            placeholder="e.g. traffic light, parked vehicles, frames with railway tracks, road with tram line",
            key="clip_text"
        )
        # run query, calculate latency and display top k results
        if clip_text_query:
            t0 = time.perf_counter()
            q_emb = clip_encode_text(clip_text_query, clip_model)
            latency_ms = (time.perf_counter() - t0) * 1000
            results = clip_search(q_emb, clip_index, clip_frame_ids, scene_docs, clip_top_k)
            st.caption(f" CLIP text encode + search — {latency_ms:.0f} ms")

            st.markdown(f"**Top {clip_top_k} visually similar frames:**")
            for r in results:
                image_path = Path(*Path(r["image_path"]).parts[1:]) # Drop the leading ".."
                img_path = parent_dir / image_path
                st.subheader(f"Frame {r['frame_id']}  —  score {r['score']:.3f}")
                st.caption(r["summary_text"])
                st.image(img_path)

    #  Image -> Images 
    else:
        st.caption(
            "Upload 1-5 example images (support set). "
            "Multiple images make the query more robust "
        )
        uploaded_files = st.file_uploader(
            "Upload query image(s)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="clip_upload"
        )
        if uploaded_files:
            pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]

            # Show thumbnails of uploaded support set
            cols = st.columns(min(len(pil_images), 5))
            for col, img in zip(cols, pil_images):
                col.image(img, use_container_width=True)

            # run query, measure latency and display top k results
            t0 = time.perf_counter()
            q_emb = clip_encode_images(pil_images, clip_model)
            latency_ms = (time.perf_counter() - t0) * 1000
            results = clip_search(q_emb, clip_index, clip_frame_ids, scene_docs, clip_top_k)
            st.caption(f" CLIP image encode ({len(pil_images)} image(s)) + search — {latency_ms:.0f} ms")
            
            st.markdown(f"**Top {clip_top_k} visually similar frames:**")
            for r in results:
                image_path = Path(*Path(r["image_path"]).parts[1:]) # Drop the leading ".."
                img_path = parent_dir / image_path
                st.subheader(f"Frame {r['frame_id']}  —  score {r['score']:.3f}")
                st.caption(r["summary_text"])
                st.image(img_path)
