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

# check groq availability 
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# enable print debug statements
print_debug = True

# get parent dir to join with images path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent


# ------------------------------------------------------------
# AUTOMATIC MODE (SCENE or ERROR) DETECTIOn
# ------------------------------------------------------------
# min cosine similarity gap between modes to trust embedding
CONFIDENCE_THRESHOLD = 0.08

# example queries per mode  --> encoded at startup for cosine similarity classification
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

# keyword fallback when embedding gap is below threshold
ERROR_KEYWORDS = {
    "missed", "false positive", "false negative", "fp", "fn",
    "error", "iou", "wrong detection", "missed detection",
    "undetected", "incorrectly detected", "wrongly detected"
}

def build_mode_embeddings(model):
    """Encode MODE_EXAMPLES once at startup and return per-mode(scene & error) embedding matrix."""
    mode_embeddings = {}
    for mode, examples in MODE_EXAMPLES.items():
        embs = model.encode(examples, convert_to_numpy=True, normalize_embeddings=True)
        mode_embeddings[mode] = embs
    return mode_embeddings

def auto_detect_mode(query, user_mode, mode_embeddings, model):
    """Classify query as Scene Search or Error Analysis via cosine similarity based on threshold"""
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    scores = {mode: float(np.mean(embs @ q_emb)) for mode, embs in mode_embeddings.items()}
    best_mode = max(scores, key=scores.get)
    score_gap = scores[best_mode] - scores[min(scores, key=scores.get)]

    if print_debug:
        print(f"\nAUTO-DETECT scores: {scores}  gap: {score_gap:.4f}")

    if score_gap >= CONFIDENCE_THRESHOLD:
        # embedding gap is large enough to trust the classifier
        effective_mode = best_mode
    else:
        # fall back to keyword check when gap is too small
        query_lower = query.lower()
        effective_mode = "Error Analysis" if any(kw in query_lower for kw in ERROR_KEYWORDS) else user_mode

    was_overridden = (effective_mode != user_mode)
    return effective_mode, was_overridden, scores


# ---------------------------------------------------------
# LLM CONFIG -> Ollama or Groq
# ---------------------------------------------------------
def call_llm(prompt, llm_config):
    """Route prompt to Groq cloud or Ollama local and return raw response string."""
    provider = llm_config.get("provider", "Ollama")
    model    = llm_config.get("model", "llama3")

    if provider == "Groq":
        # call Groq cloud API
        api_key = llm_config.get("api_key", "")
        if not api_key:
            # key is required
            raise ValueError("Groq API key is required.")
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # added temperature to get deterministic o/p with query expansion in CLIP
        )
        return response.choices[0].message.content.strip()

    # Ollama local inference with temperature=0 for deterministic output
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    )
    return response.json().get("response", "").strip()


# ------------------------------------------------------------
# FUZZY WORDS Handling for querying
# ------------------------------------------------------------
def load_fuzzy_rules():
    """Load fuzzy synonym rules from JSON."""
    with open("../data/fuzzy_rules.json", "r") as f:
        return json.load(f)

FUZZY_RULES = load_fuzzy_rules()


def expand_fuzzy_terms(query):
    """Return list of canonical fuzzy terms matched in the query via synonyms."""
    query_lower = query.lower()
    detected = []
    for key, rule in FUZZY_RULES.items():
        if key in query_lower or any(syn in query_lower for syn in rule["synonyms"]):
            detected.append(key)
    return detected


# words that carry no filter intent and should not block the fuzzy-only fast path
STOPWORDS = {
    # conjunctions / articles / prepositions
    "and", "or", "with", "the", "a", "an", "some", "of", "in", "for",
    "very", "quite", "mostly", "mainly", "also", "both",
    # scene-context nouns that describe setting but map to no numeric field
    "intersection", "scene", "frame", "image", "road", "street",
    "area", "zone", "environment", "conditions", "scenario",
    # common adjectives that describe scene mood, not measurable counts
    "active", "urban", "typical", "normal",
}

def is_fuzzy_only(query, fuzzy_hits):
    """Return True if fuzzy rules cover all meaningful (non-stopword) tokens in the query."""
    
    if not fuzzy_hits:
        return False
    residual = query.lower()
    # strip every matched fuzzy term and its synonyms from the query
    for term in fuzzy_hits:
        # strip matched fuzzy term
        residual = residual.replace(term, " ")
        for syn in FUZZY_RULES[term]["synonyms"]:
            residual = residual.replace(syn, " ")
    # check if any non-stopword tokens remain
    remaining = {w for w in residual.split() if w not in STOPWORDS}

    # if remaining is empty -> true
    return len(remaining) == 0


# ------------------------------------------------------------
# JSON FORMATING
# ------------------------------------------------------------
def format_json(text):
    """Fix malformed operator strings like '">=' or '"  >=' that LLMs sometimes emit."""
    text = re.sub(r'"\s*>=\s*"', '">="', text)
    text = re.sub(r'"\s*>\s*"', '">"', text)
    text = re.sub(r'"\s*<=\s*"', '"<="', text)
    text = re.sub(r'"\s*<\s*"', '"<"', text)
    text = text.replace('""', '"')
    return text

def extract_json_block(text):
    """Extract and parse the first JSON object from LLM output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    block = format_json(match.group(0))
    try:
        return json.loads(block)
    except:
        return None


# regex for fused operator strings like ">0" or "<= 1.5"
_OP_STRING_RE = re.compile(r'^(>=|<=|>|<|==)\s*(-?\d+(?:\.\d+)?)$')

# map MongoDB-style operators to standard ones
_MONGO_OPS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<=", "$eq": "=="}

def _coerce_numeric(val):
    """Convert string to int or float, returning original if not numeric."""
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return val

def normalize_filters(filters):
    """Normalize LLM filter output to consistent operator dicts, scalars, or lists."""
    # these fields treat scalar 0 as "> 0" (LLM shorthand)
    RANGE_FIELDS = {
        "num_cars", "num_pedestrians", "num_cyclists",
        "max_occlusion", "max_truncation",
        "occlusion_level", "truncation_value", "iou"
    }

    normalised = {}
    for key, cond in filters.items():
        if isinstance(cond, list):
            # membership list: pass through unchanged
            normalised[key] = cond
        elif isinstance(cond, str):
            # fused operator string like ">=2": parse into {">=": 2}
            m = _OP_STRING_RE.match(cond.strip())
            normalised[key] = {m.group(1): _coerce_numeric(m.group(2))} if m else cond
        elif isinstance(cond, (int, float)):
            # bare 0 on a range field means "> 0" in context
            normalised[key] = {">": 0} if (key in RANGE_FIELDS and cond == 0) else cond
        elif isinstance(cond, dict):
            # translate $mongo operators and coerce string values to numbers
            normalised[key] = {
                _MONGO_OPS.get(op, op): (_coerce_numeric(val) if isinstance(val, str) else val)
                for op, val in cond.items()
            }
        else:
            # unknown type: pass through and let apply_filters handle it
            normalised[key] = cond
    return normalised

# ---------------------------------------------------------
# LLM INTERPRETER (Scene + Error Mode) 
# ---------------------------------------------------------
def interpret_query_with_llm(query, mode, llm_config):
    """Send query to LLM and return (parsed filters + semantic query, latency_ms)."""
    
    # if query consist of fuzzy words
    fuzzy_hits = expand_fuzzy_terms(query)

    if is_fuzzy_only(query, fuzzy_hits):
        # all meaningful tokens covered by fuzzy rules — skip LLM entirely
        merged_filters = {}
        for term in fuzzy_hits:
            merged_filters.update(FUZZY_RULES[term].get("filters", {}))
        # return by skipping llm query
        return {
            "filters":        merged_filters,
            "semantic_query": query,
            "_source":        "fuzzy_only",
        }, None

    fuzzy_instructions = "".join(
        f'"{t}" -> {json.dumps(FUZZY_RULES[t]["filters"])}\n' for t in fuzzy_hits
    )

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
            # LLM returned unparseable output; treat as empty filters
            raise ValueError("No JSON found")

        # normalize operator formats to handle LLM output drift
        if "filters" in parsed and isinstance(parsed["filters"], dict):
            parsed["filters"] = normalize_filters(parsed["filters"])

        if print_debug:
            print("\nPARSED JSON:\n", parsed)

        return parsed, llm_latency_ms

    except Exception as e:
        print("LLM error:", e)
        return {"filters": {}, "semantic_query": query}, None


# ---------------------------------------------------------
# Load TEXT RAG COMPONENTS: Load files relevant for text based querying
# ---------------------------------------------------------
@st.cache_resource
def load_rag_text():
    """Load scene docs, error docs, and sentence-transformer model. Cached once."""
    with open("../data/kitti_docs.json", "r") as f:
        scene_docs = json.load(f)
    scene_index = faiss.read_index("../data/kitti_index.faiss")

    with open("../data/error_docs.json", "r") as f:
        error_docs = json.load(f)
    error_index = faiss.read_index("../data/error_index.faiss")

    with open("../data/embedding_model.txt", "r") as f:
        model_name = f.read().strip()
    model = SentenceTransformer(model_name)

    return scene_docs, scene_index, error_docs, error_index, model

# load text indexes and build mode embeddings once at startup
scene_docs, scene_index, error_docs, error_index, emb_model = load_rag_text()
mode_embeddings = build_mode_embeddings(emb_model)

# ---------------------------------------------------------
# FILTER ENGINE
# ---------------------------------------------------------
def apply_filters(docs, filters):
    """Filter docs by operator dicts, scalars, or membership lists."""
    results = []
    for d in docs:
        ok = True
        for key, cond in filters.items():
            if isinstance(cond, list):
                # membership check: field value must be in the allowed list
                if d.get(key) not in cond:
                    ok = False; break
                continue
            if not isinstance(cond, dict):
                # exact match for scalar condition
                if d.get(key) != cond:
                    ok = False; break
                continue
            for op, val in cond.items():
                dv = d.get(key)
                if dv is None:
                    # missing field always fails the filter
                    ok = False; break
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
# FALLBACK SEMANTIC SEARCH
# ---------------------------------------------------------
def semantic_search(query, docs, index, embed_model, top_k=10):
    """Encode query and return top_k docs by L2 distance in FAISS index."""
    emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(emb, top_k)
    return [docs[i] for i in I[0]]


# ---------------------------------------------------------
# Load CLIP RAG COMPONENTS: files relevant for CLIP based querying
# ---------------------------------------------------------
# CLIP model configs: label shown in sidebar -> index filenames and model id
CLIP_MODEL_CONFIGS = {
    "ViT-B-32  (512-dim, fast)": {
        "model_id":   "clip-ViT-B-32",
        "index_path": "../data/clip_index_B32.faiss",
        "ids_path":   "../data/clip_frame_ids_B32.json",
    },
    "ViT-L-14  (768-dim, better)": {
        "model_id":          "clip-ViT-L-14",
        "index_path":        "../data/clip_index_14.faiss",
        "ids_path":          "../data/clip_frame_ids_14.json", 
    },
}

@st.cache_resource
def load_rag_clip(clip_model_label: str):
    """Load CLIP index and model for given label. Cached per model so switching is instant at runtim."""
    cfg = CLIP_MODEL_CONFIGS[clip_model_label]
    index_path = cfg["index_path"]
    ids_path   = cfg["ids_path"]

    if not os.path.exists(index_path):
        # model-specific not found
        return None, [], None

    clip_index     = faiss.read_index(index_path)
    with open(ids_path, "r") as f:
        clip_frame_ids = json.load(f)
    clip_model = SentenceTransformer(cfg["model_id"])

    print(f"CLIP loaded: {cfg['model_id']}  dim={clip_index.d}  frames={len(clip_frame_ids)}")
    return clip_index, clip_frame_ids, clip_model


# ---------------------------------------------------------
# CLIP VISUAL SEARCH
# ---------------------------------------------------------
# get image query embedding
def get_embedding_images_query(pil_images, model):
    """Encode list of PIL images with CLIP and return mean L2-normalised embedding."""
    embs = []
    for img in pil_images:
        e = model.encode(img, convert_to_numpy=True).astype("float32")
        norm = np.linalg.norm(e)
        if norm > 0:
            e /= norm
        embs.append(e)
    mean_emb = np.mean(embs, axis=0).astype("float32")
    mean_emb /= np.linalg.norm(mean_emb)
    return mean_emb

# get text query embedding
def get_clip_encode_text(text, model):
    """Encode text string with CLIP and return L2-normalised embedding."""
    e = model.encode(text, convert_to_numpy=True).astype("float32")
    e /= np.linalg.norm(e)
    return e

# search based query embedding(text/image) over CLIP docs
def clip_search(query_emb, clip_index, clip_frame_ids, scene_docs, top_k=10):
    """Search CLIP image index and return top_k results with score and image path."""
    q = query_emb.reshape(1, -1)
    scores, indices = clip_index.search(q, top_k)
    id_to_doc = {d["id"]: d for d in scene_docs}
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            # FAISS returns -1 for empty slots when index has fewer vectors than top_k
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
# VISUALIZATION HELPERS for text based querying
# ------------------------------------------------------------
def draw_boxes(img, boxes, color, label_prefix):
    """Draw bounding boxes for Car, Pedestrian, Cyclist classes only. -> image with bbox"""
    VALID_CLASSES = {"Car", "Pedestrian", "Cyclist"}
    for b in boxes:
        cls = b["class"]
        if cls not in VALID_CLASSES:
            # skip DontCare, Misc, and other KITTI labels not relevant for display
            continue
        x1, y1, x2, y2 = map(int, b["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # FP/FN label text
        cv2.putText(img, f"{label_prefix} {cls}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def parse_kitti_label_file(path):
    """Parse a KITTI label .txt file -> list of object dicts."""
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

def render_side_by_side(frame_id, image_path, frame_errors):
    """Render image with boxes GT (green) and prediction (yellow) side by side with FP/FN highlights."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_path   = os.path.join("../data/training/label_2", f"{frame_id}.txt")
    pred_path = os.path.join("../runs/detect/predict/kitti_labels", f"{frame_id}.txt")

    gt_objs   = parse_kitti_label_file(gt_path)
    pred_objs = parse_kitti_label_file(pred_path)

    # copy image and draw boxes
    gt_img   = img.copy()
    pred_img = img.copy()

    # draw GT boxes
    gt_boxes = [{"bbox": o["bbox"], "class": o["type"],
                 "occlusion_level": o["occluded"], "truncation_value": o["truncated"]}
                for o in gt_objs]
    gt_img = draw_boxes(gt_img, gt_boxes, (0, 255, 0), "GT")

    # draw Pred boxes
    pred_boxes = [{"bbox": o["bbox"], "class": o["type"]} for o in pred_objs]
    pred_img = draw_boxes(pred_img, pred_boxes, (255, 255, 0), "Pred")

    # highlight error frames
    for e in frame_errors:
        if "bbox" not in e:
            continue
        # FP in red, FN in blue
        if e["error_type"] == "FP":
            pred_img = draw_boxes(pred_img, [e], (255, 0, 0), "FP")
        else:
            gt_img = draw_boxes(gt_img, [e], (0, 128, 255), "FN")
    # combined image side by side
    return np.hstack([gt_img, pred_img])


# ---------------------------------------------------------
# MAIN STREAMLIT UI
# ---------------------------------------------------------
st.title("KITTI RAG Explorer (Scene Search + Error Analysis)")
st.write("Ask natural language questions about KITTI scenes, objects, occlusion, truncation, or counts.")
st.write("Color code: FP = red, FN = blue")

# sidebar LLM provider config
st.sidebar.markdown("## LLM Provider")
llm_provider = st.sidebar.radio("Backend", ["Ollama (local)", "Groq (cloud)"])

if llm_provider == "Ollama (local)":
    # configure local Ollama with model selection
    ollama_model = st.sidebar.selectbox("Ollama Model", ["llama3", "llama3.1", "mistral"])
    st.sidebar.caption(f"Running locally via Ollama model: {ollama_model}")
    llm_config = {"provider": "Ollama", "model": ollama_model, "api_key": ""}
else:
    # configure Groq cloud selection
    if not GROQ_AVAILABLE:
        # groq package missing
        st.sidebar.error("groq package not installed")
    groq_model = st.sidebar.selectbox(
        "Groq Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    )
    groq_key = st.sidebar.text_input("Groq API Key", type="password") # api key
    st.sidebar.caption(f"Groq cloud model: {groq_model}")
    llm_config = {"provider": "Groq", "model": groq_model, "api_key": groq_key}

st.sidebar.divider()

# ---------------------------------------------------------
# sidebar CLIP model selector -> queries (txt -> image or image to image)
# ---------------------------------------------------------
# (cached per model so switching is instant after first load)
st.sidebar.markdown("### Visual Search Model")
clip_model_label = st.sidebar.radio(
    "CLIP model",
    list(CLIP_MODEL_CONFIGS.keys()),
    help="ViT-B-32 is faster. ViT-L-14 has finer-grained embeddings (needs separate index build)."
)
clip_index, clip_frame_ids, clip_model = load_rag_clip(clip_model_label)

if clip_index is not None:
    # show model and index stats in sidebar
    st.sidebar.caption(
        f"Loaded: {CLIP_MODEL_CONFIGS[clip_model_label]['model_id']} "
        f"| dim={clip_index.d} | {len(clip_frame_ids):,} frames"
    )
else:
    # index missing: guide user to build it
    st.sidebar.warning(
        "Index not found. Build it with:\n"
        f"`python generate_faiss_doc.py --clip-model "
        f"{CLIP_MODEL_CONFIGS[clip_model_label]['model_id'].replace('clip-', '')} --skip-text`"
    )

st.sidebar.divider()

# ---------------------------------------------------------
# Sidebar mode hint (auto-detection can override this)
# ---------------------------------------------------------
st.sidebar.markdown("### Mode Hint")
st.sidebar.caption("Auto-detection overrides this if error keywords (FP, FN, IoU) are found.")
query_mode = st.sidebar.selectbox("Mode Hint", ["Scene Search", "Error Analysis"])


# ---------------------------------------------------------
# Text Based Query (SCENE or ERROR mode)
# ---------------------------------------------------------
# main query input
query = st.text_input("Enter your query")
top_k = st.slider("Number of results", 1, 10, 5)

# check if pre-selected mode is correct based on query
if query:
    effective_mode, was_overridden, scores = auto_detect_mode(
        query, query_mode, mode_embeddings, emb_model
    )

    if was_overridden:
        # auto-detect changed the mode -> show which hint was overridden
        st.info(
            f"Auto-detected mode: **{effective_mode}** "
            f"(overrode sidebar hint: '{query_mode}')  "
            f"scores: Error Analysis {scores['Error Analysis']:.3f} "
            f"| Scene Search {scores['Scene Search']:.3f}"
        )
    else:
        # mode matches the sidebar hint -> just confirm it
        st.info(
            f"Mode: **{effective_mode}**  "
            f"scores: Error Analysis {scores['Error Analysis']:.3f} "
            f"| Scene Search {scores['Scene Search']:.3f}"
        )

    # convert query -> json output format via LLM
    parsed, llm_latency_ms = interpret_query_with_llm(query, effective_mode, llm_config)
    query_filters        = parsed.get("filters", {})
    semantic_query = parsed.get("semantic_query", query)

    st.markdown("### LLM Interpretation")
    if parsed.get("_source") == "fuzzy_only":
        # all terms matched fuzzy rules — LLM was not called
        st.caption("fuzzy rules only — LLM skipped (no latency)")
    elif llm_latency_ms is not None:
        # show latency badge when LLM was called successfully
        provider_label = llm_config.get("provider", "LLM")
        model_label    = llm_config.get("model", "")
        color = "green" if llm_latency_ms < 1000 else ("orange" if llm_latency_ms < 3000 else "red")
        st.caption(f"**{provider_label}** ({model_label}) :{color}[{llm_latency_ms:.0f} ms]")
    st.json({k: v for k, v in parsed.items() if k != "_source"})

    # pick relevant doc and index file
    docs  = scene_docs  if effective_mode == "Scene Search" else error_docs
    index = scene_index if effective_mode == "Scene Search" else error_index

    # search in docs based on filters
    filtered_docs = apply_filters(docs, query_filters)

    if len(filtered_docs) > 0:
        # filter matched -> display results ranked by the primary filter field
        st.success(f"Found {len(filtered_docs)} frames matching filters")

        # sort by the first range filter field found
        sort_field = None
        for field, cond in query_filters.items():
            if isinstance(cond, list) or not isinstance(cond, dict):
                sort_field = field; 
                break
            for op in cond.keys():
                if op in [">", ">=", "<", "<="]:
                    sort_field = field; 
                    break
            if sort_field:
                break
        # Sort to display results as top. ex: query as "more than 5 pedestrians" then display first result as frame with max pedestrians           
        if sort_field:
            # sort descending so highest-value frames appear first
            filtered_docs = sorted(filtered_docs, key=lambda d: d.get(sort_field, 0), reverse=True)
        # display k results 
        for d in filtered_docs[:top_k]:
            st.subheader(f"Frame {d['id']}")
            st.write(d["summary_text"])
            img_path = parent_dir / Path(*Path(d["image_path"]).parts[1:])
            if effective_mode == "Error Analysis":
                # show error metadata and GT vs prediction side-by-side overlay
                st.write(f"**Error Type:** {d['error_type']}  **Class:** {d['class']}  **IoU:** {d['iou']:.3f}")
                frame_errors = [e for e in error_docs if e["id"] == d["id"]]
                # get rendered image with GT and Predictions side by side
                combined_img = render_side_by_side(d["id"], img_path, frame_errors)
                st.image(combined_img if combined_img is not None else img_path,
                         caption="GT (left) vs Predictions (right)")
            else:
                # scene search -> just show the raw image
                st.image(img_path)
        st.stop()

    if query_filters:
        # filters present but no docs matched -> notify and fall back to embedding search
        st.warning("No matches for query_filters. Falling back to semantic search.")

    results = semantic_search(semantic_query, docs, index, emb_model, top_k)
    st.markdown("### Semantic Search Results")

    for d in results:
        st.subheader(f"Frame {d['id']}")
        st.write(d["summary_text"])
        img_path = parent_dir / Path(*Path(d["image_path"]).parts[1:])
        if effective_mode == "Error Analysis":
            # overlay FP/FN annotations on GT vs prediction comparison
            st.write(f"**Error Type:** {d['error_type']}  **Class:** {d['class']}  **IoU:** {d['iou']:.3f}")
            frame_errors = [e for e in error_docs if e["id"] == d["id"]]
            combined_img = render_side_by_side(d["id"], img_path, frame_errors)
            st.image(combined_img if combined_img is not None else img_path,
                     caption="GT (left) vs Predictions (right)")
        # scene mode -> just show the raw image
        st.image(img_path)

# ---------------------------------------------------------
# CLIP VISUAL SEARCH (modes: text -> image or image -> image)
# ---------------------------------------------------------
st.divider()
st.markdown("## Visual Search (CLIP)")

if clip_index is None:
    # CLIP index missing: prompt user to run the builder script
    st.warning("CLIP index not found. Run `python queries/generate_faiss_doc.py` to build it.")
else:
    # CLIP index ready: show visual search UI
    clip_search_mode = st.radio("Query type", ["Text2Images", "Image2Images"], horizontal=True)
    clip_top_k = st.slider("Visual results", 1, 10, 5, key="clip_k")

    # text -> image 
    if clip_search_mode == "Text2Images":
        # text query: optionally expand with Groq, then encode with CLIP
        clip_text_query = st.text_input(
            "Describe what you want to find visually",
            placeholder="e.g. pedestrian crossing sign, construction worker, road with tram tracks",
            key="clip_text"
        )

        # query expansion: Provided text query is passed to llm to improvise the query before passing to CLIP encoding
        if clip_text_query:
            # run search when user has entered a query
            groq_available = (
                llm_config.get("provider") == "Groq"
                and llm_config.get("api_key", "").strip()
            )

            if groq_available:
                # expand short query to rich visual description before CLIP encoding
                expand_prompt = f"""You are helping improve image search using CLIP embeddings.
Expand the following short query into a detailed visual description (2-3 sentences).
Focus on: colors, shapes, textures, positions, visual appearance, surrounding context.
Do NOT add information that isn't implied by the query.
Do NOT include any explanation — output only the expanded description.

Query: "{clip_text_query}"
"""
                try:
                    # query expansion and measure latency
                    t_expand = time.perf_counter()
                    expanded_query = call_llm(expand_prompt, llm_config)
                    expand_ms = (time.perf_counter() - t_expand) * 1000
                    st.caption(f"Expanded query ({expand_ms:.0f} ms): *{expanded_query}*")
                except Exception as e:
                    # expansion failed: fall back to raw query silently
                    expanded_query = clip_text_query
                    st.caption(f"Query expansion failed ({e}), using original.")
            else:
                # Groq not active: skip expansion and use raw query
                expanded_query = clip_text_query
                st.info("Select Groq in the sidebar to enable query expansion for better results.")

            # run query, measure latency and display top k results
            t0 = time.perf_counter()
            q_emb      = get_clip_encode_text(expanded_query, clip_model)
            latency_ms = (time.perf_counter() - t0) * 1000
            results    = clip_search(q_emb, clip_index, clip_frame_ids, scene_docs, clip_top_k)
            # latency display
            color = "green" if latency_ms < 500 else ("orange" if latency_ms < 2000 else "red")
            st.caption(f"CLIP encode + search :{color}[{latency_ms:.0f} ms]")
            # display results
            st.markdown(f"**Top {clip_top_k} visually similar frames:**")
            for r in results:
                img_path = parent_dir / Path(*Path(r["image_path"]).parts[1:])
                st.subheader(f"Frame {r['frame_id']}  score {r['score']:.3f}")
                st.caption(r["summary_text"])
                st.image(img_path)

    # Image -> image query (uploaded image/s is encoded and search by visual similarity)
    else:
        st.caption("Upload 1-5 example images. Multiple images make the query more robust.")
        uploaded_files = st.file_uploader(
            "Upload query image(s)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="clip_upload"
        )
        if uploaded_files:
            # encode uploaded images and run similarity search
            pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]

            # show uploaded image thumbnails
            cols = st.columns(min(len(pil_images), 5))
            for col, img in zip(cols, pil_images):
                col.image(img, width='stretch') # use_container_width=True

            # run query, measure latency and display top k results
            t0 = time.perf_counter()
            q_emb      = get_embedding_images_query(pil_images, clip_model)
            latency_ms = (time.perf_counter() - t0) * 1000
            results    = clip_search(q_emb, clip_index, clip_frame_ids, scene_docs, clip_top_k)
            # latency display
            color = "green" if latency_ms < 500 else ("orange" if latency_ms < 2000 else "red")
            st.caption(f"CLIP image encode ({len(pil_images)} image(s)) + search: {color}[{latency_ms:.0f} ms]") 
            # display results 
            st.markdown(f"**Top {clip_top_k} visually similar frames:**")
            for r in results:
                img_path = parent_dir / Path(*Path(r["image_path"]).parts[1:])
                st.subheader(f"Frame {r['frame_id']}  score {r['score']:.3f}")
                st.caption(r["summary_text"])
                st.image(img_path)
