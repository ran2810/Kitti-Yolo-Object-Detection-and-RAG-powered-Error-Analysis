# =============================================================================
# KITTI RAG Explorer — LangChain-free pipeline edition
#
# Architecture overview:
#   User query (natural language)
#       ↓
#   _interpret_query_impl()   — Ollama/llama3 converts NL → structured filters
#       ↓                       + auto-detects Scene Search vs Error Analysis
#   _filter_docs_impl()       — exact numeric filter over kitti_docs / error_docs
#       ↓  (fallback if 0 results)
#   _semantic_search_impl()   — FAISS vector search over pre-embedded doc corpus
#       ↓
#   Streamlit UI              — paginated results with bounding-box visualisation
#
# Why no LangChain agent?
#   llama3 (8B via Ollama) cannot reliably follow the ReAct Thought/Action/
#   Observation loop — it writes tool JSON inline and emits "Action: None".
#   We call the three tools directly in a deterministic sequence instead.
# =============================================================================

import streamlit as st
import json, os
import time          # used to measure per-step latency in run_pipeline()
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import re
import cv2

# LangChain @tool decorator is kept so the three impl functions can be
# re-wrapped as agent tools in the future (e.g. with a GPT-4-class model).
from langchain.tools import tool

import warnings
warnings.filterwarnings('ignore')

print_debug = True

# ---------------------------------------------------------
# FUZZY RULES
#
# fuzzy_rules.json maps human terms ("crowded", "rare cyclists") to exact
# numeric filter conditions.  Rules are authoritative: after the LLM produces
# its own filters, any field covered by a matched fuzzy rule is overwritten
# with the rule value.  This prevents the LLM from guessing wrong thresholds
# (e.g. "few" → <=2 when the rule says <=1).
# ---------------------------------------------------------
def load_fuzzy_rules():
    with open("data/fuzzy_rules.json", "r") as f:
        return json.load(f)

FUZZY_RULES = load_fuzzy_rules()

def expand_fuzzy_terms(query: str) -> list:
    """
    Return the list of fuzzy-rule keys that match the query string.
    Matches on the key itself or any of its synonyms (case-insensitive).
    """
    query_lower = query.lower()
    detected = []
    for key, rule in FUZZY_RULES.items():
        if key in query_lower:
            detected.append(key)
            continue
        for syn in rule["synonyms"]:
            if syn in query_lower:
                detected.append(key)
                break
    return detected

# ---------------------------------------------------------
# JSON SANITIZATION
#
# llama3 occasionally emits malformed JSON with spaces inside operator
# strings (e.g. " >= " instead of ">=") or doubled quotes.
# sanitize_json() normalises these before json.loads().
# extract_json_block() pulls the first {...} block from the raw LLM response,
# which may contain prose before/after the JSON object.
# ---------------------------------------------------------
def sanitize_json(text: str) -> str:
    """Fix common LLM JSON formatting errors before parsing."""
    text = re.sub(r'"\s*>=\s*"', '">="', text)
    text = re.sub(r'"\s*>\s*"',  '">"',  text)
    text = re.sub(r'"\s*<=\s*"', '"<="', text)
    text = re.sub(r'"\s*<\s*"',  '"<"',  text)
    text = text.replace('""', '"')
    return text

def extract_json_block(text: str) -> dict | None:
    """Extract and parse the first JSON object found in an LLM response string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    block = sanitize_json(match.group(0))
    try:
        return json.loads(block)
    except Exception:
        return None

# ---------------------------------------------------------
# LLM QUERY INTERPRETER (TOOL)
# accept a plain string "query || mode", keep split logic intact.
# The agent must be prompted to pass "query || mode" as a single string.
# ---------------------------------------------------------
def _interpret_query_impl(query_and_mode: str) -> str:
    """
    Convert natural language query into structured filters + semantic query.
    Input MUST be a single string in the format: "query || mode"
    where mode is either 'Scene Search' or 'Error Analysis'.
    Returns a JSON string with 'filters' and 'semantic_query'.
    """
    # robust split — guard against the LLM omitting the separator
    if "||" not in query_and_mode:
        query = query_and_mode.strip()
        mode = "Scene Search"
    else:
        parts = query_and_mode.split("||", 1)
        query = parts[0].strip()
        mode = parts[1].strip()

    fuzzy_hits = expand_fuzzy_terms(query)

    fuzzy_instructions = ""
    for term in fuzzy_hits:
        fuzzy_instructions += f'"{term}" → {json.dumps(FUZZY_RULES[term]["filters"])}\n'

    # Auto-detect mode from query when the UI mode may be wrong:
    # Error Analysis keywords → switch to Error Analysis regardless of UI setting.
    ERROR_KEYWORDS = {
        "missed", "false positive", "false negative", "fp", "fn",
        "error", "detection error", "iou", "occlusion_level",
        "truncation_value", "missed detection", "wrong detection"
    }
    query_lower = query.lower()
    if any(kw in query_lower for kw in ERROR_KEYWORDS):
        effective_mode = "Error Analysis"
    else:
        effective_mode = mode

    prompt = f"""
You are a query interpreter for a KITTI dataset explorer.

Your ONLY job is to output a JSON object with exactly two keys:
- "filters": a flat dict of field → condition (no nesting under "filters")
- "semantic_query": a short text string for semantic search

IMPORTANT — two separate doc types with DIFFERENT fields:

1. Scene Search  (use when counting objects in a scene):
   Valid fields ONLY: num_cars, num_pedestrians, num_cyclists, max_occlusion, max_truncation

2. Error Analysis  (use for missed detections, false positives, detection errors):
   Valid fields ONLY: error_type ("FP" or "FN"), class ("Car","Pedestrian","Cyclist"),
                      iou, occlusion_level (integer 0-3), truncation_value

Valid operators: ">", ">=", "<", "<=", "=="
NEVER use $gt, $gte, $lt, $lte, $eq — these are forbidden.

Rules for occlusion queries:
- "occlusion 3" or "fully occluded"  → {{"occlusion_level": {{"==": 3}}}}
- "occlusion > 1"                    → {{"occlusion_level": {{">": 1}}}}
- "missed" or "not detected"         → {{"error_type": "FN"}}
- "false positive"                   → {{"error_type": "FP"}}

Fuzzy → Numeric rules:
{fuzzy_instructions}

User query: "{query}"
Mode: "{effective_mode}"

Return ONLY a raw JSON object. No markdown, no explanation, no extra keys.
Example for "missed pedestrians with occlusion 3":
{{"filters": {{"error_type": "FN", "class": "Pedestrian", "occlusion_level": {{"==": 3}}}}, "semantic_query": "missed pedestrians occlusion 3"}}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )

    raw = response.json().get("response", "").strip()
    parsed = extract_json_block(raw)
    result = parsed if parsed else {"filters": {}, "semantic_query": query}

    # LLM sometimes wraps filters under a nested "filters" key
    # e.g. {"filters": {"num_pedestrians": ...}, "semantic_query": ...}
    # The filters dict IS already at result["filters"] — but sometimes the LLM
    # double-nests: result["filters"] = {"filters": {...}}.
    filters = result.get("filters", {})
    if "filters" in filters:
        filters = filters["filters"]
        result["filters"] = filters

    # normalize MongoDB-style operators ($gt, $lt, $gte, $lte, $eq)
    # to the app-native string operators (>, <, >=, <=, ==) that filter_docs expects.
    MONGO_TO_APP = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<=", "$eq": "=="}

    def normalize_ops(cond):
        if not isinstance(cond, dict):
            return cond
        return {MONGO_TO_APP.get(k, k): v for k, v in cond.items()}

    result["filters"] = {
        field: normalize_ops(cond) for field, cond in filters.items()
    }

    # Fuzzy rule filters are authoritative — overlay them on top of whatever
    # the LLM produced. This ensures "rare cyclists" always maps to {"<=": 1}
    # from fuzzy_rules.json, not {"<=": 2} or whatever the LLM guessed.
    # Fuzzy rules only set fields they explicitly define; LLM fills everything else.
    for term in fuzzy_hits:
        rule_filters = FUZZY_RULES[term].get("filters", {})
        for field, cond in rule_filters.items():
            result["filters"][field] = normalize_ops(cond) if isinstance(cond, dict) else cond

    # Embed effective_mode so filter_docs and semantic_search use the right doc set
    result["_mode"] = effective_mode
    return json.dumps(result)

@tool
def interpret_query(query_and_mode: str) -> str:
    """Convert natural language query into structured filters + semantic query. Input: query || mode"""
    return _interpret_query_impl(query_and_mode)

# ---------------------------------------------------------
# LOAD RAG COMPONENTS
# ---------------------------------------------------------
@st.cache_resource
def load_rag():
    with open("data/kitti_docs.json", "r") as f:
        scene_docs = json.load(f)
    scene_index = faiss.read_index("data/kitti_index.faiss")

    with open("data/error_docs.json", "r") as f:
        error_docs = json.load(f)
    error_index = faiss.read_index("data/error_index.faiss")

    with open("data/embedding_model.txt", "r") as f:
        model_name = f.read().strip()

    model = SentenceTransformer(model_name)
    return scene_docs, scene_index, error_docs, error_index, model

scene_docs, scene_index, error_docs, error_index, emb_model = load_rag()

# ---------------------------------------------------------
# FILTER TOOL
# filters arrives as a JSON string from interpret_query.
# Accept a plain string "mode || filters_json" to keep tool signatures
# consistent and avoid @tool dict-coercion issues.
# ---------------------------------------------------------
def _filter_docs_impl(mode_and_filters: str) -> dict:
    """
    Apply numeric filters to scene or error docs.
    Input MUST be a single string: "mode || filters_json"
    where filters_json is a JSON object of field conditions.
    """
    if "||" not in mode_and_filters:
        return {"results": [], "_debug": {}}
    parts = mode_and_filters.split("||", 1)
    mode = parts[0].strip()
    filters_raw = parts[1].strip()

    # filters_raw may arrive as a JSON string or as a dict str() repr
    try:
        parsed = json.loads(filters_raw)
    except json.JSONDecodeError:
        import ast
        try:
            parsed = ast.literal_eval(filters_raw)
        except Exception:
            parsed = {}

    # the agent often forwards interpret_query's full output
    # {"filters": {...}, "semantic_query": "..."} instead of just the inner
    # filters dict. Unwrap it so we iterate actual field names.
    if isinstance(parsed, dict) and "filters" in parsed:
        filters = parsed["filters"]
    else:
        filters = parsed

    docs = scene_docs if mode == "Scene Search" else error_docs

    def coerce(value, reference):
        """Coerce doc field value to the same numeric type as the filter reference.
        Fixes silent failures when JSON docs store numbers as strings e.g. "6".
        """
        if isinstance(reference, (int, float)) and isinstance(value, str):
            try:
                return float(value) if isinstance(reference, float) else int(value)
            except (ValueError, TypeError):
                return value
        return value

    results = []
    for d in docs:
        ok = True
        for key, cond in filters.items():
            if isinstance(cond, list):
                if d.get(key) not in cond:
                    ok = False
                    break
                continue
            if not isinstance(cond, dict):
                dv = coerce(d.get(key), cond)
                if dv != cond:
                    ok = False
                    break
                continue
            for op, val in cond.items():
                dv = coerce(d.get(key), val)
                if dv is None:
                    ok = False
                    break
                try:
                    if op == ">=" and not (dv >= val): ok = False
                    if op == "<=" and not (dv <= val): ok = False
                    if op == ">"  and not (dv >  val): ok = False
                    if op == "<"  and not (dv <  val): ok = False
                    if op == "==" and not (dv == val): ok = False
                except TypeError:
                    ok = False  # incompatible types even after coercion
        if ok:
            results.append(d)

    # Return dict directly — avoids @tool string truncation
    return {
        "results": results[:50],
        "_debug": {
            "total_docs": len(docs),
            "filters_applied": filters,
            "matches_found": len(results),
            "sample_doc_keys": list(docs[0].keys()) if docs else [],
            "sample_field_value": docs[0].get(list(filters.keys())[0]) if docs and filters else None,
            "sample_field_type": type(docs[0].get(list(filters.keys())[0])).__name__ if docs and filters else None,
        }
    }

@tool
def filter_docs(mode_and_filters: str) -> str:
    """Apply numeric filters to docs. Input: mode || filters_json"""
    return json.dumps(_filter_docs_impl(mode_and_filters))

# ---------------------------------------------------------
# SEMANTIC SEARCH TOOL
# Keep the same string-based input convention for consistency.
# ---------------------------------------------------------
def _semantic_search_impl(mode_and_query: str) -> list:
    """
    Perform semantic search over KITTI scene or error docs.
    Input MUST be a single string: "mode || query"
    Returns a JSON string list of the top matching documents.
    """
    if "||" not in mode_and_query:
        return []
    parts = mode_and_query.split("||", 1)
    mode = parts[0].strip()
    query = parts[1].strip()

    docs = scene_docs if mode == "Scene Search" else error_docs
    index = scene_index if mode == "Scene Search" else error_index

    emb = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(emb, 10)
    return [docs[i] for i in I[0] if i < len(docs)]

@tool
def semantic_search_tool(mode_and_query: str) -> str:
    """Semantic search over KITTI docs. Input: mode || query"""
    return json.dumps(_semantic_search_impl(mode_and_query))

# ---------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------
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

def draw_boxes(img, boxes, color, label_prefix):
    VALID_CLASSES = {"Car", "Pedestrian", "Cyclist"}
    for b in boxes:
        cls = b["class"]
        if cls not in VALID_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b["bbox"])
        label = f"{label_prefix} {cls}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def render_side_by_side(frame_id, image_path, frame_errors):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_path = os.path.join("data/training/label_2", f"{frame_id}.txt")
    pred_path = os.path.join("runs/detect/predict/kitti_labels", f"{frame_id}.txt")

    gt_objs = parse_kitti_label_file(gt_path)
    pred_objs = parse_kitti_label_file(pred_path)

    gt_img = img.copy()
    pred_img = img.copy()

    gt_boxes = [{"bbox": o["bbox"], "class": o["type"]} for o in gt_objs]
    pred_boxes = [{"bbox": o["bbox"], "class": o["type"]} for o in pred_objs]

    gt_img = draw_boxes(gt_img, gt_boxes, (0, 255, 0), "GT")
    pred_img = draw_boxes(pred_img, pred_boxes, (255, 255, 0), "Pred")

    for e in frame_errors:
        if "bbox" not in e:
            continue
        if e["error_type"] == "FP":
            pred_img = draw_boxes(pred_img, [e], (255, 0, 0), "FP")
        else:
            gt_img = draw_boxes(gt_img, [e], (0, 128, 255), "FN")

    return np.hstack([gt_img, pred_img])

# ---------------------------------------------------------
# ERROR DOC VISUALIZATION
# ---------------------------------------------------------
def resolve_image_path(raw_path: str):
    """Resolve a potentially relative / backslash path to an absolute path."""
    img_path = os.path.normpath(raw_path)
    for candidate in [
        img_path,
        os.path.join(os.getcwd(), img_path),
        os.path.join(os.path.dirname(__file__), img_path),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def add_panel_label(img, text):
    """Add a dark top-bar label to an image panel (GT / Prediction)."""
    h, w = img.shape[:2]
    bar = np.zeros((28, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    return np.vstack([bar, img])


def render_error_doc(d: dict):
    """
    Produce a side-by-side GT | Prediction image for one error doc.

    Left panel  (GT):
      - All ground-truth boxes from label_2/<id>.txt  in green
      - FN error box highlighted in blue  (missed detection)

    Right panel (Prediction):
      - All predicted boxes from kitti_labels/<id>.txt in yellow
      - FP error box highlighted in red   (false positive)

    Returns (numpy_image, error_message).  One of the two is None.
    """
    frame_id  = str(d.get("id", ""))
    raw_path  = d.get("image_path", "")
    found_path = resolve_image_path(raw_path)

    if not found_path:
        return None, f"Image not found: {os.path.normpath(raw_path)}"

    # render_side_by_side already handles GT + Pred boxes + error overlays
    composite = render_side_by_side(frame_id, found_path, [d])
    if composite is None:
        return None, f"cv2 could not read: {found_path}"

    # Split the hstack back into two panels so we can add labels
    w_half = composite.shape[1] // 2
    gt_panel   = composite[:, :w_half]
    pred_panel = composite[:, w_half:]

    gt_panel   = add_panel_label(gt_panel,   "GT  (green = ground truth  |  blue = FN missed)")
    pred_panel = add_panel_label(pred_panel, "Prediction  (yellow = predicted  |  red = FP wrong)")

    return np.hstack([gt_panel, pred_panel]), None


# ---------------------------------------------------------
# STREAMLIT UI + AGENT
# ---------------------------------------------------------
st.title("KITTI RAG Explorer (LangChain Agent Edition)")

st.sidebar.markdown("### Mode Hint")
st.sidebar.caption(
    "The pipeline auto-detects the correct mode from your query. "
    "Use this only to nudge ambiguous queries."
)
query_mode = st.sidebar.selectbox(
    "Mode Hint (auto-overridden when detected)",
    ["Scene Search", "Error Analysis"]
)
query = st.text_input("Enter your query")

# ---------------------------------------------------------
# DIRECT PIPELINE (replaces broken ReAct agent)
#
# Root cause: llama3 via Ollama was not trained for tool-use in the
# LangChain ReAct loop. It ignores "Observation:" injections and
# writes tool JSON inline into its own text, then gives up with
# "Action: None". No amount of prompt-engineering fixes this reliably.
#
# Solution: call the three tools directly in a fixed sequence.
# This is deterministic, debuggable, and produces correct results.
#
# Pipeline:
#   1. interpret_query  → structured {filters, semantic_query}
#   2. filter_docs      → exact-match results (may be empty)
#   3. semantic_search_tool → semantic results (always returns docs)
#   4. Merge: filter results first; fall back to semantic if empty.
# ---------------------------------------------------------

def run_pipeline(query: str, mode: str) -> dict:
    """
    Run the three-step pipeline and return all results plus per-step latency.

    Steps:
      1. interpret_query  — NL → structured filters (calls Ollama, slowest step)
      2. filter_docs      — exact numeric scan over the correct doc corpus
      3. semantic_search  — FAISS ANN search (only runs when filter finds nothing)

    Latency for each step is stored in the return dict under 'latency_ms'
    and shown in the debug expander in the UI.
    """
    latency_ms = {}

    # ── Step 1: NL → structured filters 
    # _interpret_query_impl calls Ollama (llama3) and returns a JSON string
    # containing {filters, semantic_query, _mode}.  The LLM response is the
    # main latency driver; typical range 1–5 s depending on hardware.
    t0 = time.perf_counter()
    interpreted_str = _interpret_query_impl(f"{query} || {mode}")
    latency_ms["Step 1 — interpret_query (Ollama)"] = round((time.perf_counter() - t0) * 1000)

    try:
        interpreted = json.loads(interpreted_str)
    except Exception:
        interpreted = {"filters": {}, "semantic_query": query}

    filters        = interpreted.get("filters", {})
    semantic_query = interpreted.get("semantic_query", query)
    # effective_mode may differ from the UI hint when auto-detection fires
    effective_mode = interpreted.get("_mode", mode)

    # ── Step 2: exact numeric filter 
    # Scans every doc in the correct corpus (scene_docs or error_docs) and
    # applies the operator conditions from Step 1.  Pure Python loop — fast
    # even for 20k+ docs.  Returns at most 50 results (pagination handles the rest).
    filter_results = []
    filter_debug   = {}
    t0 = time.perf_counter()
    if filters:
        filter_input   = f"{effective_mode} || {json.dumps(filters)}"
        filter_raw     = _filter_docs_impl(filter_input)
        filter_results = filter_raw.get("results", [])
        filter_debug   = filter_raw.get("_debug", {})
    latency_ms["Step 2 — filter_docs (exact scan)"] = round((time.perf_counter() - t0) * 1000)

    # ── Step 3: semantic fallback 
    # Only runs when the exact filter returned 0 results.  Uses a pre-built
    # FAISS index of sentence-transformer embeddings.  Top-10 by cosine similarity.
    t0 = time.perf_counter()
    if filter_results:
        semantic_results = []
        search_mode      = "filter"
    else:
        semantic_results = _semantic_search_impl(f"{effective_mode} || {semantic_query}")
        search_mode      = "semantic"
    latency_ms["Step 3 — semantic_search (FAISS)"] = round((time.perf_counter() - t0) * 1000)

    # ── Merge: filter results are always preferred
    display_docs = filter_results if filter_results else semantic_results

    return {
        "interpreted":    interpreted,
        "effective_mode": effective_mode,
        "filter_results": filter_results,
        "filter_debug":   filter_debug,
        "semantic_results": semantic_results,
        "search_mode":    search_mode,
        "display_docs":   display_docs,
        "latency_ms":     latency_ms,          # per-step timing for the debug panel
    }


# Cache pipeline output in session_state so widget interactions
# (pagination, expanders) don't re-trigger a full pipeline re-run.
cache_key = f"{query}||{query_mode}"
if query and st.session_state.get("_cache_key") != cache_key:
    with st.spinner("Running pipeline…"):
        st.session_state["_pipeline_output"] = run_pipeline(query, query_mode)
        st.session_state["_cache_key"] = cache_key

if query and "_pipeline_output" in st.session_state:
    output = st.session_state["_pipeline_output"]

    interpreted      = output["interpreted"]
    effective_mode   = output.get("effective_mode", query_mode)
    filter_results   = output["filter_results"]
    filter_debug     = output.get("filter_debug", {})
    semantic_results = output["semantic_results"]
    search_mode      = output["search_mode"]
    display_pool     = output["display_docs"]
    latency_ms       = output.get("latency_ms", {})

    # --- Mode badge (always visible, outside expander) ---
    if effective_mode != query_mode:
        st.info(
            f"🔄 Mode auto-detected: **{effective_mode}** "
            f"(your hint was '{query_mode}' — overridden)"
        )
    else:
        st.caption(f"Mode: **{effective_mode}**")

    # --- Debug expander ---
    with st.expander("🔍 Pipeline debug", expanded=False):
        # --- Per-step latency ---
        if latency_ms:
            st.markdown("**⏱ Latency**")
            total_ms = sum(latency_ms.values())
            rows = "".join(
                f"<tr><td>{step}</td><td style='text-align:right'>{ms} ms</td>"
                f"<td style='text-align:right;color:#888'>{round(ms/total_ms*100)}%</td></tr>"
                for step, ms in latency_ms.items()
                if not (step.startswith("Step 3") and search_mode == "filter")
            )
            rows += (
                f"<tr style='font-weight:bold;border-top:1px solid #555'>"
                f"<td>Total</td><td style='text-align:right'>{total_ms} ms</td>"
                f"<td style='text-align:right'></td></tr>"
            )
            st.markdown(
                f"<table style='width:100%;font-size:0.85em'>"
                f"<thead><tr><th>Step</th><th style='text-align:right'>Time</th>"
                f"<th style='text-align:right'>Share</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>",
                unsafe_allow_html=True,
            )

        st.markdown("**Step 1 — interpret_query output**")
        st.json({k: v for k, v in interpreted.items() if k != "_mode"})

        st.markdown(f"**Step 2 — filter_docs** → {len(filter_results)} match(es)")
        if filter_debug:
            total     = filter_debug.get("total_docs", "?")
            matches   = filter_debug.get("matches_found", "?")
            field_val = filter_debug.get("sample_field_value", "?")
            field_type= filter_debug.get("sample_field_type", "?")
            keys      = filter_debug.get("sample_doc_keys", [])
            st.caption(
                f"Scanned {total} docs · {matches} passed · "
                f"Field type in docs: `{field_type}` (sample value: `{field_val}`) · "
                f"Doc keys: `{keys}`"
            )
            if matches == 0:
                st.warning(
                    f"Filter returned 0 matches. Possible causes:\n"
                    f"- Field not in doc (keys: {keys})\n"
                    f"- Type mismatch: filter uses int but doc stores {field_type!r}\n"
                    "- Field name differs from what the LLM used"
                )

        if search_mode == "filter":
            st.markdown("**Step 3 — semantic_search** → skipped (filter had results ✅)")
        else:
            st.markdown(f"**Step 3 — semantic_search** → {len(semantic_results)} result(s) (filter had 0 matches)")

    # --- Results ---
    if not display_pool:
        st.warning("No matching scenes found.")
    else:
        total_matches = filter_debug.get("matches_found", len(display_pool))
        source_label  = "exact filter" if search_mode == "filter" else "semantic search (no filter matches)"
        st.success(f"Found **{total_matches}** scene(s) via **{source_label}**.")

        # --- Sort display_pool by the most relevant filter field ---
        # Determine sort key: pick the first numeric filter field from the
        # interpreted filters. For ">" / ">=" queries sort descending (most first).
        # For "<" / "<=" queries sort ascending (least / worst first).
        # Falls back to no sort for equality / list filters.
        filters_used  = interpreted.get("filters", {})
        sort_key      = None
        sort_reverse  = True   # default: highest value first

        # Error Analysis: always sort by IoU ascending (worst detections first)
        if effective_mode == "Error Analysis":
            sort_key     = "iou"
            sort_reverse = False
        else:
            # Scene Search: when multiple numeric filters exist, pick the one
            # with the highest threshold — that is the most discriminating field
            # and puts the most extreme / relevant docs first.
            # e.g. ">=2 cars AND >3 pedestrians" → sort by num_pedestrians (threshold 3 > 2)
            OP_DIRECTION = {">": True, ">=": True, "<": False, "<=": False}
            best_threshold = -1

            for field, cond in filters_used.items():
                if not isinstance(cond, dict):
                    continue
                for op, val in cond.items():
                    if op not in OP_DIRECTION:
                        continue
                    try:
                        threshold = float(val)
                    except (TypeError, ValueError):
                        continue
                    if threshold > best_threshold:
                        best_threshold = threshold
                        sort_key       = field
                        sort_reverse   = OP_DIRECTION[op]

        if sort_key:
            def _sort_val(doc):
                v = doc.get(sort_key)
                # Push None / missing to the end
                if v is None:
                    return (1, 0)
                try:
                    return (0, float(v))
                except (TypeError, ValueError):
                    return (1, 0)
            display_pool = sorted(display_pool, key=_sort_val, reverse=sort_reverse)
            sort_label = f"sorted by **{sort_key}** {'↓ highest first' if sort_reverse else '↑ lowest first'}"
        else:
            sort_label = "unsorted"

        st.caption(f"Showing {len(display_pool)} result(s) · {sort_label}")

        # Pagination
        PAGE_SIZE = 5
        total_pages = max(1, (len(display_pool) - 1) // PAGE_SIZE + 1)
        page = st.number_input(
            f"Page (1 – {total_pages}, showing {PAGE_SIZE} per page)",
            min_value=1, max_value=total_pages, value=1, step=1
        ) - 1
        page_docs = display_pool[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

        for d in page_docs:
            iou = d.get("iou", "")
            occ = d.get("occlusion_level", "")
            if effective_mode == "Error Analysis":
                meta = " · ".join(filter(None, [
                    d.get("error_type", ""), d.get("class", ""),
                    (f"IoU {iou:.2f}" if isinstance(iou, float) else f"IoU {iou}") if iou != "" else "",
                    f"Occ {occ}" if occ != "" else "",
                ]))
                st.subheader(f"Frame {d.get('id', '?')}  —  {meta}")
            else:
                st.subheader(f"Frame {d.get('id', '?')}")

            if d.get("summary_text"):
                st.write(d["summary_text"])

            if effective_mode == "Error Analysis":
                img_arr, err_msg = render_error_doc(d)
                if img_arr is not None:
                    st.image(img_arr, width="stretch")
                else:
                    st.caption(f"⚠️ {err_msg}")
            else:
                raw_path = d.get("image_path", "")
                img_path = os.path.normpath(raw_path)
                candidates = [
                    img_path,
                    os.path.join(os.getcwd(), img_path),
                    os.path.join(os.path.dirname(__file__), img_path),
                ]
                found_path = next((p for p in candidates if os.path.exists(p)), None)
                if found_path:
                    st.image(found_path)
                else:
                    st.caption(f"⚠️ Image not found. Tried: `{img_path}` (cwd: `{os.getcwd()}`)")
