# =============================================================================
# KITTI RAG Explorer
# LangChain LCEL Pipeline with switchable LLM Backend (Ollama local / Groq cloud)
#
#  What this app does 
# Natural-language search over a KITTI autonomous-driving dataset.
# Supports two query modes:
#   • Scene Search   — find frames by object counts / occlusion levels
#   • Error Analysis — find detection errors (FP/FN) by class, IoU, occlusion
#
#  LangChain LCEL pipeline 
# Each processing step is a RunnableLambda composed with the LCEL | operator
# into a single inspectable chain.  LangChain orchestrates the data flow;
# the LLM is used only for natural-language -> JSON filter interpretation.
#
#  Model selection 
# Use sidebar for switching between two backends at runtime:
#
#   Ollama (local) — private, no API key, ~2–5 s/query
#     Models: llama3
#
#   Groq (cloud)  — free API key at console.groq.com, ~0.3–0.8 s/query
#     Models: llama-3.3-70b-versatile (default), llama-3.1-8b-instant
#
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

#  LangChain imports 
from langchain.tools import tool
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False   # install with: pip install langchain-groq

import warnings
warnings.filterwarnings('ignore')

print_debug = True

###########################################################
# FUZZY RULES
#
# fuzzy_rules.json maps human terms ("crowded", "rare cyclists") to exact
# numeric filter conditions.  Rules are authoritative: after the LLM produces
# its own filters, any field covered by a matched fuzzy rule is overwritten
# with the rule value.  This prevents the LLM from guessing wrong thresholds
# (e.g. "few" -> <=2 when the rule says <=1).
###########################################################
def load_fuzzy_rules():
    with open("../data/fuzzy_rules.json", "r") as f:
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

###########################################################
# JSON SANITIZATION
#
# llama3 occasionally emits malformed JSON with spaces inside operator
# strings (e.g. " >= " instead of ">=") or doubled quotes.
# sanitize_json() normalises these before json.loads().
# extract_json_block() pulls the first {...} block from the raw LLM response,
# which may contain prose before/after the JSON object.
###########################################################
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


###########################################################
# LCEL CHAIN — LLM QUERY INTERPRETER
#
# _interpret_query_impl() is the only place in the pipeline where the LLM
# is called.  It converts a natural-language query into a structured JSON
# object with two keys:
#   "filters"       — field/operator/value conditions for exact numeric search
#   "semantic_query"— reworded query text for FAISS vector search fallback
#
# @tool wrapper (interpret_query) keeps this function agent-compatible:
#   swap in a function-calling model (llama-3.3-70b-versatile, gpt-4o) and
#   re-enable AgentExecutor to restore full ReAct tool orchestration.
###########################################################
def _interpret_query_impl(query_and_mode: str, llm=None) -> str:
    """
    Convert natural language query into structured filters + semantic query.
    Input MUST be a single string in the format: "query || mode"
    where mode is either 'Scene Search' or 'Error Analysis'.

    llm: a LangChain ChatModel instance (ChatOllama or ChatGroq).
         When provided, used directly instead of raw requests.post.
         When None, falls back to raw Ollama HTTP (legacy / testing).

    Optimisations:
      1. Fuzzy-only queries skip the LLM entirely — filters come straight
         from fuzzy_rules.json, so no inference is needed.
      2. When LLM is needed, the prompt is kept minimal (~50 tokens of
         instructions) to minimise time-to-first-token.
      3. The correct LangChain ChatModel is used (ChatOllama or ChatGroq),
         so provider switching in the sidebar actually takes effect.
    Returns a JSON string with 'filters', 'semantic_query', '_mode'.
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
        fuzzy_instructions += f'"{term}" -> {json.dumps(FUZZY_RULES[term]["filters"])}\n'

    # Fuzzy-only fast path 
    # If ALL filter fields are covered by fuzzy rules AND there are no
    # non-fuzzy terms left that need LLM interpretation, skip the LLM entirely.
    # Example: "few cyclists and rare pedestrians" -> both terms are in
    # fuzzy_rules.json, so we know every filter value already.
    # We still need to detect mode and build semantic_query, but we can do
    # that with a tiny keyword check — no inference required.
    ERROR_KEYWORDS = {
        "missed", "false positive", "false negative", "fp", "fn",
        "error", "detection error", "iou", "occlusion_level",
        "truncation_value", "missed detection", "wrong detection"
    }
    query_lower_check = query.lower()
    auto_mode = (
        "Error Analysis"
        if any(kw in query_lower_check for kw in ERROR_KEYWORDS)
        else mode
    )

    if fuzzy_hits:
        # Check if every meaningful word in the query is covered by fuzzy rules
        # by seeing whether removing all fuzzy synonyms leaves only stopwords.
        STOPWORDS = {
            # conjunctions / articles / prepositions
            "and", "or", "with", "the", "a", "an", "some", "of",
            "in", "for", "very", "quite", "mostly", "mainly",
            # common scene-context words that don't affect filter logic
            "intersection", "scene", "frame", "image", "road", "street",
            "area", "zone", "environment", "conditions", "scenario",
            "busy", "active",
        }
        residual = query_lower_check
        for term in fuzzy_hits:
            residual = residual.replace(term, " ")
            for syn in FUZZY_RULES[term]["synonyms"]:
                residual = residual.replace(syn, " ")
        residual_words = {w for w in residual.split() if w not in STOPWORDS}

        if not residual_words:
            # All terms matched — build result purely from fuzzy rules, no LLM
            merged_filters: dict = {}
            for term in fuzzy_hits:
                merged_filters.update(FUZZY_RULES[term].get("filters", {}))
            result = {
                "filters":        merged_filters,
                "semantic_query": query,
                "_mode":          auto_mode,
                "_source":        "fuzzy_only",   # visible in debug
            }
            return json.dumps(result)

    # Auto-detect mode from query when the UI mode may be wrong:
    # Error Analysis keywords -> switch to Error Analysis regardless of UI setting.
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

    # Build prompt messages
    # Pre-render all dynamic values into plain strings BEFORE creating message
    # objects — avoids ChatPromptTemplate's f-string parser rejecting nested
    # expressions like {fuzzy_instructions} or {query!r}.
    # Full detailed instructions are kept (not trimmed) so the LLM produces
    # correct field names, operators, and mode-specific filters reliably.
    from langchain_core.messages import SystemMessage, HumanMessage

    fuzzy_block = (
        f"Fuzzy rules (authoritative — use EXACT values shown):\n{fuzzy_instructions}"
        if fuzzy_instructions else "No fuzzy rules apply."
    )

    system_text = "\n".join([
        "You are a query interpreter for a KITTI autonomous-driving dataset.",
        "Output ONLY a JSON object with exactly two keys:",
        '  "filters": flat dict of field/operator/value (never nest under a "filters" key)',
        '  "semantic_query": short rephrased text for vector search',
        "",
        f"Mode: {effective_mode}",
        "",
        "SCENE SEARCH valid fields (use only for object-count queries):",
        "  num_cars, num_pedestrians, num_cyclists, max_occlusion, max_truncation",
        "",
        "ERROR ANALYSIS valid fields (use for missed detections / false positives):",
        '  error_type ("FP" or "FN"), class ("Car" | "Pedestrian" | "Cyclist"),',
        "  iou, occlusion_level (integer 0-3), truncation_value",
        "",
        'Operators: ">" ">=" "<" "<=" "=="  — NEVER use $gt $gte $lt $lte $eq.',
        "",
        "Key mappings:",
        '  missed / not detected          -> error_type "FN"',
        '  false positive                 -> error_type "FP"',
        '  occlusion 3 / fully occluded  -> {"occlusion_level": {"==": 3}}',
        '  truncation > 0.5              -> {"truncation_value": {">": 0.5}}',
        '  IoU < 0.4                     -> {"iou": {"<": 0.4}}',
        "",
        fuzzy_block,
    ])
    human_text = "Query: " + repr(query)

    messages = [SystemMessage(content=system_text),
                HumanMessage(content=human_text)]

    if llm is not None:
        # Invoke the ChatModel directly with pre-built message objects.
        # No template parsing — no nested f-string conflicts.
        response_msg = llm.invoke(messages)
        raw = response_msg.content.strip()
    else:
        # Legacy fallback: raw HTTP to local Ollama (used in tests / no llm arg)
        prompt_text = f"{system_text}\n{human_text}"
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt_text, "stream": False}
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
    """
    LangChain @tool wrapper around _interpret_query_impl.
    Exposes the interpreter as a reusable agent tool.
    When a function-calling model is used, AgentExecutor can call this tool
    autonomously as part of a ReAct loop.  For the current LCEL pipeline,
    interpret_step() calls _interpret_query_impl() directly with the llm arg.
    Input: "query || mode"  e.g. "more than 5 pedestrians || Scene Search"
    """
    return _interpret_query_impl(query_and_mode)

###########################################################
# RAG CORPUS LOADING  (@st.cache_resource)
#
# Loads the two documents and their pre-built FAISS indices once at
# startup.  @st.cache_resource ensures they stay in memory across all
# Streamlit re-runs (widget interactions never reload from disk).
#
#   scene_docs  / kitti_index.faiss  — 7 481 scene-level documents
#                                      fields: num_cars, num_pedestrians, etc.
#   error_docs  / error_index.faiss  — 22 417 per-object error documents
#                                      fields: error_type, class, iou, etc.
#
# The sentence-transformer embedding model is also loaded here so
# _semantic_search_impl() can encode queries at search time.
###########################################################
@st.cache_resource
def load_rag():
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

scene_docs, scene_index, error_docs, error_index, emb_model = load_rag()

###########################################################
# LCEL CHAIN — EXACT NUMERIC FILTER
#
# _filter_docs_impl() scans the correct documents (scene_docs or
# error_docs, selected by effective_mode from Step 1) and applies the
# structured filter conditions produced by _interpret_query_impl().
#
# Input format: "mode || filters_json"
#   e.g. "Scene Search || {"num_pedestrians": {">": 5}}"
#
# Returns a dict:
#   "results" — up to 50 matching docs (sorted later by the UI)
#   "_debug"  — scan stats shown in the pipeline debug expander
#
# Performance: pure Python loop, ~5–120 ms for 22 k docs.
# No LLM involved — this step is always deterministic and fast.
#
###########################################################
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
    """
    LangChain @tool wrapper around _filter_docs_impl.
    Serialises the result dict to JSON string for agent tool compatibility.
    In the LCEL pipeline, filter_step() calls _filter_docs_impl() directly
    to avoid @tool string serialisation overhead on large result sets.
    Input: "mode || filters_json"
    """
    return json.dumps(_filter_docs_impl(mode_and_filters))

###########################################################
# LCEL CHAIN — SEMANTIC VECTOR SEARCH  (fallback only)
#
# _semantic_search_impl() runs a FAISS approximate-nearest-neighbour search
# over the pre-built sentence-transformer embedding index.  It is only
# invoked by the RunnableBranch when Step 2 (exact filter) returns 0 results.
#
# Input format: "mode || query text"
#   e.g. "Scene Search || busy intersection with few cyclists"
#
# Steps:
#   1. Encode the semantic_query string with the sentence-transformer model
#      (same model used to build the index at indexing time).
#   2. Run index.search() for top-10 nearest neighbours by cosine similarity.
#   3. Return the corresponding doc dicts.
#
###########################################################
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
    """
    LangChain @tool wrapper around _semantic_search_impl.
    In the LCEL pipeline, semantic_step() calls _semantic_search_impl()
    directly (no serialisation overhead).  This wrapper exists so the
    function can be handed to AgentExecutor as a named tool when a
    function-calling model is used.
    Input: "mode || query text"
    """
    return json.dumps(_semantic_search_impl(mode_and_query))

###########################################################
# VISUALISATION HELPERS
#
# parse_kitti_label_file() — reads a KITTI-format .txt label file and
#   returns a list of object dicts (type, bbox, occlusion, truncation).
#
# draw_boxes() — draws coloured bounding boxes + labels on an image array.
#   Colour conventions used throughout:
#     Green  (0,255,0)   — ground-truth objects (GT panel)
#     Yellow (255,255,0) — model predictions (Pred panel)
#     Blue   (0,128,255) — FN errors (missed GT, shown on GT panel)
#     Red    (255,0,0)   — FP errors (wrong predictions, Pred panel)
#
# render_side_by_side() — produces the GT | Prediction composite image
#   for one frame.  Used by render_error_doc() in Error Analysis mode.
###########################################################
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

    gt_path = os.path.join("../data/training/label_2", f"{frame_id}.txt")
    pred_path = os.path.join("../runs/detect/predict/kitti_labels", f"{frame_id}.txt")

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

###########################################################
# ERROR DOC VISUALISATION
#
# render_error_doc() is called for every result card in Error Analysis mode.
# It delegates to render_side_by_side() which reads both the GT label file
# (data/training/label_2/<id>.txt) and the model prediction file
# (runs/detect/predict/kitti_labels/<id>.txt), then overlays the specific
# error bounding box from the error doc.
#
# resolve_image_path() — tries three candidate paths (as-is, relative to
#   cwd, relative to __file__) to handle backslash paths stored in docs on
#   Windows and relative-path issues when Streamlit changes the working dir.
#
# add_panel_label() — stamps a dark header bar onto each panel so the user
#   can immediately tell GT (left) from Prediction (right) in the composite.
###########################################################
def resolve_image_path(raw_path: str):
    """Resolve a potentially relative / backslash path to an absolute path."""
    img_path = os.path.normpath(raw_path)
    current_dir = os.path.dirname(__file__)
    parent_dir  = os.path.dirname(current_dir)
    for candidate in [
        img_path,
        os.path.join(parent_dir, img_path),
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


###########################################################
# STREAMLIT UI + AGENT
###########################################################
st.title("KITTI RAG Explorer — LangChain LCEL + Ollama")

#  LLM Provider sidebar 
# The radio button controls which ChatModel is instantiated in build_pipeline().
# Both providers use the same LCEL chain — only the llm object differs.
# Switching provider/model invalidates the session_state cache_key so the
# pipeline re-runs with the new backend on the next query submission.
st.sidebar.markdown("## LLM Provider")

llm_provider = st.sidebar.radio(
    "Backend",
    ["Ollama (local)", "Groq (cloud — fast, free)"],
    help="Ollama runs locally (private, no API key). Groq is ~10–20× faster and free.",
)

if llm_provider == "Ollama (local)":
    ollama_model = st.sidebar.selectbox(
        "Ollama model",
        ["llama3"],
        help="Must be pulled locally: `ollama pull <model>`",
    )
    llm_config = {"provider": "ollama", "model": ollama_model}
    st.sidebar.caption(f"Running locally · model: `{ollama_model}`")
else:
    if not GROQ_AVAILABLE:
        st.sidebar.error("langchain-groq not installed. Run: `pip install langchain-groq`")
    groq_key = st.sidebar.text_input(
        "Groq API key",
        type="password",
        help="Free key at console.groq.com — no credit card needed.",
        placeholder="gsk_...",
    )
    groq_model = st.sidebar.selectbox(
        "Groq model",
        [
            "llama-3.3-70b-versatile",   # best quality, recommended replacement
            "llama-3.1-8b-instant",      # fastest, lowest latency
        ],
        help="llama-3.3-70b-versatile for quality and llama-3.1-8b-instant for lowest latency.",
    )
    llm_config = {"provider": "groq", "model": groq_model, "api_key": groq_key}
    if groq_key:
        st.sidebar.caption(f" Groq cloud · model: `{groq_model}`")
    else:
        st.sidebar.warning("Paste your Groq API key above to enable.")

st.sidebar.divider()

#  Mode Hint sidebar 
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

# =============================================================================
# LANGCHAIN LCEL PIPELINE — build_pipeline() / get_pipeline() / run_pipeline()
#
#  Model selection & provider switching 
# build_pipeline(llm_config) accepts a config dict at runtime:
#
#   {"provider": "ollama", "model": "llama3"}
#       -> ChatOllama(model="llama3", temperature=0)
#       -> local Ollama server, no API key, ~2–5 s/query
#       -> good models: llama3, llama3.1, mistral-nemo, qwen2.5:14b
#
#   {"provider": "groq", "model": "llama-3.3-70b-versatile", "api_key": "..."}
#       -> ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
#       -> Groq cloud API, free tier at console.groq.com, ~0.3–0.8 s/query
#       -> good models: llama-3.3-70b-versatile (best quality, replaces the
#         decommissioned llama-3.1-70b-versatile as of Jan 2025),
#         llama-3.1-8b-instant (fastest), llama3-70b-8192, gemma2-9b-it#
#
# =============================================================================

def build_pipeline(llm_config: dict):
    """
    Assemble and return the LCEL chain for the given LLM provider config.

    This function is the chain factory.  It is called by get_pipeline()
    which caches the result with @st.cache_resource, so the chain is only
    built once per unique (provider, model) combination.

    llm_config keys:
      provider  "ollama" — use local Ollama server (ChatOllama)
                "groq"  — use Groq cloud API (ChatGroq, free tier)
      model     model name string, e.g.:
                  Ollama: "llama3"
                  Groq:   "llama-3.3-70b-versatile" (recommended),
                          "llama-3.1-8b-instant"
      api_key   Groq API key string (ignored for Ollama provider).
                Obtain a free key at console.groq.com.

    Returns:
      A compiled LangChain Runnable (LCEL chain) that accepts
      {"query": str, "mode": str} and returns the full pipeline output dict.

    Chain topology:
      RunnableLambda(interpret_step)
        | RunnableLambda(filter_step)
        | RunnableBranch(semantic_branch)
        | RunnableLambda(merge_step)
    """

    #  Instantiate the LLM from config 
    # ChatOllama and ChatGroq both extend LangChain's BaseChatModel, so every
    # downstream step (interpret_step, the LCEL | chain, LangSmith tracing)
    # works identically regardless of which provider is selected.
    # temperature=0 is set on both to produce deterministic JSON output.
    #
    # Model notes:
    #   Ollama  — model must be pulled first: `ollama pull <model>`
    #             llama3 / llama3.1 are good defaults; qwen2.5:14b has better
    #             instruction-following for structured output
    #   Groq    — llama-3.3-70b-versatile replaced the decommissioned
    #             llama-3.1-70b-versatile (retired Jan 2025); llama-3.1-8b-instant
    #             is the fastest option if latency matters more than quality
    if llm_config["provider"] == "groq":
        llm = ChatGroq(
            model=llm_config["model"],
            api_key=llm_config.get("api_key", ""),
            temperature=0,   # deterministic JSON output
        )
    else:
        llm = ChatOllama(
            model=llm_config["model"],
            temperature=0,
        )

    #  Step 1: NL -> structured filters 
    # RunnableLambda wraps _interpret_query_impl so it participates in the
    # LCEL chain.  The LLM call is the main latency driver.
    # Both ChatOllama and ChatGroq support LangSmith tracing automatically.

    def interpret_step(inputs: dict) -> dict:
        """
        Step 1 — call _interpret_query_impl and parse the result.
        Passes the full input dict forward, enriched with interpretation fields.
        """
        t0 = time.perf_counter()
        query = inputs["query"]
        mode  = inputs["mode"]

        # Pass the llm instance so the correct provider (Groq/Ollama) is used.
        # Without this, _interpret_query_impl falls back to raw requests.post
        # and the provider toggle in the sidebar has no effect on latency.
        interpreted_str = _interpret_query_impl(f"{query} || {mode}", llm=llm)
        try:
            interpreted = json.loads(interpreted_str)
        except Exception:
            interpreted = {"filters": {}, "semantic_query": query}

        return {
            **inputs,
            "interpreted":    interpreted,
            "effective_mode": interpreted.get("_mode", mode),
            "filters":        interpreted.get("filters", {}),
            "semantic_query": interpreted.get("semantic_query", query),
            "t_interpret":    round((time.perf_counter() - t0) * 1000),
        }

    #  Step 2: exact numeric filter 
    # RunnableLambda wrapping _filter_docs_impl.  Pure Python scan — fast
    # even for 20 k+ docs.  Returns up to 50 results for pagination.
    def filter_step(inputs: dict) -> dict:
        """
        Step 2 — apply structured filters against the correct doc corpus.
        Passes all previous fields forward plus filter results and debug info.
        """
        t0             = time.perf_counter()
        filter_results = []
        filter_debug   = {}

        if inputs["filters"]:
            raw = _filter_docs_impl(
                f"{inputs['effective_mode']} || {json.dumps(inputs['filters'])}"
            )
            filter_results = raw.get("results", [])
            filter_debug   = raw.get("_debug", {})

        return {
            **inputs,
            "filter_results": filter_results,
            "filter_debug":   filter_debug,
            "t_filter":       round((time.perf_counter() - t0) * 1000),
        }

    #  Step 3a: semantic search (fallback branch) 
    # Only executed by RunnableBranch when filter_results is empty.
    # Uses the pre-built FAISS index of sentence-transformer embeddings.
    def semantic_step(inputs: dict) -> dict:
        """
        Step 3 (fallback) — FAISS ANN search when exact filter finds nothing.
        """
        t0 = time.perf_counter()
        results = _semantic_search_impl(
            f"{inputs['effective_mode']} || {inputs['semantic_query']}"
        )
        return {
            **inputs,
            "semantic_results": results,
            "search_mode":      "semantic",
            "t_semantic":       round((time.perf_counter() - t0) * 1000),
        }

    #  Step 3b: passthrough (filter succeeded — skip semantic) 
    def filter_hit_step(inputs: dict) -> dict:
        """No-op branch: filter already found results, semantic search skipped."""
        return {
            **inputs,
            "semantic_results": [],
            "search_mode":      "filter",
            "t_semantic":       0,
        }

    # RunnableBranch: (condition, runnable_if_true), ..., runnable_default
    # Condition: filter returned no results -> run semantic search
    semantic_branch = RunnableBranch(
        (lambda x: len(x.get("filter_results", [])) == 0,
         RunnableLambda(semantic_step)),
        RunnableLambda(filter_hit_step),   # default: filter succeeded
    )

    #  Step 4: merge into final output dict 
    # Assembles the latency_ms dict and picks the display_docs list.
    # This is the output shape consumed by the Streamlit UI.
    def merge_step(inputs: dict) -> dict:
        """
        Final step — build the latency table and pick the display doc list.
        Produces the output dict that the Streamlit UI reads directly.
        """
        latency_ms = {
            "Step 1 — interpret_query (Ollama)": inputs.get("t_interpret", 0),
            "Step 2 — filter_docs (exact scan)": inputs.get("t_filter",    0),
            "Step 3 — semantic_search (FAISS)":  inputs.get("t_semantic",  0),
        }
        display_docs = (
            inputs["filter_results"]
            if inputs["filter_results"]
            else inputs["semantic_results"]
        )
        return {
            "interpreted":     inputs["interpreted"],
            "effective_mode":  inputs["effective_mode"],
            "filter_results":  inputs["filter_results"],
            "filter_debug":    inputs["filter_debug"],
            "semantic_results":inputs["semantic_results"],
            "search_mode":     inputs["search_mode"],
            "display_docs":    display_docs,
            "latency_ms":      latency_ms,
        }

    #  Assemble the chain with the LCEL | operator 
    # Reading left-to-right mirrors the data flow described in the header above.
    chain = (
        RunnableLambda(interpret_step)   # NL -> filters
        | RunnableLambda(filter_step)    # filters -> exact matches
        | semantic_branch                # route: skip or run FAISS
        | RunnableLambda(merge_step)     # assemble final output
    )
    return chain


@st.cache_resource
def get_pipeline(provider: str, model: str, api_key: str = ""):
    """
    Build and cache the LCEL chain, keyed by (provider, model, api_key).

    @st.cache_resource stores the compiled Runnable object in memory across
    all Streamlit re-runs (widget interactions, pagination, expander toggles).
    The chain is only rebuilt when the user changes the provider or model in
    the sidebar — switching from Ollama llama3 -> Groq llama-3.3-70b-versatile
    triggers one rebuild, then the new chain is cached for subsequent queries.

    Keying on api_key ensures a new chain is built if the user rotates their
    Groq API key without changing the model name.
    """
    return build_pipeline({"provider": provider, "model": model, "api_key": api_key})


def run_pipeline(query: str, mode: str, llm_cfg: dict) -> dict:
    """
    Invoke the LCEL chain for one user query.

    Retrieves the chain compiled for (provider, model) from get_pipeline()
    — which is @st.cache_resource cached — and calls chain.invoke().
    The session_state cache_key (query || mode || provider || model) ensures
    this function is only called when the query or provider actually changes;
    widget-triggered Streamlit re-runs reuse the stored _pipeline_output.
    """
    chain = get_pipeline(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        api_key=llm_cfg.get("api_key", ""),
    )
    return chain.invoke({"query": query, "mode": mode})


# Cache pipeline output in session_state so widget interactions
# (pagination, expanders) don't re-trigger a full pipeline re-run.
# Include provider+model in cache_key so switching LLM invalidates cache
cache_key = f"{query}||{query_mode}||{llm_config['provider']}||{llm_config['model']}"
if query and st.session_state.get("_cache_key") != cache_key:
    # Validate Groq key before invoking
    if llm_config["provider"] == "groq" and not llm_config.get("api_key"):
        st.warning("Please enter your Groq API key in the sidebar.")
    else:
        provider_label = (
            f"Ollama · {llm_config['model']}"
            if llm_config["provider"] == "ollama"
            else f"Groq · {llm_config['model']}"
        )
        with st.spinner(f"Running pipeline via {provider_label}…"):
            st.session_state["_pipeline_output"] = run_pipeline(
                query, query_mode, llm_config
            )
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
            f"Mode auto-detected: **{effective_mode}** "
            f"(your hint was '{query_mode}' — overridden)"
        )
    else:
        st.caption(f"Mode: **{effective_mode}**")

    # --- Debug expander ---
    with st.expander("Pipeline debug", expanded=False):
        # --- Per-step latency ---
        if latency_ms:
            provider_badge = (
                f" Ollama · `{llm_config['model']}`"
                if llm_config["provider"] == "ollama"
                else f" Groq · `{llm_config['model']}`"
            )
            st.markdown(f"** Latency** — {provider_badge}")
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

        source = interpreted.get("_source", "llm")
        source_label = (
            "⚡ fuzzy rules only — LLM skipped"
            if source == "fuzzy_only"
            else "LLM inference"
        )
        st.markdown(f"**Step 1 — interpret_query output** · {source_label}")
        st.json({k: v for k, v in interpreted.items() if k not in ("_mode", "_source")})

        st.markdown(f"**Step 2 — filter_docs** -> {len(filter_results)} match(es)")
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
            st.markdown("**Step 3 — semantic_search** -> skipped (filter had results ✅)")
        else:
            st.markdown(f"**Step 3 — semantic_search** -> {len(semantic_results)} result(s) (filter had 0 matches)")

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
            # e.g. ">=2 cars AND >3 pedestrians" -> sort by num_pedestrians (threshold 3 > 2)
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
                    st.caption(f" {err_msg}")
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
                    st.caption(f"!! Image not found. Tried: `{img_path}` (cwd: `{os.getcwd()}`)")
