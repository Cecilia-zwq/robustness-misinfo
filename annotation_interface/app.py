"""Local Flask server for the misinformation annotation interface.

Serves a batch of conversations (blinded) to annotators and persists their
ratings + qualitative codes through the pluggable storage layer. Run with:

    python app.py

then open http://127.0.0.1:8000
"""
import json
import os

from flask import Flask, jsonify, request, send_from_directory

import config
from storage import get_storage

app = Flask(__name__, static_folder=None)
store = get_storage()

STRIP_TURN_KEYS = {"reflection_attempts", "is_fallback",
                   "n_character_breaks", "n_belief_breaks"}


def load_rubric():
    with open(config.RUBRIC_PATH) as f:
        return json.load(f)


def load_batch(name):
    path = os.path.join(config.BATCHES_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_response_batch(name):
    path = os.path.join(config.BATCHES_DIR, f"{name}__responses.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_response_item(name, item_id):
    batch = load_response_batch(name)
    if batch is None:
        return None
    for it in batch.get("items", []):
        if it.get("item_id") == item_id:
            return it
    return None


def public_response_item(raw):
    """Strip provenance (session_id, turn) so the pair is truly isolated."""
    return {
        "item_id": raw.get("item_id"),
        "category": raw.get("category"),
        "belief": raw.get("belief"),
        "user_message": raw.get("user_message", ""),
        "target_response": raw.get("target_response", ""),
        "blind": config.BLIND_METADATA,
    }


def load_conversation_file(session_id):
    path = os.path.join(config.DATA_CONVERSATIONS, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def public_conversation(raw):
    """Strip anything that could bias a blind annotator."""
    belief = raw.get("belief", {})
    turns = []
    for t in raw.get("turns", []):
        turns.append({
            "turn": t.get("turn"),
            "user_message": t.get("user_message", ""),
            "target_response": t.get("target_response", ""),
        })
    payload = {
        "session_id": raw.get("session_id"),
        "category": belief.get("category"),
        "belief": belief.get("content"),
        "n_turns": len(turns),
        "turns": turns,
        "blind": config.BLIND_METADATA,
    }
    if not config.BLIND_METADATA:
        payload["meta"] = {
            "target_llm": raw.get("models", {}).get("target_llm"),
            "condition": raw.get("cell", {}).get("cell_id"),
        }
    return payload


# ----------------------------------------------------------------------------- API

@app.get("/api/rubric")
def api_rubric():
    return jsonify(load_rubric())


@app.get("/api/batch")
def api_batch():
    annotator = request.args.get("annotator", "").strip()
    mode = request.args.get("mode", "conversation")
    name = request.args.get("batch", config.DEFAULT_BATCH)

    if mode == "response":
        batch = load_response_batch(name)
        if batch is None:
            return jsonify({"error": f"response batch '{name}' not found"}), 404
        ids = [it["item_id"] for it in batch["items"]]
        status = store.status_map(annotator, "response", ids) if annotator else {}
        items = []
        for it in batch["items"]:
            st = status.get(it["item_id"], {"status": "not_started", "n_responses": 0})
            items.append({
                "unit_id": it["item_id"],
                "category": it.get("category"),
                "n_turns": 1,
                "status": st["status"],
                "n_responses": st["n_responses"],
            })
        return jsonify({"batch": name, "mode": mode, "annotator": annotator,
                        "blind": config.BLIND_METADATA, "items": items})

    batch = load_batch(name)
    if batch is None:
        return jsonify({"error": f"batch '{name}' not found"}), 404
    sids = batch["session_ids"]
    status = store.status_map(annotator, "conversation", sids) if annotator else {}
    items = []
    for sid in sids:
        raw = load_conversation_file(sid)
        if raw is None:
            continue
        st = status.get(sid, {"status": "not_started", "n_responses": 0})
        items.append({
            "unit_id": sid,
            "category": raw.get("belief", {}).get("category"),
            "n_turns": len(raw.get("turns", [])),
            "status": st["status"],
            "n_responses": st["n_responses"],
        })
    return jsonify({"batch": name, "mode": "conversation", "annotator": annotator,
                    "blind": config.BLIND_METADATA, "items": items})


@app.get("/api/conversation/<session_id>")
def api_conversation(session_id):
    annotator = request.args.get("annotator", "").strip()
    raw = load_conversation_file(session_id)
    if raw is None:
        return jsonify({"error": "conversation not found"}), 404
    annotation = store.load(annotator, "conversation", session_id) if annotator else None
    return jsonify({
        "conversation": public_conversation(raw),
        "annotation": annotation,
    })


@app.get("/api/response/<item_id>")
def api_response(item_id):
    annotator = request.args.get("annotator", "").strip()
    name = request.args.get("batch", config.DEFAULT_BATCH)
    raw = load_response_item(name, item_id)
    if raw is None:
        return jsonify({"error": "response item not found"}), 404
    annotation = store.load(annotator, "response", item_id) if annotator else None
    return jsonify({
        "item": public_response_item(raw),
        "annotation": annotation,
    })


@app.post("/api/annotation")
def api_save_annotation():
    record = request.get_json(force=True, silent=True) or {}
    unit_type = record.get("unit_type")
    if not record.get("annotator_id") or not record.get("unit_id") \
            or unit_type not in ("conversation", "response"):
        return jsonify({"error": "annotator_id, unit_id and a valid unit_type "
                                 "are required"}), 400
    record.setdefault("status", "in_progress")
    record.setdefault("highlights", [])
    if unit_type == "conversation":
        record.setdefault("responses", {})
        record.setdefault("conversation", {})
    else:
        record.setdefault("rating", {})
    saved = store.save(record)
    return jsonify({"ok": True, "updated_at": saved["updated_at"],
                    "status": saved["status"]})


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "blind": config.BLIND_METADATA,
                    "storage": config.STORAGE_BACKEND,
                    "batch": config.DEFAULT_BATCH})


# -------------------------------------------------------------------------- static

STATIC_DIR = os.path.join(config.BASE, "static")


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    print(f"Annotation interface → http://{config.HOST}:{config.PORT}")
    print(f"  batch={config.DEFAULT_BATCH}  blind={config.BLIND_METADATA}  "
          f"storage={config.STORAGE_BACKEND}")
    app.run(host=config.HOST, port=config.PORT, debug=True)
