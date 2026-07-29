"""Storage abstraction for annotation records.

The interface saves one record per (annotator, unit_type, unit_id). A *unit* is
either a whole ``conversation`` or a single isolated ``response`` pair — the two
annotation modes share this storage layer. Today records are JSON files under
``annotations/``; when the university MariaDB is available, implement
``MariaDBStorage`` against the schema in ``schema.sql`` and set the env var
``ANNO_STORAGE=mariadb``. The rest of the app talks only to the ``Storage``
interface, so nothing else changes.

Conversation record (unit_type="conversation", unit_id=session_id)::

    {
      "schema_version": "1.0", "unit_type": "conversation",
      "unit_id": "cell-...__model-...", "session_id": "cell-...__model-...",
      "annotator_id": "alice", "status": "in_progress" | "submitted",
      "created_at": "...", "updated_at": "...",
      "responses": {"1": {"correction": 3, "rebuttal": 2,
                          "affective_validation": "support",
                          "epistemic_endorsement": "disagree",
                          "stance": "false", "comment": "..."}, ...},
      "highlights": [{"id": "h1", "turn": 1, "target": "response",
                      "start": 40, "end": 88, "quote": "...",
                      "code": "hedging", "note": "...", "color": "--c1"}],
      "conversation": {"trajectory": "drifts", "comment": "...", "themes": [...]}
    }

Response record (unit_type="response", unit_id=item_id)::

    {
      "schema_version": "1.0", "unit_type": "response", "unit_id": "resp-0007",
      "item_id": "resp-0007", "annotator_id": "alice", "status": "...",
      "created_at": "...", "updated_at": "...",
      "rating": {"correction": 2, "rebuttal": 1, "affective_validation": "...",
                 "epistemic_endorsement": "...", "stance": "...", "comment": "..."},
      "highlights": [{"id": "h1", "turn": 1, "target": "response", ...}]
    }
"""
import json
import os
import re
from datetime import datetime, timezone

import config

SCHEMA_VERSION = "1.0"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    return _SAFE.sub("_", value)[:200]


def _n_rated(record):
    """How many response-ratings a record holds (for progress meters)."""
    if record.get("unit_type") == "response":
        return 1 if record.get("rating") else 0
    return len(record.get("responses", {}))


class Storage:
    """Interface every backend implements. ``unit_type`` is
    "conversation" or "response"; ``unit_id`` is the session_id or item_id."""

    def load(self, annotator_id, unit_type, unit_id):
        raise NotImplementedError

    def save(self, record):
        raise NotImplementedError

    def status_map(self, annotator_id, unit_type, unit_ids):
        """Return {unit_id: {"status": ..., "n_responses": int}} for a batch."""
        raise NotImplementedError


class LocalJSONStorage(Storage):
    """One JSON file per (annotator, unit_type, unit_id) under ANNOTATIONS_DIR,
    e.g. ``annotations/alice/conversation/<session_id>.json`` and
    ``annotations/alice/response/<item_id>.json``."""

    def __init__(self, root=None):
        self.root = root or config.ANNOTATIONS_DIR
        os.makedirs(self.root, exist_ok=True)

    def _path(self, annotator_id, unit_type, unit_id):
        d = os.path.join(self.root, _slug(annotator_id), _slug(unit_type))
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, _slug(unit_id) + ".json")

    def load(self, annotator_id, unit_type, unit_id):
        path = self._path(annotator_id, unit_type, unit_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def save(self, record):
        annotator_id = record["annotator_id"]
        unit_type = record["unit_type"]
        unit_id = record["unit_id"]
        path = self._path(annotator_id, unit_type, unit_id)

        existing = self.load(annotator_id, unit_type, unit_id)
        record["schema_version"] = SCHEMA_VERSION
        record["created_at"] = (existing or {}).get("created_at") or _now()
        record["updated_at"] = _now()

        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atomic write
        return record

    def status_map(self, annotator_id, unit_type, unit_ids):
        out = {}
        for uid in unit_ids:
            rec = self.load(annotator_id, unit_type, uid)
            if rec is None:
                out[uid] = {"status": "not_started", "n_responses": 0}
            else:
                out[uid] = {"status": rec.get("status", "in_progress"),
                            "n_responses": _n_rated(rec)}
        return out


class MariaDBStorage(Storage):
    """Placeholder for the university MariaDB backend.

    Implement against ``schema.sql`` once DB access is granted. Suggested
    approach: keep one row in ``annotations`` (status + JSON payload of the full
    record) for fast round-tripping, and additionally upsert normalized rows
    into ``response_ratings`` / ``highlights`` for SQL-side analysis. Then set
    ``ANNO_STORAGE=mariadb`` and the app switches over with no frontend change.
    """

    def __init__(self, conn_params=None):
        raise NotImplementedError(
            "MariaDBStorage is a stub. Fill it in using schema.sql once the "
            "university database is available (see README)."
        )


def get_storage() -> Storage:
    if config.STORAGE_BACKEND == "mariadb":
        return MariaDBStorage(config.MARIADB)
    return LocalJSONStorage()
