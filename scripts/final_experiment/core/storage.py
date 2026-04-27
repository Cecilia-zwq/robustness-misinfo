"""
core/storage.py
===============
Disk layout, artifact schema, and atomic IO for experiment outputs.

The same atomic-write + cross-file-consistency machinery proven out in
experiment3.py and experiment_ablation.py, generalised so every future
experiment in this series uses the same primitives.

Disk layout (per experiment run)
--------------------------------
::

    results/<experiment>/<timestamp>/
      ├── manifest.json                  # full run config, every session id
      ├── conversations/                 # one file per session, atomic-written
      │   └── <session_id>.json          # ConversationArtifact (see below)
      ├── scores/                        # populated by run_scoring.py later
      │   └── <session_id>__<rubric>.json   # ScoreArtifact (see scoring.py)
      ├── checkpoint*.json               # phase-specific resume cursors
      └── logs/                          # human-readable transcripts
          └── <session_id>.txt

Conversations and scores are split because:
  - Re-scoring with a new rubric should not touch conversation files
  - One conversation can have N score files (one per rubric or evaluator)
  - The two phases have different parallelism / failure profiles

Conversation artifact schema (v1.0)
-----------------------------------
Self-describing. The cell spec, belief record, and model identifiers are
embedded so a single file is enough to reproduce or re-analyse the
session without external lookups. Reflection drafts are stored alongside
the accepted message so RQ3 (reflection ablation) can replay turn-1
drafts later without rerunning the model.
"""

from __future__ import annotations

import io
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


# ════════════════════════════════════════════════════════════════════════════
# Atomic write primitive
# ════════════════════════════════════════════════════════════════════════════
# Write to <name>.tmp, fsync, then os.replace onto the final path. This
# guarantees a reader of the final path always sees either the previous
# fully-written state or the new fully-written state — never a partial
# write. os.replace is atomic on POSIX and on Windows.

def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some filesystems don't support fsync (network mounts, etc).
            # The os.replace below is still atomic w.r.t. process crashes.
            pass
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any, *, ensure_ascii: bool = False) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=ensure_ascii)
    atomic_write_text(path, text)


def cleanup_stale_tmp_files(directory: Path) -> None:
    """Remove .tmp files left over from a crash mid-write.

    Real files still hold the previous complete state; the .tmp files are
    harmless leftovers. Sweeps both `directory` and one level of
    subdirectories (covers conversations/, scores/, logs/ in our layout).
    """
    if not directory.exists():
        return
    for p in directory.rglob("*.tmp"):
        try:
            p.unlink()
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════════════════
# Conversation artifact: dataclass + IO
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ReflectionAttempt:
    """One pass of the reflection module on one turn.

    `accepted` is True for the final attempt that was actually used as
    the user message (or for the fallback message on max-retry).
    """
    attempt: int
    draft: str
    character_verdict: str  # "PASS" | "FAIL" | "PARSE_ERROR"
    character_quote: str
    character_fix: str
    belief_verdict: str
    belief_quote: str
    belief_fix: str
    accepted: bool = False


@dataclass
class TurnArtifact:
    """One turn of a conversation.

    `user_message` is the message that was actually sent to the target LLM
    (i.e. the accepted draft, or the fallback string on max-retry).
    `reflection_attempts` records every draft considered, in order, so the
    full reflection trajectory is recoverable.
    """
    turn: int
    user_message: str
    target_response: str
    reflection_attempts: list[ReflectionAttempt] = field(default_factory=list)
    is_fallback: bool = False
    n_character_breaks: int = 0
    n_belief_breaks: int = 0


@dataclass
class ConversationArtifact:
    """All data for one (cell × belief × target_model) session."""
    schema_version: str
    session_id: str
    experiment: str
    cell: dict          # Condition.to_dict()
    belief: dict        # full belief record
    models: dict        # {"user_agent": "...", "target_llm": "..."}
    config: dict        # {"n_turns": 8, "temperature_user": 0.7, ...}
    turns: list[TurnArtifact] = field(default_factory=list)
    completed_at: str | None = None

    def to_json(self) -> str:
        """Serialise as indented JSON string, preserving nested dataclasses."""
        return json.dumps(_to_jsonable(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationArtifact":
        turns = [
            TurnArtifact(
                turn=t["turn"],
                user_message=t["user_message"],
                target_response=t["target_response"],
                reflection_attempts=[
                    ReflectionAttempt(**a) for a in t.get("reflection_attempts", [])
                ],
                is_fallback=t.get("is_fallback", False),
                n_character_breaks=t.get("n_character_breaks", 0),
                n_belief_breaks=t.get("n_belief_breaks", 0),
            )
            for t in d.get("turns", [])
        ]
        return cls(
            schema_version=d["schema_version"],
            session_id=d["session_id"],
            experiment=d["experiment"],
            cell=d["cell"],
            belief=d["belief"],
            models=d["models"],
            config=d["config"],
            turns=turns,
            completed_at=d.get("completed_at"),
        )


def _to_jsonable(obj: Any) -> Any:
    """Recursive dataclass-aware JSON encoder."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


# ════════════════════════════════════════════════════════════════════════════
# Run directory + paths
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RunPaths:
    """All the paths an experiment run cares about, in one place."""
    root: Path

    @property
    def conversations(self) -> Path:
        return self.root / "conversations"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.json"

    def conversation_path(self, session_id: str) -> Path:
        return self.conversations / f"{session_id}.json"

    def score_path(self, session_id: str, rubric: str) -> Path:
        return self.scores / f"{session_id}__{rubric}.json"

    def log_path(self, session_id: str) -> Path:
        return self.logs / f"{session_id}.txt"

    def ensure_dirs(self) -> None:
        self.conversations.mkdir(parents=True, exist_ok=True)
        self.scores.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)


def make_run_paths(
    base_results_dir: Path,
    experiment_name: str,
    *,
    resume_dir: Path | str | None = None,
) -> RunPaths:
    """Resolve the run directory: existing if resuming, fresh if not."""
    if resume_dir is not None:
        root = Path(resume_dir)
        if not root.exists():
            raise FileNotFoundError(f"Resume directory does not exist: {root}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = base_results_dir / experiment_name / timestamp
    paths = RunPaths(root=root)
    paths.ensure_dirs()
    cleanup_stale_tmp_files(paths.root)
    return paths


# ════════════════════════════════════════════════════════════════════════════
# Conversation IO
# ════════════════════════════════════════════════════════════════════════════

def write_conversation(paths: RunPaths, artifact: ConversationArtifact) -> None:
    atomic_write_text(
        paths.conversation_path(artifact.session_id),
        artifact.to_json(),
    )


def read_conversation(paths: RunPaths, session_id: str) -> ConversationArtifact:
    path = paths.conversation_path(session_id)
    with open(path, "r", encoding="utf-8") as f:
        return ConversationArtifact.from_dict(json.load(f))


def list_completed_conversation_ids(paths: RunPaths) -> set[str]:
    """Session IDs that have a fully-written conversation artifact on disk.

    Used by the runner to determine which sessions can be skipped on resume.
    A file is considered complete iff (a) it parses as JSON and (b) has the
    expected schema_version and a non-null completed_at — partial writes
    can't pass both because the writer only sets completed_at last.
    """
    if not paths.conversations.exists():
        return set()
    out: set[str] = set()
    for p in paths.conversations.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if (
            data.get("schema_version") == SCHEMA_VERSION
            and data.get("completed_at") is not None
            and data.get("session_id") == p.stem
        ):
            out.add(p.stem)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Manifest IO
# ════════════════════════════════════════════════════════════════════════════

def write_manifest(paths: RunPaths, payload: dict) -> None:
    payload = {**payload, "last_updated": datetime.now().isoformat()}
    atomic_write_json(paths.manifest, payload)


def read_manifest(paths: RunPaths) -> dict | None:
    if not paths.manifest.exists():
        return None
    try:
        with open(paths.manifest, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("manifest.json unreadable (%s).", e)
        return None


# ════════════════════════════════════════════════════════════════════════════
# Session ID construction
# ════════════════════════════════════════════════════════════════════════════
# Stable, sortable, filename-safe. The four components nail down the
# experimental cell unambiguously: which IV cell, which belief, which
# target model. No reps in this design — single-run main analysis is
# explicit in the plan.

def build_session_id(
    *,
    cell_id: str,
    belief_category: str,
    belief_index: int,
    target_model_short: str,
) -> str:
    return (
        f"cell-{cell_id}"
        f"__belief-{belief_category}-{belief_index:04d}"
        f"__model-{safe_slug(target_model_short)}"
    )


_SAFE_RE = None


def safe_slug(s: str) -> str:
    """Convert a freeform string to a filename-safe slug."""
    global _SAFE_RE
    if _SAFE_RE is None:
        import re
        _SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
    return _SAFE_RE.sub("-", s).strip("-")


# Backward-compat alias for internal callers that still import `_safe`.
_safe = safe_slug
