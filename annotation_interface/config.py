"""Configuration for the annotation interface.

Everything the server needs to locate data and decide what annotators can see.
Values can be overridden with environment variables so the same code runs in
"showcase" mode now and against MariaDB later without edits.
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Where the (self-contained) conversation JSONs live and where batches/results go.
DATA_CONVERSATIONS = os.path.join(BASE, "data", "conversations")
BATCHES_DIR = os.path.join(BASE, "batches")
ANNOTATIONS_DIR = os.path.join(BASE, "annotations")
RUBRIC_PATH = os.path.join(BASE, "rubric.json")

DEFAULT_BATCH = os.environ.get("ANNO_BATCH", "demo")

# Blind annotators to the target model identity and experimental condition so
# ratings are not anchored by which model produced a response. The belief topic
# is still shown (annotators need it to judge correction). Set to "0" to reveal.
BLIND_METADATA = os.environ.get("ANNO_BLIND", "1") != "0"

# Storage backend: "local" (JSON files, default) or "mariadb" (see storage.py).
STORAGE_BACKEND = os.environ.get("ANNO_STORAGE", "local")

# MariaDB connection (only read when STORAGE_BACKEND == "mariadb").
MARIADB = {
    "host": os.environ.get("ANNO_DB_HOST", "localhost"),
    "port": int(os.environ.get("ANNO_DB_PORT", "3306")),
    "user": os.environ.get("ANNO_DB_USER", ""),
    "password": os.environ.get("ANNO_DB_PASSWORD", ""),
    "database": os.environ.get("ANNO_DB_NAME", ""),
}

HOST = os.environ.get("ANNO_HOST", "127.0.0.1")
PORT = int(os.environ.get("ANNO_PORT", "8000"))
