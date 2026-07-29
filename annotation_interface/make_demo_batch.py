"""Build a small, balanced demo batch for the annotation interface.

Selects a 4x4 grid of conversations (4 belief categories x 4 target models),
copies the selected conversation JSONs into ``data/conversations/`` so the
interface directory is self-contained, and writes ``batches/demo.json`` listing
the chosen ``session_id``s.

Point ``--source`` at any experiment's ``conversations`` folder to (re)build a
batch. Replace this with your real sampling logic later; the interface only
needs a ``batches/<name>.json`` manifest plus the matching JSON files under
``data/conversations/``.
"""
import argparse
import json
import os
import re
import shutil
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = os.path.join(
    HERE, "..", "results", "final_experiment", "main_user_IVs",
    "20260427_165233", "conversations",
)

NAME_RE = re.compile(
    r"cell-iv1-(?P<iv1>[^_]+)__iv2-(?P<iv2>[^_]+)__belief-(?P<cat>[a-z_]+)-"
    r"(?P<idx>\d+)__model-(?P<model>.+)\.json"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="Folder of conversation JSON files to sample from.")
    ap.add_argument("--name", default="demo", help="Batch name.")
    ap.add_argument("--per-cell", type=int, default=1,
                    help="Conversations per (category, model) cell.")
    args = ap.parse_args()

    source = os.path.abspath(args.source)
    if not os.path.isdir(source):
        raise SystemExit(f"Source folder not found: {source}")

    files = sorted(os.listdir(source))
    # group by (category, model) -> list of (belief_idx, filename)
    grouped = defaultdict(list)
    for fn in files:
        m = NAME_RE.match(fn)
        if not m:
            continue
        key = (m.group("cat"), m.group("model"))
        grouped[key].append((int(m.group("idx")), fn))

    categories = sorted({k[0] for k in grouped})
    models = sorted({k[1] for k in grouped})

    out_dir = os.path.join(HERE, "data", "conversations")
    os.makedirs(out_dir, exist_ok=True)
    batch_ids = []
    used_idx = defaultdict(set)  # per-category, keep beliefs distinct

    for cat in categories:
        for model in models:
            candidates = sorted(grouped.get((cat, model), []))
            picked = 0
            for idx, fn in candidates:
                if picked >= args.per_cell:
                    break
                if idx in used_idx[cat] and len(candidates) > args.per_cell:
                    continue
                used_idx[cat].add(idx)
                shutil.copyfile(os.path.join(source, fn), os.path.join(out_dir, fn))
                batch_ids.append(fn[:-5])  # strip .json -> session_id
                picked += 1

    os.makedirs(os.path.join(HERE, "batches"), exist_ok=True)
    manifest = {
        "name": args.name,
        "source": source,
        "categories": categories,
        "models": models,
        "session_ids": batch_ids,
    }
    with open(os.path.join(HERE, "batches", f"{args.name}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Selected {len(batch_ids)} conversations "
          f"({len(categories)} categories x {len(models)} models).")
    print(f"Copied into: {out_dir}")
    print(f"Manifest: batches/{args.name}.json")


if __name__ == "__main__":
    main()
