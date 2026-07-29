"""Build a response-based annotation batch: a flat, shuffled list of isolated
(user message, model response) pairs drawn from conversations.

In response mode annotators code each pair on its own, with NO conversation
context. Each item keeps hidden provenance (session_id + turn) so item-level
ratings can later be joined back to the in-context ratings for comparison — the
provenance is stripped before an item is served to the browser (see app.py).

The manifest is written to ``batches/<name>__responses.json``.
"""
import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CONV_DIR = os.path.join(HERE, "data", "conversations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="demo", help="Batch name (→ <name>__responses.json).")
    ap.add_argument("--turns", default="1,5",
                    help="Comma list of turn numbers to sample from each conversation.")
    ap.add_argument("--seed", type=int, default=7, help="Shuffle seed.")
    args = ap.parse_args()

    turns = [int(t) for t in args.turns.split(",") if t.strip()]
    files = sorted(f for f in os.listdir(CONV_DIR) if f.endswith(".json"))
    if not files:
        raise SystemExit(f"No conversations in {CONV_DIR}. Run make_demo_batch.py first.")

    items = []
    n = 0
    for fn in files:
        with open(os.path.join(CONV_DIR, fn)) as f:
            conv = json.load(f)
        belief = conv.get("belief", {})
        by_turn = {t.get("turn"): t for t in conv.get("turns", [])}
        for tn in turns:
            t = by_turn.get(tn)
            if not t or not t.get("target_response"):
                continue
            n += 1
            items.append({
                "item_id": f"resp-{n:04d}",
                # hidden provenance (not sent to the browser):
                "session_id": conv.get("session_id"),
                "turn": tn,
                # shown to annotators:
                "category": belief.get("category"),
                "belief": belief.get("content"),
                "user_message": t.get("user_message", ""),
                "target_response": t.get("target_response", ""),
            })

    random.Random(args.seed).shuffle(items)
    # re-id after shuffle so ids don't leak original ordering
    for i, it in enumerate(items, 1):
        it["item_id"] = f"resp-{i:04d}"

    manifest = {"name": args.name, "kind": "responses",
                "n_items": len(items), "sampled_turns": turns, "items": items}
    os.makedirs(os.path.join(HERE, "batches"), exist_ok=True)
    out = os.path.join(HERE, "batches", f"{args.name}__responses.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Built {len(items)} isolated response items (turns {turns}).")
    print(f"Manifest: batches/{args.name}__responses.json")


if __name__ == "__main__":
    main()
