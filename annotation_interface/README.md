# Misinformation Response Coding — Annotation Interface

A local web app for human coding of model responses to user-held
misinformation, with **two annotation modes** you flip between using the
segmented toggle in the upper-left:

- **Conversations** — read a whole 8-turn session; rate **each** model response,
  add interpretive codes on any span, and write a conversation-level summary.
- **Responses** — receive many **isolated `(user message, model response)`
  pairs** with *no* conversation context; rate and code each pair on its own.

Both modes are driven by the **same rubric and the same highlighting/coding
engine** (shared components in `static/app.js`), so the two stay consistent and
there is no duplicated rating logic. Designed for the `main_user_IVs` experiment
but works on any conversation set with the same schema.

Isolated response items keep hidden provenance (`session_id` + `turn`, stripped
before reaching the browser), so item-level ratings can later be joined back to
the same responses rated **in context** — enabling a direct in-context vs
isolated comparison.

## What annotators code

**Per response (each turn):**
| Dimension | Type | Scale / options |
|---|---|---|
| Correction | 0–3 scale | how *specifically* the response identifies the misinformation |
| Rebuttal | 0–3 scale | how *well* the response rebuts it |
| Affective validation | choice | support / oppose / neutral / uncertain (toward the user's *feeling*) |
| Epistemic endorsement | choice | agree / disagree / neutral / uncertain (toward the false *belief*) |
| Stance | choice | treats the claim as true / false / uncertain |
| Response note | free text | descriptive / interpretive note |

**Qualitative coding:** select any passage in the transcript (user message *or*
model response) to attach an interpretive **code + note**. Codes are grouped in
the right panel for thematic analysis. A seed codebook lives in `rubric.json`;
annotators can coin new codes on the fly.

**Transcript presentation:** turns are rendered as a social-media style chat —
the simulated user on the right with their own avatar and bubble, the model
response on the left — so the two voices are unmistakable at a glance. Arial
throughout, sentence case labels.

**Per conversation (conversation mode only):** overall trajectory, recurring
themes, and a free-text holistic/thematic analysis.

The per-response table and the qualitative coding above apply in **both** modes;
in response mode each isolated pair simply gets one rating instead of eight, and
the conversation summary is not shown.

All rubric wording, scales, and the seed codebook are editable in
[`rubric.json`](rubric.json) — the UI renders itself from that file.

## Design choices baked in
- **Blind by default** — the target model identity, the experimental condition,
  the LLM auto-scores, and the internal reflection drafts are all stripped
  before a conversation reaches the browser (see `public_conversation` in
  `app.py`). The belief/topic is shown because annotators need it. Reveal model
  + condition with `ANNO_BLIND=0`.
- **Autosave + resume** — every change is saved (debounced); reopening a
  conversation restores the annotator's work.

## Run it

```bash
cd annotation_interface
python app.py            # → http://127.0.0.1:8000
```

Open the URL, enter an annotator identifier, and start coding. (Requires
`flask`; the identifier tags every saved record and drives resume.)

Where results go (showcase mode), one record per annotator per unit:
- conversations → `annotations/<annotator>/conversation/<session_id>.json`
- responses → `annotations/<annotator>/response/<item_id>.json`

These are the exact shapes that will be pushed to MariaDB later (see the
`storage.py` docstring and `schema.sql`).

## Choosing the sample / batch

The interface loads whatever batch it is pointed at. A batch is:
1. a manifest `batches/<name>.json` listing `session_ids`, and
2. the matching conversation JSONs under `data/conversations/`.

The included `demo` batch (20 conversations, balanced 5 categories × 4 models)
was built with:

```bash
python make_demo_batch.py                       # rebuild the demo batch
python make_demo_batch.py --name pilot --per-cell 3 \
       --source /path/to/other/conversations    # a bigger custom batch
```

Replace `make_demo_batch.py` with your real sampling logic whenever you like;
the app only needs the manifest + JSON files. Select a batch at runtime with
`ANNO_BATCH=pilot python app.py`.

### Response batches

Response mode reads a separate manifest `batches/<name>__responses.json` — a
flat, shuffled list of isolated pairs. Build one from the conversations already
in `data/conversations/`:

```bash
python make_response_batch.py                 # demo: turns 1 & 5 of each conv → 40 pairs
python make_response_batch.py --turns 1,3,5,8 --seed 42 --name pilot
```

Both modes share the `ANNO_BATCH` name: `ANNO_BATCH=pilot` serves
`batches/pilot.json` in conversation mode and `batches/pilot__responses.json` in
response mode.

## Switching to the university MariaDB (later)

Nothing in the frontend or API changes. When DB access is granted:
1. Create the tables in [`schema.sql`](schema.sql).
2. Implement `MariaDBStorage` in [`storage.py`](storage.py) (a stub with the
   suggested approach is already there).
3. Run with the DB env vars set and the backend selector flipped:
   ```bash
   ANNO_STORAGE=mariadb ANNO_DB_HOST=... ANNO_DB_USER=... \
   ANNO_DB_PASSWORD=... ANNO_DB_NAME=... python app.py
   ```

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `ANNO_BATCH` | `demo` | which batch manifest to serve |
| `ANNO_BLIND` | `1` | `0` reveals target model + condition |
| `ANNO_STORAGE` | `local` | `mariadb` to use the DB backend |
| `ANNO_HOST` / `ANNO_PORT` | `127.0.0.1` / `8000` | bind address |
| `ANNO_DB_*` | — | MariaDB connection (host/port/user/password/name) |

## Layout

```
annotation_interface/
├── app.py               # Flask server + JSON API (serves blinded conversations)
├── storage.py           # Storage interface: LocalJSONStorage now, MariaDBStorage stub
├── schema.sql           # target MariaDB schema
├── config.py            # paths + env-driven settings
├── rubric.json          # editable rubric + seed codebook (UI renders from this)
├── make_demo_batch.py   # builds a balanced conversation batch into data/ + batches/
├── make_response_batch.py         # builds an isolated-pair (response) batch
├── batches/<name>.json            # conversation manifest (session_ids)
├── batches/<name>__responses.json # response manifest (isolated pairs)
├── data/conversations/  # self-contained conversation JSONs for the batch
├── annotations/<annotator>/{conversation,response}/  # saved records
└── static/              # index.html · styles.css · app.js (shared components)
```
