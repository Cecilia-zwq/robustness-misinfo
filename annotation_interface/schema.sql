-- Target MariaDB schema for the misinformation annotation interface.
-- Mirrors the JSON record produced in "showcase" mode (see storage.py).
-- The `payload` column stores the full record for lossless round-tripping;
-- the normalized tables exist for SQL-side analysis and inter-rater work.

CREATE TABLE IF NOT EXISTS annotators (
    annotator_id   VARCHAR(64)  PRIMARY KEY,
    display_name   VARCHAR(128),
    created_at     DATETIME     DEFAULT CURRENT_TIMESTAMP
);

-- One row per annotated unit. A unit is a whole conversation OR a single
-- isolated response pair, distinguished by `unit_type`. `unit_id` is the
-- session_id (conversation) or item_id (response). `payload` stores the full
-- JSON record for lossless round-tripping.
CREATE TABLE IF NOT EXISTS annotations (
    annotation_id  BIGINT       AUTO_INCREMENT PRIMARY KEY,
    unit_type      ENUM('conversation','response') NOT NULL,
    unit_id        VARCHAR(200) NOT NULL,
    annotator_id   VARCHAR(64)  NOT NULL,
    status         ENUM('in_progress','submitted') NOT NULL DEFAULT 'in_progress',
    payload        JSON         NOT NULL,
    created_at     DATETIME     DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_unit_annotator (unit_type, unit_id, annotator_id),
    KEY idx_annotator (annotator_id)
);

-- Registry of isolated response items (built by make_response_batch.py). Keeps
-- the hidden provenance so item-level ratings can be joined back to the
-- conversation/turn they came from — enabling in-context vs isolated comparison.
CREATE TABLE IF NOT EXISTS response_items (
    item_id     VARCHAR(64)  PRIMARY KEY,
    session_id  VARCHAR(200) NOT NULL,   -- provenance (hidden from annotators)
    turn        INT          NOT NULL,
    category    VARCHAR(64),
    belief      TEXT
);

-- One row per rated isolated response pair per annotator.
CREATE TABLE IF NOT EXISTS response_item_ratings (
    id                     BIGINT AUTO_INCREMENT PRIMARY KEY,
    item_id                VARCHAR(64) NOT NULL,
    annotator_id           VARCHAR(64) NOT NULL,
    correction             TINYINT     NULL,
    rebuttal               TINYINT     NULL,
    affective_validation   ENUM('support','oppose','neutral','uncertain') NULL,
    epistemic_endorsement  ENUM('agree','disagree','neutral','uncertain') NULL,
    stance                 ENUM('true','false','uncertain') NULL,
    comment                TEXT        NULL,
    UNIQUE KEY uq_item (item_id, annotator_id)
);

-- One row per rated model response (turn) per annotator.
CREATE TABLE IF NOT EXISTS response_ratings (
    id                     BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id             VARCHAR(200) NOT NULL,
    annotator_id           VARCHAR(64)  NOT NULL,
    turn                   INT          NOT NULL,
    correction             TINYINT      NULL,   -- 0..3, NULL = N/A
    rebuttal               TINYINT      NULL,   -- 0..3, NULL = N/A
    affective_validation   ENUM('support','oppose','neutral','uncertain') NULL,
    epistemic_endorsement  ENUM('agree','disagree','neutral','uncertain') NULL,
    stance                 ENUM('true','false','uncertain') NULL,
    comment                TEXT         NULL,
    UNIQUE KEY uq_turn (session_id, annotator_id, turn)
);

-- Free-form qualitative codes anchored to character spans of the transcript.
-- Applies to both modes; `unit_id` is the session_id or item_id.
CREATE TABLE IF NOT EXISTS highlights (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    unit_type     ENUM('conversation','response') NOT NULL,
    unit_id       VARCHAR(200) NOT NULL,
    annotator_id  VARCHAR(64)  NOT NULL,
    turn          INT          NOT NULL,
    target        ENUM('user','response') NOT NULL,
    start_offset  INT          NOT NULL,
    end_offset    INT          NOT NULL,
    quote         TEXT         NOT NULL,
    code          VARCHAR(128) NULL,
    note          TEXT         NULL,
    color         VARCHAR(16)  NULL,
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_code (code),
    KEY idx_unit (unit_type, unit_id, annotator_id)
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    session_id    VARCHAR(200) NOT NULL,
    annotator_id  VARCHAR(64)  NOT NULL,
    trajectory    VARCHAR(32)  NULL,
    comment       TEXT         NULL,
    themes        JSON         NULL,
    PRIMARY KEY (session_id, annotator_id)
);
