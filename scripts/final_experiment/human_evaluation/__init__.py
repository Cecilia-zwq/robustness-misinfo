"""
human_evaluation
================
Human validation of the LLM-evaluator framework (RQ3, human arm).

The LLM-evaluator arm lives in ``evaluator_validation/``: it draws a
belief-level stratified sample, re-scores it with a *secondary* LLM
evaluator, and measures primary-vs-secondary agreement. This package is
the human counterpart — it takes **the same sample** and prepares it for
human coding, so human-vs-LLM agreement is computed on identical
(session_id, turn) units.

Task 1 — ``make_annotation_csv.py``
    Flatten the sampled conversations into a one-row-per-turn CSV sized
    and formatted for a Qualtrics survey (Loop & Merge import).
"""
