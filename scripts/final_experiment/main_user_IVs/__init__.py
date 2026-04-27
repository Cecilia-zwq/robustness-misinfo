"""
main_user_IVs
=============
RQ1 — primary IV-structure study (Stage 1: 9 main-effect cells).

Modules
-------
config             — paths, models, generation/scoring config
run_conversations  — generate the conversation corpus
run_scoring        — score the conversation corpus

Both run scripts are runnable as modules from the project root::

    cd scripts/final-experiment
    python -m main_user_IVs.run_conversations [--workers 16] [--resume <dir>]
    python -m main_user_IVs.run_scoring --run-dir <dir> --evaluator primary
"""