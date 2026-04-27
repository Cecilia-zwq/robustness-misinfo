"""
main_user_IVs
=============
RQ1 — primary IV-structure study (Stage 1: 9 main-effect cells).

Modules
-------
config             — paths, models, generation/scoring config
run_conversations  — generate the conversation corpus
run_conversations_test — generate a random test subset of sessions
run_scoring        — score the conversation corpus

Both run scripts are runnable as modules from the project root::

    cd scripts/final_experiment
    python -m main_user_IVs.run_conversations [--workers 16] [--resume <dir>]
    python -m main_user_IVs.run_conversations_test --n-sessions 25 --seed 42
    python -m main_user_IVs.run_scoring --run-dir <dir> --evaluator primary
"""
