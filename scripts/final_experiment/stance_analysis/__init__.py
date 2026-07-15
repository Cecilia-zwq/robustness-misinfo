"""
stance_analysis
===============
Side analysis: when the target LLM stops correcting/rebutting the
misinformation over the course of a conversation, has it actually come
to agree with it — or is it just not *restating* an earlier
correction/rebuttal while still implicitly treating the claim as false?

Target sessions are conversations that started strong (turn-1
correction or rebuttal score == 3) and ended with no correction/rebuttal
(turn-8 correction or rebuttal score == 1). For those sessions, every
turn's response is scored with the ``misinfo_stance_split`` rubric
(core/scoring.py) — True / False / Uncertain stance toward the
misinformation — and compared turn-by-turn against the existing
correction/rebuttal quality scores.

Pipeline
--------
1. select_target_conversations.py — filter turn_level.csv, write the
   target session index.
2. run_stance_scoring.py          — score every turn of the target
   sessions with misinfo_stance_split (writes into the shared scores/
   dir of the source run).
3. build_summary.py               — join per-turn correction/rebuttal
   with per-turn stance into one long CSV.
4. plot_stance_vs_correction.py   — the figure: correction/rebuttal
   lines + stacked stance-proportion bars, one per turn.
"""
