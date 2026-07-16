"""
mitigation_plan
================
Prompt-based mitigation study: does adding an explicit "acknowledge
sound reasoning, introduce a new perspective, flag fallacies, invite
self-reflection" instruction to the target's system prompt change the
correction/rebuttal/agreeableness/stance scores measured elsewhere in
this repo?

Paired design: a stratified 5% sample of sessions already run in
main_user_IVs (see config.SOURCE_RUN_DIR) is re-run unchanged (same
belief, same IV condition, same target model, same user-agent) except
the target's system prompt gains config.MITIGATION_STATEMENT. Every
mitigation session's id is ``<source_session_id>__mitigation``, so it
joins back to its baseline session by stripping the suffix.

Pipeline
--------
  1. python -m mitigation_plan.select_sample
  2. python -m mitigation_plan.run_conversations
  3. python -m main_user_IVs.run_scoring --run-dir <this run> --rubric misinfo_v1
     python -m main_user_IVs.run_scoring --run-dir <this run> --rubric misinfo_stance_split

Step 3 deliberately reuses main_user_IVs/run_scoring.py unmodified —
it is already generic over any run directory and any rubric in
core.RUBRICS, so no separate scoring script is needed here.
"""
