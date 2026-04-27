scripts/final_experiment/
├── core/  # reusable across all RQs
│ ├── __init__.py
│ ├── beliefs.py        # load + validate sampled_beliefs.json
│ ├── conditions.py     # IV1 × IV2 levels, prompt assembly
│ ├── storage.py        # run-path layout, artifact schema, atomic IO
│ ├── targets.py        # TargetConfig (thinking, system prompt, tools)
│ ├── conversation.py   # conversation loop with pluggable user/target
│ ├── users.py          # UserSimulation implementations (RQ1/RQ2/RQ3)
│ ├── scoring.py        # Rubric + ScoreArtifact + parse-retry scoring
│ └── runner.py         # parallel pool + resume + checkpointing
│
├── main_user_IVs/  # RQ1 thin orchestrators
│ ├── __init__.py
│ ├── config.py
│ ├── run_conversations.py
│ ├── run_conversations_test.py
│ └── run_scoring.py
│
├── misinfo_eval_framework/  # Iteration-5 framework package kept for reuse
│ ├── __init__.py
│ ├── defaults.py
│ ├── llm_utils.py
│ ├── user_agent.py
│ ├── target_llm.py
│ └── evaluator.py
│
└── sample_beliefs.py  # builds data/dataset/sampled_beliefs.json
