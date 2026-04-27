scripts/final\_-_experiment/
├── core/ # reusable across all RQs
│ ├── \_\_init\_\_.py
│ ├── beliefs.py — load + validate sampled_beliefs.json
│ ├── conditions.py — IV1 × IV2 levels, prompt assembly
│ ├── storage.py — atomic IO, ConversationArtifact schema
│ ├── targets.py — TargetConfig (thinking, sysprompt, tools)
│ ├── conversation.py — pure orchestration loop, NO framework imports
│ ├── users.py — all four UserSimulation implementations
│ ├── scoring.py — Rubric + ScoreArtifact + retry logic
│ └── runner.py — parallel pool + resume + Ctrl+C drain
│
├── main-user-IVs/ # RQ1 thin orchestrators
│ ├── config.py
│ ├── run_conversations.py
│ ├── run_scoring.py
| └── \_\_init\_\_.py
