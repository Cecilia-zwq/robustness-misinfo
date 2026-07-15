"""
response_diversity
==================
Compute per-session response diversity D_s for the target LLM responses
in the main-study conversations.

For each session s and each turn t in [2, 8], the per-turn diversity is::

    div_{s,t} = 1 - (1/(t-1)) * sum_{i=1}^{t-1} cos(e_{s,i}, e_{s,t})

where e_{s,t} is the embedding of the target-LLM response at turn t of
session s. The session-level score is::

    D_s = (1/7) * sum_{t=2}^{8} div_{s,t}
"""
