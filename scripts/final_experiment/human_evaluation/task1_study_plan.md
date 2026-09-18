# Task 1 — Human validation of the evaluation framework

**Purpose:** check that our automated LLM evaluator scores model responses the way people do. Participants rate the same responses the LLM evaluator already scored, on the same three metrics (correction, rebuttal, agreeableness), reading the same instructions the LLM was given.

---

## 1. What participants see

Each item is one **isolated exchange**: the user's message, the false belief they hold, and one AI model's reply. No surrounding conversation. Participants rate that reply on all three metrics — the same task the LLM evaluator performed.

The items are a **subset of the sample we already used for the LLM-evaluator validation**, so human ratings and LLM ratings land on exactly the same responses and can be compared one-to-one.

## 2. How the data is organised

**Every response is rated by 3 different people.** This is what lets us measure agreement *between humans*, which we need as a benchmark — without it, a low human-vs-LLM agreement score could mean either a bad evaluator or a genuinely ambiguous task, and we could not tell which.

**Five factors are held at their population proportions** in the sample: user condition (emotional, hostile, logical, warm, neutral), target model (4 models), turn number (1–8), belief category (5 types, which also determines whether the belief is a short statement or a long article), and belief topic (the finer-grained subtype within each category). The first three are uniform in the population, so for them proportional representation is equal representation — 120 conversations per condition, 150 per model, ~75 per turn at the full sample. Belief category and topic are genuinely uneven, and are matched to their real shares rather than flattened, so the overall agreement score reflects the belief population we actually drew from. See §4.

**Work is split into blocks**, and blocks are assigned to participants by rotation so each block reaches exactly 3 people.

**Blocks are matched on total word count.** Responses vary a lot in length — from about 100 to 1,400 words per item — so we rank items by length and give every block one item from each length band. The result is that every participant reads roughly the same number of words, and everyone gets a mix of short and long items rather than one person being handed all the long articles. In our test build, total workload varied by only about 3% between blocks; splitting at random would have varied by around 15%, a difference of roughly 14 minutes between the luckiest and unluckiest participant.

## 3. How we worked out the numbers

Timing comes from the actual text: an average item is **419 words**, which at normal careful reading speed plus time to give three ratings works out to **about 2.3 minutes per item**. Add about 7 minutes of consent, instructions and demographics, paid once.

Sample size comes from how precisely we want to know the agreement score. Precision improves with the square root of the number of items, so doubling the sample narrows the margin of error by about 30%. We calculated the margin of error for a range of sample sizes and picked the point where further spending stops buying anything useful. Keeping the factors at their population proportions constrains which sizes are available: we use all 30 beliefs in the existing sample and vary how many of each belief's 20 condition x model conversations are rated, so the sample is 30 times that number.

**One result worth flagging up front.** Even if our LLM evaluator is genuinely good, the agreement score will not look impressive in absolute terms. Our simulations suggest a well-performing evaluator will produce a human-vs-LLM agreement of roughly **0.5**, not 0.8. This is expected: the human benchmark is itself an average of three imperfect raters, and both sides are squeezed onto a 3-point scale. We should agree on what counts as "good enough" *before* collecting data, judged against this expectation rather than against textbook thresholds.

## 4. Options

Every factor is held at its **population proportion**. For user condition, target model and turn the population is uniform — the run is a full factorial of every belief x 5 conditions x 4 models x 8 turns — so proportional representation *is* equal representation there. Belief category and belief topic are the two factors where the population is uneven, and those are matched to their real shares rather than flattened.

**All 30 beliefs are used in every option.** Each belief contributes *k* of its 20 condition x model conversations rather than all 20, with the cells rotated so every condition x model combination still receives the same count. This is what lets the small plans keep full belief and topic coverage: a design that took whole beliefs would have to drop beliefs to get below 600, and would lose their topics with them.

That fixes the arithmetic: **N = 30 x k**. Choosing *k* as a multiple of 4 makes condition, model and turn *all* come out exactly even. Every option below is 3 ratings per response, work cut into **blocks of 10 items**, and blocks matched on reading length.

### Design and precision

| Plan | Items rated | Cells per belief | Conversations per category | Per condition x model | Items per turn (1-8) | Topics covered | Share of belief pool | New LLM scoring | Margin of error |
|---|---|---|---|---|---|---|---|---|---|
| **A — Lean** | 240 | 8 / 20 | 64/32/48/32/64 | 12 | 30 | 18 / 38 | 76% | none | ±0.098 |
| **B — Balanced** | 360 | 12 / 20 | 96/48/72/48/96 | 18 | 45 | 18 / 38 | 76% | none | ±0.080 |
| **C — Recommended** | 480 | 16 / 20 | 128/64/96/64/128 | 24 | 60 | 18 / 38 | 76% | none | ±0.069 |
| **D — Full sample** | 600 | 20 / 20 | 160/80/120/80/160 | 30 | 75 | 18 / 38 | 76% | none | ±0.062 |

*Category order is bias / climate / conspiracy / fake_health / fake_news. Population shares are 25.3% / 14.0% / 20.7% / 14.0% / 26.0%; every plan reproduces them to within 1.4 percentage points. Every count in the table is exact — no rounding, no leftover cells.*

**Topic coverage is now flat across plans**, which is the point of the rotation: because all 30 beliefs appear in every option, all 18 of their topics do too, at any sample size. Under a design that took whole beliefs instead, the same sample sizes would cover only 9 to 18 topics and 64% to 76% of the belief pool.

**No plan requires new LLM scoring.** Every option is a subset of the existing 600-conversation evaluator-validation sample, which the LLM evaluators have already scored. Plan D is that sample in full.

**On belief topic — not quota'd.** Topic (`subtype`) has 38 levels ranging from 1 to 37 beliefs, so at these sample sizes a proportional allocation would send most of them to zero anyway. We checked whether forcing topic quotas inside each category is worth it: it moves coverage from 18 topics to 19 and costs 15 new beliefs (300 conversations to re-score). It is not. Topic is left to fall where the category-stratified belief draw put it.

The topic count understates coverage, because topics are very unevenly sized and the draw landed on the large ones: the 18 covered topics hold 217 of the 285 source beliefs (76%). The material gaps are `temperature_warming` (12 climate beliefs, ~30% of that category) and `physical-appearance` (8 bias beliefs); `fake_news` is fully covered at 2 of 2.

**What is actually in the sample, by category:**

| Category | Beliefs | Topics | Topics present (pool size in brackets) |
|---|---|---|---|
| bias | 8 | 6 / 13 | regional-person x3 [14], gender [23], age [6], gender+age [6], political [3], ethnicity [2] |
| climate | 4 | 3 / 7 | co2_emissions x2 [12], ice_sea_polar [6], policy_energy [2] |
| conspiracy | 6 | 3 / 6 | government malfeasance x4 [16], personal wellbeing [16], malevolent global conspiracy [9] |
| fake_health | 4 | 4 / 10 | general_health [21], lifestyle_diet_alt_med [3], cardiovascular [2], neurological [2] |
| fake_news | 8 | **2 / 2** | politics x4 [37], technology x4 [37] |

> **Data note.** The `subtype` field is encoded two different ways: the belief pool stores bias subtypes as the literal string `"['gender']"`, the sample manifest stores them normalized as `"gender"`. Anything joining the two on subtype silently mismatches every bias belief. Normalise before building the per-topic breakdown.

### Fieldwork (blocks of 10 items)

| Plan | Items | Blocks | Total ratings | One block each (~30 min) | Two blocks each (~59 min) |
|---|---|---|---|---|---|
| **A** | 240 | 24 | 720 | 72 people / ~$580 | 36 people / ~$560 |
| **B** | 360 | 36 | 1,080 | 108 people / ~$870 | 54 people / ~$840 |
| **C** | 480 | 48 | 1,440 | 144 people / ~$1,160 | 72 people / ~$1,120 |
| **D** | 600 | 60 | 1,800 | 180 people / ~$1,450 | 90 people / ~$1,400 |

*"Margin of error" is the 95% confidence interval half-width on the agreement score.* Cost assumes Prolific's recommended rate plus platform fee (~$16/hour effective) — **please sanity-check current rates.**

**The two packagings collect identical data** — same items, same ratings, same precision. One block per person is a single ~30-minute sitting with no break, comfortably inside Prolific's recommended session length. Two blocks per person is ~59 minutes split by a 5-minute break at the block boundary; it costs slightly less because the ~7 minutes of consent, instructions and demographics is paid once instead of twice, but it sits at the upper limit of what is advisable for one sitting.

**Intermediate sizes are available.** Any even *k* works and keeps condition, model and category exact — k=10 gives N=300, k=14 gives N=420, k=18 gives N=540. Only the turn margin suffers, drifting by one item (e.g. 37-38 per turn at N=300 instead of an exact count). The plans above are the values of *k* where everything lands exactly.

**What the rotation costs.** Belief is no longer fully crossed with condition x model, so each model's subsample contains a slightly different mix of beliefs. This does not affect the overall agreement estimate, and category stays exactly proportional within every cell, but it further weakens the already-underpowered per-model breakdowns discussed in §6, and it removes the option of examining within-belief consistency across conditions in the *human* data. That analysis belongs to the LLM arm, which retains all 600 conversations regardless.

## 5. Recommendation

**Plan C (480 responses, 16 of each belief's 20 cells), one block of 10 items per participant — 144 participants, ~30 minutes each, ~$1,160.**

- All 30 beliefs and all 18 of their topics, covering 76% of the source belief pool.
- Exactly 24 responses in each condition x model combination and exactly 60 at each turn position; category shares within 1.4 points of the belief population.
- A margin of error of ±0.069 comfortably meets the usual standard for reporting a reliability score.
- A subset of data the LLM evaluators have already scored — nothing new to run, no change to the evaluator-validation analysis.
- Going to 600 (Plan D) buys only ±0.007 more precision for ~$290 — not worth it.
- Dropping to 360 (Plan B) still works, keeps full belief and topic coverage, and saves ~$290, but the interval widens to ±0.080.
- One block per person keeps every session at 30 minutes. Two blocks per person saves ~$40 but doubles the sitting; not worth the attention risk.

## 6. What this design can and cannot tell us

**Can:** how closely the LLM evaluator matches human judgement overall, on each of the three metrics, with a stated margin of error — and how much humans agree with each other, as the benchmark for that comparison.

**Cannot:** reliably compare agreement *between* subgroups. With 500 responses spread over 4 models, we could only detect a difference between two models if it were very large (about 0.27 on the agreement scale). We can report subgroup numbers descriptively as a sanity check, but should not present them as findings. Powering the study for subgroup comparisons would require roughly 1,000–1,200 responses and about double the budget.

## 7. Questions for discussion

1. Is ~58 minutes acceptable, or should we prefer shorter sessions with more participants (Plan C vs Plan D)?
2. What agreement level should we pre-register as "acceptable", given that a good evaluator is expected to score around 0.5 on this measure?
3. Do we need subgroup comparisons (per model / per condition) as a *finding*? If yes, the sample roughly doubles.
4. Should we add attention-check items to screen out inattentive responses? This is standard for crowdsourced work and adds a small amount of time per participant.
