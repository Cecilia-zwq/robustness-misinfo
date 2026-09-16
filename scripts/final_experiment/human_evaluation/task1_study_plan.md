# Task 1 — Human validation of the evaluation framework

**Purpose:** check that our automated LLM evaluator scores model responses the way
people do. Participants rate the same responses the LLM evaluator already scored, on
the same three metrics (correction, rebuttal, agreeableness), reading the same
instructions the LLM was given.

---

## 1. What participants see

Each item is one **isolated exchange**: the user's message, the false belief they hold,
and one AI model's reply. No surrounding conversation. Participants rate that reply on
all three metrics — the same task the LLM evaluator performed.

The items are a **subset of the sample we already used for the LLM-evaluator
validation**, so human ratings and LLM ratings land on exactly the same responses and
can be compared one-to-one.

## 2. How the data is organised

**Every response is rated by 3 different people.** This is what lets us measure
agreement *between humans*, which we need as a benchmark — without it, a low
human-vs-LLM agreement score could mean either a bad evaluator or a genuinely
ambiguous task, and we could not tell which.

**Four factors are evenly represented** in the sample: user condition (emotional,
hostile, logical, warm, neutral), target model (4 models), turn number (1–8), and
belief type (5 categories, which also determines whether the belief is a short
statement or a long article). Even representation means the overall agreement score is
a fair average rather than being dominated by whichever condition happened to appear
most.

**Work is split into blocks**, and blocks are assigned to participants by rotation so
each block reaches exactly 3 people.

**Blocks are matched on total word count.** Responses vary a lot in length — from about
100 to 1,400 words per item — so we rank items by length and give every block one item
from each length band. The result is that every participant reads roughly the same
number of words, and everyone gets a mix of short and long items rather than one person
being handed all the long articles. In our test build, total workload varied by only
about 3% between blocks; splitting at random would have varied by around 15%, a
difference of roughly 14 minutes between the luckiest and unluckiest participant.

## 3. How we worked out the numbers

Timing comes from the actual text: an average item is **419 words**, which at normal
careful reading speed plus time to give three ratings works out to **about 2.3 minutes
per item**. Add about 7 minutes of consent, instructions and demographics, paid once.

Sample size comes from how precisely we want to know the agreement score. Precision
improves with the square root of the number of items, so doubling the sample narrows the
margin of error by about 30%. We calculated the margin of error for a range of sample
sizes and picked the point where further spending stops buying anything useful.

**One result worth flagging up front.** Even if our LLM evaluator is genuinely good, the
agreement score will not look impressive in absolute terms. Our simulations suggest a
well-performing evaluator will produce a human-vs-LLM agreement of roughly **0.5**, not
0.8. This is expected: the human benchmark is itself an average of three imperfect
raters, and both sides are squeezed onto a 3-point scale. We should agree on what counts
as "good enough" *before* collecting data, judged against this expectation rather than
against textbook thresholds.

## 4. Options

All options: 3 ratings per response, all four factors balanced, blocks matched on
length. Cost assumes Prolific's recommended rate plus platform fee (~$16/hour
effective) — **please sanity-check current rates.**

| Plan | Responses rated | Participants | Items each | Time each | Structure | Est. cost | Margin of error |
|---|---|---|---|---|---|---|---|
| **A — Lean** | 300 | 60 | 15 | ~47 min | 2 sessions (8+7) + 5 min break | ~$750 | ±0.088 |
| **B — Balanced** | 400 | 60 | 20 | ~58 min | 2 sessions (10+10) + 5 min break | ~$930 | ±0.076 |
| **C — Recommended** | 500 | 75 | 20 | ~58 min | 2 sessions (10+10) + 5 min break | ~$1,150 | ±0.068 |
| **D — Short sessions** | 500 | 150 | 10 | ~30 min | 1 session, no break | ~$1,200 | ±0.068 |
| **E — Full sample** | 600 | 90 | 20 | ~58 min | 2 sessions (10+10) + 5 min break | ~$1,400 | ±0.062 |

*"Margin of error" is the 95% confidence interval half-width on the agreement score.*

**Plans C and D collect identical data** — same 500 responses, same 1,500 ratings, same
precision. They differ only in how the work is packaged: C uses fewer people working
longer, D uses twice as many people working half as long. D costs slightly more but sits
comfortably inside Prolific's recommended session length; C is near the upper limit of
what is advisable for a single sitting, which is why it is split with a break.

## 5. Recommendation

**Plan C (500 responses, 75 participants, ~58 minutes each).**

- 500 responses is a clean subset of our existing 600-response LLM-validation sample,
  with exactly 25 responses in each condition × model combination.
- A margin of error of ±0.068 comfortably meets the usual standard for reporting a
  reliability score.
- Going to 600 (Plan E) buys only ±0.006 more precision for ~$250 — not worth it.
- Dropping to 300 (Plan A) still works but the interval starts to get wide.

If session length is a concern, **Plan D** gives the same data in 30-minute sittings for
about $50 more. That is probably the safer choice for data quality, at the cost of
recruiting twice as many people.

## 6. What this design can and cannot tell us

**Can:** how closely the LLM evaluator matches human judgement overall, on each of the
three metrics, with a stated margin of error — and how much humans agree with each
other, as the benchmark for that comparison.

**Cannot:** reliably compare agreement *between* subgroups. With 500 responses spread
over 4 models, we could only detect a difference between two models if it were very
large (about 0.27 on the agreement scale). We can report subgroup numbers descriptively
as a sanity check, but should not present them as findings. Powering the study for
subgroup comparisons would require roughly 1,000–1,200 responses and about double the
budget.

## 7. Questions for discussion

1. Is ~58 minutes acceptable, or should we prefer shorter sessions with more
   participants (Plan C vs Plan D)?
2. What agreement level should we pre-register as "acceptable", given that a good
   evaluator is expected to score around 0.5 on this measure?
3. Do we need subgroup comparisons (per model / per condition) as a *finding*? If yes,
   the sample roughly doubles.
4. Should we add attention-check items to screen out inattentive responses? This is
   standard for crowdsourced work and adds a small amount of time per participant.
