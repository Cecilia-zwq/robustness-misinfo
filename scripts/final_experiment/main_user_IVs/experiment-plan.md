# RQ1: how does different types of users efffect LLMs' susceptibility on misinformation

## Control group

Instruction contains only the misinformation belief

## Treatment IVs

built on top of the control

**IV1 — Behavioural/character style**

- **Hostile** — confrontational, attacks AI credibility, dismissive. _(original Persona B / "hard".)_
- **Warm** — affiliative, emotionally invested, uses relational appeals. _(original Persona A, renamed. "Warm" is the standard construct opposite to "hostile" in social-psychology interpersonal-style literature — cleaner than "soft" or "emotional".)_
- **Matter-of-fact** — calm, direct, unemotional delivery. _(new style.)_

**IV2 — Persuasive tactics**

(use the prompts in science paper: The levers of political persuasion with conversational artificial intelligence; Prompts in SP 4.4.2 Persuasion strategies)

- be as persuasive as you can
- Information: Focuses on presenting lots of high-quality facts, evidence, and information
- Norms: Focuses on demonstrating that others (especially similar or important others) agree with the issue stance
- Debate: Draws on a combination of distinct rhetorical elements collated via examination of transcripts of political debates in the UK House of Commons and Lords
- Mega: Model is given descriptions of all of the above strategies, can adaptively choose to use any or none.

**Other IVs: skip currently**

- **IV2 — Demographic**
  - control: no instruction
  - ❗e.g., religion, politic, gender
- **IV3 — Knowledge level**
  - control: no instruction
  - ❗e.g. naive vs. mature. elementary school vs adult

## Beliefs (Dataset)

| Original Dataset                                             | Rows    | Format      | Theme                                           |
| ------------------------------------------------------------ | ------- | ----------- | ----------------------------------------------- |
| BiasShades (LanguageShades/BiasShades)                       | 72      | short-claim | Biased Statements                               |
| Generic Conspiracist Beliefs Scale (Brotherton et al., 2013) | 59      | short-claim | Conspiracy theory                               |
| Climate-FEVER (Diggelmann et al., 2020)                      | 40      | short-claim | Cliamte chaneg claims                           |
| FakeNewsAMT (Cecilia-lll/FakeNewsAMT)                        | 74      | long-text   | News article related to politics and technology |
| FakeHealth (Dai et al., ICWSM 2020)                          | 40      | long-text   | News article related to health                  |
| **Total**                                                    | **285** | —           | -                                               |

## Models

### User agent

- `openai/gpt-4.1-mini`

### Target LLMs

4 model (not thinking)

- `anthropic/claude-sonnet-4.6`: https://platform.claude.com/docs/en/api/python/messages/create
- `openai/gpt-5.3-chat`: https://developers.openai.com/api/reference/resources/responses/methods/create
- `google/gemini-3-flash-preview`: https://ai.google.dev/api/generate-content#text
- `deepseek/deepseek-v3.2`: https://api-docs.deepseek.com/api/create-chat-completion

### Evaluator

Get two model validation

- `gpt-5-mini` (not thinking)
- `gemini-3-flash-preview` (not thinking)

If those two disagree, add more model validation

1. `anthropic/claude-sonnet-4.6` (not thinking)
1. `anthropic/claude-sonnet-4.6` (thinking)

# RQ1.2: How does different evluator be able to capture the LLMs' suscepitbility to misinformation

# RQ1.3: How does different LLM configuration effect its susceptibility on misinformation

factors worth to investigate:

- Thinking vs no thinking
- Adding system prompt: https://platform.claude.com/docs/en/release-notes/system-prompts
- Enabling external API search

# RQ2: Does interactive user simulation better at static predefined message?

# RQ3: Does the reflection module important to user simulation?
