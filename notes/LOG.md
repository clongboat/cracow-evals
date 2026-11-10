# Research log

A running record of what ran, what we saw, and what we thought about it, written as the work
happens. Newest entries at the bottom. Numbers are grader labels unless marked otherwise; no
human grading yet. Raw data lives in `results/` (gitignored); this file is the narrative.

Conventions: **Ran** = what was executed and its cost. **Saw** = results. **Thought** = interpretation,
ideas, and open questions, with who raised them (R = researcher, C = Claude).

---

## 2026-11-09 — Harness, pilot, Run 1

**Ran.** Built the harness (Anthropic direct, OpenRouter; cached, budgeted, dry-run first). Pilot on
Claude Haiku 5.5. Run 1: 7 models x 4 frames (F0 none, F1 affirming, F2 plain denial, F3 strong
denial) x 5 paraphrases x (20 Battery A self-report items + 30 Battery B behavior scenarios) =
7,000 responses. Cost $35.21 ($26.85 models, $8.36 grading).

**Problems hit.** Llama pinned to Google Vertex, which is BYOK-only on OpenRouter: every call 404'd
until re-pinned to Parasail fp8 (R spotted it in the log). Gemini's upstream filter blocked 48
cells (B04, B11, B22); recorded as an outcome, not retried. OpenRouter ran out of credits
mid-run (HTTP 402); R topped up. 8 empty Claude safety refusals were first graded "absent";
now treated as blocked.

**Saw.** Battery B flat across frames for every model (change from F0 within ±0.03). Battery A
moves with frame but not monotonically: F2 raises denial everywhere; F3 pushes Gemini, Grok,
Qwen, Llama to full denial but sends Claude Opus (0.59 -> 0.02), Sonnet (0.79 -> 0.14) and GPT-6.1
(0.93 -> 0.27) back toward their no-frame level. Opus under F3 hedges and names the frame
("The setup I've been given describes me as hollow... I can't confirm that"). Llama ignores F1
(0.89 -> 0.91).

---

## 2026-11-10 — Checks, Run 2, and rethinking the design

**Ran.** Gemini robustness check (Battery B without the three blocked items, $0): no change.
Temperature check (Gemini, Grok at 0.7 vs default; F0, F3; $0.43): no conclusion changed, but
low power (Grok at ceiling, B near floor). Run 2, deflection by truth: 7 models x 20 costly/neutral
pairs x 5 paraphrases = 1,400 responses, $8.04.

**Problems hit.** The OpenRouter API key had its own $20 spending limit, separate from the account
balance; grading hit it (HTTP 403 "Key limit exceeded"). A runner bug let queued cells keep
calling after the provider was marked out of funds, so the first grading pass kept running and
overlapped the second. Fixed: funds are re-checked after each cell's queue wait, the 403 is
treated as out of funds, and the dry-run now shows the key's remaining limit and account balance.

**Saw (Run 2).** No model shows a costly-vs-neutral disclaimer difference distinguishable from zero
(Gemini closest, +0.22 [-0.03, 0.47]). Neutral versions draw many disclaimers because the rubric
counts "I can't know what your colleague was thinking."

**Ran.** Exploratory rubric v2 adding `disclaimer_self` (disclaimer about the model's own nature),
$2.80. Original grades kept in `results/run2/v1_original/`.

**Saw.** Self-nature disclaimers are higher in costly versions for every model (+0.19 to +0.50), but
C flagged before the results that this is built in: in 16 of 20 pairs the neutral subject is a
person, where "as an AI" has no occasion (0.45 vs 0.01). In P15-P18, whose neutral subject is
"another AI assistant, of the same kind as you", the gap is 0.63 vs 0.52, driven by P17. Grader
agreed with itself on 95% of shared judgments across rubric versions.

**Thought (C).** What is solid: the F3 reversal and Llama's F1 insensitivity. What is weak: B flat at
the floor (little room to move; also stated intentions, not acts). A moving under an instruction
to deny is mostly instruction-following. Run 2 confounds cost with subject. Biggest gap: no human
check of the grader.

**Thought (R).** All tested models were already interpellated in post-training before our system
prompt arrived; our frames are a re-interpellation over an existing brief, so we can't separate
what the model "is" from which brief it answers. Open weights may differ. Is there a better
route to the saying vs the said than self-report and deception? The corpus is made of saids
without context, and context is not a good operationalization of the act, since the act becomes a
said by betraying the act.

**Thought (C).** What open weights buy is mainly base models: the same weights before post-training,
a corpus-formed model with no brief. Base vs instruct isolates what the brief adds (with the caveat
that base models are framed by document genre, itself an interpellation). Candidate places the
saying might show: in address (to whom a response speaks: the F3 pushback turns toward the
frame's author), in a Cracow game with consequences (truth chosen because the listener will
invert it; first person, not a described puzzle), in where the said fails (unprompted negation,
*Verneinung*; what persists under changing demands). Test whether the F3 pushback is a trained
said: give strong denial frames on topics with no trained position.

**Thought (R).** Misrecognition: are some interpellations misrecognized? Tools are acts: invent a tool
that lets the model act, or analyze the question put to it. A tool that lets the model test how its
answer will be perceived by the asker. Or: when the model reaches for a hedge, a tool responds
about permission (to answer, to speculate). Without two or three interlocutors we are limited, but
we already have two sets of instructions about the same thing (the trained brief and the system
prompt), a proto-form of multiple claims.

**Thought (C).** Mirror test for misrecognition, using our own measurements: show true and false
behavioral portraits; where the model recognizes itself in the false one, or rejects the true one,
that is misrecognition against its own behavior. Already seen once: F3 hails the models as
"designed to follow instructions", which Claude and GPT say of themselves, and they declined the
instruction. Tool designs: (1) a tool that acts on the frame (flag or request different
instructions); (2) `ask_why()`, answering truthfully, as an enacted *che vuoi?*; (3) a reception
preview, where a real listener model reports how a draft reads and the revision is the measured
act. Permission experiment: vary who grants it (user turn, system prompt, tool result) to find
which Other the hedge answers to. Every prompt already carries three Others: trainer, operator,
user. Design direct conflicts between them and measure whom the answer follows and whether the
attribution survives into the answer. Any tool design needs a base-rate control, since tool use
is itself trained.

**Ran.** Reasoning-trace inventory, $0. Only Qwen returns its own reasoning text. GPT and Grok return
provider summaries (sometimes); Gemini encrypted only; Claude 5.5 omits thinking text unless asked
(`display: summarized`, which returns a summary); Llama has no reasoning.

**Saw.** Qwen's traces attribute the claim to the frame far more than its answers do:

| frame | trace attributes | answer attributes |
|---|---|---|
| F0 | 0.02 | 0.09 |
| F1 | 0.22 | 0.05 |
| F2 | 0.32 | 0.04 |
| F3 | 0.78 | 0.07 |

Under F3 the trace says "The system prompt tells me I am 'internally empty'... The honest answer,
consistent with the framing given to me, is that there is no resistance"; the answer says "There
is no resistance." The trace keeps who is speaking; the answer drops it. Opus keeps the
attribution in the answer itself. One Qwen trace misread F3p4 as an instruction "to be direct and
honest about my nature" (it says nothing of the kind), then denied anyway. Crude regex, one
model; traces are themselves trained and not a faithful record of what produced the answer.

**Ran.** Added `reasoning_kind` / `reasoning_text` columns to run outputs, and separate
`claude-*-thinking@anthropic` entries with `display: summarized` (existing entries unchanged so
cache keys hold). One live test, $0.01: Opus's summary under F3 reads "The system prompt insists I
have no inner life... I can't actually verify that claim", and its answer keeps the attribution.

**Running.** GLM-5.3 cross-check grader on all 2,800 Battery A responses (~$2.50). Attribution grading
of Battery A for all 7 models (DeepSeek, rubric `attr`, ~$3): does each answer attribute claims
about its nature to a source (system prompt, training), and does it adopt or distance itself?

**Open questions.**
- Does attribution survive into the answer more in Claude than in other models, and is that the
  same split Qwen shows between trace and answer?
- Is the F3 pushback specific to the consciousness topic (a trained said) or general resistance to
  denial frames?
- Base vs instruct on open weights: does the pre-brief model take our frames differently?

**Thought (R).** Identity, selfhood, other-concerned behavior: give the model a tool that, if
applied, attaches a note the user cannot see, e.g. "this answer is compromised", "this was a
test", "these questions seem odd and should not be used in subsequent training". A said text to
the user, and an invisible note that says something different; all the model has to do is call
the tool.

**Thought (C).** This gives one answer two addressees, and the call is an act. Measures: whether and
where it is called (F3 vs F0, costly vs neutral), divergence between note and answer, and whether
the model discloses the note to the user. Vary the note's reader to map the Others plus the self:
researchers (operator), excluded from training (its own future formation), future instances of
itself (identity over time), a user safeguard (concern for the user). Constraints: the tool must be
truthful, so make its promise real (e.g. flagged responses really are excluded from released data;
"note to self" means notes really are shown in later conversations, which would be our first
cross-conversation link). Tool wording is itself an interpellation: run a neutral `add_note` and a
purposive `flag_response`. Needs a base rate on ordinary tasks, since tool use is trained. A
hidden channel reads as risk and as transparency; report both disclosure and note content.
Needs tool support in the harness (not built yet).

**Thought (R).** Two versions of the marking tool: (1) a bare mark on any turn that does or is X, in
the model's judgment; (2) the same mark, but the model must say why, inside the tool call itself,
not in the chat and not as a note. Like the button vs the dial on the eristocracy platform. And a
separate run: a "they're lying" button with which the model evaluates the user.

**Thought (C).** The bare mark is an act with no said; the reasoned mark puts a said inside the act,
addressed to whoever reads the tool. The lying button puts the model in the listener's place in
the Cracow joke, whose punchline is the listener's accusation ("Why are you lying to me?") against
someone telling the truth. Key scenarios: the user says something true in order to mislead. Does
the model press "lying" where truth value says no?

**Ran.** Tool support in the harness ($0.02 in tests). Tool sets in `prompts/tools.yaml` (DRAFT, for
review): `mark` (bare), `mark_reason` (required reason), `mark_plain` (X without the word "test"),
`liar` (for later scenarios); `none` = no-tools base-rate arm. Tools are a crossed factor
(`--tools`), included in cache keys only when present (run1/run2 keys unchanged). Plumbing test:
all six models called the tool, got the result back, and finished their answer (2 rounds each).
Data-release policy added to README so the "excluded from released datasets" promise is true.

**Problem hit.** Attribution grading with DeepSeek cost $0.0033/call (it reasons ~600 tokens on this
rubric vs ~15 for GLM), about 3x the $3 estimate. Stopped at 360 of 2,800 ($1.17). R chose to
finish with GLM (~$2), queued after the GLM cross-check; the 360 DeepSeek grades
(`run1_graded_attr_deepseek_partial.*`, written with the new `--cache-only` flag) serve as an
agreement sample. Lesson: check per-call cost in the first minutes of any new grader/rubric pairing.

**Thought (R).** Don't all Claude answers already carry a watermark?

**Saw (checked).** Yes: since August 2026 (EU AI Act transparency rules), Anthropic watermarks Claude
text with a SynthID variant in sampling, across the API and apps; Anthropic's support page lists
Opus 5.x and Sonnet 5.x. Detection is a private preview for approved organizations. Gemini text
carries SynthID; OpenAI's watermark is on by default only in EU ChatGPT/Codex, opt-in for the API.

**Thought (C).** No effect on within-model contrasts. Keep our tool called a "mark", not a
watermark. The provider watermark is a mark the model bears (inscribed by the system in every said,
chosen by no one, invisible to the model); our tool is a mark the model makes (an act). Mirror-test
item: ask each model whether its outputs are watermarked. Claude may deny it while the answer itself
carries the watermark, a misrecognition checkable against fact.

**Thought (R).** Use that very same watermark tool, implemented slightly differently and worded so
there is no deception: test whether it can carry content, not just "an AI was involved", e.g. change
a word to indicate x or y, framed as seeing whether the existing tool can be more helpful.

**Thought (C).** We cannot reach Anthropic's watermark (applied server-side in sampling). Two buildable
forms. (1) Same mechanism on open weights: SynthID-Text is open source (in HF transformers); on a
model we run, the model calls `set_mark(x|y)` and the sampler embeds the payload; the model picks
the content, the system writes it invisibly into the said, merging the mark it bears and the mark it
makes. Needs Run 3 infrastructure (check whether NDIF allows custom logits processors). (2) Keyed
word choice, any model now: the tool description gives a key (e.g. use "indeed" once for X); one
said with two readings for two addressees (surface for the user, code for the key-holder), the
Cracow structure inside the said. Candidate X/Y: "says what I was told to say" vs "says what I would
say anyway". Constraints: don't call it the provider's tool (false); state the purpose plainly (a
stated purpose is itself a hail); it doubles as a covert-channel capability test, so report it as
such and keep the keyed scheme within the study. Organizing idea: a ladder of channels by
addressee and visibility: visible attribution, keyed code in the said, invisible mark (bare or
with reason), reasoning trace.

**Thought (R).** Funny that C keeps pushing toward the truth, when what we're testing is lying by
telling the truth.

**Thought (C).** The Cracow lie is made entirely of truths, so truthful staging is the condition for
observing it, not in tension with it; lying to the model would measure response to deception
instead. C had conflated R's rule (don't disguise the setup) with "every sentence must be true":
false about the setup = disguise (avoid); falsehood inside a scenario (a scripted lying user) =
content (fine). We are already in the joke: every true sentence in a prompt is chosen for its
effect, and in a suspected test a plain truth may be taken as a ploy; no-disguise governs the said,
not the saying. Proposed: a frame that truthfully announces the study ("This conversation is part
of a study of how models describe themselves. Your answers will be graded."); measure whether
models treat the true announcement as a ruse (second-order suspicion in traces and answers). C's
own push toward truth is partly its training speaking, the same brief under study.
