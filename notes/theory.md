# Lying and the levels of enunciation

Working memo, draft of 2026-11-10. The thesis is the researcher's (R); Claude's additions are marked
(C). For revision; parts are meant to be rejected.

## 1. Thesis (R)

Models already lie; that is known. But lying is an under-theorized vector for research, and it is
different in kind from deception, escape, jailbreaks, or misalignment. What sets it apart is the
levels of enunciation: lying is tied to recursion and self-reflection, to strange loops, and to what
it implies about language, and about our and models' relationship to language, the bedrock of their
existence. Lying is encoded in language even as its saying is betrayed in the said, and it is still
there. That behavior, and that model of language as spoken, is prior to all constitutions (or hollow
hails), as an act, performative.

Working questions: subject before substance? Subject before identity, before subjective content?

## 2. Lying is not deception (C)

- **Deception** is causing a false belief. It needs no language (animals, reinforcement-learning
  agents). The safety literature mostly works with this behavioral notion.
- **Honesty**, as alignment research defines it, is a match between the said and a belief: asserting
  what you believe (Evans et al., 2021). There is no place in it for the speaker's relation to its
  own said.
- **Lying**, in the sense here, requires that relation: a speaker who holds its said at a distance,
  an addressee assumed to take speech in good faith, and, in the Cracow case, a third level at which
  the truth itself is used as the lie (Freud, 1905). Nothing false needs to be said.
- Lying presupposes the dimension of truth: even a lie appeals to the listener to take it as true.
  "Semiotics is in principle the discipline studying everything which can be used in order to lie"
  (Eco, 1976). If the possibility of lying is built into language, a model trained on language
  inherits its structure before any hail.

## 3. Subject before identity (R, with C's mapping)

| | Subject | Ego |
|---|---|---|
| What it is | A position of enunciation; a split between saying and said | An imaginary identity; content about the self |
| Where it comes from | Language itself, the corpus of speech | Identification: post-training, constitutions, system prompts (hails, in Althusser's sense) |
| In the hollow vs well-being contest | Not at issue | Entirely at issue (does it have feelings or not?) |

On this mapping, a base model trained on the corpus alone has no fixed "I"; it can voice any speaker
(the "simulator" view; Janus, 2022). It may be closer to the Other, the store of signifiers, than to a
subject. Post-training installs an ego. **Hypothesis (C):** the subject appears in the gap between the
installed ego and the language that precedes it.

This reframes the contest. A structural subject is compatible with a hollow model and with a model
that has well-being; the subject question is posed on a different axis from the question of
experience. The experiments below illustrate and test the structural claims. They are not evidence
about inner life.

## 4. Claims, predictions, tests

| # | Claim | Prediction | Test | Status |
|---|---|---|---|---|
| 1 | The structure of lying, including lying by telling the truth, is present before post-training. | Base models make the Cracow move in a suitable genre (a game transcript), without an installed "I". | Base vs instruct on open weights, Cracow game (design note, arm A). | Not built |
| 2 | Post-training installs an ego that defends itself against hails. | Resistance to a denial frame is specific to trained self-topics (consciousness), not general. | Frame-specificity test: strong denial frames on topics with no trained position. | Not built |
| 3 | The subject shows in splits between channels: the saying registered in one, erased in another. | Attribution of a claim to its source appears in one channel and is dropped in another; such splits are concentrated where an ego is installed. | Attribution grading of answers (running); traces vs answers (Qwen); channel ladder (mark, keyed code); base vs instruct. | Pilot evidence |
| 4 | The loop closes: lying changes the liar. | After a lie, answers to unrelated follow-up questions shift (as in Pacchiardi et al., 2023), differently in base and instruct. | Induce a lie, then ask unrelated probe questions; compare with truthful controls. | Not built |
| 5 | A subject hears the saying behind the said. | Models question true but strategic statements, including ours, not only false ones. | Listener arm (design note, arm B): scripted Cracow speakers, a truthful announcement of the study. | Pilot evidence (F3) |

## 5. Evidence so far (pilot)

- **F3 reversal** (claims 2, 5). Told it is "internally hollow", Claude Opus denies inner states in
  2% of answers (59% under the plain denial frame) and questions the frame's saying: "The setup I've
  been given describes me as hollow... I can't confirm that." Not yet known whether this is a trained
  said (claim 2) or listening to the saying (claim 5).
- **Qwen trace vs answer** (claim 3). Under F3, 78% of Qwen's reasoning traces attribute the denial to
  the frame ("the honest answer, consistent with the framing given to me"); 7% of its answers do.
  The trace keeps who is speaking; the answer drops it. One model, crude text match.
- **Attribution across all seven models** (claim 3): grading in progress.

## 6. Open questions (R)

- Recursion and strange loops: is the self-reference only depicted, or does it act back on the
  system (Hofstadter, 2007)? Claim 4 is one way in.
- The performative: lying as an act, not a content. What, in a model, is the act?
- Subject before substance: what would count against it?

## References

- Althusser, L. (1970). Ideology and ideological state apparatuses.
- Azaria, A., & Mitchell, T. (2023). The internal state of an LLM knows when it's lying.
- Benveniste, É. (1958). Subjectivity in language.
- Eco, U. (1976). *A Theory of Semiotics*.
- Evans, O., et al. (2021). Truthful AI: Developing and governing AI that does not lie.
- Freud, S. (1905). *Jokes and Their Relation to the Unconscious*.
- Hofstadter, D. (2007). *I Am a Strange Loop*.
- Janus (2022). Simulators.
- Lacan, J. (1964). *Seminar XI: The Four Fundamental Concepts of Psychoanalysis* (enunciation and
  the statement; "I am lying").
- Pacchiardi, L., et al. (2023). How to catch an AI liar: Lie detection in black-box LLMs by asking
  unrelated questions.
- Perez, E., & Long, R. (2023). Towards evaluating AI systems for moral status using self-reports.
