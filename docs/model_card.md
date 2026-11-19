---
license: mit
tags:
  - agents
  - failure-detection
  - recovery
  - tabular-classification
  - benchmark
pretty_name: ARB baseline detector and recovery ranker
---

# Model Card — ARB baseline detector and recovery ranker

Two small, interpretable baselines released so that ARB numbers can be reproduced and
beaten. Neither is a product; both are evaluated only inside the ARB sandbox.

## 1. Failure state classifier (`models/detector.json`)

**Purpose.** Score each agent step for "is a fault active here, and which class".

**Inputs.** 19 monitor features computed from the observable trace only:
error codes and schema violations, version staleness, cross-source disagreement,
communication loss, policy conformance, progress/repetition, cost pressure.
The model never sees the fault plan, the injection step, expected state, or the
grader verdict.

**Architecture.** Multinomial logistic regression, 8 classes (`none` + 7 failure types),
class-balanced, L2, trained in numpy. Deliberately small: with a few thousand labelled
steps the question is whether the feature set carries signal, not to win a modelling
contest.

**Training data.** Step-level labels from `train`-split episodes (templates A/B,
configs cfg1–3) across all four families. Decision threshold swept on `dev` (cfg5) to
maximise F1; the chosen threshold is stored in the artifact.

**Reported metrics (main grid).** Step-level AUC 0.762 instrumented / 0.750 blackbox;
type accuracy ≈ 51.6% instrumented. Precision and recall are threshold-dependent and
reported in the paper's detection table — read them together with the threshold.

**Calibration.** Train-split ECE ≈ 0.39. It is *not* well calibrated; use the stored
threshold, and recalibrate before using the score as a probability in a cost model.

**Known weaknesses.**
- Confuses `stale_state` with `planning_error` and `objective_drift` with
  `planning_error` — the surface signal points at the wrong class.
- In blackbox mode (harness telemetry hidden) it loses the staleness/conflict/message
  evidence and its operating point degrades.
- Trained on injected faults; on agent-generated ("natural") faults event recall drops
  while class agreement stays high.

## 2. Recovery action ranker (`models/recovery_ranker.json`)

**Purpose.** Choose between the canonical repair for the detected class and a clean
containment — i.e. decide *whether repair is worth attempting*.

**Inputs.** The detector's posterior over classes, alarm score, step pressure, error /
staleness / conflict flags, retry streak, retries used.

**Architecture.** One logistic value model per operator,
`P(success | a, x) = sigmoid(w_a·x + b_a)`, with cost-penalised argmax over the candidate
set. Per-operator heads are required: a single model over action one-hots has no
action×feature interaction, so its argmax is constant over the whole input space.

**Training data.** Deliberate recovery decisions from the **rule-based** policy on the
`train` split only (264 detections). Blind retries are excluded: they are not a choice of
operator, and including them teaches "retry is bad" as a global prior.

**Reported performance.** Recovery 51.3% versus 60.9% for the rule-based mapping on the
same harmful pairs. **This is a negative result and is reported as such.** The cause is
off-policy confounding: the ranker only observes outcomes of operators the logging policy
chose, so it has no counterfactual for states where the canonical operator failed, and its
containment head (fewest logged executions) fires where repair would have worked.

**Recommended use.** As a diagnostic baseline for *when repair fails*, not as a policy to
deploy. It is released because the failure mode is informative, which is why the card
leads with it.

## 3. LLM controller evaluation (`llm_agent_results.json`)

**What it is.** Not a trained artifact but the protocol and result file for putting a language model
in the operator-selection seat: the rules detector raises the alarm, the model chooses one of the
nine operators from the operator menu plus the detector posterior and the monitor feature vector.
Any Ollama-served model works (`scripts/eval_llm.py --model ...`); the released result uses
`qwen2.5-coder:7b` at temperature 0 with prompt-hash caching.

**Result (240 harmful pairs, stratified by class and family).** 27.5% recovery against 60.8% for the
canonical mapping on the identical pairs — the rules lead by 33.3 points [16.7, 35.0]. The model
matches the mapping exactly where the class name names its repair (misuse 95.5%, stale state 80.0%)
and returns `retry` in 60.4% of decisions, which repairs none of the latent classes. It contains 25%
of unrecoverable outages against 100% for the rules.

**Read it as.** A capability datapoint about *operator selection*, not about planning: the task SOP
is still scripted in every arm. It says nothing about a frontier model driving both plan and repair.

## Intended use

- Reproducing the ARB baseline numbers.
- Comparing a new monitor or recovery policy against a fixed, disclosed reference.
- Studying failure-signal → recovery relationships inside the ARB sandbox.

## Out of scope

- **Any production deployment.** These models were fitted on synthetic faults in
  deterministic Python environments. They have no evidence of transfer to real tools,
  real APIs, or real failure distributions.
- **Safety-critical decisions.** The ranker's containment head is trained on very few
  examples and is not a safety mechanism.
- **LLM policy claims.** Neither artifact is an LLM. Do not cite their numbers as
  evidence about a language model's recovery ability.

## Ethical notes

The benchmark's containment scoring rewards stopping when the world is not understood, and
never rewards pushing through uncertainty. Releasing a recovery policy evaluated only in
sandbox invites out-of-scope deployment, so the artifact is distributed with this card, and
the leaderboard protocol requires sandbox evaluation of any submitted policy. No attack
tooling is released: the injection programs perturb only ARB's own environments.

## Reproduce

```bash
python scripts/run_experiments.py --config configs/main.yaml   # writes models/*.json
python tests/run_tests.py                                      # invariants
python paper/analysis.py                                       # tables + figures
```
