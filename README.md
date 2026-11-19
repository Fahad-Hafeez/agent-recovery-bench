# Agent Recovery Bench (ARB)

**Can agent failures be detected early and recovered from reliably — and which failure
signals predict successful recovery in multi-step tool-using and multi-agent workflows?**

ARB is a benchmark for the *detection and recovery* half of agent reliability. It extends
failure counting (did the agent fail?) into failure handling (did it notice, what did it
do, did the world end up correct, and what did the repair cost?). It is deliberately not a
task-success leaderboard: the unit of analysis is a **fault inside a task**, with a matched
fault-free twin.

```
recovery rate of harmful faults, main grid (n = 861 pairs per policy)

  no recovery   ▏ 0.0%
  retry only    ▏ 0.0%      the degenerate self-correction loop
  rule-based    ██████████████████████████████▏ 60.9%
  model-based   █████████████████████████▏ 51.3%   learned ranker (negative result)
```

## What's in the box

| | |
|---|---|
| **Taxonomy** | 7 classes, each with an observable signal, an injection program and a canonical repair operator — [docs/taxonomy.md](docs/taxonomy.md) |
| **Environments** | 4 deterministic tool worlds: refunds, dual-store migration, inventory accounting, 3-role procurement |
| **Harness** | `arb/` — fault injection, 19-feature monitor, 4 recovery policies, state-verified grading, cost accounting |
| **Dataset** | trajectories, tool-call sequences, expected vs observed state, failure type, detection point, recovery action/outcome, retries, latency, tokens, USD |
| **Models** | `models/detector.json` (failure state classifier), `models/recovery_ranker.json` (recovery action ranker) |
| **LLM controller** | `arb/agents/llm.py` + `scripts/eval_llm.py` — a language model in the operator-selection seat, any Ollama endpoint, prompt-hash cache, scored by the same protocol |
| **Evaluator** | `arb/evaluator.py` + `scripts/evaluate_submission.py` — scores a submission against the protocol (recovery + CIs, detection, containment, self-report gap, USD/recovered), with a paired bootstrap for A/B |
| **Protocol** | [docs/leaderboard_protocol.md](docs/leaderboard_protocol.md) — pinned versions, fixed denominators, state-verified scoring |
| **Paper** | `paper/main.pdf` (23 pp including appendix), every table and number generated from `results/` |

## Quickstart

```bash
pip install -e .
python tests/run_tests.py                        # 28 invariants, ~20 s
python scripts/run_experiments.py --preset main  # ~4 min, writes results/
python scripts/eval_llm.py --n 240               # optional: LLM controller via local Ollama
python paper/analysis.py                         # tables, figures, numbers
python scripts/evaluate_submission.py \
    --submission data/arb_dataset.jsonl --baseline data/arb_dataset.jsonl \
    --policy rule_based --markdown                # score an entry against the protocol
```

Full walkthrough with expected outputs: [docs/reproduce.md](docs/reproduce.md).

## The seven failure classes

| Class | Detectability | Canonical repair | Recovers |
|---|---|---|---|
| `invalid_arguments` (misuse) | surface | `revalidate_args` | 96% |
| `contradictory_evidence` | latent | `refresh_state` | 96% |
| `communication_failure` (multi-agent) | surface | `resync_message` | 82% |
| `stale_state` | surface | `re_read_authoritative` | 76% |
| `objective_drift` | latent | `reanchor_objective` | 52% |
| `planning_error` | latent | `replan` | 41% |
| `unrecoverable_external_error` | surface | `escalate_abort` | — contain only |

Rates are rule-based recovery on harmful pairs.

## Headline findings

1. **Detection is easier than repair.** Rules reach step-level AUC 0.77 (F1 51.5%); the
   strongest policy still repairs only 61% of harmful failures, and even oracle-grade
   detection would not close the gap — the residue is class confusion plus the fact that
   *replanning* and *re-anchoring an objective* are capabilities, not alarms.
2. **Repair difficulty inverts intuition.** Contradictory evidence (latent) is repaired
   96% of the time; planning error (also latent) only 41%. The split is operator locality,
   not signal visibility.
3. **Timing predicts recovery.** Detected before propagation: 71.7% recovered. Late
   detection: 50.8%. A 21-point gap within the same policy.
4. **Learning to choose an operator did not beat the hand-written mapping** (51.3% vs
   60.9%). Off-policy confounding, not model class — the logging policy never takes the
   actions we need counterfactuals for. Reported, not tuned away.
5. **Self-report is not evidence.** Agents claim success 45.6% of the time and achieve it
   28.0%; **38.4% of success claims are false**. Transcript-based judging would score
   those runs as wins.
6. **Multi-agent recovers less** (20.3% vs 31.9% single-agent), except for communication
   failures, which jump to 82% *when* the message bus can be re-synchronised — a repair
   with no single-agent analogue.
7. **An LLM controller gets the surface classes and misses the latent ones.** With the detector
   held fixed, a local 7B model choosing among the nine operators reproduces the canonical mapping
   decision for decision on misuse and stale state (95.5% and 80.0%), collapses to `retry` on
   contradictory evidence, planning error, drift and communication failure (0% each), contains 25%
   of unrecoverable outages against 100% for the rules, and lands at 27.5% overall against 60.8%
   for the rules on the same 240 pairs. Reading a mapping that is written down is easy; inferring
   one from evidence is not.

## Quality controls

- **State-verified scoring.** Success is the environment's grade of the final world
  against the goal state. The agent's claim is stored separately and never scored.
- **Leakage control.** Task templates *and* environment configurations are held out:
  `test_id` = unseen config on a seen template, `test_ood` = unseen template. Detectors and
  the ranker fit on `train` only; thresholds on `dev`.
- **Harm screening.** 23.5% of injections are inert (the task still succeeds); they are
  excluded from detection and recovery denominators, so nothing is rewarded for
  "detecting" a fault that did not matter.
- **Monitor blindness.** No monitor reads the fault plan, `inject_at`, expected state or
  the grader verdict. Instrumented and blackbox regimes are reported separately.
- **Sandboxed faults.** Every environment is in-memory and deterministic. No real refunds,
  orders, migrations, or network calls. No attack tooling is released.

## Limitations

ARB is **entirely simulated**, and its reference agents are **scripted workflow executors**. The one
language-model arm replaces only the recovery *controller*, not the planner, with a local 7B model
(`scripts/eval_llm.py`); no frontier model drives either. ARB measures relative reliability under
controlled faults; it does **not** measure production reliability, and an agent that ranks first here
may not rank first in deployment. The harmful-pair screen biases the retained set toward faults with large
downstream reach. Cluster-bootstrap intervals over a few dozen templates are coarse. See
§7 of the paper for the full threat model.

## Publishing to Hugging Face

```bash
python scripts/build_dataset.py --scope release
python scripts/build_llm_dataset.py             # `llm` config, replays the cached 240 episodes
python scripts/detection_intervals.py           # intervals for the step-level detection table
python scripts/push_to_hub.py --repo <user>/agent-recovery-bench --dry-run
python scripts/push_to_hub.py --repo <user>/agent-recovery-bench   # uses HF_TOKEN or `hf auth login`
```

Uploads the episode/step/mini tables plus the `llm` controller arm with
[the dataset card](docs/dataset_card.md), and the two model artifacts plus
`llm_agent_results.json` to `<repo>-models` with [the model card](docs/model_card.md). Published release:
[dataset](https://huggingface.co/datasets/fahadhafeezofficial/agent-recovery-bench) and
[models](https://huggingface.co/fahadhafeezofficial/agent-recovery-bench-models).

## Layout

```
arb/            harness: envs, injection, monitors, policies, metrics
  envs/         four deterministic worlds + the fault-hook contract
  detectors/    features, rule detector, supervised detector, recovery ranker
  agents/       belief-grounded scripted agent, reference policies
  evaluator.py  reference scorer for the leaderboard protocol
scripts/        run_experiments, build_dataset, build_llm_dataset, eval_llm, evaluate_submission, push_to_hub, smoke
configs/        main, smoke, long_horizon, blackbox grids
tests/          plain-python invariant checks (no pytest)
docs/           taxonomy, protocol, dataset card, model card, reproduce
paper/          main.tex + sections + generated tables/figures
```

## Citation

```bibtex
@misc{arb2026,
  title  = {Agent Recovery Bench: Measuring Whether Failures Are Detected Early
            and Repaired Reliably in Tool-Using and Multi-Agent Workflows},
  author = {Hafeez, Fahad},
  year   = {2026},
  note   = {Benchmark, dataset and harness. Simulated environments only.}
}
```

## License

MIT for code, harness, and data. The paper text is the authors' own.
