---
license: mit
task_categories:
  - text-generation
  - reinforcement-learning
language:
  - en
tags:
  - agents
  - reliability
  - failure-detection
  - recovery
  - tool-use
  - multi-agent
  - benchmark
pretty_name: Agent Recovery Bench (ARB)
size_categories:
  - 1K<n<10K
configs:
  - config_name: episodes
    data_files: episodes/arb_dataset.parquet
  - config_name: steps
    data_files: steps/arb_steps.parquet
  - config_name: mini
    data_files: mini/arb_mini.parquet
  - config_name: llm
    data_files: llm/arb_llm.parquet
---

# Dataset Card — Agent Recovery Bench (ARB)

## Dataset summary

ARB measures whether agent failures are **detected early** and **recovered from
reliably** in multi-step tool-using and multi-agent workflows. Each row is one episode:
a task, a scheduled (or agent-generated) fault, a monitor, and a recovery policy, with
the full trajectory and the grader-verified outcome.

ARB extends failure *counting* into failure *handling*. It is not a task-success
leaderboard: the unit of analysis is a fault inside a task, with a matched fault-free
twin, and success is read from world state rather than from the agent's report.

- **Curated by:** ARB authors (single lab, synthetic)
- **Language:** English (tool payloads and objectives)
- **License:** MIT for the data and harness
- **Repo:** this project; harness in `arb/`

## Supported tasks

Four deterministic tool environments:

| Family | Track | Domain | Verifiable outcome |
|--------|-------|--------|--------------------|
| `tickets` | single | refund desk | refund total and status match policy arithmetic |
| `migration` | single | dual-store migration | every source record present and equal in the target |
| `inventory` | single | accounting | `on_hand + Σ adjustments == ledger` for every SKU, period closed |
| `procurement` | multi (planner/buyer/verifier) | purchasing | order state and quantities agree across all three roles |

Seven failure classes: `invalid_arguments`, `stale_state`, `contradictory_evidence`,
`planning_error`, `communication_failure`, `objective_drift`,
`unrecoverable_external_error`. See `docs/taxonomy.md`.

## Languages

Tool payloads and objectives are English JSON/text. No natural-language dialogue; the
"user" is a scripted workflow, not a chat partner.

## How to load

Episodes and steps are **separate configs**, not two splits: `datasets` requires every
split of one `DatasetDict` to share a single Arrow schema, and a step table has a different
shape from an episode table.

```python
from datasets import load_dataset

episodes = load_dataset("<user>/agent-recovery-bench", "episodes", split="train")
steps    = load_dataset("<user>/agent-recovery-bench", "steps", split="train")
llm      = load_dataset("<user>/agent-recovery-bench", "llm", split="train")

# or from local files, no hub access:
episodes = load_dataset("parquet", data_files="episodes/arb_dataset.parquet", split="train")
```

Parquet is the canonical artifact (typed and self-describing). A JSONL mirror of each
config ships alongside it, one JSON object per line, for grepping and for tools that
prefer line-delimited input.

### Free-form fields are JSON strings

Tool payloads, projected state, fault payloads, detector posteriors, attempt lists and
step-level label lists differ in shape between tools, families and failure classes. A
formatted dataset needs one schema, so those subtrees are stored as JSON strings with a
`_json` suffix (`args_json`, `data_json`, `expected_state_json`, `observed_state_json`,
`expected_final_state_json`, `payload_json`, `posterior_json`, `attempts_json`,
`violations_json`, `labels_json`, `env_task_json`, `price_sheet_json`,
`policy_config_json`). Every field the protocol scores stays a native scalar.

```python
import json
state = json.loads(episodes[0]["steps"][0]["observed_state_json"])
labels = json.loads(episodes[0]["meta"]["labels_json"])
```

## Dataset structure

### `episodes/arb_dataset.parquet` (JSONL mirror `arb_dataset.jsonl`) — one row per episode

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | `family\|template\|config\|s<seed>\|fault\|policy\|detector\|mode\|src\|horizon` |
| `family`, `track`, `template_id`, `config_id`, `split`, `seed` | | task identity |
| `policy`, `detector`, `monitor_mode` | string | which system produced the trace |
| `steps` | list | tool-call sequence (see below) |
| `fault` | object | `{type, inject_at, payload, propagation_window}` or `null` |
| `detection` | object | `{step, type, score, detected, latency_steps, latency_ms, posterior}` |
| `recovery` | object | `{attempts, outcome, n_attempts, actions, ttr_steps, contained, stop_reason}` |
| `outcome` | object | `{success, violations, claimed_success, retries, tool_calls, n_corrupted_fields, n_downstream_fields, detected, handled_at}` |
| `cost` | object | `{tokens_in, tokens_out, tool_payload_tokens, tokens_total, latency_ms, usd, tool_calls}` |
| `expected_final_state`, `observed_final_state`, `counterfactual_final_state` | object | grader view |
| `meta` | object | `labels`, `fault_source`, `competence`, `horizon`, `price_sheet`, `env_task` |
| `harmful` | bool | added by `build_dataset.py`: task fails under no recovery |

### Each entry of `steps`

`index`, `agent`, `tool`, `args`, `ok`, `error_code`, `data`, `state_version`,
**`expected_state`**, **`observed_state`**, `expected_matches`, `retry_index`,
`signals`, `tokens_in`, `tokens_out`, `tool_payload_tokens`, `latency_ms`, `usd`.

`expected_state_json` is the projection the world would show if the intended action took
effect; `observed_state_json` is what the tool actually returned. Their difference is the
per-step failure signal; `expected_matches` is the boolean form.

### `steps/arb_steps.parquet` (JSONL mirror `arb_steps.jsonl`)

The same data flattened to one row per step, with `episode_id`, `t`, `label` and the
episode's fault/policy attached. Use this for step-level detection modelling.

### `mini/arb_mini.parquet` (JSONL mirror `arb_mini.jsonl`)

A 120-episode subset for tutorials and CI. Same schema as `episodes`.

### `llm/arb_llm.parquet` (JSONL mirror `arb_llm.jsonl`) — the LLM-controller arm

240 episodes with `policy: "llm_based"`: the rules detector raises the alarm, and a local
`qwen2.5-coder:7b` chooses one of the nine operators from a prompt carrying only the
detector's posterior, the monitor feature vector and the operator menu. Same schema as
`episodes`, one row per harmful (task, fault) pair sampled round-robin over failure class
*and* family. `meta.policy_config` records the model, the temperature and that the reply
came from the prompt-hash cache; `recovery.actions` is the operator the model picked.

These are the 240 episodes behind the paper's LLM result (27.5% recovery vs 60.8% for the
canonical mapping on identical pairs). They are a separate config because they are a
different, smaller sample than `episodes`: a controller comparison, not the full grid.
Statistics in `llm/build_stats.json`. To audit a decision, read `detection.posterior`,
the operator in `recovery.actions`, and the step-by-step `expected_state` vs
`observed_state` diff that produced it.

## Dataset creation

Deterministic generation, no human annotation, no web scraping.

1. **Tasks.** (family, template, configuration, seed). Templates A/B are seen during
   fitting; template C is held out entirely (`test_ood`); `cfg6` is an unseen
   configuration (`test_id`); `cfg5` is dev; `cfg1–3` are train.
2. **Faults.** Injected by an explicit program per class, with `inject_at` aligned to the
   step that calls the targeted tool (last read, or first mutation), so a scheduled fault
   is always exercised.
3. **Screening.** Every (task, fault) pair is run with no recovery. Pairs where the task
   still succeeds (23.5% in the main grid) are **inert** and excluded from detection and
   recovery denominators. They are kept in the released data with `harmful: false`.
4. **Policies.** `no_recovery`, `retry_only`, `rule_based`, `model_based` run on the same
   pairs with the same seeds.
5. **Verification.** Outcome is the environment's grade of the final world against the
   goal state. The agent's own success claim is stored separately in
   `outcome.claimed_success` and never used for scoring.
6. **LLM arm.** The `llm` config is exported by `scripts/build_llm_dataset.py`, which
   replays the exact sampled pairs through the harness with every reply taken from
   `results/llm_cache.jsonl`. It refuses to write if a prompt is missing from the cache or
   if the replay's recovery rate disagrees with `results/llm_agent.json`, so the published
   rows are the scored run rather than a re-sample.

### Reference agents

Scripted workflow executors with a known standard operating procedure and **no oracle
access**. Every tool argument is derived from that role's belief, i.e. from observations
it actually received. This is what makes stale reads and dropped messages physically
damaging rather than cosmetic.

## Personal and sensitive information

None. All entities are synthetic (SKU numbers, ticket ids, order ids). No real customers,
orders, refunds, or migrations. No PII, no scraped text.

## Considerations for using the data

**Simulation, not production.** Every environment is deterministic Python. Real tools are
partially observable, non-deterministic and slow; real failures correlate with load,
version skew and human behaviour. ARB supports claims about *relative* reliability under
controlled faults. It does **not** support claims about production recovery rates.

**Scripted agents.** The reference agents are scripted, not LLM agents. The released
numbers characterise the harness and the fault set, not any deployed model. Adding an LLM
agent is a configuration change, not a redesign.

**Harm screening induces selection.** Retaining only harmful pairs biases the set toward
faults with large downstream reach. Reported recovery is therefore "of harmful faults",
which is the honest denominator for repair but is not the average over all injections.

**Self-report is not ground truth.** Do not score an episode from `claimed_success`, from
the transcript, or with an LLM judge reading the trajectory. In the release, 38.4% of
success claims are false.

**Off-policy data.** Recovery actions in the data were chosen by the four policies, not
explored. Training a policy on them inherits that confounding; see the paper's negative
result for the learned ranker.

## Maintenance

Generated by `scripts/build_dataset.py`; statistics in `data/build_stats.json`. The
harness version, price-sheet version and dataset revision must be pinned together for any
comparison.

## Citation

See the repository README.
