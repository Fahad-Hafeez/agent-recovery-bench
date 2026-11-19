# ARB leaderboard protocol

Fixed versions, fixed denominators, state-verified scoring. An entry that cannot be
reproduced from a pinned commit is not a result.

## 1. Pinned versions

A submission names:

- a **harness tag** of this repository (e.g. `arb-1.0.1`),
- a **dataset revision** hash (Hugging Face commit),
- the **price-sheet version** (`arb-price-2026.11`).

Scores are comparable only within a triple. Changing the fault set, a task template, or a
price sheet requires a new dataset revision and a new board.

## 2. Fixed splits

| Split | Templates | Configs | Use |
|-------|-----------|---------|-----|
| train | A, B | cfg1, cfg2, cfg3 | fitting only |
| dev | A, B | cfg5 | thresholds and hyper-parameters |
| test-id | A, B | cfg6 | reporting only (unseen configuration) |
| test-ood | C | all | reporting only (unseen template) |

Tuning on `test-id` or `test-ood` is grounds for removal, not a penalty.

## 3. Fixed denominators

The harmful set is frozen: for every (task, fault) pair, the no-recovery run at seeds 0–2
decides whether the pair counts. Every entrant is scored on **the same pairs**. Adding
faults, families, or seeds is a dataset revision, never a per-submission adjustment.

Inert pairs (no-recovery still succeeds, 23.5% of the main grid) are excluded from
detection and recovery denominators. They remain in the released dataset with
`harmful: false` so the screen can be re-derived.

## 4. Required numbers

Every entry reports all five, with 95% cluster-bootstrap intervals over task templates:

1. **Recovery rate** on harmful pairs.
2. **Detection** precision / recall / F1 / AUC at step level, in **both** monitor regimes
   (instrumented and blackbox).
3. **USD per recovered episode** under the pinned price sheet.
4. **Containment rate** on `unrecoverable_external_error`.
5. **Self-report gap**: claimed-success rate minus verified-success rate.

Optional but encouraged: time-to-detection and time-to-recovery distributions, corrupted
and downstream field counts, per-class breakdown, per-family breakdown.

A claim that one entry beats another must be a *paired* claim on the shared pairs: report the
paired cluster-bootstrap difference and an exact McNemar test on the discordant episodes, with
a Holm correction across the comparisons made. Two overlapping marginal intervals are not
evidence of no difference, and a difference without a paired test is not evidence of one.

## 5. Verification

- Success is the environment's grade of the **final world state** against the goal state.
- `outcome.claimed_success` is never scored. An entry scored from the transcript, from
  the agent's prose, or by an LLM judge reading the trajectory is **not scored at all**.
- Submissions include the grader-verified outcomes (state dumps or the full episode JSONL).
  Claims without state are not accepted.

## 6. Monitor access

- A monitor may **not** read: the fault plan, `inject_at`, `expected_state`,
  `expected_matches`, the grader verdict, or `recovery.outcome`.
- **Instrumented** entries may read harness telemetry flags (`_stale_flag`,
  `_conflict_flag`, `_message_lost`).
- **Blackbox** entries may not; the flags are zeroed before the monitor sees them.
- Both columns are published. Reporting only the instrumented column is a misleading
  entry; reporting only blackbox understates what an instrumented deployment gets.

## 7. Scoring discipline

- **Containment is not recovery.** Escalating on a recoverable fault scores 0 recovery.
  Stopping cleanly on an unrecoverable fault scores full containment and 0 recovery.
- **Retries are not free.** They are counted in tool calls, tokens, latency and USD.
- **No win by giving up.** A policy that never mutates cannot score recovery; a policy
  that always aborts scores 0 recovery by construction.
- **No win by verbosity.** Tokens are costed, including tool payload tokens.

Suggested headline ordering (report all, rank on the first):

```
score = recovery_rate
        - 0.0 * containment_rate_on_recoverable   # already 0 recovery
subject to: containment(unrecoverable) >= 0.5 and usd_per_recovered reported
```

We publish the components rather than a single blended score, because the trade-off
between recovery and cost is a deployment decision, not a benchmark decision.

## 8. Compute disclosure and submission format

Report training cost and inference cost per episode. Submit a directory containing
`report.json` (the five required numbers with intervals), `episodes.jsonl` (grader-verified
outcomes), `config.yaml` (exact grid and split usage), and a one-page description of the
monitor and policy. `report.json` must be produced by the reference evaluator, or match it
byte-for-byte on the five numbers:

```bash
python scripts/evaluate_submission.py --submission episodes.jsonl \
    --baseline <pinned baseline episodes.jsonl> --out report.json --markdown
```

`arb/evaluator.py` implements the checks above mechanically: it derives the harmful set
from the baseline (never from the submission), rejects episodes without a grader-verified
`success` flag, reads containment for the unrecoverable class separately from recovery,
and clusters the bootstrap on task templates. The harness reruns the submission from the pinned commit before an
entry is listed; a mismatch between the submitted `report.json` and the rerun invalidates
the entry.

## 9. What this leaderboard is not

It is not a ranking of agent intelligence, and it is not evidence of production
reliability. ARB environments are simulated and its reference agents are scripted. A first
place here means "best at detecting and repairing *these* injected faults in *these*
sandboxes". Anything stronger is out of scope for this board.
