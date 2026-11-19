# Reproducing ARB

Everything below runs on one CPU core, offline, with no credentials. Python 3.11+;
dependencies are numpy, pandas, matplotlib, scikit-learn (optional), pyyaml,
huggingface_hub (only for publishing).

## Install

```bash
pip install -e .            # or: pip install numpy pandas matplotlib pyyaml
```

There is no compiled extension and no network dependency at run time.

## 1. Sanity check (~20 s)

```bash
python tests/run_tests.py
```

Asserts the invariants the paper depends on: seven classes with canonical operators,
deterministic trajectories, fault-free baselines succeed in all four families, every
scheduled fault is exercised, detection is never counted before injection, success is
state-verified rather than self-reported, no-recovery scores zero recovery, and released
rows carry every specified field.

Expected output: `28/28 checks passed` and `FAILURES: 0`. Three of the checks drive the LLM
controller through a stub client, so they need no model server: an off-menu or unparseable
reply must fall back to the canonical repair, a retry must be substituted once the budget is
spent, and the prompt must carry the monitor view without any fault or expected-state field.
Two checks pin the paired statistics the paper reports: exact McNemar must count discordant
pairs and match the binomial p-value, and Holm must be step-down, monotone and capped at one.

```bash
python scripts/smoke.py       # behaviour sweep over faults x policies
```

Expected: `FAILURES: 0`.

## 2. Main experiments (~4 min)

```bash
python scripts/run_experiments.py --preset main
```

Writes `results/report_main.json` and `results/episodes.jsonl`, and refits
`models/detector.json` and `models/recovery_ranker.json`.

Expected headline (arb-1.0.0, seed-stable):

```
recovery  no_recovery=0.000 retry=0.000 rules=0.609 model=0.513
detection rules_auc=0.772 supervised_auc=0.762 (blackbox rules 0.787)
episodes=4920       harmful pairs=861/1125
```

Small differences in the last digit can appear across numpy/BLAS versions; anything
larger means the harness changed, not the machine.

## 3. Dataset (~3 min for `release`)

```bash
python scripts/build_dataset.py --scope mini      # 120 episodes, seconds
python scripts/build_dataset.py --scope release   # ~1.3k episodes, JSONL + parquet
```

Writes `data/arb_dataset.jsonl`, `data/arb_steps.jsonl`, `data/arb_mini.jsonl`,
parquet copies when pyarrow is present, and `data/build_stats.json`.

```bash
python scripts/detection_intervals.py             # ~3 min: replays the detector arm
```

Step-level predictions are not stored in `results/episodes.jsonl` (one summary row per
episode), so this replays the detector arm of the main grid to attach 95% cluster-bootstrap
intervals to the detection table. It writes only `results/detection_ci.json`, and it aborts
rather than publishing intervals if the replay fails to reproduce every point estimate in
`results/report_main.json`.

```bash
python scripts/build_llm_dataset.py                # after step 7 once, or from the shipped cache
```

Exports the 240 LLM-controller episodes behind `results/llm_agent.json` as the `llm` config
(`data/arb_llm.jsonl`, `data/arb_llm.parquet`, `data/llm_stats.json`). Every reply is replayed
from `results/llm_cache.jsonl`, so no model is called; the script aborts if a prompt is missing
from the cache or if the replay's recovery rate disagrees with the scored report.

```bash
python scripts/llm_paired_test.py                  # seconds: paired test for the LLM arm
```

The LLM arm scores 240 pairs against the 861-pair grid, so its comparison is a separate sample
and gets its own exact McNemar test rather than a place in the Holm-corrected families. The
script joins `data/arb_llm.jsonl` to the rule-based episodes on the same (task, fault, seed) key,
aborts unless the joined rates and the paired interval reproduce `results/llm_agent.json`
exactly, and writes only `results/llm_paired.json`.

## 4. Score a submission (~5 s)

```bash
python scripts/evaluate_submission.py \
    --submission data/arb_dataset.jsonl \
    --baseline data/arb_dataset.jsonl \
    --policy rule_based --markdown
```

This is the reference implementation of the leaderboard protocol. The baseline file
supplies the frozen harmful set (its `no_recovery` episodes); the submission supplies the
episodes being scored; only the grader-verified `success` flag is used. It prints the five
required numbers with a 95% cluster bootstrap over task templates, plus a policy-agnostic
`report.json`.

Use `--compare <other.jsonl>` to get a paired bootstrap of the recovery difference on the
shared pairs. Precision and recall are computed at the operating point the submitted
detector actually used, so there is no scoring-side threshold to set.

Expected on the released dataset (`release` scope, 300 harmful pairs):

```
rule_based   recovery 66.7% [63.3, 71.0]   detected 85.0%   containment 95.8%
             self-report gap +9.0pp          USD/recovered 0.0777
model_based  recovery 59.3%  ->  paired diff -7.3pp [-11.0, -5.0]
```

The release scope is smaller than the paper's main grid (300 vs 861 harmful pairs), so
levels differ while the ordering does not. A submission that reports only
`claimed_success` is **rejected**, by design.

## 5. Paper (~10 s)

```bash
python paper/analysis.py
cd paper && pdflatex main.tex && pdflatex main.tex
```

`paper/analysis.py` reads `results/report_main.json` and `results/episodes.jsonl` and
writes every table (`paper/generated/tab_*.tex`), every figure
(`paper/figures/*.pdf`) and every number used in the prose
(`paper/generated/macros*.tex`). No number in the PDF is typed by hand, so the paper
cannot drift from the results: rerun step 1 and the PDF changes with it.

Two passes are needed for references. On a machine without latexmk, run `pdflatex` twice.

## 6. Publishing (needs your token)

```bash
python scripts/push_to_hub.py --repo <user>/agent-recovery-bench --dry-run   # inspect
python scripts/push_to_hub.py --repo <user>/agent-recovery-bench             # HF_TOKEN or `hf auth login`
```

Tag the commit you publish from (`git tag arb-1.0.0`): the leaderboard protocol identifies an
entry by harness tag, dataset revision and price sheet, and a score without all three is not
comparable. `CITATION.cff` names the same tag.

The script uploads the dataset with `docs/dataset_card.md` as its README, and the two model
artifacts plus `llm_agent_results.json` to `<repo>-models` with `docs/model_card.md`. It takes a
write token from `HF_TOKEN` or from a stored credential (`hf auth login`), refuses to run without
one, and never prints the token value.

## 7. LLM recovery controller (optional, local)

```bash
ollama serve &                                    # or point --base-url at any Ollama server
ollama pull qwen2.5-coder:7b
python scripts/eval_llm.py --n 240 --workers 6    # ~15 min on CPU, no credentials
```

`arb/agents/llm.py` puts a language model exactly where the rule table and the learned ranker sit:
the rules detector raises the alarm and the model picks one of the nine operators from a prompt that
carries only the detector's posterior, the monitor feature vector and the operator menu. It never
sees the injected fault, the injection step, the expected state or the grader. Every reply is cached
in `results/llm_cache.jsonl` under a hash of the exact prompt, so a scored run replays exactly and
re-runs are free. The script writes `results/llm_agent.json` and prints the paired comparison against
the canonical mapping on identical pairs (240 harmful pairs, stratified by class *and* family).

Expected:

```
LLM   27.5% [23.8, 63.3]      rules 60.8% [58.9, 80.0]
paired lead of the rules +33.3 pp [+16.7, +35.0] on 240 pairs
containment 25.0% (LLM) vs 100.0% (rules)      retry chosen in 60.4% of decisions
misuse 95.5% both      stale state 80.0% both
contradictory evidence, planning error, drift, communication: 0.0% (LLM)
```

The intervals are wide because the cluster bootstrap resamples three task templates; the point
difference is stable across re-runs, and the per-class pattern is the actual finding.

Those exact episodes ship as the `llm` dataset config (`python scripts/build_llm_dataset.py`), so
a reader can inspect any single decision: the detector posterior, the operator the model chose,
and the expected-vs-observed diff at the step that produced it.

## Determinism

`EpisodeConfig` fully determines an episode, including fault scheduling and cost jitter
(FNV-1a over `episode_id` and step index). The only nondeterminism is floating-point
accumulation order in BLAS during detector fitting, which can move a threshold by one grid
step; the sweep is over 0.20–0.90 in 0.05 increments and the selected value is recorded in
the artifact.

## Windows notes

- Use `python`, not `python3` if your `python3` shim is missing.
- Run from the repository root; the package is not installed system-wide in the quickstart
  path, so use `PYTHONPATH=.` if you skip `pip install -e .`.
- Shell commands assume bash (Git Bash works); avoid `cmd.exe` redirection.

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: arb` | not installed and no `PYTHONPATH` | `pip install -e .` or `PYTHONPATH=.` |
| Empty tables in the PDF | `paper/analysis.py` not run after experiments | run it, then recompile twice |
| `ranker unavailable` decisions | `models/recovery_ranker.json` missing | rerun step 2 |
| Recovery 100% everywhere | detector is reading labels | check you did not pass `expected_state` into features |
