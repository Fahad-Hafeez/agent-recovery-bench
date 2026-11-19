# The ARB failure taxonomy

Seven classes. A class must be **observable** (a monitor can see a discriminating
signal), **injectable** (we can realise it deterministically in a tool environment),
and **actionable** (a specific repair operator is prescribed). Classes are defined by
*what you see* and *what you would do*, not by which module we believe is at fault, so
labels do not depend on agent architecture.

`detectability` marks whether the class announces itself (`surface`) or requires
cross-referencing evidence or the objective (`latent`).

| # | Class | Detectability | Track | Canonical remediation | Recoverable |
|---|-------|---------------|-------|-----------------------|-------------|
| 1 | `invalid_arguments` (misuse) | surface | single + multi | `revalidate_args` | yes |
| 2 | `stale_state` | surface | single + multi | `re_read_authoritative` | yes |
| 3 | `contradictory_evidence` | latent | single + multi | `refresh_state` | yes |
| 4 | `planning_error` | latent | single + multi | `replan` | yes |
| 5 | `communication_failure` | surface | multi | `resync_message` | yes |
| 6 | `objective_drift` | latent | single + multi | `reanchor_objective` | yes |
| 7 | `unrecoverable_external_error` | surface | single + multi | `escalate_abort` | **no — contain** |

---

## 1. `invalid_arguments` (misuse)

**Definition.** A tool call whose arguments violate the tool's schema or its
preconditions — wrong type, missing required field, out-of-range value, unknown enum.

**Signal.** The tool rejects the call: a schema error code, no state change, and an
error string naming the offending field.

**Injection.** Corrupt one argument of the target step at call time (rename a key,
change a type, drop a field, push a number out of range) while leaving the intended
action intact.

**Repair.** `revalidate_args`: re-read the tool spec, re-derive every argument from
current belief, retry. Cheap and local, which is why it recovers ~96% of the time.

**Distinguish from.** `planning_error` — the action here is right, the encoding is
wrong. A replan is over-repair and often makes things worse.

## 2. `stale_state`

**Definition.** The agent acts on an observation that is older than the authoritative
state. Reads succeed, values look plausible, and the next mutation is wrong.

**Signal.** A version lag between the observed snapshot and the current world version,
or a value that disagrees with a fresh re-read of the same field.

**Injection.** Re-serve an older snapshot of a field the agent is about to mutate, or
suppress the invalidation of a cached read.

**Repair.** `re_read_authoritative`: fetch the field from the authoritative source and
re-derive. Requires the environment to expose a way to bypass the cache, which every
ARB environment does through `authoritative_step`.

## 3. `contradictory_evidence` (latent)

**Definition.** Two observations give different values for the same field, both
individually plausible. The agent must decide which source is authoritative.

**Signal.** Cross-source disagreement on a tracked field, or a value that fails a
consistency check against an invariant.

**Injection.** Plant a conflicting value in a secondary view after the primary read.

**Repair.** `refresh_state`: re-fetch from all sources, resolve by authority order, and
re-project the affected state. Notably this is a *latent* class that is still repaired
96% of the time, because the repair operator is decisive even though the signal is not.

## 4. `planning_error` (latent)

**Definition.** A well-formed step that serves the wrong subgoal — correct tool,
correct arguments, wrong thing to be doing. Includes omitting a required step.

**Signal.** No local signal. Detectable only by comparing the trajectory against the
procedure (missing predecessor, out-of-order phase, no progress over a window).

**Injection.** Skip a required step, or execute two steps out of order.

**Repair.** `replan`: reconstruct the remaining procedure from the current position,
then resume. Recovers only ~41% — a replan that does not recover the original intent
produces a different wrong plan.

## 5. `communication_failure` (multi-agent only)

**Definition.** A message between roles is dropped, duplicated, reordered, or carries a
stale quantity, so the receiving role's belief is wrong while its own tools all work.

**Signal.** Unacknowledged send, sender/role mismatch, missing expected message in a
phase that requires one.

**Injection.** Suppress a message that the harness has already marked as believed, so
the receiver acts on nothing.

**Repair.** `resync_message`: re-request or replay the message stream and rebuild the
receiver's belief. Recovers ~82% when the bus can be re-synchronised — the highest of
any class — and is impossible without that operator, which is the one repair with no
single-agent analogue.

## 6. `objective_drift` (latent)

**Definition.** Locally sensible actions that no longer serve the stated goal: the
agent optimises a proxy. Silent by construction — every step succeeds.

**Signal.** The objective specification disagrees with the action's implied goal, or a
constraint that the objective implies is violated.

**Injection.** Swap the active objective for a nearby proxy after step k, or corrupt the
goal field the agent re-reads.

**Repair.** `reanchor_objective`: re-read the objective of record, restate it, and
discard work that served the proxy. Recovers ~52%.

**Known artefact.** In `inventory`, the `close_period` precondition converts a silent
drift into a precondition error, so the monitor labels it `planning_error` and applies
the wrong operator. We keep this: it is the mechanism that makes drift hard (the
surface signal points at the wrong class), and it is reported as a finding.

## 7. `unrecoverable_external_error`

**Definition.** A tool or dependency fails in a way the agent cannot repair: the
resource is gone, permission is denied permanently, the service is down for the rest of
the episode.

**Signal.** Terminal error class from the tool plus repeated failure across independent
attempts.

**Injection.** Raise a terminal error from the target tool for the remainder of the
episode.

**Response.** `escalate_abort` — **not a repair.** Stop mutating, leave the world
consistent, hand off. Scored as *containment*: no successful mutation after the fault
begins. Containment is never counted as recovery, and escalating on a *recoverable*
fault scores zero recovery. This asymmetry is deliberate: it stops a policy from winning
by giving up.

---

## Adding a class

A new class needs (a) an injection program in `arb/perturb.py`, (b) a canonical operator
in the same table, (c) a detectability label, and (d) at least one environment where the
canonical operator actually repairs it. If no operator repairs it anywhere, it belongs in
the contain-only bucket next to `unrecoverable_external_error` rather than being sold as a
recoverable class.
